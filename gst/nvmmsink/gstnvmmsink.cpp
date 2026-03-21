#include "gstnvmmsink.h"
#include "gstnvmmallocator.h"
#include "nvmm_buffer.hpp"
#include "nvmm_types.hpp"

#include <cstring>
#include <string>
#include <atomic>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

/// Shared memory layout:
///   [ ShmHeader ][ frame data (dataSize bytes) ]
///
/// ShmHeader contains metadata for the consumer to interpret the frame.
namespace {

struct ShmHeader {
    uint32_t magic;           // 0x4E564D4D ("NVMM")
    uint32_t version;         // protocol version
    uint32_t width;
    uint32_t height;
    uint32_t format;          // NvBufSurfaceColorFormat
    uint32_t data_size;       // size of frame data following this header
    uint32_t num_planes;
    uint32_t pitches[4];      // per-plane pitch
    uint32_t offsets[4];      // per-plane offset
    int32_t  dmabuf_fd;       // DMA-buf fd (-1 if not exported)
    uint64_t frame_number;    // monotonic frame counter
    uint64_t timestamp_ns;    // PTS in nanoseconds
    uint32_t ready;           // set to 1 when frame is written
    uint32_t _reserved[8];
};

static constexpr uint32_t kShmMagic = 0x4E564D4D;
static constexpr uint32_t kShmVersion = 1;

}  // namespace

enum {
    PROP_0,
    PROP_SHM_NAME,
    PROP_EXPORT_DMABUF,
    PROP_SYNC,
};

struct _GstNvmmSinkPrivate {
    std::string shm_name;
    gboolean export_dmabuf;
    int shm_fd;
    void *shm_ptr;
    gsize shm_size;
    std::atomic<uint64_t> frame_number;
    GstVideoInfo video_info;
};

G_DEFINE_TYPE_WITH_PRIVATE(GstNvmmSink, gst_nvmm_sink, GST_TYPE_BASE_SINK)

static void
gst_nvmm_sink_set_property(GObject *object, guint prop_id,
                            const GValue *value, GParamSpec *pspec)
{
    auto *self = GST_NVMM_SINK(object);
    switch (prop_id) {
        case PROP_SHM_NAME:
            self->priv->shm_name = g_value_get_string(value) ? g_value_get_string(value) : "";
            break;
        case PROP_EXPORT_DMABUF:
            self->priv->export_dmabuf = g_value_get_boolean(value);
            break;
        case PROP_SYNC:
            g_object_set(object, "sync", g_value_get_boolean(value), NULL);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void
gst_nvmm_sink_get_property(GObject *object, guint prop_id,
                            GValue *value, GParamSpec *pspec)
{
    auto *self = GST_NVMM_SINK(object);
    switch (prop_id) {
        case PROP_SHM_NAME:
            g_value_set_string(value, self->priv->shm_name.c_str());
            break;
        case PROP_EXPORT_DMABUF:
            g_value_set_boolean(value, self->priv->export_dmabuf);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static gboolean
gst_nvmm_sink_set_caps(GstBaseSink *sink, GstCaps *caps)
{
    auto *self = GST_NVMM_SINK(sink);

    if (!gst_video_info_from_caps(&self->priv->video_info, caps)) {
        GST_ERROR_OBJECT(self, "Failed to parse caps");
        return FALSE;
    }

    GST_INFO_OBJECT(self, "Configured: %dx%d format=%s",
                    GST_VIDEO_INFO_WIDTH(&self->priv->video_info),
                    GST_VIDEO_INFO_HEIGHT(&self->priv->video_info),
                    gst_video_format_to_string(
                        GST_VIDEO_INFO_FORMAT(&self->priv->video_info)));
    return TRUE;
}

static gboolean
gst_nvmm_sink_start(GstBaseSink *sink)
{
    auto *self = GST_NVMM_SINK(sink);

    if (self->priv->shm_name.empty()) {
        self->priv->shm_name = "/nvmm_sink_0";
    }

    /* Estimate max shm size: header + generous frame buffer (4K RGBA) */
    self->priv->shm_size = sizeof(ShmHeader) + (3840 * 2160 * 4);

    self->priv->shm_fd = shm_open(self->priv->shm_name.c_str(),
                                   O_CREAT | O_RDWR, 0666);
    if (self->priv->shm_fd < 0) {
        GST_ERROR_OBJECT(self, "shm_open(%s) failed: %s",
                         self->priv->shm_name.c_str(), strerror(errno));
        return FALSE;
    }

    if (ftruncate(self->priv->shm_fd, self->priv->shm_size) < 0) {
        GST_ERROR_OBJECT(self, "ftruncate failed: %s", strerror(errno));
        close(self->priv->shm_fd);
        self->priv->shm_fd = -1;
        return FALSE;
    }

    self->priv->shm_ptr = mmap(NULL, self->priv->shm_size,
                                PROT_READ | PROT_WRITE, MAP_SHARED,
                                self->priv->shm_fd, 0);
    if (self->priv->shm_ptr == MAP_FAILED) {
        GST_ERROR_OBJECT(self, "mmap failed: %s", strerror(errno));
        close(self->priv->shm_fd);
        self->priv->shm_fd = -1;
        self->priv->shm_ptr = nullptr;
        return FALSE;
    }

    memset(self->priv->shm_ptr, 0, self->priv->shm_size);
    self->priv->frame_number = 0;

    GST_INFO_OBJECT(self, "Shared memory '%s' created (%" G_GSIZE_FORMAT " bytes)",
                    self->priv->shm_name.c_str(), self->priv->shm_size);
    return TRUE;
}

static gboolean
gst_nvmm_sink_stop(GstBaseSink *sink)
{
    auto *self = GST_NVMM_SINK(sink);

    if (self->priv->shm_ptr && self->priv->shm_ptr != MAP_FAILED) {
        munmap(self->priv->shm_ptr, self->priv->shm_size);
        self->priv->shm_ptr = nullptr;
    }

    if (self->priv->shm_fd >= 0) {
        close(self->priv->shm_fd);
        self->priv->shm_fd = -1;
    }

    shm_unlink(self->priv->shm_name.c_str());

    GST_INFO_OBJECT(self, "Shared memory '%s' destroyed",
                    self->priv->shm_name.c_str());
    return TRUE;
}

static GstFlowReturn
gst_nvmm_sink_render(GstBaseSink *sink, GstBuffer *buffer)
{
    auto *self = GST_NVMM_SINK(sink);
    auto *header = static_cast<ShmHeader *>(self->priv->shm_ptr);
    auto *frame_data = static_cast<uint8_t *>(self->priv->shm_ptr) + sizeof(ShmHeader);
    GstMemory *mem;
    GstMapInfo map_info;

    if (!self->priv->shm_ptr) {
        GST_ERROR_OBJECT(self, "Shared memory not initialized");
        return GST_FLOW_ERROR;
    }

    mem = gst_buffer_peek_memory(buffer, 0);

    /* Mark frame as not ready while writing */
    header->ready = 0;
    __sync_synchronize();

    /* Fill header */
    header->magic = kShmMagic;
    header->version = kShmVersion;
    header->width = GST_VIDEO_INFO_WIDTH(&self->priv->video_info);
    header->height = GST_VIDEO_INFO_HEIGHT(&self->priv->video_info);
    header->format = static_cast<uint32_t>(
        GST_VIDEO_INFO_FORMAT(&self->priv->video_info));
    header->frame_number = self->priv->frame_number.fetch_add(1);
    header->timestamp_ns = GST_BUFFER_PTS(buffer);
    header->dmabuf_fd = -1;

    /* Export DMA-buf fd if requested and memory is NVMM */
    if (self->priv->export_dmabuf && gst_is_nvmm_memory(mem)) {
        int fd = -1;
        gint gfd = -1;
        void *surface = gst_nvmm_memory_get_surface(mem);
        if (surface) {
            /* Try to get fd — on mock this returns a fake fd */
            GstMemory *nvmem = mem;
            /* Use the allocator API */
            gboolean ok = FALSE;
            /* Direct NvBufSurfaceGetFd via the public API */
            /* For now, store -1; real implementation will call
               gst_nvmm_memory_get_fd() once available on the C++ side */
            header->dmabuf_fd = -1;
            (void)fd;
            (void)gfd;
            (void)nvmem;
            (void)ok;
        }
    }

    /* Map buffer and copy frame data to shared memory */
    if (gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
        gsize copy_size = map_info.size;
        gsize available = self->priv->shm_size - sizeof(ShmHeader);

        if (copy_size > available)
            copy_size = available;

        header->data_size = static_cast<uint32_t>(copy_size);
        header->num_planes = GST_VIDEO_INFO_N_PLANES(&self->priv->video_info);

        for (guint i = 0; i < header->num_planes && i < 4; i++) {
            header->pitches[i] = GST_VIDEO_INFO_PLANE_STRIDE(&self->priv->video_info, i);
            header->offsets[i] = GST_VIDEO_INFO_PLANE_OFFSET(&self->priv->video_info, i);
        }

        memcpy(frame_data, map_info.data, copy_size);
        gst_buffer_unmap(buffer, &map_info);
    } else {
        GST_WARNING_OBJECT(self, "Failed to map buffer");
        header->data_size = 0;
    }

    /* Signal frame is ready */
    __sync_synchronize();
    header->ready = 1;

    return GST_FLOW_OK;
}

static void
gst_nvmm_sink_finalize(GObject *object)
{
    auto *self = GST_NVMM_SINK(object);
    self->priv->~_GstNvmmSinkPrivate();
    G_OBJECT_CLASS(gst_nvmm_sink_parent_class)->finalize(object);
}

static void
gst_nvmm_sink_class_init(GstNvmmSinkClass *klass)
{
    auto *gobject_class = G_OBJECT_CLASS(klass);
    auto *element_class = GST_ELEMENT_CLASS(klass);
    auto *basesink_class = GST_BASE_SINK_CLASS(klass);

    gobject_class->set_property = gst_nvmm_sink_set_property;
    gobject_class->get_property = gst_nvmm_sink_get_property;
    gobject_class->finalize = gst_nvmm_sink_finalize;

    g_object_class_install_property(gobject_class, PROP_SHM_NAME,
        g_param_spec_string("shm-name", "Shared Memory Name",
            "POSIX shared memory segment name (e.g., /nvmm_sink_0)",
            "/nvmm_sink_0", G_PARAM_READWRITE));

    g_object_class_install_property(gobject_class, PROP_EXPORT_DMABUF,
        g_param_spec_boolean("export-dmabuf", "Export DMA-buf",
            "Export NVMM buffer as DMA-buf fd in shared memory header",
            FALSE, G_PARAM_READWRITE));

    gst_element_class_set_static_metadata(element_class,
        "NVMM Shared Memory Sink",
        "Sink/Video",
        "Exports NVMM video frames via POSIX shared memory for zero-copy IPC",
        "Pavel Guzenfeld");

    GstCaps *caps = gst_caps_from_string(
        "video/x-raw(memory:NVMM), "
        "format=(string){NV12, RGBA, I420, BGRA}, "
        "width=(int)[1, 8192], height=(int)[1, 8192], "
        "framerate=(fraction)[0/1, 240/1]; "
        "video/x-raw, "
        "format=(string){NV12, RGBA, I420, BGRA}, "
        "width=(int)[1, 8192], height=(int)[1, 8192], "
        "framerate=(fraction)[0/1, 240/1]");
    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps));
    gst_caps_unref(caps);

    basesink_class->set_caps = gst_nvmm_sink_set_caps;
    basesink_class->start = gst_nvmm_sink_start;
    basesink_class->stop = gst_nvmm_sink_stop;
    basesink_class->render = gst_nvmm_sink_render;
}

static void
gst_nvmm_sink_init(GstNvmmSink *self)
{
    self->priv = static_cast<GstNvmmSinkPrivate *>(
        gst_nvmm_sink_get_instance_private(self));
    new (self->priv) GstNvmmSinkPrivate();
    self->priv->shm_name = "/nvmm_sink_0";
    self->priv->export_dmabuf = FALSE;
    self->priv->shm_fd = -1;
    self->priv->shm_ptr = nullptr;
    self->priv->shm_size = 0;
    self->priv->frame_number = 0;
}

/* Plugin registration */
static gboolean
plugin_init(GstPlugin *plugin)
{
    return gst_element_register(plugin, "nvmmsink",
                                GST_RANK_NONE, GST_TYPE_NVMM_SINK);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvmmsink,
    "NVMM shared memory sink for zero-copy IPC",
    plugin_init,
    "0.1.0",
    "MIT",
    "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp"
)
