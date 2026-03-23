#include "gstnvmmsink.h"
#include "gstnvmmallocator.h"
#include "nvmm_buffer.hpp"
#include "nvmm_types.hpp"

#ifdef NVMM_MOCK_API
#include "nvbufsurface_mock.h"
#else
#include <nvbufsurface.h>
#endif

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

#include "shm_protocol.h"

typedef NvmmShmHeader ShmHeader;

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

    /* Resize shm to match actual frame size */
    gsize needed = sizeof(ShmHeader) + GST_VIDEO_INFO_SIZE(&self->priv->video_info);
    if (self->priv->shm_ptr && needed > self->priv->shm_size) {
        munmap(self->priv->shm_ptr, self->priv->shm_size);
        self->priv->shm_ptr = nullptr;

        self->priv->shm_size = needed;
        if (ftruncate(self->priv->shm_fd, self->priv->shm_size) < 0) {
            GST_ERROR_OBJECT(self, "ftruncate resize failed");
            return FALSE;
        }
        self->priv->shm_ptr = mmap(NULL, self->priv->shm_size,
                                    PROT_READ | PROT_WRITE, MAP_SHARED,
                                    self->priv->shm_fd, 0);
        if (self->priv->shm_ptr == MAP_FAILED) {
            GST_ERROR_OBJECT(self, "mmap resize failed");
            self->priv->shm_ptr = nullptr;
            return FALSE;
        }
    }

    GST_INFO_OBJECT(self, "Configured: %dx%d format=%s shm=%" G_GSIZE_FORMAT,
                    GST_VIDEO_INFO_WIDTH(&self->priv->video_info),
                    GST_VIDEO_INFO_HEIGHT(&self->priv->video_info),
                    gst_video_format_to_string(
                        GST_VIDEO_INFO_FORMAT(&self->priv->video_info)),
                    self->priv->shm_size);
    return TRUE;
}

static gboolean
gst_nvmm_sink_start(GstBaseSink *sink)
{
    auto *self = GST_NVMM_SINK(sink);

    if (self->priv->shm_name.empty()) {
        self->priv->shm_name = "/nvmm_sink_0";
    }

    /* Initial size: header only. set_caps will resize to match actual frame. */
    self->priv->shm_size = sizeof(ShmHeader) + (640 * 480 * 4);

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
    header->magic = NVMM_SHM_MAGIC;
    header->version = NVMM_SHM_VERSION;
    header->width = GST_VIDEO_INFO_WIDTH(&self->priv->video_info);
    header->height = GST_VIDEO_INFO_HEIGHT(&self->priv->video_info);
    header->format = static_cast<uint32_t>(
        GST_VIDEO_INFO_FORMAT(&self->priv->video_info));
    header->frame_number = self->priv->frame_number.fetch_add(1);
    header->timestamp_ns = GST_BUFFER_PTS(buffer);
    header->dmabuf_fd = -1;

    /* Export DMA-buf fd if requested.
       On Jetson, bufferDesc contains the DMA-buf fd for SURFACE_ARRAY memory.
       Only attempt this for NVMM buffers — check both our allocator
       and the negotiated caps feature. */
    if (self->priv->export_dmabuf) {
        gboolean is_nvmm = gst_is_nvmm_memory(mem);

        /* Check caps feature for external NVMM (nvvidconv, nvv4l2) */
        if (!is_nvmm) {
            GstCaps *pad_caps = gst_pad_get_current_caps(
                GST_BASE_SINK_PAD(sink));
            if (pad_caps) {
                GstCapsFeatures *feat = gst_caps_get_features(pad_caps, 0);
                is_nvmm = feat && gst_caps_features_contains(feat, "memory:NVMM");
                gst_caps_unref(pad_caps);
            }
        }

        if (is_nvmm) {
            NvBufSurface *nvsurf = nullptr;
            if (gst_is_nvmm_memory(mem)) {
                nvsurf = static_cast<NvBufSurface *>(
                    gst_nvmm_memory_get_surface(mem));
            } else {
                /* NVIDIA convention: mapped data = NvBufSurface* */
                GstMapInfo dma_map;
                if (gst_buffer_map(buffer, &dma_map, GST_MAP_READ)) {
                    nvsurf = reinterpret_cast<NvBufSurface *>(dma_map.data);
                    gst_buffer_unmap(buffer, &dma_map);
                }
            }
            if (nvsurf && nvsurf->surfaceList) {
                int32_t fd = static_cast<int32_t>(
                    nvsurf->surfaceList[0].bufferDesc);
                if (fd > 0) header->dmabuf_fd = fd;
            }
        }
    }

    header->num_planes = GST_VIDEO_INFO_N_PLANES(&self->priv->video_info);
    for (guint i = 0; i < header->num_planes && i < 4; i++) {
        header->pitches[i] = GST_VIDEO_INFO_PLANE_STRIDE(&self->priv->video_info, i);
        header->offsets[i] = GST_VIDEO_INFO_PLANE_OFFSET(&self->priv->video_info, i);
    }

    /* Copy frame data to shared memory.
       Try per-plane NVMM map first, fall back to gst_buffer_map for CPU buffers. */
    gsize available = self->priv->shm_size - sizeof(ShmHeader);
    gsize total_copied = 0;

    if (gst_is_nvmm_memory(mem)) {
        /* NVMM: map each plane individually (planes aren't contiguous) */
        for (guint p = 0; p < header->num_planes && p < 4; p++) {
            guint8 *plane_data = nullptr;
            gsize plane_size = 0;
            if (gst_nvmm_memory_map_plane(mem, p, GST_MAP_READ,
                                           &plane_data, &plane_size)) {
                gsize copy_size = plane_size;
                if (total_copied + copy_size > available)
                    copy_size = available - total_copied;
                memcpy(frame_data + total_copied, plane_data, copy_size);
                total_copied += copy_size;
            }
        }
        gst_nvmm_memory_unmap_plane(mem);
    } else {
        /* CPU memory: standard flat map */
        if (gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
            gsize copy_size = map_info.size;
            if (copy_size > available) copy_size = available;
            memcpy(frame_data, map_info.data, copy_size);
            total_copied = copy_size;
            gst_buffer_unmap(buffer, &map_info);
        } else {
            GST_WARNING_OBJECT(self, "Failed to map buffer");
        }
    }
    header->data_size = static_cast<uint32_t>(total_copied);

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
    "LGPL",
    "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp"
)
