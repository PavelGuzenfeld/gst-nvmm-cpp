/// nvmmsink — GPU-copy NVMM IPC sink.
///
/// Allocates a pool of NVMM buffers, copies incoming frames into them via
/// NvBufSurfTransform (VIC, GPU-to-GPU, no CPU involvement; handles the
/// BLOCK_LINEAR -> PITCH_LINEAR conversion upstream NVMM producers need), and
/// shares the pool buffer DMA-buf fds with consumers via SCM_RIGHTS over a unix
/// domain socket.
///
/// Consumers (nvmmappsrc) import the fds and read directly from GPU memory
/// (zero-copy on the consumer side). Ref counts in shared memory manage buffer
/// lifecycle.

#include "gstnvmmsink.h"
#include "gstnvmmallocator.h"

#ifdef NVMM_MOCK_API
#include "nvbufsurface_mock.h"
#else
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#endif

#include <cstring>
#include <string>
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <poll.h>

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

#include "shm_protocol.h"
#include "fd_ipc.h"
#include "nvmm_caps.h"

typedef NvmmShmHeader ShmHeader;

enum {
    PROP_0,
    PROP_SHM_NAME,
    PROP_POOL_SIZE_PROP,
};

struct PoolBuffer {
    NvBufSurface *surface;
    int fd;  /* bufferDesc = DMA-buf fd */
    NvBufSurfaceMapParams map_params;  /* serializable params for cross-process import */
};

struct _GstNvmmSinkPrivate {
    std::string shm_name;
    int shm_fd;
    void *shm_ptr;
    gsize shm_size;
    std::atomic<uint64_t> frame_number;
    GstVideoInfo video_info;

    /* Buffer pool */
    PoolBuffer pool[NVMM_POOL_SIZE];
    int pool_size;
    int write_idx;
    gboolean pool_allocated;

    /* Socket server for fd passing */
    std::string socket_path;
    int listen_fd;
    std::thread accept_thread;
    std::atomic<bool> running;
    std::vector<int> client_fds;
    std::mutex clients_mutex;
};

G_DEFINE_TYPE_WITH_PRIVATE(GstNvmmSink, gst_nvmm_sink, GST_TYPE_BASE_SINK)

/* --- helpers --- */

static NvBufSurface *
get_nvbufsurface_from_buffer(GstBaseSink *sink, GstBuffer *buffer)
{
    /* Check caps for NVMM feature */
    GstCaps *caps = gst_pad_get_current_caps(GST_BASE_SINK_PAD(sink));
    if (!caps) return nullptr;
    GstCapsFeatures *feat = gst_caps_get_features(caps, 0);
    gboolean is_nvmm = feat && gst_caps_features_contains(feat, "memory:NVMM");
    gst_caps_unref(caps);
    if (!is_nvmm) return nullptr;

    /* NVIDIA convention: mapped data = NvBufSurface* */
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ))
        return nullptr;
    NvBufSurface *surf = reinterpret_cast<NvBufSurface *>(map.data);
    gst_buffer_unmap(buffer, &map);
    return surf;
}

static gboolean
allocate_pool(GstNvmmSink *self)
{
    auto *priv = self->priv;
    int w = GST_VIDEO_INFO_WIDTH(&priv->video_info);
    int h = GST_VIDEO_INFO_HEIGHT(&priv->video_info);

    NvBufSurfaceCreateParams params;
    memset(&params, 0, sizeof(params));
    params.gpuId = 0;
    params.width = w;
    params.height = h;
    params.size = 0;
    params.isContiguous = true;
    params.layout = NVBUF_LAYOUT_PITCH;
    params.memType = NVBUF_MEM_SURFACE_ARRAY;

    /* Map GstVideoFormat to NvBufSurfaceColorFormat */
    GstVideoFormat fmt = GST_VIDEO_INFO_FORMAT(&priv->video_info);
    switch (fmt) {
        case GST_VIDEO_FORMAT_NV12: params.colorFormat = NVBUF_COLOR_FORMAT_NV12; break;
        case GST_VIDEO_FORMAT_RGBA: params.colorFormat = NVBUF_COLOR_FORMAT_RGBA; break;
        case GST_VIDEO_FORMAT_BGRA: params.colorFormat = NVBUF_COLOR_FORMAT_BGRA; break;
        case GST_VIDEO_FORMAT_I420: params.colorFormat = NVBUF_COLOR_FORMAT_YUV420; break;
        default:
            GST_ERROR_OBJECT(self, "Unsupported format for pool: %s",
                             gst_video_format_to_string(fmt));
            return FALSE;
    }

    for (int i = 0; i < priv->pool_size; i++) {
        NvBufSurface *surf = nullptr;
        if (NvBufSurfaceCreate(&surf, 1, &params) != 0 || !surf) {
            GST_ERROR_OBJECT(self, "Failed to create pool buffer %d", i);
            return FALSE;
        }
        /* NvBufSurfaceCreate leaves numFilled = 0; set it so NvBufSurfTransform
           (render path) accepts buffer index 0 as a valid destination. Mirrors
           nvmm_buffer.cpp. */
        surf->numFilled = surf->batchSize ? surf->batchSize : 1;
        priv->pool[i].surface = surf;
        priv->pool[i].fd = (int)surf->surfaceList[0].bufferDesc;
        memset(&priv->pool[i].map_params, 0, sizeof(NvBufSurfaceMapParams));
        if (NvBufSurfaceGetMapParams(surf, 0, &priv->pool[i].map_params) != 0) {
            fprintf(stderr, "[nvmmsink] NvBufSurfaceGetMapParams failed for buffer %d\n", i);
            return FALSE;
        }
    }

    priv->pool_allocated = TRUE;
    return TRUE;
}

static void
destroy_pool(GstNvmmSink *self)
{
    auto *priv = self->priv;
    for (int i = 0; i < priv->pool_size; i++) {
        if (priv->pool[i].surface) {
            NvBufSurfaceDestroy(priv->pool[i].surface);
            priv->pool[i].surface = nullptr;
            priv->pool[i].fd = -1;
        }
    }
    priv->pool_allocated = FALSE;
}

static void
send_fds_to_client(GstNvmmSink *self, int client_fd)
{
    auto *priv = self->priv;

    /* Send pool_size first */
    int ps = priv->pool_size;
    if (send(client_fd, &ps, sizeof(ps), 0) != sizeof(ps)) {
        fprintf(stderr, "[nvmmsink] Failed to send pool_size to client\n");
        close(client_fd);
        return;
    }

    /* Send NvBufSurfaceMapParams for each pool buffer (serializable metadata) */
    for (int i = 0; i < priv->pool_size; i++) {
        NvBufSurfaceMapParams params = priv->pool[i].map_params;
        if (send(client_fd, &params, sizeof(params), 0) != sizeof(params)) {
            fprintf(stderr, "[nvmmsink] Failed to send map_params[%d]\n", i);
            close(client_fd);
            return;
        }
    }

    /* Send fds via SCM_RIGHTS */
    int fds[NVMM_POOL_SIZE];
    for (int i = 0; i < priv->pool_size; i++)
        fds[i] = priv->pool[i].fd;

    if (nvmm_send_fds(client_fd, fds, priv->pool_size) < 0) {
        fprintf(stderr, "[nvmmsink] Failed to send fds: %s\n", strerror(errno));
        close(client_fd);
        return;
    }

    std::lock_guard<std::mutex> lock(priv->clients_mutex);
    priv->client_fds.push_back(client_fd);
}

static void
accept_loop(GstNvmmSink *self)
{
    auto *priv = self->priv;

    while (priv->running.load()) {
        struct pollfd pfd = { priv->listen_fd, POLLIN, 0 };
        int ret = poll(&pfd, 1, 200 /* ms */);
        if (ret <= 0) continue;

        int client = accept(priv->listen_fd, nullptr, nullptr);
        if (client < 0) continue;

        /* Wait for pool to be allocated before sending fds */
        while (!priv->pool_allocated && priv->running.load())
            g_usleep(10000);

        if (!priv->running.load()) {
            close(client);
            break;
        }

        send_fds_to_client(self, client);
    }
}

/* --- GstBaseSink vfuncs --- */

static void
gst_nvmm_sink_set_property(GObject *object, guint prop_id,
                            const GValue *value, GParamSpec *pspec)
{
    auto *self = GST_NVMM_SINK(object);
    switch (prop_id) {
        case PROP_SHM_NAME:
            self->priv->shm_name = g_value_get_string(value) ? g_value_get_string(value) : "";
            break;
        case PROP_POOL_SIZE_PROP:
            /* Min NVMM_MIN_POOL_SIZE: a consumer (nvmmappsrc) holds a ref on up
               to RELEASE_DELAY (12) in-flight buffers, so a smaller pool lets a
               steady consumer hold every slot and starve the producer. */
            self->priv->pool_size =
                CLAMP(g_value_get_int(value), NVMM_MIN_POOL_SIZE, NVMM_POOL_SIZE);
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
        case PROP_POOL_SIZE_PROP:
            g_value_set_int(value, self->priv->pool_size);
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

    /* Allocate the NVMM buffer pool now that we know the format */
    if (!self->priv->pool_allocated) {
        if (!allocate_pool(self)) {
            GST_ERROR_OBJECT(self, "Failed to allocate buffer pool");
            return FALSE;
        }

        /* Update shm header with pool info */
        auto *header = static_cast<ShmHeader *>(self->priv->shm_ptr);
        header->width = GST_VIDEO_INFO_WIDTH(&self->priv->video_info);
        header->height = GST_VIDEO_INFO_HEIGHT(&self->priv->video_info);
        header->format = (uint32_t)GST_VIDEO_INFO_FORMAT(&self->priv->video_info);
        header->pool_size = self->priv->pool_size;
        header->num_planes = GST_VIDEO_INFO_N_PLANES(&self->priv->video_info);
        /* Store the actual NVMM pitches from pool buffers */
        for (guint i = 0; i < header->num_planes && i < 4; i++) {
            header->pitches[i] = self->priv->pool[0].surface->surfaceList[0].planeParams.pitch[i];
            header->offsets[i] = self->priv->pool[0].surface->surfaceList[0].planeParams.offset[i];
        }
        strncpy(header->socket_path, self->priv->socket_path.c_str(),
                sizeof(header->socket_path) - 1);
        __sync_synchronize();
    }

    GST_INFO_OBJECT(self, "Configured: %dx%d format=%s pool=%d",
                    GST_VIDEO_INFO_WIDTH(&self->priv->video_info),
                    GST_VIDEO_INFO_HEIGHT(&self->priv->video_info),
                    gst_video_format_to_string(
                        GST_VIDEO_INFO_FORMAT(&self->priv->video_info)),
                    self->priv->pool_size);
    return TRUE;
}

static gboolean
gst_nvmm_sink_start(GstBaseSink *sink)
{
    auto *self = GST_NVMM_SINK(sink);
    auto *priv = self->priv;

    if (priv->shm_name.empty())
        priv->shm_name = "/nvmm_sink_0";

    /* Shared memory: header only (no frame data — GPU-copy pool) */
    priv->shm_size = sizeof(ShmHeader);
    priv->shm_fd = shm_open(priv->shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (priv->shm_fd < 0) {
        fprintf(stderr, "[nvmmsink] shm_open FAILED: %s\n", strerror(errno));
        return FALSE;
    }
    if (ftruncate(priv->shm_fd, priv->shm_size) < 0) {
        fprintf(stderr, "[nvmmsink] ftruncate FAILED: %s\n", strerror(errno));
        close(priv->shm_fd);
        priv->shm_fd = -1;
        return FALSE;
    }
    priv->shm_ptr = mmap(NULL, priv->shm_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, priv->shm_fd, 0);
    if (priv->shm_ptr == MAP_FAILED) {
        fprintf(stderr, "[nvmmsink] mmap FAILED: %s\n", strerror(errno));
        close(priv->shm_fd);
        priv->shm_fd = -1;
        priv->shm_ptr = nullptr;
        return FALSE;
    }
    auto *header = static_cast<ShmHeader *>(priv->shm_ptr);
    memset(header, 0, sizeof(ShmHeader));
    header->magic = NVMM_SHM_MAGIC;
    header->version = NVMM_SHM_VERSION;
    header->ready = 0;
    for (int i = 0; i < NVMM_POOL_SIZE; i++)
        header->ref_counts[i] = 0;

    /* Unix socket for fd passing — path derived from shm name */
    /* Socket path: /tmp/nvmm_<name>.sock (flatten the shm name to avoid subdirectories) */
    std::string flat_name = priv->shm_name;
    for (auto &c : flat_name) { if (c == '/') c = '_'; }
    priv->socket_path = std::string("/tmp/nvmm") + flat_name + ".sock";
    priv->listen_fd = nvmm_server_listen(priv->socket_path.c_str());
    if (priv->listen_fd < 0) {
        fprintf(stderr, "[nvmmsink] socket listen FAILED: %s\n", strerror(errno));
        return FALSE;
    }
    /* Start accept thread */
    priv->running = true;
    priv->accept_thread = std::thread(accept_loop, self);

    priv->frame_number = 0;
    priv->write_idx = 0;

    GST_INFO_OBJECT(self, "Started: shm='%s' socket='%s'",
                    priv->shm_name.c_str(), priv->socket_path.c_str());
    return TRUE;
}

static gboolean
gst_nvmm_sink_stop(GstBaseSink *sink)
{
    auto *self = GST_NVMM_SINK(sink);
    auto *priv = self->priv;

    /* Stop accept thread */
    priv->running = false;
    if (priv->accept_thread.joinable())
        priv->accept_thread.join();

    /* Close client connections */
    {
        std::lock_guard<std::mutex> lock(priv->clients_mutex);
        for (int fd : priv->client_fds)
            close(fd);
        priv->client_fds.clear();
    }

    /* Close listen socket */
    if (priv->listen_fd >= 0) {
        close(priv->listen_fd);
        priv->listen_fd = -1;
    }
    if (!priv->socket_path.empty()) {
        unlink(priv->socket_path.c_str());
        priv->socket_path.clear();
    }

    /* Destroy pool */
    destroy_pool(self);

    /* Clean up shm */
    if (priv->shm_ptr && priv->shm_ptr != MAP_FAILED) {
        munmap(priv->shm_ptr, priv->shm_size);
        priv->shm_ptr = nullptr;
    }
    if (priv->shm_fd >= 0) {
        close(priv->shm_fd);
        priv->shm_fd = -1;
    }
    shm_unlink(priv->shm_name.c_str());

    GST_INFO_OBJECT(self, "Stopped");
    return TRUE;
}

static GstFlowReturn
gst_nvmm_sink_render(GstBaseSink *sink, GstBuffer *buffer)
{
    auto *self = GST_NVMM_SINK(sink);
    auto *priv = self->priv;
    auto *header = static_cast<ShmHeader *>(priv->shm_ptr);

    if (!priv->shm_ptr || !priv->pool_allocated)
        return GST_FLOW_ERROR;

    /* Get NvBufSurface from incoming buffer */
    NvBufSurface *src_surf = get_nvbufsurface_from_buffer(sink, buffer);
    if (!src_surf) {
        GST_WARNING_OBJECT(self, "Buffer is not NVMM — GPU-copy requires NVMM input");
        return GST_FLOW_ERROR;
    }

    /* Find and claim next free pool buffer.
       Use CAS to atomically set ref_count from 0 to -1 (writer lock).
       -1 means "being written" — consumers will skip this buffer.
       After writing, set ref_count back to 0 and update write_idx. */
    int target = -1;
    for (int i = 0; i < priv->pool_size; i++) {
        int idx = (priv->write_idx + 1 + i) % priv->pool_size;
        if (__sync_bool_compare_and_swap(&header->ref_counts[idx], 0, -1)) {
            target = idx;
            break;
        }
    }

    if (target < 0) {
        fprintf(stderr, "[nvmmsink] All pool buffers busy, dropping frame\n");
        return GST_FLOW_OK;
    }

    /* GPU copy: incoming → pool[target].
     *
     * Use NvBufSurfTransform (VIC), NOT NvBufSurfaceCopy. Upstream NVMM
     * producers (e.g. nvvidconv, nvv4l2decoder) hand us BLOCK_LINEAR tiled
     * surfaces, while the pool is allocated PITCH_LINEAR. NvBufSurfaceCopy is a
     * raw memory copy that does not de-tile, so a block-linear -> pitch-linear
     * copy scrambles the image. NvBufSurfTransform runs through VIC and converts
     * layout (and, if ever needed, format/size) correctly. */
    NvBufSurfTransformParams xform;
    memset(&xform, 0, sizeof(xform));
    xform.transform_flag = 0;  /* full-surface convert; no crop/flip/scale */
    if (NvBufSurfTransform(src_surf, priv->pool[target].surface, &xform)
            != NvBufSurfTransformError_Success) {
        fprintf(stderr, "[nvmmsink] NvBufSurfTransform failed for pool buffer %d\n", target);
        /* Release writer lock on failure */
        __sync_lock_test_and_set(&header->ref_counts[target], 0);
        return GST_FLOW_ERROR;
    }

    /* Release writer lock (set ref_count from -1 back to 0) and publish */
    __sync_lock_test_and_set(&header->ref_counts[target], 0);
    __sync_synchronize();
    header->write_idx = target;
    header->timestamp_ns = GST_BUFFER_PTS(buffer);
    header->frame_number = priv->frame_number.fetch_add(1);
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
            "POSIX shared memory segment name (e.g., /cam1)",
            "/nvmm_sink_0", G_PARAM_READWRITE));

    g_object_class_install_property(gobject_class, PROP_POOL_SIZE_PROP,
        g_param_spec_int("pool-size", "Pool Size",
            "Number of NVMM buffers in the pool (13-16). Must exceed the "
            "consumer's RELEASE_DELAY (12) or a steady consumer starves the producer.",
            NVMM_MIN_POOL_SIZE, NVMM_POOL_SIZE, NVMM_POOL_SIZE,
            G_PARAM_READWRITE));

    gst_element_class_set_static_metadata(element_class,
        "NVMM GPU-Copy IPC Sink",
        "Sink/Video",
        "GPU-copy NVMM IPC via buffer pool + SCM_RIGHTS fd passing",
        "Pavel Guzenfeld, Stereolabs");

    GstCaps *caps = gst_caps_from_string(NVMM_CAPS_STRING);
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
    self->priv->shm_fd = -1;
    self->priv->shm_ptr = nullptr;
    self->priv->shm_size = 0;
    self->priv->frame_number = 0;
    self->priv->pool_size = NVMM_POOL_SIZE;
    self->priv->write_idx = 0;
    self->priv->pool_allocated = FALSE;
    self->priv->listen_fd = -1;
    self->priv->running = false;
    for (int i = 0; i < NVMM_POOL_SIZE; i++) {
        self->priv->pool[i].surface = nullptr;
        self->priv->pool[i].fd = -1;
    }
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
    "NVMM GPU-copy IPC sink",
    plugin_init,
    "1.2.0",
    "LGPL",
    "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp"
)
