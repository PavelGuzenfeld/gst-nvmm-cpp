/// JetPack 6 IPC backend.
///
/// Wire format: NvmmShmPoolHeader (code NVMM_SHM_PROTO_POOL). Producer holds
/// an NVMM buffer pool whose DMA-buf fds are exported to consumers via
/// SCM_RIGHTS over a unix-domain socket. Consumers NvBufSurfaceImport the
/// fds and read directly from GPU memory.
///
/// Two render paths:
///   1. propose_allocation accepted — upstream writes straight into our pool;
///      render() retains the buffer ref, publishes its index, and releases
///      only when consumers signal done (true zero-copy). [stage c]
///   2. propose_allocation not accepted — render() NvBufSurfaceCopy's the
///      incoming buffer into our next free pool slot (GPU-to-GPU copy).

#include "ipc_backend.h"
#include "shm_protocol.h"
#include "fd_ipc.h"

#ifdef NVMM_MOCK_API
#include "nvbufsurface_mock.h"
#else
#include <nvbufsurface.h>
#endif

#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_ipc_debug);
#define GST_CAT_DEFAULT gst_nvmm_ipc_debug

extern "C" void
nvmm_ipc_backend_init_debug(void)
{
    GST_DEBUG_CATEGORY_INIT(gst_nvmm_ipc_debug, "nvmmipc.jp6", 0,
                            "NVMM IPC (JetPack 6 pool-backend)");
}

typedef NvmmShmPoolHeader ShmHeader;

/* ================================================================== */
/* Shared atomic view over shm ref_counts[].
 *
 * NvmmShmPoolHeader::ref_counts[] is declared `volatile int32_t` for C
 * compatibility; here we treat each slot as an atomic via __atomic_*
 * primitives. Both paths produce the same machine code on Linux/aarch64;
 * the wrappers just make the intent explicit.
 */
static inline int32_t
refc_load(volatile int32_t *p)
{
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
static inline void
refc_store(volatile int32_t *p, int32_t v)
{
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}
static inline bool
refc_cas(volatile int32_t *p, int32_t expect, int32_t desired)
{
    return __atomic_compare_exchange_n(p, &expect, desired, false,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}
static inline int32_t
refc_fetch_add(volatile int32_t *p, int32_t d)
{
    return __atomic_fetch_add(p, d, __ATOMIC_ACQ_REL);
}
/* ================================================================== */

struct PoolSlot {
    NvBufSurface          *surface    = nullptr;
    int                    fd         = -1;
    NvBufSurfaceMapParams  map_params{};
};

/* ------------------------------------------------------------------ */
/*                            Producer                                 */
/* ------------------------------------------------------------------ */

struct NvmmIpcProducer {
    std::string shm_name;
    std::string socket_path;

    int   shm_fd   = -1;
    void *shm_ptr  = nullptr;
    gsize shm_size = 0;

    int pool_size_requested = 8;
    std::vector<PoolSlot> pool;
    gboolean pool_allocated = FALSE;

    std::atomic<uint64_t> frame_number{0};
    int                   write_idx = 0;
    GstVideoInfo          video_info{};
    gboolean              caps_set = FALSE;

    /* Socket accept loop for fd passing. */
    int                       listen_fd = -1;
    std::thread               accept_thread;
    std::atomic<bool>         running{false};
    std::vector<int>          client_fds;
    std::mutex                clients_mutex;
};

static gboolean
v2_allocate_pool(NvmmIpcProducer *self, GstElement *owner)
{
    const int w = GST_VIDEO_INFO_WIDTH(&self->video_info);
    const int h = GST_VIDEO_INFO_HEIGHT(&self->video_info);

    NvBufSurfaceCreateParams params{};
    params.gpuId        = 0;
    params.width        = (uint32_t)w;
    params.height       = (uint32_t)h;
    params.isContiguous = true;
    params.layout       = NVBUF_LAYOUT_PITCH;
    params.memType      = NVBUF_MEM_SURFACE_ARRAY;

    switch (GST_VIDEO_INFO_FORMAT(&self->video_info)) {
        case GST_VIDEO_FORMAT_NV12: params.colorFormat = NVBUF_COLOR_FORMAT_NV12;   break;
        case GST_VIDEO_FORMAT_RGBA: params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;   break;
        case GST_VIDEO_FORMAT_BGRA: params.colorFormat = NVBUF_COLOR_FORMAT_BGRA;   break;
        case GST_VIDEO_FORMAT_I420: params.colorFormat = NVBUF_COLOR_FORMAT_YUV420; break;
        default:
            GST_ERROR_OBJECT(owner, "unsupported format for NVMM pool: %s",
                gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(&self->video_info)));
            return FALSE;
    }

    self->pool.resize(self->pool_size_requested);
    for (int i = 0; i < self->pool_size_requested; i++) {
        NvBufSurface *surf = nullptr;
        if (NvBufSurfaceCreate(&surf, 1, &params) != 0 || !surf) {
            GST_ERROR_OBJECT(owner, "NvBufSurfaceCreate failed for pool slot %d", i);
            return FALSE;
        }
        self->pool[i].surface = surf;
        self->pool[i].fd      = (int)surf->surfaceList[0].bufferDesc;
        if (NvBufSurfaceGetMapParams(surf, 0, &self->pool[i].map_params) != 0) {
            GST_ERROR_OBJECT(owner, "NvBufSurfaceGetMapParams failed for slot %d", i);
            return FALSE;
        }
    }
    self->pool_allocated = TRUE;
    GST_INFO_OBJECT(owner, "allocated NVMM pool: %d slots @ %dx%d",
                    self->pool_size_requested, w, h);
    return TRUE;
}

static void
v2_destroy_pool(NvmmIpcProducer *self)
{
    for (auto &slot : self->pool) {
        if (slot.surface) {
            NvBufSurfaceDestroy(slot.surface);
            slot.surface = nullptr;
            slot.fd      = -1;
        }
    }
    self->pool.clear();
    self->pool_allocated = FALSE;
}

static void
v2_send_fds_to_client(NvmmIpcProducer *self, GstElement *owner, int client_fd)
{
    int ps = (int)self->pool.size();
    if (send(client_fd, &ps, sizeof(ps), 0) != (ssize_t)sizeof(ps)) {
        GST_WARNING_OBJECT(owner, "failed to send pool_size to client: %s",
                           g_strerror(errno));
        close(client_fd);
        return;
    }

    for (int i = 0; i < ps; i++) {
        NvBufSurfaceMapParams p = self->pool[i].map_params;
        if (send(client_fd, &p, sizeof(p), 0) != (ssize_t)sizeof(p)) {
            GST_WARNING_OBJECT(owner, "failed to send map_params[%d]: %s",
                               i, g_strerror(errno));
            close(client_fd);
            return;
        }
    }

    std::vector<int> fds(ps);
    for (int i = 0; i < ps; i++) fds[i] = self->pool[i].fd;
    if (nvmm_send_fds(client_fd, fds.data(), ps) < 0) {
        GST_WARNING_OBJECT(owner, "failed to SCM_RIGHTS send fds: %s",
                           g_strerror(errno));
        close(client_fd);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(self->clients_mutex);
        self->client_fds.push_back(client_fd);
    }
    GST_INFO_OBJECT(owner, "handed pool (%d fds) to client fd=%d", ps, client_fd);
}

static void
v2_accept_loop(NvmmIpcProducer *self, GstElement *owner)
{
    while (self->running.load()) {
        struct pollfd pfd { self->listen_fd, POLLIN, 0 };
        int ret = poll(&pfd, 1, 200 /* ms */);
        if (ret <= 0) continue;

        int client = accept(self->listen_fd, nullptr, nullptr);
        if (client < 0) continue;

        /* Wait until the pool is ready (set_caps has been called). */
        while (!self->pool_allocated && self->running.load())
            g_usleep(10000);

        if (!self->running.load()) { close(client); break; }
        v2_send_fds_to_client(self, owner, client);
    }
}

/* --- Public interface -------------------------------------------- */

NvmmIpcProducer *
nvmm_ipc_producer_new(const gchar *shm_name, int pool_size)
{
    auto *self = new NvmmIpcProducer();
    self->shm_name = (shm_name && *shm_name) ? shm_name : "/nvmm_sink_0";
    self->pool_size_requested = CLAMP(pool_size, 4, NVMM_POOL_SIZE_MAX);
    gst_video_info_init(&self->video_info);
    return self;
}

void
nvmm_ipc_producer_free(NvmmIpcProducer *self) { delete self; }

gboolean
nvmm_ipc_producer_start(NvmmIpcProducer *self, GstElement *owner)
{
    self->shm_size = sizeof(ShmHeader);
    self->shm_fd = shm_open(self->shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (self->shm_fd < 0) {
        GST_ERROR_OBJECT(owner, "shm_open(%s) failed: %s",
                         self->shm_name.c_str(), g_strerror(errno));
        return FALSE;
    }
    if (ftruncate(self->shm_fd, self->shm_size) < 0) {
        GST_ERROR_OBJECT(owner, "ftruncate failed: %s", g_strerror(errno));
        close(self->shm_fd); self->shm_fd = -1;
        return FALSE;
    }
    self->shm_ptr = mmap(nullptr, self->shm_size,
                         PROT_READ | PROT_WRITE, MAP_SHARED,
                         self->shm_fd, 0);
    if (self->shm_ptr == MAP_FAILED) {
        GST_ERROR_OBJECT(owner, "mmap failed: %s", g_strerror(errno));
        close(self->shm_fd); self->shm_fd = -1;
        self->shm_ptr = nullptr;
        return FALSE;
    }

    auto *header = static_cast<ShmHeader *>(self->shm_ptr);
    memset(header, 0, sizeof(*header));
    header->magic   = NVMM_SHM_MAGIC;
    header->version = NVMM_SHM_PROTO_POOL;
    for (int i = 0; i < NVMM_POOL_SIZE_MAX; i++) refc_store(&header->ref_counts[i], 0);

    /* Socket path derived from shm name; flatten slashes. */
    std::string flat = self->shm_name;
    for (auto &c : flat) if (c == '/') c = '_';
    self->socket_path = std::string("/tmp/nvmm") + flat + ".sock";

    self->listen_fd = nvmm_server_listen(self->socket_path.c_str());
    if (self->listen_fd < 0) {
        GST_ERROR_OBJECT(owner, "socket listen(%s) failed: %s",
                         self->socket_path.c_str(), g_strerror(errno));
        return FALSE;
    }

    self->running.store(true);
    self->accept_thread = std::thread(v2_accept_loop, self, owner);
    self->frame_number.store(0);
    self->write_idx = 0;

    GST_INFO_OBJECT(owner, "jp6 producer started: shm='%s' socket='%s'",
                    self->shm_name.c_str(), self->socket_path.c_str());
    return TRUE;
}

gboolean
nvmm_ipc_producer_stop(NvmmIpcProducer *self, GstElement *owner)
{
    self->running.store(false);
    if (self->accept_thread.joinable()) self->accept_thread.join();

    {
        std::lock_guard<std::mutex> lock(self->clients_mutex);
        for (int fd : self->client_fds) close(fd);
        self->client_fds.clear();
    }
    if (self->listen_fd >= 0) { close(self->listen_fd); self->listen_fd = -1; }
    if (!self->socket_path.empty()) { unlink(self->socket_path.c_str()); self->socket_path.clear(); }

    v2_destroy_pool(self);

    if (self->shm_ptr && self->shm_ptr != MAP_FAILED) {
        munmap(self->shm_ptr, self->shm_size);
        self->shm_ptr = nullptr;
    }
    if (self->shm_fd >= 0) { close(self->shm_fd); self->shm_fd = -1; }
    shm_unlink(self->shm_name.c_str());

    GST_INFO_OBJECT(owner, "jp6 producer stopped");
    return TRUE;
}

gboolean
nvmm_ipc_producer_set_caps(NvmmIpcProducer *self, GstElement *owner,
                           const GstVideoInfo *info,
                           gboolean /*caps_have_nvmm_feature*/)
{
    self->video_info = *info;
    self->caps_set   = TRUE;

    if (!self->pool_allocated && !v2_allocate_pool(self, owner))
        return FALSE;

    auto *header = static_cast<ShmHeader *>(self->shm_ptr);
    header->width      = GST_VIDEO_INFO_WIDTH(info);
    header->height     = GST_VIDEO_INFO_HEIGHT(info);
    header->format     = (uint32_t)GST_VIDEO_INFO_FORMAT(info);
    header->pool_size  = (uint32_t)self->pool.size();
    header->num_planes = GST_VIDEO_INFO_N_PLANES(info);
    for (guint i = 0; i < header->num_planes && i < 4; i++) {
        header->pitches[i] = self->pool[0].surface->surfaceList[0].planeParams.pitch[i];
        header->offsets[i] = self->pool[0].surface->surfaceList[0].planeParams.offset[i];
    }
    g_strlcpy(header->socket_path, self->socket_path.c_str(),
              sizeof(header->socket_path));
    __atomic_thread_fence(__ATOMIC_RELEASE);

    GST_INFO_OBJECT(owner, "jp6 configured: %dx%d %s pool=%u",
                    header->width, header->height,
                    gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(info)),
                    header->pool_size);
    return TRUE;
}

gboolean
nvmm_ipc_producer_propose_allocation(NvmmIpcProducer * /*self*/,
                                     GstElement      * /*owner*/,
                                     GstQuery        * /*query*/)
{
    /* Stage (c) will offer the pool here; until then we always copy. */
    return FALSE;
}

/* Helper: fetch NvBufSurface* from an incoming NVMM GstBuffer. */
static NvBufSurface *
v2_buffer_to_surface(GstBuffer *buffer, gboolean caps_is_nvmm)
{
    if (!caps_is_nvmm) return nullptr;
    GstMapInfo info;
    if (!gst_buffer_map(buffer, &info, GST_MAP_READ)) return nullptr;
    NvBufSurface *s = reinterpret_cast<NvBufSurface *>(info.data);
    gst_buffer_unmap(buffer, &info);
    return s;  /* NVIDIA convention: mapped data == stable NvBufSurface* */
}

GstFlowReturn
nvmm_ipc_producer_render(NvmmIpcProducer *self, GstElement *owner,
                         GstBuffer *buffer)
{
    if (!self->shm_ptr || !self->pool_allocated)
        return GST_FLOW_ERROR;

    auto *header = static_cast<ShmHeader *>(self->shm_ptr);

    NvBufSurface *src = v2_buffer_to_surface(buffer, TRUE);
    if (!src) {
        GST_WARNING_OBJECT(owner, "render: buffer is not NVMM (pool backend requires NVMM input)");
        return GST_FLOW_ERROR;
    }

    /* Claim next idle slot via CAS 0 -> -1 (writer lock). */
    const int ps = (int)self->pool.size();
    int target = -1;
    for (int i = 0; i < ps; i++) {
        int idx = (self->write_idx + 1 + i) % ps;
        if (refc_cas(&header->ref_counts[idx], 0, -1)) { target = idx; break; }
    }
    if (target < 0) {
        GST_WARNING_OBJECT(owner, "all %d pool buffers busy, dropping frame", ps);
        return GST_FLOW_OK;
    }

    if (NvBufSurfaceCopy(src, self->pool[target].surface) != 0) {
        GST_WARNING_OBJECT(owner, "NvBufSurfaceCopy failed for slot %d", target);
        refc_store(&header->ref_counts[target], 0);
        return GST_FLOW_ERROR;
    }

    /* Release writer lock, publish. */
    refc_store(&header->ref_counts[target], 0);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    header->write_idx    = target;
    header->timestamp_ns = GST_BUFFER_PTS(buffer);
    header->frame_number = self->frame_number.fetch_add(1);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    header->ready = 1;
    self->write_idx = target;

    GST_LOG_OBJECT(owner, "published frame #%" G_GUINT64_FORMAT " into slot %d",
                   (guint64)header->frame_number, target);
    return GST_FLOW_OK;
}

/* ------------------------------------------------------------------ */
/*                            Consumer                                 */
/* ------------------------------------------------------------------ */

/* Delayed release of imported surfaces: the hardware encoder may still be
   reading a DMA buffer after the consuming GstBuffer is unref'd. We decrement
   the shm ref_count only after RELEASE_DELAY frames have passed. */
#define V2_RELEASE_DELAY 4

struct NvmmIpcConsumer {
    std::string shm_name;
    int   shm_fd   = -1;
    void *shm_ptr  = nullptr;
    gsize shm_size = 0;

    int socket_fd = -1;
    int pool_size = 0;
    std::vector<NvBufSurface *> imported;

    uint64_t     last_frame = 0;
    GstVideoInfo video_info{};
    gboolean     caps_ready = FALSE;

    /* Delayed release ring — see note above. */
    volatile int32_t *ring[V2_RELEASE_DELAY] = {};
    int ring_head = 0;
};

NvmmIpcConsumer *
nvmm_ipc_consumer_new(const gchar *shm_name)
{
    auto *self = new NvmmIpcConsumer();
    self->shm_name = (shm_name && *shm_name) ? shm_name : "/nvmm_sink_0";
    gst_video_info_init(&self->video_info);
    return self;
}

void
nvmm_ipc_consumer_free(NvmmIpcConsumer *self) { delete self; }

gboolean
nvmm_ipc_consumer_start(NvmmIpcConsumer *self, GstElement *owner)
{
    self->shm_fd = shm_open(self->shm_name.c_str(), O_RDWR, 0);
    if (self->shm_fd < 0) {
        GST_ERROR_OBJECT(owner, "shm_open(%s) failed: %s",
                         self->shm_name.c_str(), g_strerror(errno));
        return FALSE;
    }
    struct stat st;
    if (fstat(self->shm_fd, &st) < 0 || st.st_size == 0) {
        GST_ERROR_OBJECT(owner, "shm empty or fstat failed");
        close(self->shm_fd); self->shm_fd = -1;
        return FALSE;
    }
    self->shm_size = st.st_size;
    self->shm_ptr = mmap(nullptr, self->shm_size,
                         PROT_READ | PROT_WRITE, MAP_SHARED,
                         self->shm_fd, 0);
    if (self->shm_ptr == MAP_FAILED) {
        GST_ERROR_OBJECT(owner, "mmap failed: %s", g_strerror(errno));
        close(self->shm_fd); self->shm_fd = -1; self->shm_ptr = nullptr;
        return FALSE;
    }

    auto *header = static_cast<ShmHeader *>(self->shm_ptr);

    /* Wait for producer ready (magic + socket_path populated). */
    int waited = 0;
    while (header->magic != NVMM_SHM_MAGIC || header->socket_path[0] == '\0') {
        if (++waited > 5000) {
            GST_ERROR_OBJECT(owner, "timeout waiting for producer");
            return FALSE;
        }
        g_usleep(1000);
    }
    if (header->version != NVMM_SHM_PROTO_POOL) {
        GST_ERROR_OBJECT(owner, "shm version mismatch: got %u, want %u",
                         header->version, NVMM_SHM_PROTO_POOL);
        return FALSE;
    }

    self->socket_fd = nvmm_client_connect(header->socket_path);
    if (self->socket_fd < 0) {
        GST_ERROR_OBJECT(owner, "connect(%s) failed: %s",
                         header->socket_path, g_strerror(errno));
        return FALSE;
    }

    int ps = 0;
    if (recv(self->socket_fd, &ps, sizeof(ps), MSG_WAITALL) != (ssize_t)sizeof(ps)
        || ps <= 0 || ps > NVMM_POOL_SIZE_MAX) {
        GST_ERROR_OBJECT(owner, "bad pool_size from producer: %d", ps);
        return FALSE;
    }
    self->pool_size = ps;

    std::vector<NvBufSurfaceMapParams> params(ps);
    for (int i = 0; i < ps; i++) {
        if (recv(self->socket_fd, &params[i], sizeof(params[i]), MSG_WAITALL)
            != (ssize_t)sizeof(params[i])) {
            GST_ERROR_OBJECT(owner, "failed to receive map_params[%d]", i);
            return FALSE;
        }
    }
    std::vector<int> fds(ps);
    if (nvmm_recv_fds(self->socket_fd, fds.data(), ps) < 0) {
        GST_ERROR_OBJECT(owner, "SCM_RIGHTS recv failed: %s", g_strerror(errno));
        return FALSE;
    }

    self->imported.resize(ps);
    for (int i = 0; i < ps; i++) {
        params[i].fd = fds[i];
        NvBufSurface *s = nullptr;
        if (NvBufSurfaceImport(&s, &params[i]) != 0 || !s) {
            GST_ERROR_OBJECT(owner, "NvBufSurfaceImport failed for slot %d fd=%d",
                             i, fds[i]);
            return FALSE;
        }
        self->imported[i] = s;
    }
    GST_INFO_OBJECT(owner, "jp6 consumer started: pool=%d, imported %d surfaces",
                    ps, (int)self->imported.size());
    return TRUE;
}

gboolean
nvmm_ipc_consumer_stop(NvmmIpcConsumer *self, GstElement *owner)
{
    /* Drain release ring. */
    for (int i = 0; i < V2_RELEASE_DELAY; i++) {
        if (self->ring[i]) {
            refc_fetch_add(self->ring[i], -1);
            self->ring[i] = nullptr;
        }
    }
    for (auto *s : self->imported)
        if (s) NvBufSurfaceDestroy(s);
    self->imported.clear();

    if (self->socket_fd >= 0) { close(self->socket_fd); self->socket_fd = -1; }
    if (self->shm_ptr && self->shm_ptr != MAP_FAILED) {
        munmap(self->shm_ptr, self->shm_size);
        self->shm_ptr = nullptr;
    }
    if (self->shm_fd >= 0) { close(self->shm_fd); self->shm_fd = -1; }

    GST_INFO_OBJECT(owner, "jp6 consumer stopped");
    return TRUE;
}

gboolean
nvmm_ipc_consumer_peek_caps(NvmmIpcConsumer *self, GstVideoInfo *out_info,
                            gboolean *out_is_nvmm)
{
    if (!self->shm_ptr) return FALSE;
    auto *header = static_cast<const ShmHeader *>(self->shm_ptr);
    if (header->magic != NVMM_SHM_MAGIC || !header->ready ||
        header->width == 0 || header->height == 0)
        return FALSE;
    gst_video_info_set_format(out_info, (GstVideoFormat)header->format,
                              header->width, header->height);
    if (out_is_nvmm) *out_is_nvmm = TRUE;   /* pool backend always presents NVMM buffers */
    return TRUE;
}

/* --- Wrap an imported NVMM surface in a GstBuffer ---------------- */
/*
 * Downstream elements get the NvBufSurface pointer by mapping the buffer's
 * single memory (NVIDIA convention). We use gst_memory_new_wrapped rather
 * than registering a custom GstAllocator GType so we don't run into double-
 * registration when both the sink and appsrc plugins are loaded in the same
 * GStreamer process.
 */
static GstMemory *
v2_wrap_imported(NvBufSurface *surf)
{
    return gst_memory_new_wrapped(
        GST_MEMORY_FLAG_NO_SHARE,
        surf, sizeof(NvBufSurface),
        0, sizeof(NvBufSurface),
        nullptr, nullptr);
}

GstFlowReturn
nvmm_ipc_consumer_fetch(NvmmIpcConsumer *self, GstElement *owner,
                        GstPad *src_pad, GstBuffer **out_buffer)
{
    if (!self->shm_ptr) return GST_FLOW_ERROR;
    auto *header = static_cast<ShmHeader *>(self->shm_ptr);

    int attempts = 0;
    while (!header->ready || header->frame_number == self->last_frame) {
        if (GST_PAD_IS_FLUSHING(src_pad)) return GST_FLOW_FLUSHING;
        if (++attempts > 2000) {
            GST_INFO_OBJECT(owner, "no new frame for 2s, returning EOS");
            return GST_FLOW_EOS;
        }
        g_usleep(1000);
    }
    __atomic_thread_fence(__ATOMIC_ACQUIRE);

    if (header->magic != NVMM_SHM_MAGIC) return GST_FLOW_ERROR;

    /* Set caps on first frame. */
    if (!self->caps_ready && header->width > 0 && header->height > 0) {
        gst_video_info_set_format(&self->video_info,
                                  (GstVideoFormat)header->format,
                                  header->width, header->height);
        GstCaps *caps = gst_video_info_to_caps(&self->video_info);
        gst_caps_set_features(caps, 0,
            gst_caps_features_new("memory:NVMM", NULL));
        gst_pad_set_caps(src_pad, caps);
        gst_caps_unref(caps);
        self->caps_ready = TRUE;
    }

    /* CAS-increment ref_count on current write_idx, avoiding writer lock. */
    uint32_t idx = header->write_idx;
    if (idx >= (uint32_t)self->pool_size) return GST_FLOW_ERROR;

    int cas_attempts = 0;
    while (true) {
        int32_t cur = refc_load(&header->ref_counts[idx]);
        if (cur < 0) {                         /* producer writing this slot */
            g_usleep(100);
            idx = header->write_idx;
            if (idx >= (uint32_t)self->pool_size) return GST_FLOW_ERROR;
            if (++cas_attempts > 1000) return GST_FLOW_ERROR;
            continue;
        }
        if (refc_cas(&header->ref_counts[idx], cur, cur + 1)) break;
    }
    __atomic_thread_fence(__ATOMIC_ACQUIRE);

    /* Delayed release: decrement the slot we acquired RELEASE_DELAY frames ago. */
    volatile int32_t *old = self->ring[self->ring_head];
    if (old) refc_fetch_add(old, -1);
    self->ring[self->ring_head] = &header->ref_counts[idx];
    self->ring_head = (self->ring_head + 1) % V2_RELEASE_DELAY;

    GstBuffer *buf = gst_buffer_new();
    gst_buffer_append_memory(buf, v2_wrap_imported(self->imported[idx]));
    GST_BUFFER_PTS(buf)      = header->timestamp_ns;
    GST_BUFFER_DURATION(buf) = GST_CLOCK_TIME_NONE;

    self->last_frame = header->frame_number;
    GST_LOG_OBJECT(owner, "fetched frame #%" G_GUINT64_FORMAT " from slot %u",
                   (guint64)header->frame_number, idx);

    *out_buffer = buf;
    return GST_FLOW_OK;
}
