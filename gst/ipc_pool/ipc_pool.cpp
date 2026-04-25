/// Pool + SCM_RIGHTS NVMM IPC backend.
///
/// Single backend for both JetPack 5 (>= R35.3.1 / JP 5.1.1, March 2023) and
/// JetPack 6 (any). Earlier L4T 35.x revisions cannot use this code path —
/// `NvBufSurfaceImport` shipped in R35.3.1; meson rejects older toolchains.
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
#include "nvmm_config.h"

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

#include <dlfcn.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_ipc_debug);
#define GST_CAT_DEFAULT gst_nvmm_ipc_debug

extern "C" void
nvmm_ipc_backend_init_debug(void)
{
    GST_DEBUG_CATEGORY_INIT(gst_nvmm_ipc_debug, "nvmmipc.pool", 0,
                            "NVMM IPC (pool + SCM_RIGHTS backend)");
}

/* Runtime guard. The meson configure step verifies that NvBufSurfaceImport
 * exists in the headers. This catches the deploy-time mismatch where a
 * binary built against R35.3.1+ headers ends up running on a host whose
 * libnvbufsurface.so is older — the symbol won't resolve. With direct
 * symbol references the linker would fail loudly, but in some scenarios
 * (containers with --runtime nvidia mounting host libs, lazy linking,
 * stub libs in NVIDIA's l4t-jetpack image) the failure surfaces only
 * when the symbol is actually called. Probe via dlsym at start() so the
 * error message names the cause instead of the SoC just dying. */
static gboolean
runtime_check_import_api(GstElement *owner)
{
#ifdef NVMM_MOCK_API
    (void)owner;
    return TRUE;  /* mock provides the symbol unconditionally */
#else
    static gboolean checked = FALSE;
    static gboolean ok      = FALSE;
    if (checked) return ok;
    checked = TRUE;

    /* libnvbufsurface is already loaded by the time we get here (the
     * backend's own static refs to NvBufSurfaceCreate etc. pulled it in).
     * RTLD_NOLOAD just gives us a handle without re-loading. */
    void *h = dlopen("libnvbufsurface.so.1.0.0", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libnvbufsurface.so.1",   RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libnvbufsurface.so",     RTLD_NOW | RTLD_NOLOAD);
    if (!h) {
        GST_ERROR_OBJECT(owner,
            "NVMM IPC: libnvbufsurface not loaded (this should be impossible "
            "if any NVMM code ran above this point). Refusing to start.");
        return FALSE;
    }
    void *sym = dlsym(h, "NvBufSurfaceImport");
    dlclose(h);
    if (!sym) {
        GST_ERROR_OBJECT(owner,
            "NVMM IPC: libnvbufsurface.so on this host does NOT export "
            "NvBufSurfaceImport. Cross-process NVMM IPC requires L4T "
            "R35.3.1 (JetPack 5.1.1) or newer on the JP5 line, or any JP6. "
            "This binary was built against newer headers but is running on "
            "an older BSP — upgrade jetson-multimedia-api on the host.");
        return FALSE;
    }
    ok = TRUE;
    GST_INFO_OBJECT(owner, "NVMM IPC: NvBufSurfaceImport present in libnvbufsurface");
    return TRUE;
#endif
}

typedef NvmmShmPoolHeader ShmHeader;

/* Cross-process futex on a shared-memory uint32_t. FUTEX (not
 * FUTEX_PRIVATE) so the kernel knows the address is shared between
 * processes — required for SCM_RIGHTS-style IPC. */
static inline int futex_wait(uint32_t *addr, uint32_t expected,
                              const struct timespec *timeout)
{
    return (int)syscall(SYS_futex, addr, FUTEX_WAIT, expected,
                        timeout, nullptr, 0);
}
static inline int futex_wake(uint32_t *addr, int n_waiters)
{
    return (int)syscall(SYS_futex, addr, FUTEX_WAKE, n_waiters,
                        nullptr, nullptr, 0);
}

/* Atomic accessors for the shm slot ref_counts shared with remote
 * consumers. See shm_protocol.h for memory-order rules. */
static inline int32_t refc_load(int32_t *p)
{
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
static inline void refc_store(int32_t *p, int32_t v)
{
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}
static inline bool refc_cas(int32_t *p, int32_t expect, int32_t desired)
{
    return __atomic_compare_exchange_n(p, &expect, desired, false,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}
static inline int32_t refc_fetch_add(int32_t *p, int32_t d)
{
    return __atomic_fetch_add(p, d, __ATOMIC_ACQ_REL);
}
/* ================================================================== */

/* ================================================================== */
/* GstNvmmIpcUpstreamPool — GstBufferPool subclass that exposes our
 * pre-allocated IPC slot surfaces directly to upstream. When upstream
 * accepts the pool via propose_allocation, every frame upstream renders
 * lands directly in one of our slots; render() then publishes the slot
 * index without an NvBufSurfaceCopy.
 *
 * Lifecycle dance: a slot is acquireable only when BOTH
 *   (a) it's in the pool's standard free queue (no local GstBuffer
 *       holding it), AND
 *   (b) the shm refcount slots[i].ref_count == 0 (no remote consumer
 *       still reading it).
 * Producer's render() ref's the buffer through a delayed-release ring
 * to keep (a) false long enough for (b) to drain.
 * ================================================================== */

struct PoolSlot {
    NvBufSurface          *surface    = nullptr;
    int                    fd         = -1;
    NvBufSurfaceMapParams  map_params{};
};

/* Forward decl — the upstream-facing GstBufferPool subclass below. */
struct _GstNvmmIpcUpstreamPool;
typedef struct _GstNvmmIpcUpstreamPool GstNvmmIpcUpstreamPool;

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

    /* Upstream-facing buffer pool: when upstream agrees, it allocates
     * straight into our IPC slot surfaces and render() skips the copy. */
    GstBufferPool            *upstream_pool = nullptr;

    /* Producer-side delayed-release ring for zero-copy buffers — keep
     * the upstream GstBuffer ref'd for pool_release_delay frames after
     * publishing so the slot doesn't get reissued while remote consumers
     * are reading. */
    std::vector<GstBuffer *>  zc_release_ring;
    int                       zc_ring_head = 0;
};

/* Per-buffer marker so render() can recognize "this came from our pool"
 * and read the slot index. */
static GQuark
nvmm_pool_slot_quark(void)
{
    static GQuark q = 0;
    if (!q) q = g_quark_from_static_string("nvmm-ipc-pool-slot-idx");
    return q;
}

struct _GstNvmmIpcUpstreamPool {
    GstBufferPool       parent;
    NvmmIpcProducer    *producer;        /* back-ref; not owned */
    int                 next_slot_hint;  /* round-robin acquire start */
};

typedef struct _GstNvmmIpcUpstreamPoolClass {
    GstBufferPoolClass parent_class;
} GstNvmmIpcUpstreamPoolClass;

G_DEFINE_TYPE(GstNvmmIpcUpstreamPool, nvmm_ipc_upstream_pool, GST_TYPE_BUFFER_POOL)

#define NVMM_IPC_UPSTREAM_POOL(obj) \
    G_TYPE_CHECK_INSTANCE_CAST((obj), nvmm_ipc_upstream_pool_get_type(), GstNvmmIpcUpstreamPool)

/* alloc_buffer: hand out the pre-allocated slot surfaces in round-robin
 * order. After all N slots have been alloc'd once, the pool's standard
 * free-queue mechanism handles further acquire/release. */
static GstFlowReturn
nvmm_ipc_upstream_pool_alloc(GstBufferPool *pool, GstBuffer **out,
                              GstBufferPoolAcquireParams * /*params*/)
{
    auto *self = NVMM_IPC_UPSTREAM_POOL(pool);
    NvmmIpcProducer *prod = self->producer;
    if (!prod || !prod->pool_allocated) return GST_FLOW_ERROR;

    const int n = (int)prod->pool.size();
    int idx = self->next_slot_hint;
    self->next_slot_hint = (idx + 1) % n;
    if (idx >= n) return GST_FLOW_ERROR;

    NvBufSurface *surf = prod->pool[idx].surface;
    GstBuffer *buf = gst_buffer_new();
    GstMemory *mem = gst_memory_new_wrapped(
        GST_MEMORY_FLAG_NO_SHARE,
        surf, sizeof(NvBufSurface),
        0, sizeof(NvBufSurface),
        nullptr, nullptr);
    gst_buffer_append_memory(buf, mem);
    /* Tag the buffer with its slot index so render() can find it. */
    gst_mini_object_set_qdata(GST_MINI_OBJECT_CAST(buf),
                              nvmm_pool_slot_quark(),
                              GINT_TO_POINTER(idx + 1),  /* +1 to distinguish from NULL */
                              nullptr);
    *out = buf;
    return GST_FLOW_OK;
}

/* acquire_buffer: rejected slots whose shm refcount is still nonzero —
 * remote consumers are reading them, can't reissue. Falls through to the
 * default acquire (which pops from the free queue + may call alloc_buffer
 * for the first N) and re-checks the refcount; if pinned, releases and
 * tries again up to N times. */
static GstFlowReturn
nvmm_ipc_upstream_pool_acquire(GstBufferPool *pool, GstBuffer **out,
                                GstBufferPoolAcquireParams *params)
{
    auto *self = NVMM_IPC_UPSTREAM_POOL(pool);
    NvmmIpcProducer *prod = self->producer;
    if (!prod || !prod->shm_ptr) return GST_FLOW_ERROR;

    auto *header = static_cast<ShmHeader *>(prod->shm_ptr);
    const int n_attempts = (int)prod->pool.size() * 2;

    for (int i = 0; i < n_attempts; i++) {
        GstBuffer *buf = nullptr;
        GstFlowReturn r = GST_BUFFER_POOL_CLASS(
            nvmm_ipc_upstream_pool_parent_class)->acquire_buffer(pool, &buf, params);
        if (r != GST_FLOW_OK) return r;

        gpointer tag = gst_mini_object_get_qdata(GST_MINI_OBJECT_CAST(buf),
                                                 nvmm_pool_slot_quark());
        int idx = GPOINTER_TO_INT(tag) - 1;
        if (idx < 0 || idx >= (int)prod->pool.size()) {
            /* Stranger buffer — shouldn't happen in our pool. Pass it back. */
            *out = buf;
            return GST_FLOW_OK;
        }

        if (refc_load(&header->slots[idx].ref_count) == 0) {
            *out = buf;
            return GST_FLOW_OK;  /* slot is remote-free, take it */
        }

        /* Slot still pinned remotely. Release back and try the next. */
        GST_BUFFER_POOL_CLASS(nvmm_ipc_upstream_pool_parent_class)
            ->release_buffer(pool, buf);
        g_usleep(nvmm::config::busy_slot_backoff_us);
    }
    return GST_FLOW_EOS;  /* every slot is remote-pinned, give up */
}

static void
nvmm_ipc_upstream_pool_init(GstNvmmIpcUpstreamPool *self)
{
    self->producer = nullptr;
    self->next_slot_hint = 0;
}

static void
nvmm_ipc_upstream_pool_class_init(GstNvmmIpcUpstreamPoolClass *klass)
{
    auto *pool_class = GST_BUFFER_POOL_CLASS(klass);
    pool_class->alloc_buffer   = nvmm_ipc_upstream_pool_alloc;
    pool_class->acquire_buffer = nvmm_ipc_upstream_pool_acquire;
}

static GstBufferPool *
nvmm_ipc_upstream_pool_new(NvmmIpcProducer *prod)
{
    auto *p = (GstNvmmIpcUpstreamPool *)
        g_object_new(nvmm_ipc_upstream_pool_get_type(), nullptr);
    p->producer = prod;
    return GST_BUFFER_POOL(p);
}

static gboolean
allocate_pool(NvmmIpcProducer *self, GstElement *owner)
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
destroy_pool(NvmmIpcProducer *self)
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
send_fds_to_client(NvmmIpcProducer *self, GstElement *owner, int client_fd)
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
accept_loop(NvmmIpcProducer *self, GstElement *owner)
{
    while (self->running.load()) {
        struct pollfd pfd { self->listen_fd, POLLIN, 0 };
        int ret = poll(&pfd, 1, nvmm::config::accept_poll_ms);
        if (ret <= 0) continue;

        int client = accept(self->listen_fd, nullptr, nullptr);
        if (client < 0) continue;

        /* Wait until the pool is ready (set_caps has been called). */
        while (!self->pool_allocated && self->running.load())
            g_usleep(nvmm::config::accept_pool_wait_us);

        if (!self->running.load()) { close(client); break; }
        send_fds_to_client(self, owner, client);
    }
}

/* --- Public interface -------------------------------------------- */

NvmmIpcProducer *
nvmm_ipc_producer_new(const gchar *shm_name, int pool_size)
{
    auto *self = new NvmmIpcProducer();
    self->shm_name = (shm_name && *shm_name) ? shm_name : nvmm::config::default_shm_name;
    self->pool_size_requested = CLAMP(pool_size, nvmm::config::min_pool_size, NVMM_POOL_SIZE_MAX);
    gst_video_info_init(&self->video_info);
    return self;
}

void
nvmm_ipc_producer_free(NvmmIpcProducer *self) { delete self; }

gboolean
nvmm_ipc_producer_start(NvmmIpcProducer *self, GstElement *owner)
{
    if (!runtime_check_import_api(owner)) return FALSE;

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
    for (int i = 0; i < NVMM_POOL_SIZE_MAX; i++) refc_store(&header->slots[i].ref_count, 0);

    /* Socket path derived from shm name; flatten slashes. */
    std::string flat = self->shm_name;
    for (auto &c : flat) if (c == '/') c = '_';
    self->socket_path = std::string(nvmm::config::socket_path_prefix) + flat + ".sock";

    self->listen_fd = nvmm_server_listen(self->socket_path.c_str());
    if (self->listen_fd < 0) {
        GST_ERROR_OBJECT(owner, "socket listen(%s) failed: %s",
                         self->socket_path.c_str(), g_strerror(errno));
        return FALSE;
    }

    self->running.store(true);
    self->accept_thread = std::thread(accept_loop, self, owner);
    /* Start at 1 so the first published frame_number (fetch_add returns
     * pre-increment = 1) is distinguishable from a consumer's default
     * last_frame=0. */
    self->frame_number.store(1);
    self->write_idx = 0;

    GST_INFO_OBJECT(owner, "producer started: shm='%s' socket='%s'",
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

    /* Drain the zero-copy delayed-release ring before destroying the
     * pool — those buffers wrap our slot surfaces and must not be
     * unref'd after we tear the pool down. */
    for (auto *&b : self->zc_release_ring) {
        if (b) { gst_buffer_unref(b); b = nullptr; }
    }
    self->zc_release_ring.clear();
    if (self->upstream_pool) {
        gst_buffer_pool_set_active(self->upstream_pool, FALSE);
        gst_object_unref(self->upstream_pool);
        self->upstream_pool = nullptr;
    }

    destroy_pool(self);

    if (self->shm_ptr && self->shm_ptr != MAP_FAILED) {
        munmap(self->shm_ptr, self->shm_size);
        self->shm_ptr = nullptr;
    }
    if (self->shm_fd >= 0) { close(self->shm_fd); self->shm_fd = -1; }
    shm_unlink(self->shm_name.c_str());

    GST_INFO_OBJECT(owner, "producer stopped");
    return TRUE;
}

gboolean
nvmm_ipc_producer_set_caps(NvmmIpcProducer *self, GstElement *owner,
                           const GstVideoInfo *info,
                           gboolean /*caps_have_nvmm_feature*/)
{
    /* Reject caps renegotiation. The pool slot surfaces were sized and
     * formatted at first set_caps and the fds are already in remote
     * consumers' hands. Reallocating here would break those consumers
     * silently. Tear the pipeline down and start fresh if caps change. */
    if (self->caps_set) {
        const GstVideoInfo *cur = &self->video_info;
        if (GST_VIDEO_INFO_FORMAT(cur) != GST_VIDEO_INFO_FORMAT(info) ||
            GST_VIDEO_INFO_WIDTH(cur)  != GST_VIDEO_INFO_WIDTH(info)  ||
            GST_VIDEO_INFO_HEIGHT(cur) != GST_VIDEO_INFO_HEIGHT(info)) {
            GST_ERROR_OBJECT(owner,
                "caps renegotiation not supported: had %dx%d %s, got %dx%d %s. "
                "Pool slots are already shared with consumers; restart the "
                "pipeline to pick up new caps.",
                GST_VIDEO_INFO_WIDTH(cur), GST_VIDEO_INFO_HEIGHT(cur),
                gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(cur)),
                GST_VIDEO_INFO_WIDTH(info), GST_VIDEO_INFO_HEIGHT(info),
                gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(info)));
            return FALSE;
        }
        return TRUE;  /* same caps, nothing to do */
    }

    self->video_info = *info;
    self->caps_set   = TRUE;

    if (!self->pool_allocated && !allocate_pool(self, owner))
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

    /* Publish "header complete" via RELEASE on `ready`; pairs with the
     * consumer's ACQUIRE load in start() and makes all the non-atomic
     * field writes above visible. */
    __atomic_store_n(&header->ready, 1u, __ATOMIC_RELEASE);

    GST_INFO_OBJECT(owner, "configured: %dx%d %s pool=%u",
                    header->width, header->height,
                    gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(info)),
                    header->pool_size);
    return TRUE;
}

gboolean
nvmm_ipc_producer_propose_allocation(NvmmIpcProducer *self, GstElement *owner,
                                     GstQuery *query)
{
    /* Need caps + pool to advertise. set_caps must have run first. */
    if (!self->pool_allocated) return FALSE;

    GstCaps *caps = nullptr;
    gboolean need_pool = FALSE;
    gst_query_parse_allocation(query, &caps, &need_pool);
    if (!need_pool || !caps) return FALSE;

    /* Lazily build the upstream-facing pool. Initial buffer count = our
     * slot count; min/max cap so upstream can size its internal queues. */
    if (!self->upstream_pool) {
        self->upstream_pool = nvmm_ipc_upstream_pool_new(self);
        GstStructure *config = gst_buffer_pool_get_config(self->upstream_pool);
        gst_buffer_pool_config_set_params(
            config, caps,
            sizeof(NvBufSurface),                /* "size" — opaque to upstream */
            (guint)self->pool.size(),            /* min */
            (guint)self->pool.size());           /* max */
        if (!gst_buffer_pool_set_config(self->upstream_pool, config)) {
            GST_WARNING_OBJECT(owner, "upstream pool set_config failed");
            gst_object_unref(self->upstream_pool);
            self->upstream_pool = nullptr;
            return FALSE;
        }
        self->zc_release_ring.assign(nvmm::config::pool_release_delay, nullptr);
    }

    gst_query_add_allocation_pool(query, self->upstream_pool,
                                  sizeof(NvBufSurface),
                                  (guint)self->pool.size(),
                                  (guint)self->pool.size());
    GST_INFO_OBJECT(owner, "offered NVMM pool (%d slots) to upstream for zero-copy",
                    (int)self->pool.size());
    return TRUE;
}

/* Helper: fetch NvBufSurface* from an incoming NVMM GstBuffer. */
static NvBufSurface *
buffer_to_surface(GstBuffer *buffer, gboolean caps_is_nvmm)
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

    NvBufSurface *src = buffer_to_surface(buffer, TRUE);
    if (!src) {
        GST_WARNING_OBJECT(owner, "render: buffer is not NVMM (pool backend requires NVMM input)");
        return GST_FLOW_ERROR;
    }

    /* Zero-copy fast path: did upstream write into one of our pool slots?
     * The pool tags every buffer it hands out with the slot index in
     * qdata. If present, skip the copy and publish the slot directly;
     * we then ref the buffer through a delayed-release ring so the slot
     * doesn't get reissued until remote consumers have had time to drop
     * their refs. */
    int target = -1;
    gpointer tag = gst_mini_object_get_qdata(GST_MINI_OBJECT_CAST(buffer),
                                              nvmm_pool_slot_quark());
    if (tag) {
        target = GPOINTER_TO_INT(tag) - 1;
        if (target < 0 || target >= (int)self->pool.size())
            target = -1;  /* corrupt qdata, fall through to copy path */
    }

    const int ps = (int)self->pool.size();

    if (target >= 0) {
        /* Zero-copy: upstream rendered straight into slot[target]. No
         * NvBufSurfaceCopy. Ref the buffer for delayed release. */
        if (!self->zc_release_ring.empty()) {
            const int delay = (int)self->zc_release_ring.size();
            GstBuffer *old = self->zc_release_ring[self->zc_ring_head];
            if (old) gst_buffer_unref(old);
            self->zc_release_ring[self->zc_ring_head] = gst_buffer_ref(buffer);
            self->zc_ring_head = (self->zc_ring_head + 1) % delay;
        }
        GST_LOG_OBJECT(owner, "render zero-copy into slot %d", target);
    } else {
        /* Copy path: claim next idle slot via CAS 0 -> -1 (writer lock),
         * GPU-to-GPU NvBufSurfaceCopy, then unlock. */
        for (int i = 0; i < ps; i++) {
            int idx = (self->write_idx + 1 + i) % ps;
            if (refc_cas(&header->slots[idx].ref_count, 0, -1)) { target = idx; break; }
        }
        if (target < 0) {
            GST_WARNING_OBJECT(owner, "all %d pool buffers busy, dropping frame", ps);
            return GST_FLOW_OK;
        }

        if (NvBufSurfaceCopy(src, self->pool[target].surface) != 0) {
            GST_WARNING_OBJECT(owner, "NvBufSurfaceCopy failed for slot %d", target);
            refc_store(&header->slots[target].ref_count, 0);
            return GST_FLOW_ERROR;
        }
        refc_store(&header->slots[target].ref_count, 0);
    }

    /* Publish the new frame. RELEASE store on `ready` makes the preceding
     * writes to write_idx / frame_number / timestamp_ns visible. */
    const uint64_t fn = self->frame_number.fetch_add(1);
    __atomic_store_n(&header->timestamp_ns, (uint64_t)GST_BUFFER_PTS(buffer), __ATOMIC_RELAXED);
    __atomic_store_n(&header->frame_number, fn,                                __ATOMIC_RELAXED);
    __atomic_store_n(&header->write_idx,    (uint32_t)target,                  __ATOMIC_RELAXED);
    __atomic_store_n(&header->ready,        1u,                                __ATOMIC_RELEASE);
    self->write_idx = target;

    /* Bump the futex address with RELEASE so consumers' ACQUIRE-load
     * sees a fresh value, then wake every waiter. INT_MAX = unbounded —
     * we want all consumers, not just one. */
    __atomic_add_fetch(&header->wake_counter, 1, __ATOMIC_RELEASE);
    futex_wake(&header->wake_counter, INT_MAX);

    GST_LOG_OBJECT(owner, "published frame #%" G_GUINT64_FORMAT " into slot %d",
                   (guint64)fn, target);
    return GST_FLOW_OK;
}

/* ------------------------------------------------------------------ */
/*                            Consumer                                 */
/* ------------------------------------------------------------------ */

/* Delayed release of imported surfaces: the hardware encoder may still be
   reading a DMA buffer after the consuming GstBuffer is unref'd. We decrement
   the shm ref_count only after pool_release_delay frames have passed
   (see nvmm_config.h). */

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

    /* Delayed release ring — see note above. Pointers into shm ref_counts[];
     * accessed through refc_* helpers. */
    int32_t *ring[nvmm::config::pool_release_delay] = {};
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
    if (!runtime_check_import_api(owner)) return FALSE;

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

    /* Wait for producer to publish "header complete". ACQUIRE load on
     * `ready` pairs with the RELEASE store in the producer's set_caps;
     * after the load returns 1, all header fields (magic, version,
     * socket_path, pool_size, pitches, offsets) are guaranteed visible. */
    int waited = 0;
    while (__atomic_load_n(&header->ready, __ATOMIC_ACQUIRE) == 0) {
        if (++waited > nvmm::config::start_wait_ticks) {
            GST_ERROR_OBJECT(owner, "timeout waiting for producer");
            return FALSE;
        }
        g_usleep(nvmm::config::poll_interval_us);
    }
    if (header->magic != NVMM_SHM_MAGIC) {
        GST_ERROR_OBJECT(owner, "shm magic mismatch: got 0x%08x", header->magic);
        return FALSE;
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
    GST_INFO_OBJECT(owner, "consumer started: pool=%d, imported %d surfaces",
                    ps, (int)self->imported.size());
    return TRUE;
}

gboolean
nvmm_ipc_consumer_stop(NvmmIpcConsumer *self, GstElement *owner)
{
    /* Drain release ring. */
    for (int i = 0; i < nvmm::config::pool_release_delay; i++) {
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

    GST_INFO_OBJECT(owner, "consumer stopped");
    return TRUE;
}

gboolean
nvmm_ipc_consumer_peek_caps(NvmmIpcConsumer *self, GstVideoInfo *out_info,
                            gboolean *out_is_nvmm)
{
    if (!self->shm_ptr) return FALSE;
    auto *header = static_cast<ShmHeader *>(self->shm_ptr);
    if (header->magic != NVMM_SHM_MAGIC ||
        __atomic_load_n(&header->ready, __ATOMIC_ACQUIRE) == 0 ||
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
wrap_imported(NvBufSurface *surf)
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

    /* Block until producer publishes a new frame. The futex wait on
     * wake_counter pairs with the producer's __atomic_add_fetch +
     * FUTEX_WAKE in render(). Each iteration:
     *   1. ACQUIRE-load wake_counter; if changed since last frame we
     *      saw, the new frame's writes are visible — go grab it.
     *   2. Honor flushing — return immediately if the pad is going down.
     *   3. Apply the EOS deadline — short timeouts so we periodically
     *      re-check flushing and the deadline rather than blocking
     *      forever on a stuck producer.
     *   4. futex_wait on wake_counter == observed value. If producer
     *      bumped it between (1) and (4), wait returns EAGAIN and we
     *      loop; that's fine, the next ACQUIRE-load picks up the new
     *      frame. */
    uint32_t ready;
    uint64_t fn;
    int      ticks  = 0;
    const long tick_us = nvmm::config::poll_interval_us;
    struct timespec wait_to = {
        tick_us / 1000000,
        (tick_us % 1000000) * 1000
    };

    while (true) {
        const uint32_t wc = __atomic_load_n(&header->wake_counter, __ATOMIC_ACQUIRE);
        ready = __atomic_load_n(&header->ready,        __ATOMIC_ACQUIRE);
        fn    = __atomic_load_n(&header->frame_number, __ATOMIC_RELAXED);
        if (ready && fn != self->last_frame) break;
        if (GST_PAD_IS_FLUSHING(src_pad)) return GST_FLOW_FLUSHING;
        if (nvmm::config::fetch_idle_ticks > 0 &&
            ++ticks > nvmm::config::fetch_idle_ticks) {
            GST_INFO_OBJECT(owner, "no new frame for ~%d ms, returning EOS",
                            nvmm::config::fetch_idle_ticks * (int)tick_us / 1000);
            return GST_FLOW_EOS;
        }
        /* Block until producer bumps wake_counter or our tick expires.
         * Spurious wakes (EINTR, EAGAIN) loop normally. */
        futex_wait(&header->wake_counter, wc, &wait_to);
    }

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
    uint32_t idx = __atomic_load_n(&header->write_idx, __ATOMIC_ACQUIRE);
    if (idx >= (uint32_t)self->pool_size) return GST_FLOW_ERROR;

    int cas_attempts = 0;
    while (true) {
        int32_t cur = refc_load(&header->slots[idx].ref_count);
        if (cur < 0) {                         /* producer writing this slot */
            g_usleep(nvmm::config::busy_slot_backoff_us);
            idx = __atomic_load_n(&header->write_idx, __ATOMIC_ACQUIRE);
            if (idx >= (uint32_t)self->pool_size) return GST_FLOW_ERROR;
            if (++cas_attempts > nvmm::config::cas_retry_limit) return GST_FLOW_ERROR;
            continue;
        }
        if (refc_cas(&header->slots[idx].ref_count, cur, cur + 1)) break;
    }

    /* Delayed release: decrement the slot we acquired RELEASE_DELAY frames ago. */
    int32_t *old = self->ring[self->ring_head];
    if (old) refc_fetch_add(old, -1);
    self->ring[self->ring_head] = &header->slots[idx].ref_count;
    self->ring_head = (self->ring_head + 1) % nvmm::config::pool_release_delay;

    GstBuffer *buf = gst_buffer_new();
    gst_buffer_append_memory(buf, wrap_imported(self->imported[idx]));
    GST_BUFFER_PTS(buf)      = __atomic_load_n(&header->timestamp_ns, __ATOMIC_RELAXED);
    GST_BUFFER_DURATION(buf) = GST_CLOCK_TIME_NONE;

    self->last_frame = fn;
    GST_LOG_OBJECT(owner, "fetched frame #%" G_GUINT64_FORMAT " from slot %u",
                   (guint64)fn, idx);

    *out_buffer = buf;
    return GST_FLOW_OK;
}
