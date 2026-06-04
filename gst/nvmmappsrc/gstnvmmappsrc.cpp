/// nvmmappsrc — Zero-copy NVMM IPC source.
///
/// Connects to nvmmsink via unix socket to receive pool buffer DMA-buf fds
/// and NvBufSurfaceMapParams, imports them with NvBufSurfaceImport, and
/// pushes NVMM GstBuffers downstream.
///
/// Ref counts in shared memory manage buffer lifecycle.

#include "gstnvmmappsrc.h"

#ifdef NVMM_MOCK_API
#include "nvbufsurface_mock.h"
#else
#include <nvbufsurface.h>
#endif

#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <gst/video/video.h>

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
    PROP_IS_LIVE,
};

/* --- Custom GstAllocator for imported NVMM pool buffers --- */

typedef struct {
    GstAllocator parent;
} NvmmImportedAllocator;

typedef struct {
    GstAllocatorClass parent_class;
} NvmmImportedAllocatorClass;

#define NVMM_TYPE_IMPORTED_ALLOCATOR (nvmm_imported_allocator_get_type())

G_DEFINE_TYPE(NvmmImportedAllocator, nvmm_imported_allocator, GST_TYPE_ALLOCATOR)

struct NvmmImportedMemory {
    GstMemory parent;
    NvBufSurface *surface;
    volatile int32_t *ref_count_ptr;
};

static gpointer
nvmm_imported_mem_map(GstMemory *mem, gsize /*maxsize*/, GstMapFlags /*flags*/)
{
    auto *m = reinterpret_cast<NvmmImportedMemory *>(mem);
    return m->surface;  /* NVIDIA convention: mapped data = NvBufSurface* */
}

static void
nvmm_imported_mem_unmap(GstMemory * /*mem*/) {}

static void
nvmm_imported_mem_free(GstAllocator * /*alloc*/, GstMemory *mem)
{
    /* Do NOT decrement ref_count here — the hardware encoder may still be
       reading the DMA buffer after GstBuffer unref. Ref counts are managed
       by the create() function via a delayed release ring. */
    auto *m = reinterpret_cast<NvmmImportedMemory *>(mem);
    (void)m;
    g_free(m);
}

static void
nvmm_imported_allocator_class_init(NvmmImportedAllocatorClass *klass)
{
    auto *alloc_class = GST_ALLOCATOR_CLASS(klass);
    alloc_class->alloc = nullptr;
    alloc_class->free = nvmm_imported_mem_free;
}

static void
nvmm_imported_allocator_init(NvmmImportedAllocator *self)
{
    GstAllocator *alloc = GST_ALLOCATOR(self);
    alloc->mem_type = "NvmmImportedMemory";
    alloc->mem_map = nvmm_imported_mem_map;
    alloc->mem_unmap = nvmm_imported_mem_unmap;
    GST_OBJECT_FLAG_SET(alloc, GST_ALLOCATOR_FLAG_CUSTOM_ALLOC);
}

static GstAllocator *imported_allocator_singleton = nullptr;

static GstAllocator *
get_imported_allocator(void)
{
    if (!imported_allocator_singleton) {
        imported_allocator_singleton = (GstAllocator *)g_object_new(
            NVMM_TYPE_IMPORTED_ALLOCATOR, nullptr);
        gst_object_ref_sink(imported_allocator_singleton);
    }
    return imported_allocator_singleton;
}

/* --- Main element --- */

/* Delay ring: hold ref counts for RELEASE_DELAY frames before decrementing.
   This accounts for hardware encoder pipeline depth — the encoder may still
   be reading the DMA buffer after unreffing the GstBuffer. */
#define RELEASE_DELAY 12  /* ~100ms at 120fps, covers encoder pipeline depth */

struct RefSlot {
    volatile int32_t *ptr;
    int active;
};

struct _GstNvmmAppSrcPrivate {
    std::string shm_name;
    int shm_fd;
    void *shm_ptr;
    gsize shm_size;
    uint64_t last_frame_number;
    GstVideoInfo video_info;
    gboolean is_live;
    gboolean caps_set;

    /* Zero-copy pool */
    int socket_fd;
    int pool_size;
    NvBufSurface *imported_surfaces[NVMM_POOL_SIZE];

    /* Delayed ref count release ring */
    RefSlot release_ring[RELEASE_DELAY];
    int ring_head;
};

G_DEFINE_TYPE_WITH_PRIVATE(GstNvmmAppSrc, gst_nvmm_app_src, GST_TYPE_PUSH_SRC)

static void
gst_nvmm_app_src_set_property(GObject *object, guint prop_id,
                               const GValue *value, GParamSpec *pspec)
{
    auto *self = GST_NVMM_APP_SRC(object);
    switch (prop_id) {
        case PROP_SHM_NAME:
            self->priv->shm_name = g_value_get_string(value) ? g_value_get_string(value) : "";
            break;
        case PROP_IS_LIVE:
            self->priv->is_live = g_value_get_boolean(value);
            gst_base_src_set_live(GST_BASE_SRC(self), self->priv->is_live);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void
gst_nvmm_app_src_get_property(GObject *object, guint prop_id,
                               GValue *value, GParamSpec *pspec)
{
    auto *self = GST_NVMM_APP_SRC(object);
    switch (prop_id) {
        case PROP_SHM_NAME:
            g_value_set_string(value, self->priv->shm_name.c_str());
            break;
        case PROP_IS_LIVE:
            g_value_set_boolean(value, self->priv->is_live);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static gboolean
gst_nvmm_app_src_start(GstBaseSrc *src)
{
    auto *self = GST_NVMM_APP_SRC(src);
    auto *priv = self->priv;

    if (priv->shm_name.empty())
        priv->shm_name = "/nvmm_sink_0";

    /* Open shared memory */
    priv->shm_fd = shm_open(priv->shm_name.c_str(), O_RDWR, 0);
    if (priv->shm_fd < 0) {
        fprintf(stderr, "[nvmmappsrc] shm_open(%s) failed: %s\n",
                priv->shm_name.c_str(), strerror(errno));
        return FALSE;
    }

    struct stat st;
    if (fstat(priv->shm_fd, &st) < 0 || st.st_size == 0) {
        fprintf(stderr, "[nvmmappsrc] shm empty or stat failed\n");
        close(priv->shm_fd);
        priv->shm_fd = -1;
        return FALSE;
    }
    priv->shm_size = st.st_size;

    priv->shm_ptr = mmap(NULL, priv->shm_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, priv->shm_fd, 0);
    if (priv->shm_ptr == MAP_FAILED) {
        fprintf(stderr, "[nvmmappsrc] mmap failed\n");
        close(priv->shm_fd);
        priv->shm_fd = -1;
        priv->shm_ptr = nullptr;
        return FALSE;
    }

    auto *header = static_cast<ShmHeader *>(priv->shm_ptr);

    /* Wait for producer header */
    int wait = 0;
    while (header->magic != NVMM_SHM_MAGIC || header->socket_path[0] == '\0') {
        if (++wait > 5000) {
            fprintf(stderr, "[nvmmappsrc] Timeout waiting for producer\n");
            return FALSE;
        }
        g_usleep(1000);
    }
    if (header->version != NVMM_SHM_VERSION) {
        fprintf(stderr, "[nvmmappsrc] Bad version %u\n", header->version);
        return FALSE;
    }

    /* Connect to producer socket */
    priv->socket_fd = nvmm_client_connect(header->socket_path);
    if (priv->socket_fd < 0) {
        fprintf(stderr, "[nvmmappsrc] Connect to %s failed: %s\n",
                header->socket_path, strerror(errno));
        return FALSE;
    }
    /* Receive pool_size */
    int ps = 0;
    if (recv(priv->socket_fd, &ps, sizeof(ps), MSG_WAITALL) != sizeof(ps) ||
        ps <= 0 || ps > NVMM_POOL_SIZE) {
        fprintf(stderr, "[nvmmappsrc] Failed to receive pool_size (got %d)\n", ps);
        close(priv->socket_fd);
        priv->socket_fd = -1;
        return FALSE;
    }
    priv->pool_size = ps;

    /* Receive NvBufSurfaceMapParams for each buffer */
    NvBufSurfaceMapParams params[NVMM_POOL_SIZE];
    for (int i = 0; i < priv->pool_size; i++) {
        if (recv(priv->socket_fd, &params[i], sizeof(params[i]), MSG_WAITALL)
            != (ssize_t)sizeof(params[i])) {
            fprintf(stderr, "[nvmmappsrc] Failed to receive map_params[%d]\n", i);
            close(priv->socket_fd);
            priv->socket_fd = -1;
            return FALSE;
        }
    }
    /* Receive DMA-buf fds via SCM_RIGHTS */
    int fds[NVMM_POOL_SIZE];
    if (nvmm_recv_fds(priv->socket_fd, fds, priv->pool_size) < 0) {
        fprintf(stderr, "[nvmmappsrc] Failed to receive fds: %s\n", strerror(errno));
        close(priv->socket_fd);
        priv->socket_fd = -1;
        return FALSE;
    }
    /* Import each fd using NvBufSurfaceImport with the received params */
    for (int i = 0; i < priv->pool_size; i++) {
        /* Update the fd in params to the one we received via SCM_RIGHTS */
        params[i].fd = fds[i];

        NvBufSurface *surf = nullptr;
        if (NvBufSurfaceImport(&surf, &params[i]) != 0 || !surf) {
            fprintf(stderr, "[nvmmappsrc] NvBufSurfaceImport failed for buffer %d (fd=%d)\n",
                    i, fds[i]);
            return FALSE;
        }
        priv->imported_surfaces[i] = surf;
    }

    priv->last_frame_number = 0;
    priv->caps_set = FALSE;

    return TRUE;
}

static gboolean
gst_nvmm_app_src_stop(GstBaseSrc *src)
{
    auto *self = GST_NVMM_APP_SRC(src);
    auto *priv = self->priv;

    /* Flush delayed release ring — decrement all held ref counts */
    for (int i = 0; i < RELEASE_DELAY; i++) {
        if (priv->release_ring[i].active && priv->release_ring[i].ptr) {
            __sync_fetch_and_sub(priv->release_ring[i].ptr, 1);
            priv->release_ring[i].active = 0;
            priv->release_ring[i].ptr = nullptr;
        }
    }

    for (int i = 0; i < NVMM_POOL_SIZE; i++) {
        if (priv->imported_surfaces[i]) {
            NvBufSurfaceDestroy(priv->imported_surfaces[i]);
            priv->imported_surfaces[i] = nullptr;
        }
    }

    if (priv->socket_fd >= 0) {
        close(priv->socket_fd);
        priv->socket_fd = -1;
    }

    if (priv->shm_ptr && priv->shm_ptr != MAP_FAILED) {
        munmap(priv->shm_ptr, priv->shm_size);
        priv->shm_ptr = nullptr;
    }
    if (priv->shm_fd >= 0) {
        close(priv->shm_fd);
        priv->shm_fd = -1;
    }

    return TRUE;
}

static GstFlowReturn
gst_nvmm_app_src_create(GstPushSrc *push_src, GstBuffer **buf)
{
    auto *self = GST_NVMM_APP_SRC(push_src);
    auto *priv = self->priv;
    auto *header = static_cast<ShmHeader *>(priv->shm_ptr);

    if (!priv->shm_ptr)
        return GST_FLOW_ERROR;

    /* Wait for a new frame */
    int attempts = 0;
    while (!header->ready || header->frame_number == priv->last_frame_number) {
        if (GST_PAD_IS_FLUSHING(GST_BASE_SRC_PAD(push_src)))
            return GST_FLOW_FLUSHING;
        if (++attempts > 2000) {
            fprintf(stderr, "[nvmmappsrc] No new frame for 2s\n");
            return GST_FLOW_EOS;
        }
        g_usleep(1000);
    }

    __sync_synchronize();

    if (header->magic != NVMM_SHM_MAGIC)
        return GST_FLOW_ERROR;

    /* Set NVMM caps on first frame */
    if (!priv->caps_set && header->width > 0 && header->height > 0) {
        GstVideoFormat fmt = static_cast<GstVideoFormat>(header->format);
        gst_video_info_set_format(&priv->video_info, fmt,
                                  header->width, header->height);

        GstCaps *caps = gst_video_info_to_caps(&priv->video_info);
        gst_caps_set_features(caps, 0,
            gst_caps_features_new("memory:NVMM", NULL));
        gst_base_src_set_caps(GST_BASE_SRC(self), caps);
        gst_caps_unref(caps);
        priv->caps_set = TRUE;

    }

    /* Read current buffer index and safely increment ref count.
       Use CAS to ensure we don't increment a buffer being written (-1). */
    uint32_t idx = header->write_idx;
    if (idx >= (uint32_t)priv->pool_size)
        return GST_FLOW_ERROR;

    /* CAS loop: increment ref_count only if >= 0 (not being written) */
    int attempts_cas = 0;
    while (true) {
        int32_t old_val = __sync_add_and_fetch(&header->ref_counts[idx], 0);
        if (old_val < 0) {
            /* Buffer being written — re-read write_idx, producer may have moved on */
            g_usleep(100);
            idx = header->write_idx;
            if (++attempts_cas > 1000) return GST_FLOW_ERROR;
            continue;
        }
        if (__sync_bool_compare_and_swap(&header->ref_counts[idx], old_val, old_val + 1))
            break;
    }
    __sync_synchronize();

    /* Delayed release: decrement the ref count of the buffer we acquired
       RELEASE_DELAY frames ago. This gives the hardware encoder enough time
       to finish reading the DMA buffer before the producer recycles it. */
    RefSlot *old_slot = &priv->release_ring[priv->ring_head];
    if (old_slot->active && old_slot->ptr) {
        __sync_fetch_and_sub(old_slot->ptr, 1);
    }
    old_slot->ptr = &header->ref_counts[idx];
    old_slot->active = 1;
    priv->ring_head = (priv->ring_head + 1) % RELEASE_DELAY;

    /* Create GstBuffer with custom memory wrapping the imported surface. */
    NvmmImportedMemory *mem = (NvmmImportedMemory *)g_malloc0(sizeof(NvmmImportedMemory));
    gst_memory_init(GST_MEMORY_CAST(mem), GST_MEMORY_FLAG_NO_SHARE,
                    get_imported_allocator(), nullptr,
                    sizeof(NvBufSurface), 0, 0, sizeof(NvBufSurface));
    mem->surface = priv->imported_surfaces[idx];
    mem->ref_count_ptr = nullptr;  /* managed by ring, not by free callback */

    GstBuffer *buffer = gst_buffer_new();
    gst_buffer_append_memory(buffer, GST_MEMORY_CAST(mem));

    GST_BUFFER_PTS(buffer) = header->timestamp_ns;
    GST_BUFFER_DURATION(buffer) = GST_CLOCK_TIME_NONE;

    priv->last_frame_number = header->frame_number;

    *buf = buffer;
    return GST_FLOW_OK;
}

static void
gst_nvmm_app_src_finalize(GObject *object)
{
    auto *self = GST_NVMM_APP_SRC(object);
    self->priv->~_GstNvmmAppSrcPrivate();
    G_OBJECT_CLASS(gst_nvmm_app_src_parent_class)->finalize(object);
}

static void
gst_nvmm_app_src_class_init(GstNvmmAppSrcClass *klass)
{
    auto *gobject_class = G_OBJECT_CLASS(klass);
    auto *element_class = GST_ELEMENT_CLASS(klass);
    auto *basesrc_class = GST_BASE_SRC_CLASS(klass);
    auto *pushsrc_class = GST_PUSH_SRC_CLASS(klass);

    gobject_class->set_property = gst_nvmm_app_src_set_property;
    gobject_class->get_property = gst_nvmm_app_src_get_property;
    gobject_class->finalize = gst_nvmm_app_src_finalize;

    g_object_class_install_property(gobject_class, PROP_SHM_NAME,
        g_param_spec_string("shm-name", "Shared Memory Name",
            "POSIX shared memory segment name to read from",
            "/nvmm_sink_0", G_PARAM_READWRITE));

    g_object_class_install_property(gobject_class, PROP_IS_LIVE,
        g_param_spec_boolean("is-live", "Is Live",
            "Whether this source is a live source",
            TRUE, G_PARAM_READWRITE));

    gst_element_class_set_static_metadata(element_class,
        "NVMM Zero-Copy IPC Source",
        "Source/Video",
        "Zero-copy NVMM IPC via imported pool buffer fds",
        "Pavel Guzenfeld, Stereolabs");

    GstCaps *caps = gst_caps_from_string(NVMM_CAPS_STRING);
    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
    gst_caps_unref(caps);

    basesrc_class->start = gst_nvmm_app_src_start;
    basesrc_class->stop = gst_nvmm_app_src_stop;
    pushsrc_class->create = gst_nvmm_app_src_create;
}

static void
gst_nvmm_app_src_init(GstNvmmAppSrc *self)
{
    self->priv = static_cast<GstNvmmAppSrcPrivate *>(
        gst_nvmm_app_src_get_instance_private(self));
    new (self->priv) GstNvmmAppSrcPrivate();
    self->priv->shm_name = "/nvmm_sink_0";
    self->priv->shm_fd = -1;
    self->priv->shm_ptr = nullptr;
    self->priv->shm_size = 0;
    self->priv->last_frame_number = 0;
    self->priv->is_live = TRUE;
    self->priv->caps_set = FALSE;
    self->priv->socket_fd = -1;
    self->priv->pool_size = 0;
    for (int i = 0; i < NVMM_POOL_SIZE; i++)
        self->priv->imported_surfaces[i] = nullptr;
    for (int i = 0; i < RELEASE_DELAY; i++) {
        self->priv->release_ring[i].ptr = nullptr;
        self->priv->release_ring[i].active = 0;
    }
    self->priv->ring_head = 0;

    gst_base_src_set_live(GST_BASE_SRC(self), TRUE);
    gst_base_src_set_format(GST_BASE_SRC(self), GST_FORMAT_TIME);
}

/* Plugin registration */
static gboolean
plugin_init(GstPlugin *plugin)
{
    return gst_element_register(plugin, "nvmmappsrc",
                                GST_RANK_NONE, GST_TYPE_NVMM_APP_SRC);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvmmappsrc,
    "NVMM zero-copy IPC source",
    plugin_init,
    "1.1.0",
    "LGPL",
    "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp"
)
