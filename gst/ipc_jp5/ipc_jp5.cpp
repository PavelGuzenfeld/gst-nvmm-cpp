/// JetPack 5 IPC backend.
///
/// Wire format: NvmmShmCopyHeader (code NVMM_SHM_PROTO_COPY). Header +
/// plane-interleaved pixel data share one shm segment. Producer memcpy's
/// each rendered buffer into shm; consumer memcpy's out. No GPU-side
/// sharing — this backend exists because the JetPack 5 SDK does not
/// expose NvBufSurfaceImport / NvBufSurfaceMapParams.

#include "ipc_backend.h"
#include "shm_protocol.h"
#include "gstnvmmallocator.h"

#ifdef NVMM_MOCK_API
#include "nvbufsurface_mock.h"
#else
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#endif

#include <atomic>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_ipc_debug);
#define GST_CAT_DEFAULT gst_nvmm_ipc_debug

extern "C" void
nvmm_ipc_backend_init_debug(void)
{
    GST_DEBUG_CATEGORY_INIT(gst_nvmm_ipc_debug, "nvmmipc.jp5", 0,
                            "NVMM IPC (JetPack 5 copy-backend)");
}

typedef NvmmShmCopyHeader ShmHeader;

/* ------------------------------------------------------------------ */
/*                            Producer                                 */
/* ------------------------------------------------------------------ */

struct NvmmIpcProducer {
    /* Scratch pitch-linear NVMM surface used as an intermediate when the
     * upstream buffer arrives in block-linear layout. Allocated lazily on
     * first block-linear frame and reused thereafter. */
    NvBufSurface *scratch_pitch = nullptr;
    int           scratch_w     = 0;
    int           scratch_h     = 0;
    NvBufSurfaceColorFormat scratch_fmt = NVBUF_COLOR_FORMAT_INVALID;

    std::string shm_name;
    int   shm_fd    = -1;
    void *shm_ptr   = nullptr;
    gsize shm_size  = 0;
    std::atomic<uint64_t> frame_number{0};
    GstVideoInfo video_info;
    gboolean caps_set = FALSE;
};

NvmmIpcProducer *
nvmm_ipc_producer_new(const gchar *shm_name, int /*pool_size*/)
{
    auto *self = new NvmmIpcProducer();
    self->shm_name = (shm_name && *shm_name) ? shm_name : "/nvmm_sink_0";
    gst_video_info_init(&self->video_info);
    return self;
}

void
nvmm_ipc_producer_free(NvmmIpcProducer *self)
{
    delete self;
}

gboolean
nvmm_ipc_producer_start(NvmmIpcProducer *self, GstElement *owner)
{
    /* Header + worst-case 1080p RGBA until set_caps resizes to match. */
    self->shm_size = sizeof(ShmHeader) + (1920 * 1080 * 4);

    self->shm_fd = shm_open(self->shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (self->shm_fd < 0) {
        GST_ERROR_OBJECT(owner, "shm_open(%s) failed: %s",
                         self->shm_name.c_str(), g_strerror(errno));
        return FALSE;
    }

    if (ftruncate(self->shm_fd, self->shm_size) < 0) {
        GST_ERROR_OBJECT(owner, "ftruncate(%" G_GSIZE_FORMAT ") failed: %s",
                         self->shm_size, g_strerror(errno));
        close(self->shm_fd);
        self->shm_fd = -1;
        return FALSE;
    }

    self->shm_ptr = mmap(nullptr, self->shm_size,
                         PROT_READ | PROT_WRITE, MAP_SHARED,
                         self->shm_fd, 0);
    if (self->shm_ptr == MAP_FAILED) {
        GST_ERROR_OBJECT(owner, "mmap failed: %s", g_strerror(errno));
        close(self->shm_fd);
        self->shm_fd  = -1;
        self->shm_ptr = nullptr;
        return FALSE;
    }

    memset(self->shm_ptr, 0, self->shm_size);
    self->frame_number.store(0);

    GST_INFO_OBJECT(owner, "jp5 producer started: shm='%s' size=%" G_GSIZE_FORMAT,
                    self->shm_name.c_str(), self->shm_size);
    return TRUE;
}

gboolean
nvmm_ipc_producer_stop(NvmmIpcProducer *self, GstElement *owner)
{
    if (self->scratch_pitch) {
        NvBufSurfaceDestroy(self->scratch_pitch);
        self->scratch_pitch = nullptr;
    }
    if (self->shm_ptr && self->shm_ptr != MAP_FAILED) {
        munmap(self->shm_ptr, self->shm_size);
        self->shm_ptr = nullptr;
    }
    if (self->shm_fd >= 0) {
        close(self->shm_fd);
        self->shm_fd = -1;
    }
    shm_unlink(self->shm_name.c_str());
    GST_INFO_OBJECT(owner, "jp5 producer stopped: shm='%s'", self->shm_name.c_str());
    return TRUE;
}

gboolean
nvmm_ipc_producer_set_caps(NvmmIpcProducer *self, GstElement *owner,
                           const GstVideoInfo *info,
                           gboolean /*caps_have_nvmm_feature*/)
{
    self->video_info = *info;
    self->caps_set = TRUE;

    gsize needed = sizeof(ShmHeader) + GST_VIDEO_INFO_SIZE(info);
    if (self->shm_ptr && needed > self->shm_size) {
        munmap(self->shm_ptr, self->shm_size);
        self->shm_size = needed;
        if (ftruncate(self->shm_fd, self->shm_size) < 0) {
            GST_ERROR_OBJECT(owner, "ftruncate resize failed: %s", g_strerror(errno));
            self->shm_ptr = nullptr;
            return FALSE;
        }
        self->shm_ptr = mmap(nullptr, self->shm_size,
                             PROT_READ | PROT_WRITE, MAP_SHARED,
                             self->shm_fd, 0);
        if (self->shm_ptr == MAP_FAILED) {
            GST_ERROR_OBJECT(owner, "mmap resize failed: %s", g_strerror(errno));
            self->shm_ptr = nullptr;
            return FALSE;
        }
    }

    GST_INFO_OBJECT(owner, "jp5 configured: %dx%d %s (shm=%" G_GSIZE_FORMAT ")",
                    GST_VIDEO_INFO_WIDTH(info), GST_VIDEO_INFO_HEIGHT(info),
                    gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(info)),
                    self->shm_size);
    return TRUE;
}

gboolean
nvmm_ipc_producer_propose_allocation(NvmmIpcProducer * /*self*/,
                                     GstElement      * /*owner*/,
                                     GstQuery        * /*query*/)
{
    /* The JP5 copy backend has no upstream pool to offer — the consumer
       always receives a CPU copy out of shm. */
    return FALSE;
}

/* Ensure the producer has a pitch-linear scratch surface matching the
 * given size/format. Reused across frames. */
static NvBufSurface *
jp5_get_scratch_pitch(NvmmIpcProducer *self, GstElement *owner,
                     int w, int h, NvBufSurfaceColorFormat fmt)
{
    if (self->scratch_pitch &&
        self->scratch_w == w && self->scratch_h == h &&
        self->scratch_fmt == fmt)
        return self->scratch_pitch;

    if (self->scratch_pitch) {
        NvBufSurfaceDestroy(self->scratch_pitch);
        self->scratch_pitch = nullptr;
    }

    NvBufSurfaceCreateParams params{};
    params.gpuId        = 0;
    params.width        = (uint32_t)w;
    params.height       = (uint32_t)h;
    params.colorFormat  = fmt;
    params.layout       = NVBUF_LAYOUT_PITCH;
    params.memType      = NVBUF_MEM_SURFACE_ARRAY;
    params.isContiguous = true;

    NvBufSurface *s = nullptr;
    if (NvBufSurfaceCreate(&s, 1, &params) != 0 || !s) {
        GST_ERROR_OBJECT(owner, "scratch pitch-linear surface allocation failed");
        return nullptr;
    }
    self->scratch_pitch = s;
    self->scratch_w     = w;
    self->scratch_h     = h;
    self->scratch_fmt   = fmt;
    GST_INFO_OBJECT(owner,
        "allocated pitch-linear scratch for block-linear->pitch conversion: %dx%d",
        w, h);
    return s;
}

/* Copy an external NVMM surface's planes into tightly-packed shm, matching
 * the GstVideoInfo layout the consumer will emit. If the source is
 * BLOCK_LINEAR (the default for nvvidconv/nvv4l2decoder output on Tegra),
 * first convert to a pitch-linear scratch via NvBufSurfTransform, then
 * CPU-map + memcpy. */
static gsize
jp5_copy_external_nvmm(NvmmIpcProducer *self, GstElement *owner,
                      const GstVideoInfo *vi,
                      ShmHeader *header, uint8_t *frame_data, gsize available,
                      NvBufSurface *surf)
{
    if (!surf || !surf->surfaceList) {
        GST_WARNING_OBJECT(owner, "external NVMM: surf=%p list=%p",
                           (void*)surf, surf ? (void*)surf->surfaceList : nullptr);
        return 0;
    }
    const NvBufSurfaceParams *sp = &surf->surfaceList[0];
    const guint nplanes = GST_VIDEO_INFO_N_PLANES(vi);
    if (nplanes == 0 || nplanes > 4) {
        GST_WARNING_OBJECT(owner, "external NVMM: bad plane count %u", nplanes);
        return 0;
    }

    /* Destination plane sizes derived from GstVideoInfo — these are the
     * tight-packed byte counts the consumer will expect. */
    gsize dst_plane_size[4] = {0,0,0,0};
    for (guint pl = 0; pl < nplanes; pl++) {
        gsize off  = GST_VIDEO_INFO_PLANE_OFFSET(vi, pl);
        gsize next = (pl + 1 < nplanes)
                       ? (gsize)GST_VIDEO_INFO_PLANE_OFFSET(vi, pl + 1)
                       : (gsize)GST_VIDEO_INFO_SIZE(vi);
        dst_plane_size[pl] = next - off;
    }

    GST_LOG_OBJECT(owner,
                   "external NVMM %dx%d fmt=%s nplanes=%u layout=%d memType=%d "
                   "src_pitch=[%u,%u] dst_stride=[%d,%d] dst_plane_size=[%zu,%zu]",
                   GST_VIDEO_INFO_WIDTH(vi), GST_VIDEO_INFO_HEIGHT(vi),
                   gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(vi)),
                   nplanes, sp->layout, surf->memType,
                   sp->planeParams.pitch[0], sp->planeParams.pitch[1],
                   GST_VIDEO_INFO_PLANE_STRIDE(vi, 0),
                   nplanes > 1 ? GST_VIDEO_INFO_PLANE_STRIDE(vi, 1) : 0,
                   dst_plane_size[0], dst_plane_size[1]);

    /* If the source is block-linear (Tegra tiled), go via a pitch-linear
     * scratch surface first. VIC handles the detile; we then CPU-map the
     * scratch and memcpy rows out. */
    NvBufSurface *src_surf = surf;
    if (sp->layout == NVBUF_LAYOUT_BLOCK_LINEAR) {
        NvBufSurface *scratch = jp5_get_scratch_pitch(
            self, owner,
            sp->width, sp->height, sp->colorFormat);
        if (!scratch) return 0;

        NvBufSurfTransformParams xp{};
        xp.transform_flag = 0;
        if (NvBufSurfTransform(surf, scratch, &xp) != NvBufSurfTransformError_Success) {
            GST_WARNING_OBJECT(owner, "NvBufSurfTransform block->pitch failed");
            return 0;
        }
        src_surf = scratch;
        sp = &scratch->surfaceList[0];  /* use scratch's params from here on */
    }

    if (NvBufSurfaceMap(src_surf, 0, -1, NVBUF_MAP_READ) != 0) {
        GST_WARNING_OBJECT(owner, "NvBufSurfaceMap(all-planes) failed");
        return 0;
    }
    NvBufSurfaceSyncForCpu(src_surf, 0, -1);

    gsize total = 0;
    for (guint pl = 0; pl < nplanes; pl++) {
        const uint8_t *src = (const uint8_t *)sp->mappedAddr.addr[pl];
        if (!src) {
            GST_WARNING_OBJECT(owner, "plane %u mappedAddr null after map", pl);
            continue;
        }

        const gsize  src_pitch  = sp->planeParams.pitch[pl];
        const gsize  dst_pitch  = GST_VIDEO_INFO_PLANE_STRIDE(vi, pl);
        const gsize  plane_rows = (dst_pitch > 0) ? dst_plane_size[pl] / dst_pitch : 0;
        const gsize  row_bytes  = MIN(src_pitch, dst_pitch);

        if (!src_pitch || !dst_pitch || !plane_rows) {
            GST_WARNING_OBJECT(owner,
                "plane %u bad geom src_pitch=%zu dst_pitch=%zu rows=%zu",
                pl, src_pitch, dst_pitch, plane_rows);
            continue;
        }

        const gsize offset_before = total;
        for (gsize row = 0; row < plane_rows && total + row_bytes <= available;
             row++) {
            memcpy(frame_data + total, src + row * src_pitch, row_bytes);
            total += row_bytes;
        }

        header->pitches[pl] = (uint32_t)dst_pitch;
        header->offsets[pl] = (uint32_t)offset_before;
    }

    NvBufSurfaceUnMap(src_surf, 0, -1);
    return total;
}

GstFlowReturn
nvmm_ipc_producer_render(NvmmIpcProducer *self, GstElement *owner,
                         GstBuffer *buffer)
{
    if (!self->shm_ptr) {
        GST_ERROR_OBJECT(owner, "shm not initialized");
        return GST_FLOW_ERROR;
    }

    auto *header = static_cast<ShmHeader *>(self->shm_ptr);
    auto *frame_data = static_cast<uint8_t *>(self->shm_ptr) + sizeof(ShmHeader);

    GstMemory *mem = gst_buffer_peek_memory(buffer, 0);

    /* Take the `ready` flag to 0 with RELEASE so any stale reader that
     * still sees 1 at least observes the pre-update values consistently. */
    __atomic_store_n(&header->ready, 0u, __ATOMIC_RELEASE);

    header->magic        = NVMM_SHM_MAGIC;
    header->version      = NVMM_SHM_PROTO_COPY;
    header->width        = GST_VIDEO_INFO_WIDTH(&self->video_info);
    header->height       = GST_VIDEO_INFO_HEIGHT(&self->video_info);
    header->format       = (uint32_t)GST_VIDEO_INFO_FORMAT(&self->video_info);
    const uint64_t fn    = self->frame_number.fetch_add(1);
    header->frame_number = fn;
    header->timestamp_ns = GST_BUFFER_PTS(buffer);
    header->dmabuf_fd    = -1;
    header->num_planes   = GST_VIDEO_INFO_N_PLANES(&self->video_info);
    for (guint i = 0; i < header->num_planes && i < 4; i++) {
        header->pitches[i] = GST_VIDEO_INFO_PLANE_STRIDE(&self->video_info, i);
        header->offsets[i] = GST_VIDEO_INFO_PLANE_OFFSET(&self->video_info, i);
    }

    const gsize available = self->shm_size - sizeof(ShmHeader);
    gsize total = 0;

    /* Detect which of three input shapes we received:
         (A) our own NVMM allocator — map per-plane via gst_nvmm_memory_*
         (B) external NVMM buffer (nvvidconv, zedsrc, nvv4l2decoder) —
             gst_buffer_map gives us an NvBufSurface*, use NvBufSurfaceMap
         (C) plain CPU video/x-raw — standard gst_buffer_map */
    GstCaps *pad_caps = gst_pad_get_current_caps(GST_BASE_SINK_PAD(GST_BASE_SINK(owner)));
    GstCapsFeatures *feat = pad_caps ? gst_caps_get_features(pad_caps, 0) : nullptr;
    gboolean caps_is_nvmm = feat && gst_caps_features_contains(feat, "memory:NVMM");
    if (pad_caps) gst_caps_unref(pad_caps);

    if (gst_is_nvmm_memory(mem)) {
        /* Case (A) */
        for (guint p = 0; p < header->num_planes && p < 4; p++) {
            guint8 *plane = nullptr;
            gsize   psize = 0;
            if (gst_nvmm_memory_map_plane(mem, p, GST_MAP_READ, &plane, &psize)) {
                gsize n = MIN(psize, available - total);
                memcpy(frame_data + total, plane, n);
                total += n;
            }
        }
        gst_nvmm_memory_unmap_plane(mem);
    } else if (caps_is_nvmm) {
        /* Case (B) — external NVMM buffer. mapped data is an NvBufSurface*.
         * Keep the GstBuffer map alive across the NvBufSurfaceMap / read /
         * UnMap sequence; unmapping the GstBuffer first triggers NVIDIA's
         * allocator cleanup and leaves surf->mappedAddr stale. */
        GstMapInfo info;
        if (gst_buffer_map(buffer, &info, GST_MAP_READ)) {
            NvBufSurface *surf = reinterpret_cast<NvBufSurface *>(info.data);
            if (surf)
                total = jp5_copy_external_nvmm(self, owner, &self->video_info,
                                              header, frame_data,
                                              available, surf);
            else
                GST_WARNING_OBJECT(owner, "external NVMM map returned null surf");
            gst_buffer_unmap(buffer, &info);
        } else {
            GST_WARNING_OBJECT(owner, "gst_buffer_map failed on external NVMM");
        }
    } else {
        /* Case (C) — plain CPU raw */
        GstMapInfo info;
        if (gst_buffer_map(buffer, &info, GST_MAP_READ)) {
            gsize n = MIN((gsize)info.size, available);
            memcpy(frame_data, info.data, n);
            total = n;
            gst_buffer_unmap(buffer, &info);
        } else {
            GST_WARNING_OBJECT(owner, "buffer map failed, rendering empty frame");
        }
    }
    header->data_size = (uint32_t)total;

    /* RELEASE on `ready` pairs with the consumer's ACQUIRE load; every
     * write above is visible once the consumer observes ready == 1. */
    __atomic_store_n(&header->ready, 1u, __ATOMIC_RELEASE);

    GST_LOG_OBJECT(owner, "published frame #%" G_GUINT64_FORMAT " (%" G_GSIZE_FORMAT " bytes)",
                   (guint64)fn, total);
    return GST_FLOW_OK;
}

/* ------------------------------------------------------------------ */
/*                            Consumer                                 */
/* ------------------------------------------------------------------ */

struct NvmmIpcConsumer {
    std::string shm_name;
    int   shm_fd  = -1;
    void *shm_ptr = nullptr;
    gsize shm_size = 0;
    uint64_t last_frame = 0;
    GstVideoInfo video_info;
    gboolean caps_ready = FALSE;
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
nvmm_ipc_consumer_free(NvmmIpcConsumer *self)
{
    delete self;
}

gboolean
nvmm_ipc_consumer_start(NvmmIpcConsumer *self, GstElement *owner)
{
    self->shm_fd = shm_open(self->shm_name.c_str(), O_RDONLY, 0);
    if (self->shm_fd < 0) {
        GST_ERROR_OBJECT(owner, "shm_open(%s) failed: %s",
                         self->shm_name.c_str(), g_strerror(errno));
        return FALSE;
    }

    struct stat st;
    if (fstat(self->shm_fd, &st) < 0 || st.st_size == 0) {
        GST_ERROR_OBJECT(owner, "shm empty or fstat failed");
        close(self->shm_fd);
        self->shm_fd = -1;
        return FALSE;
    }
    self->shm_size = st.st_size;

    self->shm_ptr = mmap(nullptr, self->shm_size, PROT_READ, MAP_SHARED,
                         self->shm_fd, 0);
    if (self->shm_ptr == MAP_FAILED) {
        GST_ERROR_OBJECT(owner, "mmap failed: %s", g_strerror(errno));
        close(self->shm_fd);
        self->shm_fd  = -1;
        self->shm_ptr = nullptr;
        return FALSE;
    }

    self->last_frame = 0;
    GST_INFO_OBJECT(owner, "jp5 consumer started: shm='%s'", self->shm_name.c_str());
    return TRUE;
}

gboolean
nvmm_ipc_consumer_stop(NvmmIpcConsumer *self, GstElement *owner)
{
    if (self->shm_ptr && self->shm_ptr != MAP_FAILED) {
        munmap(self->shm_ptr, self->shm_size);
        self->shm_ptr = nullptr;
    }
    if (self->shm_fd >= 0) {
        close(self->shm_fd);
        self->shm_fd = -1;
    }
    GST_INFO_OBJECT(owner, "jp5 consumer stopped: shm='%s'", self->shm_name.c_str());
    return TRUE;
}

gboolean
nvmm_ipc_consumer_peek_caps(NvmmIpcConsumer *self, GstVideoInfo *out_info,
                            gboolean *out_is_nvmm)
{
    if (!self->shm_ptr) return FALSE;
    auto *header = static_cast<const ShmHeader *>(self->shm_ptr);
    if (header->magic != NVMM_SHM_MAGIC ||
        __atomic_load_n(&header->ready, __ATOMIC_ACQUIRE) == 0 ||
        header->width == 0 || header->height == 0)
        return FALSE;

    gst_video_info_set_format(out_info, (GstVideoFormat)header->format,
                              header->width, header->height);
    if (out_is_nvmm) *out_is_nvmm = FALSE;   /* copy backend never emits NVMM buffers */
    return TRUE;
}

GstFlowReturn
nvmm_ipc_consumer_fetch(NvmmIpcConsumer *self, GstElement *owner,
                        GstPad *src_pad, GstBuffer **out_buffer)
{
    if (!self->shm_ptr) return GST_FLOW_ERROR;
    auto *header = static_cast<ShmHeader *>(self->shm_ptr);

    /* Poll `ready` with ACQUIRE semantics; that makes the producer's
     * prior writes (frame_number, width, height, pixel data in shm)
     * visible once the load returns 1. */
    int attempts = 0;
    uint32_t ready;
    uint64_t fn;
    while (true) {
        ready = __atomic_load_n(&header->ready,        __ATOMIC_ACQUIRE);
        fn    = __atomic_load_n(&header->frame_number, __ATOMIC_RELAXED);
        if (ready && fn != self->last_frame) break;
        if (GST_PAD_IS_FLUSHING(src_pad)) return GST_FLOW_FLUSHING;
        if (++attempts > 500) {
            GST_INFO_OBJECT(owner, "no new frame for 500ms, returning EOS");
            return GST_FLOW_EOS;
        }
        g_usleep(1000);
    }

    if (header->magic != NVMM_SHM_MAGIC) {
        GST_ERROR_OBJECT(owner, "invalid shm magic: 0x%08x", header->magic);
        return GST_FLOW_ERROR;
    }

    /* Set caps on first frame from header — the copy backend emits plain
       video/x-raw (no memory:NVMM feature) since pixel data is CPU memcpy'd
       out of shm. */
    if (!self->caps_ready && header->width > 0 && header->height > 0) {
        gst_video_info_set_format(&self->video_info,
                                  (GstVideoFormat)header->format,
                                  header->width, header->height);
        GstCaps *caps = gst_video_info_to_caps(&self->video_info);
        gst_pad_set_caps(src_pad, caps);
        GST_INFO_OBJECT(owner, "caps locked from shm: %" GST_PTR_FORMAT, caps);
        gst_caps_unref(caps);
        self->caps_ready = TRUE;
    }

    gsize data_size = header->data_size;
    if (data_size == 0 || data_size > self->shm_size - sizeof(ShmHeader)) {
        GST_WARNING_OBJECT(owner, "invalid data_size: %u", header->data_size);
        return GST_FLOW_ERROR;
    }

    GstBuffer *buf = gst_buffer_new_allocate(nullptr, data_size, nullptr);
    if (!buf) return GST_FLOW_ERROR;

    GstMapInfo info;
    if (gst_buffer_map(buf, &info, GST_MAP_WRITE)) {
        const auto *frame_data =
            static_cast<const uint8_t *>(self->shm_ptr) + sizeof(ShmHeader);
        memcpy(info.data, frame_data, data_size);
        gst_buffer_unmap(buf, &info);
    }

    /* Attach GstVideoMeta with the exact strides the producer wrote.
     * Without this, downstream elements fall back to
     * gst_video_info_to_caps defaults which may differ from what we
     * actually wrote (especially for multi-plane formats like NV12). */
    if (self->caps_ready) {
        gsize offsets[GST_VIDEO_MAX_PLANES] = {0};
        gint  strides[GST_VIDEO_MAX_PLANES] = {0};
        guint n = GST_VIDEO_INFO_N_PLANES(&self->video_info);
        for (guint p = 0; p < n && p < 4; p++) {
            offsets[p] = header->offsets[p];
            strides[p] = (gint)header->pitches[p];
        }
        gst_buffer_add_video_meta_full(
            buf, GST_VIDEO_FRAME_FLAG_NONE,
            GST_VIDEO_INFO_FORMAT(&self->video_info),
            GST_VIDEO_INFO_WIDTH(&self->video_info),
            GST_VIDEO_INFO_HEIGHT(&self->video_info),
            n, offsets, strides);
    }

    GST_BUFFER_PTS(buf)      = __atomic_load_n(&header->timestamp_ns, __ATOMIC_RELAXED);
    GST_BUFFER_DURATION(buf) = GST_CLOCK_TIME_NONE;

    self->last_frame = fn;
    GST_LOG_OBJECT(owner, "fetched frame #%" G_GUINT64_FORMAT " (%" G_GSIZE_FORMAT " bytes)",
                   (guint64)fn, data_size);

    *out_buffer = buf;
    return GST_FLOW_OK;
}
