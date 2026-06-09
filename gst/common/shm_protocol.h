/// NVMM shared memory IPC protocol.
/// Defines the header layout for frames shared between nvmmsink and nvmmappsrc
/// (or any external consumer such as a ROS2 node).
///
/// Producer allocates a pool of NVMM buffers and GPU-copies each incoming frame
/// into the pool via NvBufSurfaceCopy. Pool DMA-buf fds are handed to consumers
/// over a unix-domain socket (SCM_RIGHTS); consumers import the fds and read
/// directly from GPU memory — no further copies.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NVMM_SHM_MAGIC   0x4E564D4D  /* "NVMM" */
#define NVMM_SHM_VERSION 3           /* v3: optional per-frame metadata side-channel */

#define NVMM_POOL_SIZE 16  /* number of buffers in the GPU-copy pool */
/* Minimum pool size. A consumer (nvmmappsrc) holds a ref on up to its
   RELEASE_DELAY (12) most-recent in-flight buffers, so the pool must have at
   least RELEASE_DELAY + 1 slots or a steady consumer starves the producer. */
#define NVMM_MIN_POOL_SIZE 13

typedef struct NvmmShmHeader {
    uint32_t magic;           /* NVMM_SHM_MAGIC */
    uint32_t version;         /* NVMM_SHM_VERSION */
    uint32_t width;
    uint32_t height;
    uint32_t format;          /* GstVideoFormat enum value */
    uint32_t pool_size;       /* number of pool buffers */
    uint32_t num_planes;
    uint32_t pitches[4];      /* per-plane pitch from pool buffers */
    uint32_t offsets[4];
    char     socket_path[108]; /* unix socket path for fd passing */

    /* Updated per frame (use memory barriers when reading/writing) */
    volatile uint32_t write_idx;       /* pool index of latest frame */
    volatile uint64_t frame_number;    /* monotonic frame counter */
    volatile uint64_t timestamp_ns;    /* PTS in nanoseconds */
    volatile uint32_t ready;           /* 1 after first frame written */

    /* Per-buffer ref counts: producer waits for 0 before reusing.
       Consumers increment before reading, decrement when done. */
    volatile int32_t ref_counts[NVMM_POOL_SIZE];

    /* Optional per-frame metadata side-channel (v3+). When meta_enabled == 1 the
       segment is grown by NVMM_POOL_SIZE NvmmFrameMeta records placed immediately
       after this header (see nvmm_shm_meta()); slot i holds the detections for
       pool buffer i and is protected by the same ref_counts[i] as the pixels. */
    volatile uint32_t meta_enabled;   /* 1 if producer writes the metadata region */
    uint32_t meta_max_objects;        /* objects/frame the region was sized for */
    uint32_t _reserved[14];
} NvmmShmHeader;

/* ----- Flat, POD detection metadata: the cross-process interface -----
   This is deliberately DeepStream-free. A non-GStreamer consumer (e.g. a ROS2
   node) reads these structs straight out of the shared segment; only a consumer
   that is itself a DeepStream pipeline needs to re-hydrate them into
   NvDsBatchMeta (see nvmm_det_meta.h, optional). All fields are fixed-size with
   no pointers so the records are self-contained across processes. */

#define NVMM_META_LABEL_LEN   64u
#define NVMM_META_MAX_OBJECTS 256u  /* per-frame cap; overflow is truncated + flagged */

#define NVMM_FRAME_META_FLAG_TRUNCATED 0x1u  /* >NVMM_META_MAX_OBJECTS detections */

typedef struct NvmmDetObject {
    float    left, top, width, height; /* bbox in INFERENCE-frame pixel space */
    int32_t  class_id;
    float    confidence;
    uint64_t tracker_id;               /* 0 when no tracker is present */
    char     label[NVMM_META_LABEL_LEN]; /* NUL-terminated class label */
} NvmmDetObject;

typedef struct NvmmFrameMeta {
    uint64_t frame_number;   /* correlation key: matches header->frame_number */
    uint32_t infer_width;    /* coordinate space of the bboxes — REQUIRED so the */
    uint32_t infer_height;   /* consumer can rescale to the published surface */
    uint32_t num_objects;    /* valid entries in objects[] (<= NVMM_META_MAX_OBJECTS) */
    uint32_t flags;          /* NVMM_FRAME_META_FLAG_* */
    NvmmDetObject objects[NVMM_META_MAX_OBJECTS];
} NvmmFrameMeta;

#include <stddef.h>

/* Byte size of the shared segment. With metadata the per-buffer NvmmFrameMeta
   records follow the header; the consumer learns the real size from fstat() so
   it never has to compute this, but the producer uses it to ftruncate(). */
static inline size_t nvmm_shm_segment_size(int meta_enabled)
{
    size_t base = sizeof(NvmmShmHeader);
    if (meta_enabled)
        base += (size_t)NVMM_POOL_SIZE * sizeof(NvmmFrameMeta);
    return base;
}

/* Pointer to the metadata record for pool buffer `idx`. Caller must have checked
   header->meta_enabled and that the mapped segment is large enough. */
static inline NvmmFrameMeta *nvmm_shm_meta(void *base, uint32_t idx)
{
    unsigned char *p = (unsigned char *)base + sizeof(NvmmShmHeader);
    return (NvmmFrameMeta *)p + idx;
}

#ifdef __cplusplus
}
#endif
