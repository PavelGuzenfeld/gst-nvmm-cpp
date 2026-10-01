#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Wire format between nvmmsink and nvmmappsrc or any external reader.
/// Pool DMA-buf fds travel over a unix socket (SCM_RIGHTS); readers import
/// them and read GPU memory in place.
#define NVMM_SHM_MAGIC   0x4E564D4D
#define NVMM_SHM_VERSION 3

#define NVMM_POOL_SIZE 16
/// nvmmappsrc holds refs on its RELEASE_DELAY (12) newest buffers, so the
/// pool needs RELEASE_DELAY + 1 slots or a steady reader starves the writer.
#define NVMM_MIN_POOL_SIZE 13

typedef struct NvmmShmHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    /// GstVideoFormat enum value.
    uint32_t format;
    uint32_t pool_size;
    uint32_t num_planes;
    uint32_t pitches[4];
    uint32_t offsets[4];
    char     socket_path[108];

    /// Rewritten every frame: pair each access with a memory barrier.
    volatile uint32_t write_idx;
    volatile uint64_t frame_number;
    /// PTS in nanoseconds.
    volatile uint64_t timestamp_ns;
    volatile uint32_t ready;

    /// Readers increment before reading and decrement when done; the writer
    /// waits for 0 before reusing a slot. Also guards meta slot i.
    volatile int32_t ref_counts[NVMM_POOL_SIZE];

    /// When 1, NVMM_POOL_SIZE NvmmFrameMeta records follow this header and
    /// record i belongs to pool buffer i (see nvmm_shm_meta()).
    volatile uint32_t meta_enabled;
    /// Objects per frame the meta region was sized for.
    uint32_t meta_max_objects;
    uint32_t _reserved[14];
} NvmmShmHeader;

/// Fixed-size, pointer-free records read straight out of the shared segment;
/// no DeepStream dependency.
#define NVMM_META_LABEL_LEN   64u
#define NVMM_META_MAX_OBJECTS 256u

/// Set when a frame had more than NVMM_META_MAX_OBJECTS detections; the
/// rest were dropped.
#define NVMM_FRAME_META_FLAG_TRUNCATED 0x1u

typedef struct NvmmDetObject {
    /// Pixels in the inference frame (NvmmFrameMeta::infer_width/height).
    float    left, top, width, height;
    int32_t  class_id;
    float    confidence;
    /// 0 when no tracker ran.
    uint64_t tracker_id;
    char     label[NVMM_META_LABEL_LEN];
} NvmmDetObject;

typedef struct NvmmFrameMeta {
    /// Matches NvmmShmHeader::frame_number of the frame it describes.
    uint64_t frame_number;
    /// Bbox coordinate space; readers rescale to the published surface.
    uint32_t infer_width;
    uint32_t infer_height;
    uint32_t num_objects;
    uint32_t flags;
    NvmmDetObject objects[NVMM_META_MAX_OBJECTS];
} NvmmFrameMeta;

#include <stddef.h>

/// Producer-side size for ftruncate(); readers take the size from fstat().
static inline size_t nvmm_shm_segment_size(int meta_enabled)
{
    size_t base = sizeof(NvmmShmHeader);
    if (meta_enabled)
        base += (size_t)NVMM_POOL_SIZE * sizeof(NvmmFrameMeta);
    return base;
}

/// Caller must have checked header->meta_enabled and the mapped size.
static inline NvmmFrameMeta *nvmm_shm_meta(void *base, uint32_t idx)
{
    unsigned char *p = (unsigned char *)base + sizeof(NvmmShmHeader);
    return (NvmmFrameMeta *)p + idx;
}

#ifdef __cplusplus
}
#endif
