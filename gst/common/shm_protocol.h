#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NVMM_SHM_MAGIC   0x4E564D4D
#define NVMM_SHM_VERSION 3

#define NVMM_POOL_SIZE 16
#define NVMM_MIN_POOL_SIZE 13

typedef struct NvmmShmHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t format;
    uint32_t pool_size;
    uint32_t num_planes;
    uint32_t pitches[4];
    uint32_t offsets[4];
    char     socket_path[108];

    volatile uint32_t write_idx;
    volatile uint64_t frame_number;
    volatile uint64_t timestamp_ns;
    volatile uint32_t ready;

    volatile int32_t ref_counts[NVMM_POOL_SIZE];

    volatile uint32_t meta_enabled;
    uint32_t meta_max_objects;
    uint32_t _reserved[14];
} NvmmShmHeader;

#define NVMM_META_LABEL_LEN   64u
#define NVMM_META_MAX_OBJECTS 256u

#define NVMM_FRAME_META_FLAG_TRUNCATED 0x1u

typedef struct NvmmDetObject {
    float    left, top, width, height;
    int32_t  class_id;
    float    confidence;
    uint64_t tracker_id;
    char     label[NVMM_META_LABEL_LEN];
} NvmmDetObject;

typedef struct NvmmFrameMeta {
    uint64_t frame_number;
    uint32_t infer_width;
    uint32_t infer_height;
    uint32_t num_objects;
    uint32_t flags;
    NvmmDetObject objects[NVMM_META_MAX_OBJECTS];
} NvmmFrameMeta;

#include <stddef.h>

static inline size_t nvmm_shm_segment_size(int meta_enabled)
{
    size_t base = sizeof(NvmmShmHeader);
    if (meta_enabled)
        base += (size_t)NVMM_POOL_SIZE * sizeof(NvmmFrameMeta);
    return base;
}

static inline NvmmFrameMeta *nvmm_shm_meta(void *base, uint32_t idx)
{
    unsigned char *p = (unsigned char *)base + sizeof(NvmmShmHeader);
    return (NvmmFrameMeta *)p + idx;
}

#ifdef __cplusplus
}
#endif
