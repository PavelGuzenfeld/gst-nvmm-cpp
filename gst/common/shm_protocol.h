/// NVMM shared memory IPC protocol.
/// Defines the header layout for frames shared between nvmmsink and nvmmappsrc
/// (or any external consumer such as a ROS2 node).
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NVMM_SHM_MAGIC   0x4E564D4D  /* "NVMM" */
#define NVMM_SHM_VERSION 1

typedef struct NvmmShmHeader {
    uint32_t magic;           /* NVMM_SHM_MAGIC */
    uint32_t version;         /* protocol version */
    uint32_t width;
    uint32_t height;
    uint32_t format;          /* GstVideoFormat enum value */
    uint32_t data_size;       /* size of frame data following this header */
    uint32_t num_planes;
    uint32_t pitches[4];      /* per-plane pitch in bytes */
    uint32_t offsets[4];      /* per-plane offset in bytes */
    int32_t  dmabuf_fd;       /* DMA-buf fd (-1 if not exported) */
    uint64_t frame_number;    /* monotonic frame counter */
    uint64_t timestamp_ns;    /* PTS in nanoseconds */
    uint32_t ready;           /* set to 1 when frame data is valid */
    uint32_t _reserved[8];
} NvmmShmHeader;

#ifdef __cplusplus
}
#endif
