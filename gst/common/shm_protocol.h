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
#define NVMM_SHM_VERSION 2           /* bump on wire-layout changes */

#define NVMM_POOL_SIZE 16  /* number of buffers in the GPU-copy pool */

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

    uint32_t _reserved[16];
} NvmmShmHeader;

#ifdef __cplusplus
}
#endif
