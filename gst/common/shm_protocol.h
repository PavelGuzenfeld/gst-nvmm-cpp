/// NVMM shared memory IPC protocol.
/// Defines the header layout for frames shared between nvmmsink and nvmmappsrc
/// (or any external consumer such as a ROS2 node).
///
/// Version 1: CPU-copy mode (frame data follows header in shm)
/// Version 2: Zero-copy mode (pool of NVMM buffers, fds passed via unix socket)
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NVMM_SHM_MAGIC   0x4E564D4D  /* "NVMM" */
#define NVMM_SHM_VERSION 1
#define NVMM_SHM_VERSION_ZC 2  /* zero-copy version */

#define NVMM_POOL_SIZE 16  /* number of buffers in the zero-copy pool */

/* Version 1: CPU-copy header (frame data in shm after header) */
typedef struct NvmmShmHeader {
    uint32_t magic;           /* NVMM_SHM_MAGIC */
    uint32_t version;         /* NVMM_SHM_VERSION or NVMM_SHM_VERSION_ZC */
    uint32_t width;
    uint32_t height;
    uint32_t format;          /* GstVideoFormat enum value */
    uint32_t data_size;       /* size of frame data following this header (v1 only) */
    uint32_t num_planes;
    uint32_t pitches[4];      /* per-plane pitch in bytes */
    uint32_t offsets[4];      /* per-plane offset in bytes */
    int32_t  dmabuf_fd;       /* DMA-buf fd (-1 if not exported) */
    uint64_t frame_number;    /* monotonic frame counter */
    uint64_t timestamp_ns;    /* PTS in nanoseconds */
    uint32_t ready;           /* set to 1 when frame data is valid */
    uint32_t _reserved[8];
} NvmmShmHeader;

/* Version 2: Zero-copy header (no frame data in shm, buffers shared via fds) */
typedef struct NvmmShmHeaderZC {
    uint32_t magic;           /* NVMM_SHM_MAGIC */
    uint32_t version;         /* NVMM_SHM_VERSION_ZC */
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
} NvmmShmHeaderZC;

#ifdef __cplusplus
}
#endif
