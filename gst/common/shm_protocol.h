/// NVMM shared-memory IPC wire formats.
///
/// Two byte-layouts coexist. Each carries an explicit `version` byte code so
/// a consumer can inspect the header and reject or support the protocol
/// before touching any other fields.
///
///   "copy"  (NvmmShmCopyHeader, NVMM_SHM_PROTO_COPY = 1)
///     Header + plane-interleaved pixel data live in one shm segment. The
///     producer memcpy's each frame in; the consumer memcpy's each frame
///     out. Works on any Linux + NVIDIA SDK but pays two CPU copies per
///     frame. Used by the JetPack 5 / L4T 35.x backend where neither
///     NvBufSurfaceImport nor NvBufSurfaceMapParams is available.
///
///   "pool"  (NvmmShmPoolHeader, NVMM_SHM_PROTO_POOL = 2)
///     Header holds metadata + per-slot ref counts only; no pixels in shm.
///     The producer exports a pool of NVMM DMA-buf fds over a unix-domain
///     socket (SCM_RIGHTS). Consumers import the fds with NvBufSurfaceImport
///     and read directly from GPU memory. Used by the JetPack 6 / L4T 36.x+
///     backend.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NVMM_SHM_MAGIC        0x4E564D4D    /* "NVMM" */

/* Wire-format version codes. These integers go on disk — do not renumber. */
#define NVMM_SHM_PROTO_COPY   1
#define NVMM_SHM_PROTO_POOL   2

#define NVMM_POOL_SIZE_MAX    32            /* compile-time cap for pool protocol */

/* -------------------------------------------------------------- */
/* "copy" protocol — header + inline pixel data in the same shm    */
/* -------------------------------------------------------------- */
typedef struct NvmmShmCopyHeader {
    uint32_t magic;             /* NVMM_SHM_MAGIC */
    uint32_t version;           /* NVMM_SHM_PROTO_COPY */
    uint32_t width;
    uint32_t height;
    uint32_t format;            /* GstVideoFormat enum value */
    uint32_t data_size;         /* bytes of pixel data following this header */
    uint32_t num_planes;
    uint32_t pitches[4];
    uint32_t offsets[4];
    int32_t  dmabuf_fd;         /* -1 when no fd exported */
    uint64_t frame_number;
    uint64_t timestamp_ns;
    uint32_t ready;
    uint32_t _reserved[8];
} NvmmShmCopyHeader;

/* -------------------------------------------------------------- */
/* "pool" protocol — metadata-only shm, fds over unix socket       */
/* -------------------------------------------------------------- */
/*
 * Fields below are shared across processes. They are NOT declared `volatile`
 * — `volatile` blocks compiler register-caching but provides no atomicity
 * or cross-CPU ordering, which is the wrong tool for IPC synchronization.
 *
 * Access them from C/C++ code using the __atomic_* builtins with explicit
 * memory order:
 *
 *   reads       -> __atomic_load_n(&field, __ATOMIC_ACQUIRE)
 *   publishes   -> __atomic_store_n(&field, v, __ATOMIC_RELEASE)
 *   CAS         -> __atomic_compare_exchange_n(..., __ATOMIC_ACQ_REL, ...)
 *   fences      -> __atomic_thread_fence(__ATOMIC_ACQUIRE/RELEASE)
 *
 * Rationale for __atomic_* over std::atomic<T> on C++14: the C++14
 * standard requires an atomic object to be constructed via its
 * constructor, and does not guarantee its layout matches plain T.
 * Reinterpret-casting a shm byte range to std::atomic<T>* is
 * technically UB and ABI-implementation-defined. std::atomic_ref<T>
 * (C++20) legitimizes the pattern but the project is pinned to
 * cpp_std=c++14. __atomic_* works on plain integral fields, lowers to
 * the same machine code std::atomic<T> would on supported architectures
 * (aarch64 ldar/stlr, x86 mov+mfence), and compiles identically in C
 * and C++.
 *
 * TODO: when we drop pre-C++20 toolchain support, migrate these to
 * std::atomic_ref with the same memory_order arguments.
 *
 * uint32_t / int32_t / uint64_t are lock-free on every Linux target
 * this project supports (aarch64 + x86_64 glibc).
 */
typedef struct NvmmShmPoolHeader {
    uint32_t magic;
    uint32_t version;           /* NVMM_SHM_PROTO_POOL */
    uint32_t width;
    uint32_t height;
    uint32_t format;
    uint32_t pool_size;         /* number of pool buffers, <= NVMM_POOL_SIZE_MAX */
    uint32_t num_planes;
    uint32_t pitches[4];
    uint32_t offsets[4];
    char     socket_path[108];  /* unix socket path for SCM_RIGHTS fd passing */

    /* Updated per frame. Access via __atomic_* — see note above. */
    uint32_t write_idx;
    uint64_t frame_number;
    uint64_t timestamp_ns;
    uint32_t ready;

    /* Per-slot ref counts — access via __atomic_* (see note above):
         0    = idle; producer may claim via CAS 0 -> -1
        -1    = writer lock held; consumers skip this slot
         >0   = N consumers currently reading
       The producer recycles a slot only when its ref == 0. */
    int32_t ref_counts[NVMM_POOL_SIZE_MAX];

    uint32_t _reserved[16];
} NvmmShmPoolHeader;

#ifdef __cplusplus
}
#endif
