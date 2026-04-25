/// NVMM shared-memory IPC wire format.
///
/// Single protocol: pool + SCM_RIGHTS fd passing. The shm segment carries
/// metadata + per-slot ref counts only; pixel data lives in NVMM DMA-buf
/// surfaces whose fds are passed over a unix-domain socket. Consumers
/// `NvBufSurfaceImport` the fds and read directly from GPU memory — no
/// further copies.
///
/// Requires the NVMM cross-process import API (NvBufSurfaceImport,
/// NvBufSurfaceGetMapParams), which ships in L4T R35.3.1+ (JetPack 5.1.1)
/// and any JetPack 6.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NVMM_SHM_MAGIC        0x4E564D4D    /* "NVMM" */

/* Wire-format version. Bytes on disk — never renumber, only bump.
 *   1 (legacy "copy" protocol) — removed; never released
 *   2 (legacy "pool" v1)        — removed; slots shared cache lines
 *   3 (legacy "pool" v2)        — removed; no wake_counter, polled wait
 *   4 = current "pool" protocol — adds wake_counter for cross-process
 *       futex wakeup; consumer no longer polls in 1ms increments. */
#define NVMM_SHM_PROTO_POOL   4

#define NVMM_POOL_SIZE_MAX    32            /* compile-time cap for pool protocol */
#define NVMM_CACHE_LINE       64            /* aarch64 + x86_64 mainstream size */

/*
 * IPC synchronization rules — READ BEFORE TOUCHING THIS STRUCT.
 *
 * (1) Use __atomic_* with explicit memory order on plain integers, not
 *     std::atomic<T>. Reinterpret-casting an shm byte range to
 *     std::atomic<T>* is UB and ABI-implementation-defined in C++14.
 *     std::atomic_ref<T> (C++20) is the standard answer — TODO when we
 *     bump the language level.
 *
 * (2) Cache-line aware layout — avoid false sharing on SMP.
 *     - Setup-time fields (magic/version/width/...) go first, read-mostly.
 *     - Hot publish fields (ready/write_idx/frame_number/timestamp_ns)
 *       occupy their own cache line; the producer's RELEASE on ready
 *       does not invalidate any consumer's ref_count line.
 *     - Per-slot ref_counts each sit on their OWN cache line. Two
 *       consumers hitting different pool slots never ping-pong a line.
 *
 * (3) uint32_t / int32_t / uint64_t are lock-free on every Linux
 *     target this project supports (aarch64 + x86_64 glibc).
 */

/* A single ref-count padded out to a full cache line so concurrent
 * consumers on different slots don't thrash each other's L1. */
typedef struct __attribute__((aligned(NVMM_CACHE_LINE))) NvmmPoolSlotState {
    int32_t ref_count;
    /* Filler so each slot ends on a cache-line boundary. The compiler
     * will pad to a full cache line thanks to the struct alignment, but
     * we add explicit bytes so sizeof is defined and `static_assert`s
     * can pin it. */
    char _pad[NVMM_CACHE_LINE - sizeof(int32_t)];
} NvmmPoolSlotState;

typedef struct NvmmShmPoolHeader {
    /* ---------- setup-time fields (read-mostly) ---------- */
    uint32_t magic;
    uint32_t version;           /* NVMM_SHM_PROTO_POOL */
    uint32_t width;
    uint32_t height;
    uint32_t format;
    uint32_t pool_size;         /* <= NVMM_POOL_SIZE_MAX */
    uint32_t num_planes;
    uint32_t pitches[4];
    uint32_t offsets[4];
    char     socket_path[108];  /* unix socket path for SCM_RIGHTS */

    /* ---------- hot publish line (producer writes, consumers read) ---------- */
    /* Dedicated cache line. The RELEASE store on `ready` is the sync
     * edge for every non-atomic setup field above. wake_counter is
     * the futex address — producer increments + FUTEX_WAKEs after
     * every publish, consumers FUTEX_WAIT on it. */
    __attribute__((aligned(NVMM_CACHE_LINE))) uint32_t ready;
    uint32_t write_idx;
    uint64_t frame_number;
    uint64_t timestamp_ns;
    uint32_t wake_counter;
    /* Pad to end-of-cache-line so the next member starts fresh. */
    char _hot_pad[NVMM_CACHE_LINE
                  - sizeof(uint32_t) * 3   /* ready + write_idx + wake_counter */
                  - sizeof(uint64_t) * 2]; /* frame_number + timestamp_ns */

    /* ---------- per-slot ref counts ----------
     *   0    = idle; producer may claim via CAS 0 -> -1
     *  -1    = writer lock held; consumers skip this slot
     *  >0    = N consumers currently reading
     * Producer recycles a slot only when its ref == 0.
     * Each slot is on its own cache line (see NvmmPoolSlotState). */
    NvmmPoolSlotState slots[NVMM_POOL_SIZE_MAX];

    /* Future-proofing. */
    uint32_t _reserved[16];
} NvmmShmPoolHeader;

#ifdef __cplusplus
/* Pin the layout at compile time; any accidental struct change breaks
 * the build and prevents a silent wire-incompat change. */
#include <cstddef>
static_assert(sizeof(NvmmPoolSlotState) == NVMM_CACHE_LINE,
              "PoolSlotState must be exactly one cache line");
static_assert(alignof(NvmmPoolSlotState) == NVMM_CACHE_LINE,
              "PoolSlotState must be cache-line aligned");
static_assert(offsetof(NvmmShmPoolHeader, ready) % NVMM_CACHE_LINE == 0,
              "ready must start a fresh cache line");
static_assert(offsetof(NvmmShmPoolHeader, slots) % NVMM_CACHE_LINE == 0,
              "slots[] must start a fresh cache line");
#endif

#ifdef __cplusplus
}
#endif
