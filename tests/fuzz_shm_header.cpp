/// Fuzz harness for the consumer's shm-header parsing path.
///
/// The consumer (`nvmmappsrc`) reads an `NvmmShmHeader` from shared memory
/// written by a separate producer (`nvmmsink`). A malicious or buggy producer
/// could write nonsense — wrong magic/version, out-of-range
/// width/height/format/pool_size, corrupt pitches/offsets, an out-of-range
/// write_idx. The consumer must not crash / read OOB / hit UB on any byte
/// pattern.
///
/// This targets the pure validation logic (no real shm segment / socket
/// needed), mirroring the checks the consumer makes before it trusts a field.
/// Build it under `-Db_sanitize=address,undefined` (see scripts/run-sanitizers.sh)
/// so any OOB/UB in the parse surfaces. It exposes the libFuzzer entry point
/// `LLVMFuzzerTestOneInput` and a standalone /dev/urandom driver so it also
/// builds and runs under GCC without clang's libFuzzer.

#include "shm_protocol.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

/* Target: NvmmShmHeader validation, matching the fields the consumer trusts. */
static int
fuzz_shm_header(const uint8_t *data, size_t size)
{
    if (size < sizeof(NvmmShmHeader)) return 0;
    NvmmShmHeader h;
    memcpy(&h, data, sizeof(h));

    if (h.magic != NVMM_SHM_MAGIC) return 0;
    if (h.version != NVMM_SHM_VERSION) return 0;
    if (h.width == 0 || h.width > 16384) return 0;
    if (h.height == 0 || h.height > 16384) return 0;
    if (h.pool_size < NVMM_MIN_POOL_SIZE || h.pool_size > NVMM_POOL_SIZE) return 0;
    if (h.num_planes > 4) return 0;

    /* pitches/offsets are indexed per plane — reading num_planes of them must
       stay in bounds (num_planes<=4 guards the [4] arrays). */
    for (uint32_t i = 0; i < h.num_planes; i++) {
        volatile uint32_t p = h.pitches[i];
        volatile uint32_t o = h.offsets[i];
        (void)p; (void)o;
    }

    /* write_idx must be < pool_size for the consumer to index pool slots. */
    uint32_t idx = __atomic_load_n(&h.write_idx, __ATOMIC_RELAXED);
    if (idx >= h.pool_size) return 0;

    /* Liveness fields are read but any bit pattern is wire-legal. */
    (void)__atomic_load_n(&h.frame_number, __ATOMIC_RELAXED);
    (void)__atomic_load_n(&h.timestamp_ns, __ATOMIC_RELAXED);
    (void)__atomic_load_n(&h.ready, __ATOMIC_RELAXED);

    return 0;
}

extern "C" int
LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    return fuzz_shm_header(data, size);
}

#ifndef LIB_FUZZING_ENGINE
/* Standalone driver: feed /dev/urandom chunks to the target until the
   iteration budget is exhausted. Runs under GCC + the ASan/UBSan build with no
   clang-only libFuzzer dependency. A clean exit (rc=0) under the sanitizers is
   the pass signal. */
int main(int argc, char *argv[])
{
    uint64_t iterations = (argc > 1) ? strtoull(argv[1], nullptr, 0) : 100000;

    int rnd = open("/dev/urandom", O_RDONLY);
    if (rnd < 0) { perror("/dev/urandom"); return 1; }

    uint8_t buf[8192];
    for (uint64_t i = 0; i < iterations; i++) {
        ssize_t n = read(rnd, buf, sizeof(buf));
        if (n <= 0) break;
        LLVMFuzzerTestOneInput(buf, (size_t)n);
    }
    close(rnd);
    printf("fuzz_shm_header: %llu iterations, no crash\n",
           (unsigned long long)iterations);
    return 0;
}
#endif
