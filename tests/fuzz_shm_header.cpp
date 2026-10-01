#include "shm_protocol.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

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

    for (uint32_t i = 0; i < h.num_planes; i++) {
        volatile uint32_t p = h.pitches[i];
        volatile uint32_t o = h.offsets[i];
        (void)p; (void)o;
    }

    uint32_t idx = __atomic_load_n(&h.write_idx, __ATOMIC_RELAXED);
    if (idx >= h.pool_size) return 0;

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
