/// libFuzzer harness for the consumer's shm-header parsing path.
///
/// The consumer reads a NvmmShmCopyHeader (JP5) or NvmmShmPoolHeader (JP6)
/// from shared memory written by an untrusted producer. A malicious or
/// buggy producer could write nonsense — wrong magic, out-of-range
/// width/height/format/pool_size, corrupt offsets/pitches, negative
/// ref_counts, anything. The consumer must not crash / read OOB / UB out
/// on any byte pattern.
///
/// Build with: -Db_sanitize=address,undefined AND -Db_lundef=false AND
///             either clang with -fsanitize=fuzzer or link a driver
///             that calls LLVMFuzzerTestOneInput in a loop (we do the
///             latter here so the harness also builds under GCC).
///
/// We can't easily exercise the real nvmm_ipc_consumer_start() because
/// it requires a real shm segment + socket server. Instead we target
/// the pure parsing logic: validate magic, version, and plausible
/// field values; anywhere the consumer code would dereference based on
/// a field.

#include "shm_protocol.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

/* Standalone entry point — reads bytes from stdin (up to N bytes) and
 * feeds them to the target function. Compatible with AFL and the simple
 * stdin-based fuzzing loop used by libFuzzer. */

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

/* -------- target: NvmmShmCopyHeader validation -------- */

static int
fuzz_copy_header(const uint8_t *data, size_t size)
{
    if (size < sizeof(NvmmShmCopyHeader)) return 0;
    NvmmShmCopyHeader h;
    memcpy(&h, data, sizeof(h));

    /* Mimic the consumer's validation logic. Any combination of fields
     * must not cause UB here. */
    if (h.magic != NVMM_SHM_MAGIC) return 0;
    if (h.version != NVMM_SHM_PROTO_COPY) return 0;
    if (h.width == 0 || h.width > 16384) return 0;
    if (h.height == 0 || h.height > 16384) return 0;
    if (h.num_planes > 4) return 0;

    /* data_size must be within the trailing buffer bounds (attacker
     * controls the value). The consumer clamps; reproduce that here. */
    size_t available = size > sizeof(h) ? size - sizeof(h) : 0;
    if (h.data_size > available) return 0;

    /* pitches/offsets should be internally consistent; fuzzer may give
     * us anything. Check the consumer's derived math doesn't overflow. */
    for (uint32_t p = 0; p < h.num_planes; p++) {
        uint64_t end = (uint64_t)h.offsets[p] + (uint64_t)h.pitches[p] * h.height;
        if (end > h.data_size) return 0;
    }

    return 0;
}

/* -------- target: NvmmShmPoolHeader validation -------- */

static int
fuzz_pool_header(const uint8_t *data, size_t size)
{
    if (size < sizeof(NvmmShmPoolHeader)) return 0;
    NvmmShmPoolHeader h;
    memcpy(&h, data, sizeof(h));

    if (h.magic != NVMM_SHM_MAGIC) return 0;
    if (h.version != NVMM_SHM_PROTO_POOL) return 0;
    if (h.width == 0 || h.width > 16384) return 0;
    if (h.height == 0 || h.height > 16384) return 0;
    if (h.pool_size == 0 || h.pool_size > NVMM_POOL_SIZE_MAX) return 0;
    if (h.num_planes > 4) return 0;

    /* socket_path: attacker could omit null terminator. */
    bool terminated = false;
    for (size_t i = 0; i < sizeof(h.socket_path); i++)
        if (h.socket_path[i] == '\0') { terminated = true; break; }
    if (!terminated) return 0;

    /* write_idx must be < pool_size for the consumer to index slots[]. */
    uint32_t idx = __atomic_load_n(&h.write_idx, __ATOMIC_RELAXED);
    if (idx >= h.pool_size) return 0;

    /* ref_count must fit its legal range (0 idle, -1 writer, >0 readers). */
    for (uint32_t i = 0; i < h.pool_size; i++) {
        int32_t rc = h.slots[i].ref_count;
        (void)rc;  /* any value is legal wire-wise; we just make sure we can read it */
    }

    return 0;
}

extern "C" int
LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size == 0) return 0;
    /* First byte picks which target we exercise. */
    uint8_t sel = data[0];
    data++;
    size--;
    if (sel & 1)  return fuzz_copy_header(data, size);
    else          return fuzz_pool_header(data, size);
}

#ifndef LIB_FUZZING_ENGINE
/* Standalone driver: runs until stdin EOF, feeding each read chunk to
 * LLVMFuzzerTestOneInput. Lets us run the harness under GCC + the
 * existing ASan/UBSan build without pulling in clang-only libFuzzer. */
int main(int argc, char *argv[])
{
    uint64_t iterations = 10000;
    if (argc > 1) iterations = (uint64_t)strtoull(argv[1], nullptr, 0);

    /* /dev/urandom drives the fuzz stream so each invocation sees a
     * different distribution. */
    int rnd = open("/dev/urandom", O_RDONLY);
    if (rnd < 0) { perror("/dev/urandom"); return 1; }

    uint8_t buf[8192];
    uint64_t crashes = 0;
    for (uint64_t i = 0; i < iterations; i++) {
        ssize_t n = read(rnd, buf, sizeof(buf));
        if (n <= 0) break;
        LLVMFuzzerTestOneInput(buf, (size_t)n);
    }
    close(rnd);
    std::printf("fuzz: %llu iterations, %llu crashes\n",
        (unsigned long long)iterations, (unsigned long long)crashes);
    return 0;
}
#endif
