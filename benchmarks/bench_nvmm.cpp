/// Benchmark framework for NVMM operations.
/// Measures alloc/free, map/unmap, and transform throughput.
/// On host: uses mock API (measures framework overhead).
/// On Jetson: measures real VIC hardware performance.
///
/// Output: CSV to stdout for easy plotting.

#include "nvmm_buffer.hpp"
#include "nvmm_transform.hpp"
#include "nvmm_types.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::micro>;

struct BenchResult {
    const char *name;
    int iterations;
    double total_us;
    double avg_us;
    double min_us;
    double max_us;
};

static void print_csv_header() {
    printf("benchmark,iterations,total_us,avg_us,min_us,max_us\n");
}

static void print_csv_row(const BenchResult &r) {
    printf("%s,%d,%.2f,%.2f,%.2f,%.2f\n",
           r.name, r.iterations, r.total_us, r.avg_us, r.min_us, r.max_us);
}

/* --- Benchmarks --- */

static BenchResult bench_alloc_free(int iterations, uint32_t w, uint32_t h,
                                     nvmm::ColorFormat fmt) {
    BenchResult result = {};
    result.name = "alloc_free";
    result.iterations = iterations;
    result.min_us = 1e12;

    nvmm::SurfaceParams params;
    params.width = w;
    params.height = h;
    params.color_format = fmt;
    params.mem_type = nvmm::MemoryType::kSystemHeap;

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        auto iter_start = Clock::now();
        auto buf = nvmm::NvmmBuffer::create(params);
        auto iter_end = Clock::now();

        double us = Duration(iter_end - iter_start).count();
        if (us < result.min_us) result.min_us = us;
        if (us > result.max_us) result.max_us = us;
    }
    auto end = Clock::now();
    result.total_us = Duration(end - start).count();
    result.avg_us = result.total_us / iterations;

    return result;
}

static BenchResult bench_map_unmap(int iterations, uint32_t w, uint32_t h) {
    BenchResult result = {};
    result.name = "map_unmap";
    result.iterations = iterations;
    result.min_us = 1e12;

    nvmm::SurfaceParams params;
    params.width = w;
    params.height = h;
    params.color_format = nvmm::ColorFormat::kNV12;
    params.mem_type = nvmm::MemoryType::kSystemHeap;

    auto buf_result = nvmm::NvmmBuffer::create(params);
    if (!buf_result) {
        fprintf(stderr, "Failed to create buffer for benchmark\n");
        return result;
    }

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        auto iter_start = Clock::now();
        auto map = buf_result.value().map_read(0);
        buf_result.value().unmap();
        auto iter_end = Clock::now();

        double us = Duration(iter_end - iter_start).count();
        if (us < result.min_us) result.min_us = us;
        if (us > result.max_us) result.max_us = us;
    }
    auto end = Clock::now();
    result.total_us = Duration(end - start).count();
    result.avg_us = result.total_us / iterations;

    return result;
}

static BenchResult bench_transform(int iterations, uint32_t src_w, uint32_t src_h,
                                    uint32_t dst_w, uint32_t dst_h) {
    BenchResult result = {};
    result.name = "transform_scale";
    result.iterations = iterations;
    result.min_us = 1e12;

    nvmm::SurfaceParams src_params;
    src_params.width = src_w;
    src_params.height = src_h;
    src_params.color_format = nvmm::ColorFormat::kNV12;
    src_params.mem_type = nvmm::MemoryType::kSystemHeap;

    nvmm::SurfaceParams dst_params;
    dst_params.width = dst_w;
    dst_params.height = dst_h;
    dst_params.color_format = nvmm::ColorFormat::kNV12;
    dst_params.mem_type = nvmm::MemoryType::kSystemHeap;

    auto src = nvmm::NvmmBuffer::create(src_params);
    auto dst = nvmm::NvmmBuffer::create(dst_params);
    if (!src || !dst) {
        fprintf(stderr, "Failed to create buffers for benchmark\n");
        return result;
    }

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        auto iter_start = Clock::now();
        nvmm::NvmmTransform::scale(src.value(), dst.value());
        auto iter_end = Clock::now();

        double us = Duration(iter_end - iter_start).count();
        if (us < result.min_us) result.min_us = us;
        if (us > result.max_us) result.max_us = us;
    }
    auto end = Clock::now();
    result.total_us = Duration(end - start).count();
    result.avg_us = result.total_us / iterations;

    return result;
}

int main() {
    const int N = 1000;

    fprintf(stderr, "NVMM Benchmark (mock=%s, iterations=%d)\n",
#ifdef NVMM_MOCK_API
            "yes",
#else
            "no",
#endif
            N);

    print_csv_header();

    /* Allocation benchmarks */
    print_csv_row(bench_alloc_free(N, 1920, 1080, nvmm::ColorFormat::kNV12));
    print_csv_row(bench_alloc_free(N, 3840, 2160, nvmm::ColorFormat::kRGBA));

    /* Map/unmap benchmarks */
    print_csv_row(bench_map_unmap(N, 1920, 1080));
    print_csv_row(bench_map_unmap(N, 3840, 2160));

    /* Transform benchmarks */
    print_csv_row(bench_transform(N, 1920, 1080, 640, 480));
    print_csv_row(bench_transform(N, 3840, 2160, 1920, 1080));

    return 0;
}
