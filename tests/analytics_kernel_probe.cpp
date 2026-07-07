/// Parity-checks the CUDA low_texture_motion (analytics_kernels.cu) against the
/// host implementation (itself golden-validated vs OpenCV) on deterministic
/// synthetic scenes — self-contained, no golden data. Same metric family as the
/// golden tests: the mask threshold is a hard nonlinearity, so a handful of
/// knife-edge pixels may flip (FMA contraction differs between host and
/// device); bound the flip fraction and demand tight agreement elsewhere.
/// Real-GPU test; runs only in the -Danalytics_cuda lane (suite nvidia_hwlib).
///
/// Tests run from main(), NOT via the self-registering TEST macro: that macro
/// executes in static constructors, and cross-TU static-init order would let
/// them race the .cu translation unit's CUDA symbol registration ("invalid
/// device symbol") — the same reason samurai_kernel_probe is main()-driven.
#include "analytics_kernels.hpp"
#include "analytics_scene.h"
#include "low_texture_motion.hpp"
#include "test_harness.h"

#include <cmath>

namespace {

#define RUN_TEST(fn) \
    do { \
        printf("  TEST %s ... ", #fn); \
        try { fn(); printf("PASS\n"); tests_passed++; } \
        catch (...) { printf("FAIL (exception)\n"); tests_failed++; } \
    } while (0)

nvmm::img::Image<uint8_t> make_scene(int w, int h, unsigned seed) {
    scene::Rng rng(seed);
    nvmm::img::Image<uint8_t> f(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) f.at(y, x) = scene::clamp_u8(110.f + rng.gauss(3.f));
    for (int y = h / 8; y < h / 3; y++)
        for (int x = w / 8; x < w / 3; x++) f.at(y, x) = (uint8_t)rng.uniform(0, 256);
    for (int i = 0; i < 60; i++)
        scene::fill_circle(f, rng.uniform(8, w - 8), rng.uniform(8, h - 8),
                           rng.uniform(2, 6), (uint8_t)rng.uniform(60, 200));
    return f;
}

void parity_case(int w, int h, unsigned seed, int diff_blur) {
    nvmm::img::Image<uint8_t> cur = make_scene(w, h, seed);
    scene::Rng rng(seed * 31 + 7);
    nvmm::img::Image<uint8_t> ra(w, h), rb(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            ra.at(y, x) = scene::clamp_u8((float)cur.at(y, x) - 25.f + rng.gauss(2.f));
            rb.at(y, x) = scene::clamp_u8((float)cur.at(y, x) - 10.f + rng.gauss(2.f));
        }

    nvmm::motion::LowTextureMotionParams p;
    p.diff_blur = diff_blur;
    nvmm::img::Image<float> host = nvmm::motion::low_texture_motion(cur, ra, rb, p);

    nvmm::motion::LowTextureMotionCuda gpu;
    nvmm::img::Image<float> dev;
    if (!gpu.run(cur, ra, rb, p, dev)) {
        printf("[cuda: %s] ", gpu.last_error());
        ASSERT_TRUE(false);
    }

    long over = 0;
    double worst_ok = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const double d = std::fabs((double)host.at(y, x) - dev.at(y, x));
            if (d > 0.5) over++;
            else worst_ok = std::max(worst_ok, d);
        }
    const double frac = (double)over / ((double)w * h);
    printf("[%dx%d blur=%d flips=%.4f%% worst=%.2e] ", w, h, diff_blur, 100.0 * frac,
           worst_ok);
    ASSERT_TRUE(frac <= 0.001);     // knife-edge threshold flips only
    ASSERT_TRUE(worst_ok <= 1e-3);  // tight agreement everywhere else
}

void parity_256_with_output_blur() { parity_case(256, 256, 3, 3); }
void parity_256_no_output_blur() { parity_case(256, 256, 17, 0); }
void parity_odd_size_1080p() { parity_case(1919, 1079, 5, 3); }

// zero-copy path: pitched device planes in, pitched device plane out, explicit
// stream — must agree exactly with the host-wrapper path on the same input.
void device_api_pitched_zero_copy() {
    const int w = 253, h = 199;
    nvmm::img::Image<uint8_t> cur = make_scene(w, h, 13);
    nvmm::motion::LowTextureMotionParams p;

    nvmm::motion::LowTextureMotionCuda gpu;
    nvmm::img::Image<float> host_path;
    ASSERT_TRUE(gpu.run(cur, cur, cur, p, host_path));

    cudaStream_t stream;
    ASSERT_TRUE(cudaStreamCreate(&stream) == cudaSuccess);
    const size_t in_pitch = 320, out_pitch = 512;   // deliberately > width
    uint8_t *d_in = nullptr;
    float *d_out = nullptr;
    ASSERT_TRUE(cudaMalloc(&d_in, in_pitch * h) == cudaSuccess);
    ASSERT_TRUE(cudaMalloc(&d_out, out_pitch * h * sizeof(float)) == cudaSuccess);
    ASSERT_TRUE(cudaMemcpy2D(d_in, in_pitch, cur.data(), (size_t)w, (size_t)w, (size_t)h,
                             cudaMemcpyHostToDevice) == cudaSuccess);

    nvmm::motion::DevicePlane<const uint8_t> in{d_in, w, h, (std::ptrdiff_t)in_pitch};
    nvmm::motion::DevicePlane<float> out{d_out, w, h, (std::ptrdiff_t)out_pitch};
    nvmm::motion::LowTextureMotionCuda gpu2;
    ASSERT_TRUE(gpu2.run_device(in, in, in, p, out, stream));
    ASSERT_TRUE(cudaStreamSynchronize(stream) == cudaSuccess);

    nvmm::img::Image<float> dev(w, h);
    ASSERT_TRUE(cudaMemcpy2D(dev.data(), (size_t)w * sizeof(float), d_out,
                             out_pitch * sizeof(float), (size_t)w * sizeof(float),
                             (size_t)h, cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);

    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) ASSERT_NEAR(host_path.at(y, x), dev.at(y, x), 0.0);
}

void strided_view_upload() {
    // run() must honor a stride > width (e.g. a mapped NVMM plane)
    nvmm::img::Image<uint8_t> big = make_scene(320, 240, 9);
    nvmm::img::View<const uint8_t> v(big.row(10) + 16, 256, 200, big.width());
    nvmm::img::Image<uint8_t> packed(256, 200);
    for (int y = 0; y < 200; y++)
        for (int x = 0; x < 256; x++) packed.at(y, x) = v.at(y, x);

    nvmm::motion::LowTextureMotionParams p;
    nvmm::motion::LowTextureMotionCuda gpu;
    nvmm::img::Image<float> a, b;
    ASSERT_TRUE(gpu.run(v, v, v, p, a));
    ASSERT_TRUE(gpu.run(packed, packed, packed, p, b));
    for (int y = 0; y < 200; y++)
        for (int x = 0; x < 256; x++) ASSERT_NEAR(a.at(y, x), b.at(y, x), 0.0);
}

}  // namespace

int main() {
    printf("== analytics CUDA kernel parity ==\n");
    RUN_TEST(parity_256_with_output_blur);
    RUN_TEST(parity_256_no_output_blur);
    RUN_TEST(parity_odd_size_1080p);
    RUN_TEST(device_api_pitched_zero_copy);
    RUN_TEST(strided_view_upload);
    return tests_failed > 0 ? 1 : 0;
}
