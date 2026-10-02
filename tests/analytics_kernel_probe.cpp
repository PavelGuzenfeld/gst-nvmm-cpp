#include "analytics_kernels.hpp"
#include "analytics_scene.h"
#include "low_texture_motion.hpp"
#include "test_harness.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

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

/// FMA contraction differs between host and device, so knife-edge mask pixels may
/// flip; bound the flip fraction and demand tight agreement everywhere else.
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
    ASSERT_TRUE(frac <= 0.001);
    ASSERT_TRUE(worst_ok <= 1e-3);
}

void parity_256_with_output_blur() { parity_case(256, 256, 3, 3); }
void parity_256_no_output_blur() { parity_case(256, 256, 17, 0); }
void parity_odd_size_1080p() { parity_case(1919, 1079, 5, 3); }

void device_api_pitched_zero_copy() {
    const int w = 253, h = 199;
    nvmm::img::Image<uint8_t> cur = make_scene(w, h, 13);
    nvmm::motion::LowTextureMotionParams p;

    nvmm::motion::LowTextureMotionCuda gpu;
    nvmm::img::Image<float> host_path;
    ASSERT_TRUE(gpu.run(cur, cur, cur, p, host_path));

    cudaStream_t stream;
    ASSERT_TRUE(cudaStreamCreate(&stream) == cudaSuccess);
    const size_t in_pitch = 320, out_pitch = 512;
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

struct Frames {
    nvmm::img::Image<uint8_t> cur, ra, rb;
};

int toggled(const std::vector<int> &steps, int i, int height) {
    int v = 0;
    for (int s : steps)
        if (i >= s) v = height - v;
    return v;
}

/// cur = f(x) + g(y) with steps of 112 on one axis and 15 on the other, so every Sobel
/// magnitude is 4 * {0, 15, 112, 113} and every blur tap is dyadic: host and device
/// agree bit for bit, and a 112 step lands blur-7 exactly on grad_thresh 14 three px away.
Frames exact_scene(int w, int h, bool major_on_x, unsigned seed) {
    const int major_len = major_on_x ? w : h, minor_len = major_on_x ? h : w;
    const std::vector<int> major = {1, 9, 20, 23, 26, 29, 32, 35, 38, 52, major_len - 4,
                                    major_len - 1};
    const std::vector<int> minor = {1, 12, 30, minor_len - 1};
    scene::Rng rng(seed);
    Frames f{nvmm::img::Image<uint8_t>(w, h), nvmm::img::Image<uint8_t>(w, h),
             nvmm::img::Image<uint8_t>(w, h)};
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const int a = major_on_x ? x : y, b = major_on_x ? y : x;
            const int c = 40 + toggled(major, a, 112) + toggled(minor, b, 15);
            f.cur.at(y, x) = (uint8_t)c;
            f.ra.at(y, x) = (uint8_t)(c + rng.uniform(-30, 31));
            f.rb.at(y, x) = (uint8_t)(c + rng.uniform(-30, 31));
        }
    return f;
}

long mismatches(const nvmm::img::Image<float> &a, const nvmm::img::Image<float> &b) {
    if (a.width() != b.width() || a.height() != b.height()) return -1;
    long n = 0;
    for (int y = 0; y < a.height(); y++)
        for (int x = 0; x < a.width(); x++) n += a.at(y, x) != b.at(y, x);
    return n;
}

constexpr uint32_t kUnwritten = 0xFFFFFFFFu;

/// Output rows carry 7 padding floats and one spare row, all pre-filled with
/// kUnwritten; `untouched` reports whether any write landed outside the w x h frame.
bool run_pitched(nvmm::motion::LowTextureMotionCuda &gpu, const Frames &f,
                 const nvmm::motion::LowTextureMotionParams &p, nvmm::img::Image<float> &out,
                 bool &untouched) {
    const int w = f.cur.width(), h = f.cur.height();
    const size_t in_pitch = (size_t)w + 13, out_pitch = (size_t)w + 7;
    uint8_t *d_in[3] = {nullptr, nullptr, nullptr};
    const nvmm::img::Image<uint8_t> *src[3] = {&f.cur, &f.ra, &f.rb};
    for (int i = 0; i < 3; i++) {
        cudaMalloc(&d_in[i], in_pitch * h);
        cudaMemcpy2D(d_in[i], in_pitch, src[i]->data(), (size_t)w, (size_t)w, (size_t)h,
                     cudaMemcpyHostToDevice);
    }
    const size_t out_words = out_pitch * (h + 1);
    float *d_out = nullptr;
    cudaMalloc(&d_out, out_words * sizeof(float));
    cudaMemset(d_out, 0xFF, out_words * sizeof(float));
    nvmm::motion::DevicePlane<const uint8_t> pc{d_in[0], w, h, (std::ptrdiff_t)in_pitch},
        pa{d_in[1], w, h, (std::ptrdiff_t)in_pitch}, pb{d_in[2], w, h, (std::ptrdiff_t)in_pitch};
    nvmm::motion::DevicePlane<float> po{d_out, w, h, (std::ptrdiff_t)out_pitch};
    const bool ok = gpu.run_device(pc, pa, pb, p, po, nullptr) &&
                    cudaStreamSynchronize(nullptr) == cudaSuccess;
    std::vector<uint32_t> raw(out_words);
    cudaMemcpy(raw.data(), d_out, out_words * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 3; i++) cudaFree(d_in[i]);
    cudaFree(d_out);
    out = nvmm::img::Image<float>(w, h);
    untouched = true;
    for (size_t i = 0; i < out_words; i++) {
        const int y = (int)(i / out_pitch), x = (int)(i % out_pitch);
        if (y < h && x < w) std::memcpy(&out.at(y, x), &raw[i], sizeof(float));
        else untouched = untouched && raw[i] == kUnwritten;
    }
    return ok;
}

nvmm::motion::LowTextureMotionParams params(int grad_blur, int close_k, int diff_blur,
                                            int border) {
    nvmm::motion::LowTextureMotionParams p;
    p.grad_blur = grad_blur;
    p.close_k = close_k;
    p.diff_blur = diff_blur;
    p.border = border;
    return p;
}

void expect_bit_exact(const Frames &f, const nvmm::motion::LowTextureMotionParams &p) {
    const nvmm::img::Image<float> host = nvmm::motion::low_texture_motion(f.cur, f.ra, f.rb, p);
    nvmm::motion::LowTextureMotionCuda packed_gpu, pitched_gpu;
    nvmm::img::Image<float> packed, pitched;
    bool untouched = false;
    ASSERT_TRUE(packed_gpu.run(f.cur, f.ra, f.rb, p, packed));
    ASSERT_TRUE(run_pitched(pitched_gpu, f, p, pitched, untouched));
    printf("[k=%d close=%d blur=%d border=%d diff=%ld/%ld] ", p.grad_blur, p.close_k,
           p.diff_blur, p.border, mismatches(host, packed), mismatches(host, pitched));
    ASSERT_EQ(mismatches(host, packed), 0);
    ASSERT_EQ(mismatches(host, pitched), 0);
    ASSERT_TRUE(untouched);
}

const std::vector<nvmm::motion::LowTextureMotionParams> &exact_cases() {
    static const std::vector<nvmm::motion::LowTextureMotionParams> cases = {
        nvmm::motion::LowTextureMotionParams(), params(7, 15, 3, 0), params(7, 1, 0, 0),
        params(5, 3, 0, 0), params(3, 7, 5, 5), params(7, 3, 1, 1)};
    return cases;
}

void exact_parity_x_steps_partial_warp_width() {
    const Frames f = exact_scene(77, 64, true, 21);
    for (const auto &p : exact_cases()) expect_bit_exact(f, p);
}

void exact_parity_y_steps_partial_block_height() {
    const Frames f = exact_scene(64, 77, false, 22);
    for (const auto &p : exact_cases()) expect_bit_exact(f, p);
}

void run_device_rejects_degenerate_or_oversized_input() {
    const Frames f = exact_scene(64, 64, true, 23);
    nvmm::motion::LowTextureMotionCuda gpu;
    uint8_t *d_in = nullptr;
    float *d_out = nullptr;
    ASSERT_TRUE(cudaMalloc(&d_in, 64 * 64) == cudaSuccess);
    ASSERT_TRUE(cudaMalloc(&d_out, 64 * 64 * sizeof(float)) == cudaSuccess);
    cudaMemcpy(d_in, f.cur.data(), 64 * 64, cudaMemcpyHostToDevice);
    auto run = [&](int w, int h, const nvmm::motion::LowTextureMotionParams &p,
                   const uint8_t *c, const uint8_t *a, const uint8_t *b, float *o) {
        const bool ok = gpu.run_device({c, w, h, w}, {a, w, h, w}, {b, w, h, w}, p, {o, w, h, w},
                                       nullptr);
        ASSERT_TRUE(cudaStreamSynchronize(nullptr) == cudaSuccess);
        return ok;
    };
    const nvmm::motion::LowTextureMotionParams p, tiny = params(1, 1, 0, 0);
    ASSERT_TRUE(!run(1, 64, tiny, d_in, d_in, d_in, d_out));
    ASSERT_TRUE(!run(64, 1, tiny, d_in, d_in, d_in, d_out));
    ASSERT_TRUE(run(2, 2, tiny, d_in, d_in, d_in, d_out));
    ASSERT_TRUE(!run(64, 64, params(32, 15, 3, 12), d_in, d_in, d_in, d_out));
    ASSERT_TRUE(!run(64, 64, params(7, 15, 32, 12), d_in, d_in, d_in, d_out));
    ASSERT_TRUE(run(64, 64, params(31, 15, 31, 12), d_in, d_in, d_in, d_out));
    ASSERT_TRUE(!run(64, 64, p, nullptr, d_in, d_in, d_out));
    ASSERT_TRUE(!run(64, 64, p, d_in, nullptr, d_in, d_out));
    ASSERT_TRUE(!run(64, 64, p, d_in, d_in, nullptr, d_out));
    ASSERT_TRUE(!run(64, 64, p, d_in, d_in, d_in, nullptr));
    cudaFree(d_in);
    cudaFree(d_out);
    ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);
    ASSERT_TRUE(std::string(gpu.last_error()) == cudaGetErrorString(cudaSuccess));
}

void two_by_two_frame_matches_host() {
    Frames f{nvmm::img::Image<uint8_t>(2, 2), nvmm::img::Image<uint8_t>(2, 2),
             nvmm::img::Image<uint8_t>(2, 2)};
    const uint8_t cur[4] = {10, 50, 90, 130}, ra[4] = {20, 45, 100, 110}, rb[4] = {5, 52, 80, 150};
    for (int i = 0; i < 4; i++) {
        f.cur.data()[i] = cur[i];
        f.ra.data()[i] = ra[i];
        f.rb.data()[i] = rb[i];
    }
    expect_bit_exact(f, params(1, 1, 0, 0));
}

void wide_kernel_taps_do_not_leak_into_narrow_blur() {
    const Frames first = exact_scene(64, 64, true, 28), f = exact_scene(77, 64, true, 29);
    nvmm::motion::LowTextureMotionCuda gpu;
    nvmm::img::Image<float> out;
    ASSERT_TRUE(gpu.run(first.cur, first.ra, first.rb, params(31, 15, 31, 12), out));
    expect_bit_exact(f, params(7, 3, 3, 0));
}

/// The first error sticks in last_error(); a failed call must not fail the next one.
void failures_keep_first_error_then_recover() {
    nvmm::motion::LowTextureMotionCuda gpu;
    static const uint8_t byte = 0;
    static float word = 0.f;
    const int huge = 1 << 20;
    const bool ok = gpu.run_device({&byte, huge, huge, huge}, {&byte, huge, huge, huge},
                                   {&byte, huge, huge, huge},
                                   nvmm::motion::LowTextureMotionParams(),
                                   {&word, huge, huge, huge}, nullptr);
    printf("[%s] ", gpu.last_error());
    ASSERT_TRUE(!ok);
    const std::string oom = cudaGetErrorString(cudaErrorMemoryAllocation);
    ASSERT_TRUE(gpu.last_error() == oom);

    const Frames f = exact_scene(77, 64, true, 24);
    const nvmm::img::View<const uint8_t> short_stride(f.cur.data(), 77, 64, 76);
    const nvmm::motion::LowTextureMotionParams p;
    nvmm::img::Image<float> dev;
    ASSERT_TRUE(!gpu.run(short_stride, f.ra, f.rb, p, dev));
    ASSERT_TRUE(!gpu.run(f.cur, short_stride, f.rb, p, dev));
    ASSERT_TRUE(!gpu.run(f.cur, f.ra, short_stride, p, dev));
    ASSERT_TRUE(gpu.last_error() == oom);
    ASSERT_TRUE(gpu.run(f.cur, f.ra, f.rb, p, dev));
    ASSERT_EQ(mismatches(nvmm::motion::low_texture_motion(f.cur, f.ra, f.rb, p), dev), 0);
}

void one_instance_follows_size_and_path_changes() {
    const Frames small = exact_scene(77, 64, true, 25), tall = exact_scene(77, 200, true, 26),
                 wide = exact_scene(150, 64, false, 27);
    const nvmm::motion::LowTextureMotionParams p;
    nvmm::motion::LowTextureMotionCuda gpu;
    nvmm::img::Image<float> out;
    bool untouched = false;
    auto host = [&](const Frames &f) { return nvmm::motion::low_texture_motion(f.cur, f.ra, f.rb, p); };
    ASSERT_TRUE(run_pitched(gpu, small, p, out, untouched));
    ASSERT_EQ(mismatches(host(small), out), 0);
    for (const Frames *f : {&small, &tall, &small, &wide}) {
        ASSERT_TRUE(gpu.run(f->cur, f->ra, f->rb, p, out));
        ASSERT_EQ(mismatches(host(*f), out), 0);
    }
    ASSERT_TRUE(run_pitched(gpu, wide, p, out, untouched));
    ASSERT_EQ(mismatches(host(wide), out), 0);
}

}

/// Driven from main(), not TEST: static init across TUs would race the .cu TU's
/// CUDA symbol registration ("invalid device symbol").
int main() {
    printf("== analytics CUDA kernel parity ==\n");
    RUN_TEST(parity_256_with_output_blur);
    RUN_TEST(parity_256_no_output_blur);
    RUN_TEST(parity_odd_size_1080p);
    RUN_TEST(device_api_pitched_zero_copy);
    RUN_TEST(strided_view_upload);
    RUN_TEST(exact_parity_x_steps_partial_warp_width);
    RUN_TEST(exact_parity_y_steps_partial_block_height);
    RUN_TEST(run_device_rejects_degenerate_or_oversized_input);
    RUN_TEST(two_by_two_frame_matches_host);
    RUN_TEST(wide_kernel_taps_do_not_leak_into_narrow_blur);
    RUN_TEST(failures_keep_first_error_then_recover);
    RUN_TEST(one_instance_follows_size_and_path_changes);
    return tests_failed > 0 ? 1 : 0;
}
