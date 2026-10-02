#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "preprocess.hpp"
#include "roi_preprocess.hpp"
#include "nvmm_frame.h"
#include "test_harness.h"

namespace {

struct GstInit { GstInit() { gst_init(nullptr, nullptr); } } _gst_init;

constexpr float kYoloScale = 1.f / 255.f;

struct NetInput {
    int w, h;
    float *dev = nullptr;
    std::vector<float> host;
    NetInput(int w_, int h_) : w(w_), h(h_), host((size_t)3 * w_ * h_)
    {
        if (cudaMalloc((void **)&dev, host.size() * sizeof(float)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed");
    }
    ~NetInput() { cudaFree(dev); }
    void fetch(cudaStream_t s)
    {
        cudaStreamSynchronize(s);
        cudaMemcpy(host.data(), dev, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    }
    float at(int c, int x, int y) const { return host[((size_t)c * h + y) * w + x]; }
};

using Expect = std::function<float(int c, int x, int y)>;

int mismatches(const NetInput &in, const Expect &expect)
{
    int wrong = 0;
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < in.h; y++)
            for (int x = 0; x < in.w; x++)
                if (in.at(c, x, y) != expect(c, x, y)) wrong++;
    return wrong;
}

struct Letterboxed {
    NetInput in{32, 32};
    nvmm::LetterboxInfo lb;
    Letterboxed(int fw, int fh, Yuv colour, bool rgb, float scale)
    {
        NvBufSurface *src = pitch_nv12(fw, fh, solid(colour));
        nvmm::Preprocessor pre;
        std::string err;
        if (!pre.configure(32, 32, fw, fh, rgb, scale, nullptr, err) ||
            !pre.run(src, in.dev, lb, err))
            throw std::runtime_error(err);
        in.fetch(nullptr);
        NvBufSurfaceDestroy(src);
    }
};

TEST(wide_frame_is_letterboxed_with_pad_114_rows_above_and_below_the_image) {
    Letterboxed r(64, 32, kRed, true, kYoloScale);
    const Rgba red = vic_rgba_at(solid(kRed), 64, 32, 0, 0);
    ASSERT_EQ(r.lb.scale, 0.5f);
    ASSERT_EQ(r.lb.pad_x, 0.f);
    ASSERT_EQ(r.lb.pad_y, 8.f);
    ASSERT_EQ(r.lb.frame_w, 64);
    ASSERT_EQ(r.lb.frame_h, 32);
    ASSERT_EQ(mismatches(r.in, [&](int c, int, int y) {
        return (y < 8 || y >= 24 ? 114.f : (float)red[c]) * kYoloScale;
    }), 0);
}

TEST(tall_frame_is_pillarboxed_with_pad_114_columns_left_and_right) {
    Letterboxed r(32, 64, kRed, true, 1.f);
    const Rgba red = vic_rgba_at(solid(kRed), 32, 64, 0, 0);
    ASSERT_EQ(r.lb.pad_x, 8.f);
    ASSERT_EQ(r.lb.pad_y, 0.f);
    ASSERT_EQ(mismatches(r.in, [&](int c, int x, int) {
        return x < 8 || x >= 24 ? 114.f : (float)red[c];
    }), 0);
}

TEST(bgr_colour_order_puts_blue_in_plane_zero_and_red_in_plane_two) {
    Letterboxed r(64, 32, kRed, false, 1.f);
    const Rgba red = vic_rgba_at(solid(kRed), 64, 32, 0, 0);
    ASSERT_EQ(mismatches(r.in, [&](int c, int, int y) {
        return y < 8 || y >= 24 ? 114.f : (float)red[2 - c];
    }), 0);
}

Nv12Painter red_left_blue_right()
{
    return [](int x, int) { return x < 32 ? kRed : kBlue; };
}

std::string roi_run(NetInput &in, float l, float t, float w, float h,
                    const float *offsets = nullptr, const float *stds = nullptr)
{
    NvBufSurface *src = pitch_nv12(64, 32, red_left_blue_right());
    nvmm::RoiPreprocessor pre;
    std::string err;
    if (pre.configure(in.w, in.h, true, 1.f, offsets, stds, nullptr, err) &&
        pre.run(src, l, t, w, h, in.dev, err))
        in.fetch(nullptr);
    NvBufSurfaceDestroy(src);
    return err;
}

TEST(roi_input_holds_only_the_pixels_inside_the_requested_rect) {
    const Rgba red = vic_rgba_at(red_left_blue_right(), 64, 32, 0, 0);
    const Rgba blue = vic_rgba_at(red_left_blue_right(), 64, 32, 63, 0);
    NetInput left(16, 16), right(16, 16);
    ASSERT_EQ(roi_run(left, 8, 8, 16, 16), std::string());
    ASSERT_EQ(roi_run(right, 40, 8, 16, 16), std::string());
    ASSERT_EQ(mismatches(left, [&](int c, int, int) { return (float)red[c]; }), 0);
    ASSERT_EQ(mismatches(right, [&](int c, int, int) { return (float)blue[c]; }), 0);
}

TEST(roi_offsets_and_std_normalise_each_channel_as_x_minus_offset_over_std) {
    const Rgba red = vic_rgba_at(red_left_blue_right(), 64, 32, 0, 0);
    const float off[3] = {0.5f, 1.5f, 2.5f};
    const float sd[3] = {2.f, 4.f, 8.f};
    NetInput in(16, 16);
    ASSERT_EQ(roi_run(in, 8, 8, 16, 16, off, sd), std::string());
    ASSERT_EQ(mismatches(in, [&](int c, int, int) {
        return ((float)red[c] - off[c]) / sd[c];
    }), 0);
}

TEST(roi_std_without_offsets_still_divides_each_channel) {
    const Rgba red = vic_rgba_at(red_left_blue_right(), 64, 32, 0, 0);
    const float sd[3] = {2.f, 4.f, 8.f};
    NetInput in(16, 16);
    ASSERT_EQ(roi_run(in, 8, 8, 16, 16, nullptr, sd), std::string());
    ASSERT_EQ(mismatches(in, [&](int c, int, int) { return (float)red[c] / sd[c]; }), 0);
}

TEST(roi_std_value_of_zero_is_rejected_at_configure) {
    const float sd[3] = {1.f, 0.f, 1.f};
    NetInput in(16, 16);
    ASSERT_EQ(roi_run(in, 8, 8, 16, 16, nullptr, sd), std::string("std-values contains 0"));
}

TEST(roi_clamped_to_under_two_pixels_is_rejected_as_degenerate) {
    NetInput in(16, 16);
    ASSERT_EQ(roi_run(in, 64, 8, 8, 16), std::string("ROI degenerate after clamping"));
}

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
