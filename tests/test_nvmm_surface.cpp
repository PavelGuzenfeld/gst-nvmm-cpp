#include "nvmm_buffer.hpp"
#include "nvmm_transform.hpp"
#include "nvmm_types.hpp"

#ifdef NVMM_MOCK_API
#include "nvbufsurface_mock.h"
#else
#include <nvbufsurface.h>
#endif

#include <cstdio>
#include <cstring>

#include "test_harness.h"

namespace {

nvmm::Result<nvmm::NvmmBuffer> make_buffer(uint32_t w, uint32_t h, nvmm::ColorFormat fmt) {
    nvmm::SurfaceParams params;
    params.width = w;
    params.height = h;
    params.color_format = fmt;
    params.mem_type = nvmm::MemoryType::kDefault;
    return nvmm::NvmmBuffer::create(params);
}

TEST(create_reports_size_format_and_plane_count) {
    struct Case { uint32_t w, h; nvmm::ColorFormat fmt; uint32_t planes; };
    const Case cases[] = {
        {1920, 1080, nvmm::ColorFormat::kNV12, 2},
        {640, 480, nvmm::ColorFormat::kRGBA, 1},
        {640, 480, nvmm::ColorFormat::kI420, 3},
    };
    for (const Case &c : cases) {
        auto result = make_buffer(c.w, c.h, c.fmt);
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().width(), c.w);
        ASSERT_EQ(result.value().height(), c.h);
        ASSERT_EQ(result.value().format(), c.fmt);
        ASSERT_EQ(result.value().num_planes(), c.planes);
    }
}

TEST(create_zero_size_fails) {
    nvmm::SurfaceParams params;
    params.width = 0;
    params.height = 0;

    auto result = nvmm::NvmmBuffer::create(params);
    ASSERT_TRUE(!result.has_value());
    ASSERT_EQ(result.error().code, nvmm::ErrorCode::kInvalidParam);
}

TEST(move_semantics) {
    auto result = make_buffer(320, 240, nvmm::ColorFormat::kNV12);
    ASSERT_TRUE(result.has_value());

    nvmm::NvmmBuffer moved = std::move(result.value());
    ASSERT_TRUE(moved.valid());
    ASSERT_EQ(moved.width(), 320u);
}

TEST(release_prevents_destroy) {
    NvBufSurface* raw_ptr = nullptr;
    {
        auto result = make_buffer(64, 64, nvmm::ColorFormat::kNV12);
        ASSERT_TRUE(result.has_value());
        raw_ptr = result.value().release();
        ASSERT_TRUE(raw_ptr != nullptr);
        ASSERT_TRUE(!result.value().valid());
    }
    ASSERT_TRUE(raw_ptr->surfaceList != nullptr);
    ASSERT_EQ(raw_ptr->surfaceList[0].width, 64u);

    NvBufSurfaceDestroy(raw_ptr);
}

TEST(map_write_then_read_round_trips) {
    auto result = make_buffer(16, 16, nvmm::ColorFormat::kRGBA);
    ASSERT_TRUE(result.has_value());

    auto map_w = result.value().map_write(0);
    ASSERT_TRUE(map_w.has_value());
    ASSERT_TRUE(map_w.value().size() > 0);
    std::memset(map_w.value().data(), 0xAB, map_w.value().size());
    ASSERT_TRUE(result.value().unmap().has_value());

    auto map_r = result.value().map_read(0);
    ASSERT_TRUE(map_r.has_value());
    ASSERT_EQ(map_r.value().data()[0], 0xAB);
    ASSERT_TRUE(result.value().unmap().has_value());
}

TEST(export_fd) {
    auto result = make_buffer(64, 64, nvmm::ColorFormat::kNV12);
    ASSERT_TRUE(result.has_value());

    auto fd_result = result.value().export_fd();
    ASSERT_TRUE(fd_result.has_value());
    ASSERT_TRUE(fd_result.value() >= 0);
}

TEST(plane_info_nv12) {
    auto result = make_buffer(1920, 1080, nvmm::ColorFormat::kNV12);
    ASSERT_TRUE(result.has_value());

    auto p0 = result.value().plane_info(0);
    ASSERT_EQ(p0.width, 1920u);
    ASSERT_EQ(p0.height, 1080u);

    auto p1 = result.value().plane_info(1);
    ASSERT_TRUE(p1.width > 0);
    ASSERT_TRUE(p1.height > 0);
    ASSERT_TRUE(p1.height <= p0.height);
}

TEST(scale_crop_and_convert_succeed) {
    auto src = make_buffer(1920, 1080, nvmm::ColorFormat::kNV12);
    auto dst = make_buffer(640, 480, nvmm::ColorFormat::kNV12);
    auto src_rgba = make_buffer(1920, 1080, nvmm::ColorFormat::kRGBA);
    auto dst_rgba = make_buffer(640, 480, nvmm::ColorFormat::kRGBA);
    ASSERT_TRUE(src.has_value() && dst.has_value());
    ASSERT_TRUE(src_rgba.has_value() && dst_rgba.has_value());

    auto map = src.value().map_write(0);
    ASSERT_TRUE(map.has_value());
    std::memset(map.value().data(), 0x42, map.value().size());
    src.value().unmap();

    ASSERT_TRUE(nvmm::NvmmTransform::scale(src.value(), dst.value()).has_value());
    nvmm::CropRect crop{100, 100, 800, 600};
    ASSERT_TRUE(nvmm::NvmmTransform::crop_and_scale(src_rgba.value(), dst_rgba.value(), crop)
                    .has_value());
    ASSERT_TRUE(nvmm::NvmmTransform::convert(dst.value(), dst_rgba.value()).has_value());
}

/// Rotating by 90 or 270 swaps the destination's width and height.
TEST(transform_succeeds_for_each_flip_interpolation_and_compute) {
    struct Case { uint32_t src_w, src_h, dst_w, dst_h; nvmm::TransformParams params; };
    Case cases[6];
    cases[0] = {640, 480, 640, 480, {}};
    cases[0].params.flip = nvmm::FlipMethod::kRotate180;
    cases[1] = {640, 480, 480, 640, {}};
    cases[1].params.flip = nvmm::FlipMethod::kRotate90CW;
    cases[2] = {640, 480, 480, 640, {}};
    cases[2].params.flip = nvmm::FlipMethod::kRotate90CCW;
    cases[3] = {1920, 1080, 640, 480, {}};
    cases[3].params.interpolation = nvmm::Interpolation::k5Tap;
    cases[4] = {1920, 1080, 640, 480, {}};
    cases[4].params.compute = nvmm::ComputeMode::kVic;
    cases[5] = {1920, 1080, 320, 240, {}};
    cases[5].params.src_crop = {0, 0, 960, 540};
    cases[5].params.flip = nvmm::FlipMethod::kFlipHorizontal;

    for (const Case &c : cases) {
        auto src = make_buffer(c.src_w, c.src_h, nvmm::ColorFormat::kNV12);
        auto dst = make_buffer(c.dst_w, c.dst_h, nvmm::ColorFormat::kNV12);
        ASSERT_TRUE(src.has_value() && dst.has_value());
        ASSERT_TRUE(nvmm::NvmmTransform::transform(src.value(), dst.value(), c.params).has_value());
    }
}

TEST(null_surface_fails) {
    nvmm::NvmmBuffer null_buf{nullptr};
    auto dst = make_buffer(640, 480, nvmm::ColorFormat::kNV12);
    ASSERT_TRUE(dst.has_value());

    ASSERT_TRUE(!nvmm::NvmmTransform::scale(null_buf, dst.value()).has_value());
}

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
