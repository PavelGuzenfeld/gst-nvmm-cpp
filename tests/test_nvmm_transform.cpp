#include "nvmm_buffer.hpp"
#include "nvmm_transform.hpp"
#include "nvmm_types.hpp"

#include <cstdio>
#include <cstring>

#include "test_harness.h"

namespace {

nvmm::Result<nvmm::NvmmBuffer> make_buffer(uint32_t w, uint32_t h,
                                             nvmm::ColorFormat fmt) {
    nvmm::SurfaceParams params;
    params.width = w;
    params.height = h;
    params.color_format = fmt;
    params.mem_type = nvmm::MemoryType::kDefault;
    return nvmm::NvmmBuffer::create(params);
}

TEST(scale_nv12) {
    auto src = make_buffer(1920, 1080, nvmm::ColorFormat::kNV12);
    auto dst = make_buffer(640, 480, nvmm::ColorFormat::kNV12);
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());

    auto map = src.value().map_write(0);
    ASSERT_TRUE(map.has_value());
    std::memset(map.value().data(), 0x42, map.value().size());
    src.value().unmap();

    auto result = nvmm::NvmmTransform::scale(src.value(), dst.value());
    ASSERT_TRUE(result.has_value());
}

TEST(crop_and_scale) {
    auto src = make_buffer(1920, 1080, nvmm::ColorFormat::kRGBA);
    auto dst = make_buffer(640, 480, nvmm::ColorFormat::kRGBA);
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());

    nvmm::CropRect crop{100, 100, 800, 600};
    auto result = nvmm::NvmmTransform::crop_and_scale(src.value(), dst.value(), crop);
    ASSERT_TRUE(result.has_value());
}

TEST(format_convert) {
    auto src = make_buffer(640, 480, nvmm::ColorFormat::kNV12);
    auto dst = make_buffer(640, 480, nvmm::ColorFormat::kRGBA);
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());

    auto result = nvmm::NvmmTransform::convert(src.value(), dst.value());
    ASSERT_TRUE(result.has_value());
}

TEST(transform_with_flip) {
    auto src = make_buffer(640, 480, nvmm::ColorFormat::kNV12);
    auto dst = make_buffer(640, 480, nvmm::ColorFormat::kNV12);
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());

    nvmm::TransformParams params;
    params.flip = nvmm::FlipMethod::kRotate180;
    auto result = nvmm::NvmmTransform::transform(src.value(), dst.value(), params);
    ASSERT_TRUE(result.has_value());
}

TEST(transform_rotate90) {
    auto src = make_buffer(640, 480, nvmm::ColorFormat::kNV12);
    auto dst = make_buffer(480, 640, nvmm::ColorFormat::kNV12);
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());

    nvmm::TransformParams params;
    params.flip = nvmm::FlipMethod::kRotate90CW;
    auto result = nvmm::NvmmTransform::transform(src.value(), dst.value(), params);
    ASSERT_TRUE(result.has_value());
}

TEST(transform_rotate270) {
    auto src = make_buffer(640, 480, nvmm::ColorFormat::kNV12);
    auto dst = make_buffer(480, 640, nvmm::ColorFormat::kNV12);
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());

    nvmm::TransformParams params;
    params.flip = nvmm::FlipMethod::kRotate90CCW;
    auto result = nvmm::NvmmTransform::transform(src.value(), dst.value(), params);
    ASSERT_TRUE(result.has_value());
}

TEST(transform_with_interpolation) {
    auto src = make_buffer(1920, 1080, nvmm::ColorFormat::kNV12);
    auto dst = make_buffer(640, 480, nvmm::ColorFormat::kNV12);
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());

    nvmm::TransformParams params;
    params.interpolation = nvmm::Interpolation::k5Tap;
    auto result = nvmm::NvmmTransform::transform(src.value(), dst.value(), params);
    ASSERT_TRUE(result.has_value());
}

TEST(transform_with_compute_mode) {
    auto src = make_buffer(1920, 1080, nvmm::ColorFormat::kNV12);
    auto dst = make_buffer(640, 480, nvmm::ColorFormat::kNV12);
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());

    nvmm::TransformParams params;
    params.compute = nvmm::ComputeMode::kVic;
    auto result = nvmm::NvmmTransform::transform(src.value(), dst.value(), params);
    ASSERT_TRUE(result.has_value());
}

TEST(transform_with_crop_and_flip) {
    auto src = make_buffer(1920, 1080, nvmm::ColorFormat::kNV12);
    auto dst = make_buffer(320, 240, nvmm::ColorFormat::kNV12);
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());

    nvmm::TransformParams params;
    params.src_crop = {0, 0, 960, 540};
    params.flip = nvmm::FlipMethod::kFlipHorizontal;
    auto result = nvmm::NvmmTransform::transform(src.value(), dst.value(), params);
    ASSERT_TRUE(result.has_value());
}

TEST(null_surface_fails) {
    nvmm::NvmmBuffer null_buf{nullptr};
    auto dst = make_buffer(640, 480, nvmm::ColorFormat::kNV12);
    ASSERT_TRUE(dst.has_value());

    auto result = nvmm::NvmmTransform::scale(null_buf, dst.value());
    ASSERT_TRUE(!result.has_value());
}

}

int main() {
    printf("=== NvmmTransform Tests ===\n");
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
