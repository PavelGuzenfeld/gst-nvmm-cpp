/// Unit tests for nvmm::NvmmTransform (NvBufSurfTransform wrapper).

#include "nvmm_buffer.hpp"
#include "nvmm_transform.hpp"
#include "nvmm_types.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>

namespace {

int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    struct test_reg_##name { test_reg_##name() { \
        printf("  TEST %s ... ", #name); \
        try { test_##name(); printf("PASS\n"); tests_passed++; } \
        catch (...) { printf("FAIL (exception)\n"); tests_failed++; } \
    } } test_reg_inst_##name; \
    static void test_##name()

#define ASSERT_TRUE(expr) do { \
    if (!(expr)) { printf("FAIL at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
                    tests_failed++; return; } } while(0)

// Helper: create a test buffer
nvmm::Result<nvmm::NvmmBuffer> make_buffer(uint32_t w, uint32_t h,
                                             nvmm::ColorFormat fmt) {
    nvmm::SurfaceParams params;
    params.width = w;
    params.height = h;
    params.color_format = fmt;
    params.mem_type = nvmm::MemoryType::kSystemHeap;
    return nvmm::NvmmBuffer::create(params);
}

// --- Tests ---

TEST(scale_nv12) {
    auto src = make_buffer(1920, 1080, nvmm::ColorFormat::kNV12);
    auto dst = make_buffer(640, 480, nvmm::ColorFormat::kNV12);
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());

    // Write a pattern to src
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

}  // namespace

int main() {
    printf("=== NvmmTransform Tests ===\n");
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
