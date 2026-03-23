/// Unit tests for nvmm::NvmmBuffer (RAII wrapper for NvBufSurface).
/// Uses mock API on host, real API on Jetson.

#include "nvmm_buffer.hpp"
#include "nvmm_types.hpp"

#ifdef NVMM_MOCK_API
#include "nvbufsurface_mock.h"
#else
#include <nvbufsurface.h>
#endif

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

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { printf("FAIL at %s:%d: %s != %s\n", __FILE__, __LINE__, #a, #b); \
                       tests_failed++; return; } } while(0)

// --- Tests ---

TEST(create_nv12_buffer) {
    nvmm::SurfaceParams params;
    params.width = 1920;
    params.height = 1080;
    params.color_format = nvmm::ColorFormat::kNV12;
    params.mem_type = nvmm::MemoryType::kDefault;
    params.num_surfaces = 1;

    auto result = nvmm::NvmmBuffer::create(params);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().width(), 1920u);
    ASSERT_EQ(result.value().height(), 1080u);
    ASSERT_EQ(result.value().format(), nvmm::ColorFormat::kNV12);
    ASSERT_EQ(result.value().num_planes(), 2u);  // Y + UV
}

TEST(create_rgba_buffer) {
    nvmm::SurfaceParams params;
    params.width = 640;
    params.height = 480;
    params.color_format = nvmm::ColorFormat::kRGBA;
    params.mem_type = nvmm::MemoryType::kDefault;

    auto result = nvmm::NvmmBuffer::create(params);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().width(), 640u);
    ASSERT_EQ(result.value().height(), 480u);
    ASSERT_EQ(result.value().num_planes(), 1u);  // Single plane RGBA
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
    nvmm::SurfaceParams params;
    params.width = 320;
    params.height = 240;
    params.color_format = nvmm::ColorFormat::kNV12;
    params.mem_type = nvmm::MemoryType::kDefault;

    auto result = nvmm::NvmmBuffer::create(params);
    ASSERT_TRUE(result.has_value());

    nvmm::NvmmBuffer moved = std::move(result.value());
    ASSERT_TRUE(moved.valid());
    ASSERT_EQ(moved.width(), 320u);
}

TEST(release_prevents_destroy) {
    nvmm::SurfaceParams params;
    params.width = 64;
    params.height = 64;
    params.color_format = nvmm::ColorFormat::kNV12;
    params.mem_type = nvmm::MemoryType::kDefault;

    NvBufSurface* raw_ptr = nullptr;
    {
        auto result = nvmm::NvmmBuffer::create(params);
        ASSERT_TRUE(result.has_value());
        raw_ptr = result.value().release();
        ASSERT_TRUE(raw_ptr != nullptr);
        ASSERT_TRUE(!result.value().valid());
        /* Destructor runs here — must NOT call NvBufSurfaceDestroy */
    }
    /* Surface should still be valid after NvmmBuffer is destroyed */
    ASSERT_TRUE(raw_ptr->surfaceList != nullptr);
    ASSERT_EQ(raw_ptr->surfaceList[0].width, 64u);

    /* Clean up manually */
    NvBufSurfaceDestroy(raw_ptr);
}

TEST(map_read) {
    nvmm::SurfaceParams params;
    params.width = 64;
    params.height = 64;
    params.color_format = nvmm::ColorFormat::kRGBA;
    params.mem_type = nvmm::MemoryType::kDefault;

    auto result = nvmm::NvmmBuffer::create(params);
    ASSERT_TRUE(result.has_value());

    auto map_result = result.value().map_read(0);
    ASSERT_TRUE(map_result.has_value());
    ASSERT_TRUE(map_result.value().data() != nullptr);
    ASSERT_TRUE(map_result.value().size() > 0);

    auto unmap_result = result.value().unmap();
    ASSERT_TRUE(unmap_result.has_value());
}

TEST(map_write_and_verify) {
    nvmm::SurfaceParams params;
    params.width = 16;
    params.height = 16;
    params.color_format = nvmm::ColorFormat::kRGBA;
    params.mem_type = nvmm::MemoryType::kDefault;

    auto result = nvmm::NvmmBuffer::create(params);
    ASSERT_TRUE(result.has_value());

    // Write pattern
    auto map_w = result.value().map_write(0);
    ASSERT_TRUE(map_w.has_value());
    std::memset(map_w.value().data(), 0xAB, map_w.value().size());
    result.value().unmap();

    // Read back and verify
    auto map_r = result.value().map_read(0);
    ASSERT_TRUE(map_r.has_value());
    ASSERT_EQ(map_r.value().data()[0], 0xAB);
    result.value().unmap();
}

TEST(export_fd) {
    nvmm::SurfaceParams params;
    params.width = 64;
    params.height = 64;
    params.color_format = nvmm::ColorFormat::kNV12;
    params.mem_type = nvmm::MemoryType::kDefault;

    auto result = nvmm::NvmmBuffer::create(params);
    ASSERT_TRUE(result.has_value());

    auto fd_result = result.value().export_fd();
    // Mock returns a fake fd, real API returns a real one
    ASSERT_TRUE(fd_result.has_value());
    ASSERT_TRUE(fd_result.value() >= 0);
}

TEST(plane_info_nv12) {
    nvmm::SurfaceParams params;
    params.width = 1920;
    params.height = 1080;
    params.color_format = nvmm::ColorFormat::kNV12;
    params.mem_type = nvmm::MemoryType::kDefault;

    auto result = nvmm::NvmmBuffer::create(params);
    ASSERT_TRUE(result.has_value());

    auto p0 = result.value().plane_info(0);
    ASSERT_EQ(p0.width, 1920u);
    ASSERT_EQ(p0.height, 1080u);

    auto p1 = result.value().plane_info(1);
    ASSERT_TRUE(p1.width > 0);
    ASSERT_TRUE(p1.height > 0);
    // UV plane height is half of Y plane
    ASSERT_TRUE(p1.height <= p0.height);
}

TEST(i420_three_planes) {
    nvmm::SurfaceParams params;
    params.width = 640;
    params.height = 480;
    params.color_format = nvmm::ColorFormat::kI420;
    params.mem_type = nvmm::MemoryType::kDefault;

    auto result = nvmm::NvmmBuffer::create(params);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().num_planes(), 3u);  // Y + U + V
}

}  // namespace

int main() {
    printf("=== NvmmBuffer Tests ===\n");
    // Tests are auto-registered via static constructors above.
    // After all static constructors run, report results.
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
