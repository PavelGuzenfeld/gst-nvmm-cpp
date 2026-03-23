/// Integration test for GstNvmmAllocator — tests the GStreamer allocator
/// interface backed by NVMM buffers.

#include <gst/gst.h>

#include "gstnvmmallocator.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

static int tests_passed = 0;
static int tests_failed = 0;

#define RUN_TEST(name) do { \
    printf("  TEST %s ... ", #name); \
    test_##name(); \
} while(0)

#define ASSERT_TRUE(expr) do { \
    if (!(expr)) { printf("FAIL at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
                    tests_failed++; return; } } while(0)

#define ASSERT_NOT_NULL(ptr) ASSERT_TRUE((ptr) != NULL)

#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)

static void test_allocator_creates() {
    GstAllocator* alloc = gst_nvmm_allocator_new(0 /* default */);
    ASSERT_NOT_NULL(alloc);
    ASSERT_TRUE(GST_IS_NVMM_ALLOCATOR(alloc));
    gst_object_unref(alloc);
    PASS();
}

static void test_allocator_alloc_free() {
    GstAllocator* alloc = gst_nvmm_allocator_new(0 /* default */);
    ASSERT_NOT_NULL(alloc);

    GstMemory* mem = gst_allocator_alloc(alloc, 1920 * 1080 * 3 / 2, NULL);
    ASSERT_NOT_NULL(mem);
    ASSERT_TRUE(gst_is_nvmm_memory(mem));

    void* surface = gst_nvmm_memory_get_surface(mem);
    ASSERT_NOT_NULL(surface);

    gst_memory_unref(mem);
    gst_object_unref(alloc);
    PASS();
}

static void test_direct_map_returns_surface() {
    GstAllocator* alloc = gst_nvmm_allocator_new(0 /* default */);
    GstMemory* mem = gst_allocator_alloc(alloc, 640 * 480 * 3 / 2, NULL);
    ASSERT_NOT_NULL(mem);

    /* NVIDIA convention: gst_memory_map returns NvBufSurface* (not pixels).
       This is required for interop with nvvidconv, nvv4l2decoder etc. */
    GstMapInfo map_info;
    gboolean ok = gst_memory_map(mem, &map_info, GST_MAP_READ);
    ASSERT_TRUE(ok);
    ASSERT_NOT_NULL(map_info.data);
    /* map_info.data should equal the NvBufSurface* */
    ASSERT_TRUE(map_info.data == gst_nvmm_memory_get_surface(mem));
    gst_memory_unmap(mem, &map_info);

    gst_memory_unref(mem);
    gst_object_unref(alloc);
    PASS();
}

static void test_per_plane_map() {
    GstAllocator* alloc = gst_nvmm_allocator_new(0 /* default */);
    GstMemory* mem = gst_allocator_alloc(alloc, 640 * 480 * 3 / 2, NULL);
    ASSERT_NOT_NULL(mem);

    /* Per-plane map should work */
    guint8* data = NULL;
    gsize size = 0;
    gboolean ok = gst_nvmm_memory_map_plane(mem, 0, GST_MAP_READ, &data, &size);
    ASSERT_TRUE(ok);
    ASSERT_NOT_NULL(data);
    ASSERT_TRUE(size > 0);

    gst_nvmm_memory_unmap_plane(mem);
    gst_memory_unref(mem);
    gst_object_unref(alloc);
    PASS();
}

static void test_per_plane_write_read_roundtrip() {
    GstAllocator* alloc = gst_nvmm_allocator_new(0 /* default */);
    GstMemory* mem = gst_allocator_alloc(alloc, 64 * 64 * 4, NULL);
    ASSERT_NOT_NULL(mem);

    /* Write pattern via per-plane map */
    guint8* data = NULL;
    gsize size = 0;
    gboolean ok = gst_nvmm_memory_map_plane(mem, 0, GST_MAP_WRITE, &data, &size);
    ASSERT_TRUE(ok);
    memset(data, 0xCD, size);
    gst_nvmm_memory_unmap_plane(mem);

    /* Read back and verify */
    ok = gst_nvmm_memory_map_plane(mem, 0, GST_MAP_READ, &data, &size);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(data[0] == 0xCD);
    gst_nvmm_memory_unmap_plane(mem);

    gst_memory_unref(mem);
    gst_object_unref(alloc);
    PASS();
}

static void test_non_nvmm_memory_rejected() {
    GstAllocator* sys_alloc = gst_allocator_find(GST_ALLOCATOR_SYSMEM);
    GstMemory* mem = gst_allocator_alloc(sys_alloc, 1024, NULL);
    ASSERT_TRUE(!gst_is_nvmm_memory(mem));
    ASSERT_TRUE(gst_nvmm_memory_get_surface(mem) == NULL);

    guint8* data = NULL;
    gsize size = 0;
    ASSERT_TRUE(!gst_nvmm_memory_map_plane(mem, 0, GST_MAP_READ, &data, &size));

    gst_memory_unref(mem);
    PASS();
}

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);
    printf("=== GstNvmmAllocator Tests ===\n");

    RUN_TEST(allocator_creates);
    RUN_TEST(allocator_alloc_free);
    RUN_TEST(direct_map_returns_surface);
    RUN_TEST(per_plane_map);
    RUN_TEST(per_plane_write_read_roundtrip);
    RUN_TEST(non_nvmm_memory_rejected);

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    gst_deinit();
    return tests_failed > 0 ? 1 : 0;
}
