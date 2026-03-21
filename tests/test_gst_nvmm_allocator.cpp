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
    GstAllocator* alloc = gst_nvmm_allocator_new(6 /* system heap for mock */);
    ASSERT_NOT_NULL(alloc);
    ASSERT_TRUE(GST_IS_NVMM_ALLOCATOR(alloc));
    gst_object_unref(alloc);
    PASS();
}

static void test_allocator_alloc_free() {
    GstAllocator* alloc = gst_nvmm_allocator_new(6);
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

static void test_allocator_map_unmap() {
    GstAllocator* alloc = gst_nvmm_allocator_new(6);
    GstMemory* mem = gst_allocator_alloc(alloc, 640 * 480 * 4, NULL);
    ASSERT_NOT_NULL(mem);

    GstMapInfo map_info;
    gboolean ok = gst_memory_map(mem, &map_info, GST_MAP_READ);
    ASSERT_TRUE(ok);
    ASSERT_NOT_NULL(map_info.data);
    ASSERT_TRUE(map_info.size > 0);

    gst_memory_unmap(mem, &map_info);
    gst_memory_unref(mem);
    gst_object_unref(alloc);
    PASS();
}

static void test_allocator_write_read_round_trip() {
    GstAllocator* alloc = gst_nvmm_allocator_new(6);
    GstMemory* mem = gst_allocator_alloc(alloc, 64 * 64 * 4, NULL);
    ASSERT_NOT_NULL(mem);

    // Write
    GstMapInfo write_info;
    gboolean ok = gst_memory_map(mem, &write_info, GST_MAP_WRITE);
    ASSERT_TRUE(ok);
    memset(write_info.data, 0xCD, write_info.size);
    gst_memory_unmap(mem, &write_info);

    // Read back
    GstMapInfo read_info;
    ok = gst_memory_map(mem, &read_info, GST_MAP_READ);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(((uint8_t*)read_info.data)[0] == 0xCD);
    gst_memory_unmap(mem, &read_info);

    gst_memory_unref(mem);
    gst_object_unref(alloc);
    PASS();
}

static void test_non_nvmm_memory_rejected() {
    GstAllocator* sys_alloc = gst_allocator_find(GST_ALLOCATOR_SYSMEM);
    GstMemory* mem = gst_allocator_alloc(sys_alloc, 1024, NULL);
    ASSERT_TRUE(!gst_is_nvmm_memory(mem));
    ASSERT_TRUE(gst_nvmm_memory_get_surface(mem) == NULL);
    gst_memory_unref(mem);
    PASS();
}

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);
    printf("=== GstNvmmAllocator Tests ===\n");

    RUN_TEST(allocator_creates);
    RUN_TEST(allocator_alloc_free);
    RUN_TEST(allocator_map_unmap);
    RUN_TEST(allocator_write_read_round_trip);
    RUN_TEST(non_nvmm_memory_rejected);

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    gst_deinit();
    return tests_failed > 0 ? 1 : 0;
}
