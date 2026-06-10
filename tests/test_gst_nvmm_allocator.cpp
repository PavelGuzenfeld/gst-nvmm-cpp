/// Integration test for GstNvmmAllocator — tests the custom video
/// allocation path (not GstAllocator::alloc).

#include <gst/gst.h>
#include <gst/video/video.h>

#include "gstnvmmallocator.h"
#include "gstnvmmbufferpool.h"

#ifdef NVMM_MOCK_API
#include "nvbufsurface_mock.h"
#else
#include <nvbufsurface.h>
#endif

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

static void test_alloc_video_nv12() {
    GstAllocator* alloc = gst_nvmm_allocator_new(0 /* default */);
    ASSERT_NOT_NULL(alloc);

    GstMemory* mem = gst_nvmm_allocator_alloc_video(alloc,
        GST_VIDEO_FORMAT_NV12, 1920, 1080);
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
    GstMemory* mem = gst_nvmm_allocator_alloc_video(alloc,
        GST_VIDEO_FORMAT_NV12, 640, 480);
    ASSERT_NOT_NULL(mem);

    GstMapInfo map_info;
    gboolean ok = gst_memory_map(mem, &map_info, GST_MAP_READ);
    ASSERT_TRUE(ok);
    ASSERT_NOT_NULL(map_info.data);
    ASSERT_TRUE(map_info.data == gst_nvmm_memory_get_surface(mem));
    gst_memory_unmap(mem, &map_info);

    gst_memory_unref(mem);
    gst_object_unref(alloc);
    PASS();
}

static void test_per_plane_map() {
    GstAllocator* alloc = gst_nvmm_allocator_new(0 /* default */);
    GstMemory* mem = gst_nvmm_allocator_alloc_video(alloc,
        GST_VIDEO_FORMAT_NV12, 640, 480);
    ASSERT_NOT_NULL(mem);

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
    GstMemory* mem = gst_nvmm_allocator_alloc_video(alloc,
        GST_VIDEO_FORMAT_RGBA, 64, 64);
    ASSERT_NOT_NULL(mem);

    guint8* data = NULL;
    gsize size = 0;
    gboolean ok = gst_nvmm_memory_map_plane(mem, 0, GST_MAP_WRITE, &data, &size);
    ASSERT_TRUE(ok);
    memset(data, 0xCD, size);
    gst_nvmm_memory_unmap_plane(mem);

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

static void test_alloc_video_rgba() {
    GstAllocator* alloc = gst_nvmm_allocator_new(0 /* default */);
    GstMemory* mem = gst_nvmm_allocator_alloc_video(alloc,
        GST_VIDEO_FORMAT_RGBA, 1280, 720);
    ASSERT_NOT_NULL(mem);
    ASSERT_TRUE(gst_is_nvmm_memory(mem));
    ASSERT_TRUE(mem->size > 0);

    gst_memory_unref(mem);
    gst_object_unref(alloc);
    PASS();
}

static void test_alloc_video_invalid() {
    GstAllocator* alloc = gst_nvmm_allocator_new(0 /* default */);

    /* Zero dimensions should fail */
    GstMemory* mem = gst_nvmm_allocator_alloc_video(alloc,
        GST_VIDEO_FORMAT_NV12, 0, 0);
    ASSERT_TRUE(mem == NULL);

    gst_object_unref(alloc);
    PASS();
}

/* The buffer pool must stamp GstVideoMeta with the surface's REAL NVMM strides
   (planeParams.pitch/offset), not the GstVideoInfo defaults — hardware alignment
   makes them differ on Jetson (e.g. 640-wide NV12 -> 768 pitch). Production plan 4.3. */
static void test_pool_video_meta_real_strides() {
    GstBufferPool* pool = gst_nvmm_buffer_pool_new();
    ASSERT_NOT_NULL(pool);

    GstCaps* caps = gst_caps_from_string(
        "video/x-raw(memory:NVMM), format=(string)NV12, "
        "width=(int)1920, height=(int)1080");
    GstVideoInfo vinfo;
    ASSERT_TRUE(gst_video_info_from_caps(&vinfo, caps));

    GstStructure* config = gst_buffer_pool_get_config(pool);
    gst_buffer_pool_config_set_params(config, caps,
        (guint)GST_VIDEO_INFO_SIZE(&vinfo), 2, 4);
    ASSERT_TRUE(gst_buffer_pool_set_config(pool, config));
    ASSERT_TRUE(gst_buffer_pool_set_active(pool, TRUE));

    GstBuffer* buf = NULL;
    ASSERT_TRUE(gst_buffer_pool_acquire_buffer(pool, &buf, NULL) == GST_FLOW_OK);
    ASSERT_NOT_NULL(buf);

    GstVideoMeta* vmeta = gst_buffer_get_video_meta(buf);
    ASSERT_NOT_NULL(vmeta);

    void* surface = gst_nvmm_memory_get_surface(gst_buffer_peek_memory(buf, 0));
    ASSERT_NOT_NULL(surface);
    NvBufSurface* nvsurf = static_cast<NvBufSurface*>(surface);
    NvBufSurfacePlaneParams& pp = nvsurf->surfaceList[0].planeParams;

    /* GstVideoMeta strides/offsets must match the surface's planeParams. */
    for (guint i = 0; i < vmeta->n_planes; i++) {
        ASSERT_TRUE(vmeta->stride[i] == (gint)pp.pitch[i]);
        ASSERT_TRUE(vmeta->offset[i] == (gsize)pp.offset[i]);
    }

    gst_buffer_unref(buf);
    gst_buffer_pool_set_active(pool, FALSE);
    gst_object_unref(pool);
    gst_caps_unref(caps);
    PASS();
}

/* Phase 2: NVMM memory must be share-capable so tee fan-out + make_writable
   stay zero-copy (shallow buffer copy referencing the SAME NvBufSurface). */

static void test_memory_is_shareable() {
    GstAllocator* alloc = gst_nvmm_allocator_new(0);
    GstMemory* mem = gst_nvmm_allocator_alloc_video(alloc,
        GST_VIDEO_FORMAT_NV12, 640, 480);
    ASSERT_NOT_NULL(mem);
    /* NO_SHARE would force a deep copy on make_writable — must be cleared. */
    ASSERT_TRUE(!GST_MEMORY_FLAG_IS_SET(mem, GST_MEMORY_FLAG_NO_SHARE));
    gst_memory_unref(mem);
    gst_object_unref(alloc);
    PASS();
}

static void test_share_references_same_surface() {
    GstAllocator* alloc = gst_nvmm_allocator_new(0);
    GstMemory* mem = gst_nvmm_allocator_alloc_video(alloc,
        GST_VIDEO_FORMAT_NV12, 640, 480);
    ASSERT_NOT_NULL(mem);

    GstMemory* shared = gst_memory_share(mem, 0, -1);
    ASSERT_NOT_NULL(shared);
    ASSERT_TRUE(gst_is_nvmm_memory(shared));
    /* Zero-copy: the share resolves to the SAME NvBufSurface, not a copy. */
    ASSERT_TRUE(gst_nvmm_memory_get_surface(shared) ==
                gst_nvmm_memory_get_surface(mem));

    gst_memory_unref(shared);
    gst_memory_unref(mem);
    gst_object_unref(alloc);
    PASS();
}

static void test_tee_make_writable_zero_copy() {
    GstAllocator* alloc = gst_nvmm_allocator_new(0);
    GstMemory* mem = gst_nvmm_allocator_alloc_video(alloc,
        GST_VIDEO_FORMAT_NV12, 640, 480);
    ASSERT_NOT_NULL(mem);
    void* surf0 = gst_nvmm_memory_get_surface(mem);

    GstBuffer* buf = gst_buffer_new();
    gst_buffer_append_memory(buf, mem);  /* buf takes ownership of mem */

    /* gst_buffer_copy() shallow-copies: shares the memory by ref when the
       memory is shareable, deep-copies when NO_SHARE. Assert it shared. */
    GstBuffer* buf2 = gst_buffer_copy(buf);
    void* surf1 = gst_nvmm_memory_get_surface(gst_buffer_peek_memory(buf2, 0));
    ASSERT_TRUE(surf1 == surf0);  /* same surface across the two buffers */

    gst_buffer_unref(buf2);
    gst_buffer_unref(buf);
    gst_object_unref(alloc);
    PASS();
}

static void test_share_outlives_parent() {
    GstAllocator* alloc = gst_nvmm_allocator_new(0);
    GstMemory* mem = gst_nvmm_allocator_alloc_video(alloc,
        GST_VIDEO_FORMAT_NV12, 640, 480);
    ASSERT_NOT_NULL(mem);
    void* surf = gst_nvmm_memory_get_surface(mem);

    GstMemory* shared = gst_memory_share(mem, 0, -1);
    ASSERT_NOT_NULL(shared);
    gst_memory_unref(mem);  /* drop the original ref; share holds a parent ref */

    /* The surface is still valid: the share kept the owner alive. */
    ASSERT_TRUE(gst_nvmm_memory_get_surface(shared) == surf);

    gst_memory_unref(shared);
    gst_object_unref(alloc);
    PASS();
}

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);
    printf("=== GstNvmmAllocator Tests ===\n");

    RUN_TEST(allocator_creates);
    RUN_TEST(alloc_video_nv12);
    RUN_TEST(alloc_video_rgba);
    RUN_TEST(alloc_video_invalid);
    RUN_TEST(direct_map_returns_surface);
    RUN_TEST(per_plane_map);
    RUN_TEST(per_plane_write_read_roundtrip);
    RUN_TEST(non_nvmm_memory_rejected);
    RUN_TEST(pool_video_meta_real_strides);
    RUN_TEST(memory_is_shareable);
    RUN_TEST(share_references_same_surface);
    RUN_TEST(tee_make_writable_zero_copy);
    RUN_TEST(share_outlives_parent);

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    gst_deinit();
    return tests_failed > 0 ? 1 : 0;
}
