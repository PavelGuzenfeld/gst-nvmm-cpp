/// Unit tests for NvmmOpticalFlowMeta — CI-safe (no VPI/NVMM): the meta is pure
/// GStreamer. Exercises register/add/get, the S10.5 decode round-trip, copy
/// transform, and free.

#include <gst/gst.h>

#include "nvmm_optical_flow_meta.h"

#include <cstdio>
#include <cstdint>

static int tests_passed = 0;
static int tests_failed = 0;

#define RUN_TEST(name) do { printf("  TEST %s ... ", #name); test_##name(); } while(0)
#define ASSERT_TRUE(expr) do { \
    if (!(expr)) { printf("FAIL at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
                    tests_failed++; return; } } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)

/* Build a host flow buffer of mv_w*mv_h cells, each (dx,dy) as int16 S10.5. */
static GstMemory *make_flow(gint w, gint h, int16_t dx, int16_t dy) {
    GstMemory *mem = gst_allocator_alloc(nullptr, (gsize)w * h * 4, nullptr);
    GstMapInfo map;
    gst_memory_map(mem, &map, GST_MAP_WRITE);
    int16_t *v = reinterpret_cast<int16_t *>(map.data);
    for (gint i = 0; i < w * h; i++) { v[2 * i] = dx; v[2 * i + 1] = dy; }
    gst_memory_unmap(mem, &map);
    return mem;
}

static void test_api_type_registered() {
    GType t = nvmm_optical_flow_meta_api_get_type();
    ASSERT_TRUE(t != 0);
    ASSERT_TRUE(nvmm_optical_flow_meta_get_info() != nullptr);
    PASS();
}

static void test_add_and_get() {
    GstBuffer *buf = gst_buffer_new();
    GstMemory *flow = make_flow(16, 12, 0, 0);
    NvmmOpticalFlowMeta *m = gst_buffer_add_nvmm_optical_flow_meta(
        buf, flow, 16, 12, 4, 64, 48);
    gst_memory_unref(flow);  /* meta keeps its own ref */
    ASSERT_TRUE(m != nullptr);

    NvmmOpticalFlowMeta *got = gst_buffer_get_nvmm_optical_flow_meta(buf);
    ASSERT_TRUE(got == m);
    ASSERT_TRUE(got->mv_width == 16 && got->mv_height == 12);
    ASSERT_TRUE(got->grid_size == 4);
    ASSERT_TRUE(got->frame_width == 64 && got->frame_height == 48);
    ASSERT_TRUE(got->mv != nullptr);
    gst_buffer_unref(buf);  /* must free meta (and unref its mv) without leak */
    PASS();
}

static void test_s10_5_decode() {
    /* dx = 64 -> 2.0 px, dy = -48 -> -1.5 px in S10.5 (raw/32). */
    GstBuffer *buf = gst_buffer_new();
    GstMemory *flow = make_flow(4, 4, 64, -48);
    gst_buffer_add_nvmm_optical_flow_meta(buf, flow, 4, 4, 8, 32, 32);
    gst_memory_unref(flow);

    NvmmOpticalFlowMeta *m = gst_buffer_get_nvmm_optical_flow_meta(buf);
    GstMapInfo map;
    ASSERT_TRUE(gst_memory_map(m->mv, &map, GST_MAP_READ));
    const int16_t *v = reinterpret_cast<const int16_t *>(map.data);
    const double dx = v[0] / 32.0, dy = v[1] / 32.0;
    gst_memory_unmap(m->mv, &map);
    ASSERT_TRUE(dx == 2.0 && dy == -1.5);
    gst_buffer_unref(buf);
    PASS();
}

static void test_copy_transform() {
    GstBuffer *buf = gst_buffer_new();
    GstMemory *flow = make_flow(8, 8, 32, 0);
    gst_buffer_add_nvmm_optical_flow_meta(buf, flow, 8, 8, 4, 32, 32);
    gst_memory_unref(flow);

    /* gst_buffer_copy runs meta transform with the COPY quark. */
    GstBuffer *copy = gst_buffer_copy(buf);
    NvmmOpticalFlowMeta *cm = gst_buffer_get_nvmm_optical_flow_meta(copy);
    ASSERT_TRUE(cm != nullptr);
    ASSERT_TRUE(cm->mv_width == 8 && cm->mv_height == 8 && cm->grid_size == 4);
    ASSERT_TRUE(cm->mv != nullptr);

    gst_buffer_unref(buf);   /* original gone; copy still holds its mv ref */
    GstMapInfo map;
    ASSERT_TRUE(gst_memory_map(cm->mv, &map, GST_MAP_READ));
    ASSERT_TRUE(reinterpret_cast<const int16_t *>(map.data)[0] == 32);
    gst_memory_unmap(cm->mv, &map);
    gst_buffer_unref(copy);
    PASS();
}

int main(int argc, char **argv) {
    gst_init(&argc, &argv);
    printf("Running NvmmOpticalFlowMeta tests:\n");
    RUN_TEST(api_type_registered);
    RUN_TEST(add_and_get);
    RUN_TEST(s10_5_decode);
    RUN_TEST(copy_transform);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
