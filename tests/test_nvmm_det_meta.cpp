/// Unit tests for the IPC metadata side-channel: the flat wire layout
/// (shm_protocol.h) and the DeepStream-free GstNvmmDetMeta (nvmm_det_meta.h).
/// No NvBufSurface or DeepStream needed — pure POD + GstMeta round-trips.

#include "shm_protocol.h"
#include "nvmm_det_meta.h"

#include <gst/gst.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {

/* gst_init must run before the static-registered TEST constructors below; in a
   single TU static objects initialize in declaration order, so this goes first. */
struct GstInit { GstInit() { gst_init(nullptr, nullptr); } } _gst_init;

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

/* ASSERT_* throw so a failing assertion unwinds into TEST's catch (which counts
   the failure) instead of returning into the PASS path — otherwise a failed test
   prints both "FAIL at ..." and "PASS" and bumps both counters. */
#define ASSERT_TRUE(expr) do { \
    if (!(expr)) { printf("FAIL at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
                    throw std::runtime_error("assertion failed"); } } while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { printf("FAIL at %s:%d: %s != %s\n", __FILE__, __LINE__, #a, #b); \
                       throw std::runtime_error("assertion failed"); } } while(0)

static void fill_frame(NvmmFrameMeta *f, guint32 n)
{
    memset(f, 0, sizeof(*f));
    f->frame_number = 42;
    f->infer_width = 1920;
    f->infer_height = 1080;
    f->num_objects = n;
    for (guint32 i = 0; i < n && i < NVMM_META_MAX_OBJECTS; i++) {
        f->objects[i].left = (float)i;
        f->objects[i].top = (float)(i * 2);
        f->objects[i].width = 10.0f + i;
        f->objects[i].height = 20.0f + i;
        f->objects[i].class_id = (int)i;
        f->objects[i].confidence = 0.5f + (float)i / 1000.0f;
        f->objects[i].tracker_id = 1000 + i;
        snprintf(f->objects[i].label, NVMM_META_LABEL_LEN, "class_%u", i);
    }
}

// --- wire-layout / pointer math ---

TEST(segment_size_without_meta) {
    ASSERT_EQ(nvmm_shm_segment_size(0), sizeof(NvmmShmHeader));
}

TEST(segment_size_with_meta) {
    size_t expect = sizeof(NvmmShmHeader)
                  + (size_t)NVMM_POOL_SIZE * sizeof(NvmmFrameMeta);
    ASSERT_EQ(nvmm_shm_segment_size(1), expect);
}

TEST(meta_slot_pointer_math) {
    /* Slot i must sit at header + i*sizeof(NvmmFrameMeta), right after header. */
    unsigned char *base = (unsigned char *)g_malloc0(nvmm_shm_segment_size(1));
    NvmmFrameMeta *slot0 = nvmm_shm_meta(base, 0);
    NvmmFrameMeta *slot3 = nvmm_shm_meta(base, 3);
    ASSERT_EQ((unsigned char *)slot0, base + sizeof(NvmmShmHeader));
    ASSERT_EQ((size_t)((unsigned char *)slot3 - (unsigned char *)slot0),
              3u * sizeof(NvmmFrameMeta));
    g_free(base);
}

// --- GstNvmmDetMeta round-trips ---

TEST(add_get_roundtrip) {
    NvmmFrameMeta f;
    fill_frame(&f, 3);

    GstBuffer *buf = gst_buffer_new();
    GstNvmmDetMeta *m = gst_buffer_add_nvmm_det_meta(buf, &f);
    ASSERT_TRUE(m != nullptr);

    GstNvmmDetMeta *got = gst_buffer_get_nvmm_det_meta(buf);
    ASSERT_TRUE(got != nullptr);
    ASSERT_EQ(got->frame_number, 42u);
    ASSERT_EQ(got->infer_width, 1920u);
    ASSERT_EQ(got->infer_height, 1080u);
    ASSERT_EQ(got->num_objects, 3u);
    ASSERT_TRUE(got->objects != nullptr);
    ASSERT_EQ(got->objects[2].class_id, 2);
    ASSERT_EQ(got->objects[2].tracker_id, 1002u);
    ASSERT_TRUE(got->objects[1].width == 11.0f);
    ASSERT_TRUE(strcmp(got->objects[2].label, "class_2") == 0);

    gst_buffer_unref(buf);  /* must free m->objects without leaking/crashing */
}

TEST(empty_detections) {
    NvmmFrameMeta f;
    fill_frame(&f, 0);
    GstBuffer *buf = gst_buffer_new();
    GstNvmmDetMeta *m = gst_buffer_add_nvmm_det_meta(buf, &f);
    ASSERT_TRUE(m != nullptr);
    ASSERT_EQ(m->num_objects, 0u);
    ASSERT_TRUE(m->objects == nullptr);
    gst_buffer_unref(buf);
}

TEST(object_count_clamped) {
    NvmmFrameMeta f;
    /* Claim more than the cap; the objects[] array only holds MAX, so the
       serializer must clamp to avoid reading past the fixed array. */
    fill_frame(&f, NVMM_META_MAX_OBJECTS);
    f.num_objects = NVMM_META_MAX_OBJECTS + 100;  /* lie about the count */

    GstBuffer *buf = gst_buffer_new();
    GstNvmmDetMeta *m = gst_buffer_add_nvmm_det_meta(buf, &f);
    ASSERT_TRUE(m != nullptr);
    ASSERT_EQ(m->num_objects, NVMM_META_MAX_OBJECTS);
    gst_buffer_unref(buf);
}

TEST(survives_buffer_copy) {
    NvmmFrameMeta f;
    fill_frame(&f, 5);
    GstBuffer *buf = gst_buffer_new();
    gst_buffer_add_nvmm_det_meta(buf, &f);

    GstBuffer *copy = gst_buffer_copy(buf);  /* triggers meta transform (copy) */
    GstNvmmDetMeta *got = gst_buffer_get_nvmm_det_meta(copy);
    ASSERT_TRUE(got != nullptr);
    ASSERT_EQ(got->num_objects, 5u);
    ASSERT_EQ(got->objects[4].class_id, 4);
    ASSERT_TRUE(strcmp(got->objects[3].label, "class_3") == 0);

    gst_buffer_unref(buf);
    gst_buffer_unref(copy);
}

}  // namespace

int main() {
    printf("=== NVMM Detection Metadata Tests ===\n");
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
