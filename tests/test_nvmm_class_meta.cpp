/// Unit tests for GstNvmmClassMeta (the secondary-classifier sibling meta):
/// attach/read-back, empty case, and copy-transform behavior. Pure GstMeta —
/// no NvBufSurface/CUDA — so it runs on x86 CI.

#include "nvmm_class_meta.h"

#include <gst/gst.h>

#include <cmath>
#include <cstdio>
#include <cstring>

#include "test_harness.h"

namespace {

/* gst_init must run before the static-registered TEST constructors below; in a
   single TU static objects initialize in declaration order, so this goes first. */
struct GstInit { GstInit() { gst_init(nullptr, nullptr); } } _gst_init;

NvmmClassEntry entry(gint32 id, gfloat conf, guint32 fresh, const char *label) {
    NvmmClassEntry e{};
    e.class_id = id;
    e.confidence = conf;
    e.fresh = fresh;
    snprintf(e.label, sizeof e.label, "%s", label);
    return e;
}

}  // namespace

TEST(attach_and_read_back) {
    GstBuffer *buf = gst_buffer_new();
    NvmmClassEntry in[2] = { entry(3, 0.9f, 1, "sitting"), entry(-1, 0.f, 0, "") };
    ASSERT_NOT_NULL(gst_buffer_add_nvmm_class_meta(buf, in, 2));

    GstNvmmClassMeta *m = gst_buffer_get_nvmm_class_meta(buf);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->num_objects, 2u);
    ASSERT_EQ(m->objects[0].class_id, 3);
    ASSERT_NEAR(m->objects[0].confidence, 0.9f, 1e-6);
    ASSERT_EQ(m->objects[0].fresh, 1u);
    ASSERT_TRUE(strcmp(m->objects[0].label, "sitting") == 0);
    ASSERT_EQ(m->objects[1].class_id, -1);
    gst_buffer_unref(buf);
}

TEST(empty_meta_has_null_objects) {
    GstBuffer *buf = gst_buffer_new();
    GstNvmmClassMeta *m = gst_buffer_add_nvmm_class_meta(buf, nullptr, 0);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->num_objects, 0u);
    ASSERT_TRUE(m->objects == nullptr);
    gst_buffer_unref(buf);
}

// gst_buffer_copy propagates the meta via the copy-transform (deep entry copy).
TEST(copy_transform_carries_entries) {
    GstBuffer *buf = gst_buffer_new();
    NvmmClassEntry in[1] = { entry(7, 0.42f, 0, "walking") };
    gst_buffer_add_nvmm_class_meta(buf, in, 1);

    GstBuffer *copy = gst_buffer_copy(buf);
    gst_buffer_unref(buf);  // copy must own its own entries

    GstNvmmClassMeta *m = gst_buffer_get_nvmm_class_meta(copy);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->num_objects, 1u);
    ASSERT_EQ(m->objects[0].class_id, 7);
    ASSERT_TRUE(strcmp(m->objects[0].label, "walking") == 0);
    gst_buffer_unref(copy);
}

int main()
{
    printf("nvmm_class_meta unit tests\n");
    printf("%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
