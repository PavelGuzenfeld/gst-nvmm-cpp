#include "nvmm_class_meta.h"

#include <gst/gst.h>

#include <cmath>
#include <cstdio>
#include <cstring>

#include "test_harness.h"

namespace {

/// Declared first: within one TU, static objects initialize in declaration order.
struct GstInit { GstInit() { gst_init(nullptr, nullptr); } } _gst_init;

NvmmClassEntry entry(gint32 id, gfloat conf, guint32 fresh, const char *label) {
    NvmmClassEntry e{};
    e.class_id = id;
    e.confidence = conf;
    e.fresh = fresh;
    snprintf(e.label, sizeof e.label, "%s", label);
    return e;
}

}

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

TEST(copy_transform_carries_entries) {
    GstBuffer *buf = gst_buffer_new();
    NvmmClassEntry in[1] = { entry(7, 0.42f, 0, "walking") };
    gst_buffer_add_nvmm_class_meta(buf, in, 1);

    GstBuffer *copy = gst_buffer_copy(buf);
    gst_buffer_unref(buf);

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
