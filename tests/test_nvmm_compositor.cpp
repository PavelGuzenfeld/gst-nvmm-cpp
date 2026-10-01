#include <gst/gst.h>
#include <gst/base/gstaggregator.h>

#include <cstdio>

#include "test_harness.h"

namespace {

/// Declared first: within one TU, static objects initialize in declaration order.
struct GstInit { GstInit() { gst_init(nullptr, nullptr); } } _gst_init;

/// aggregate() needs real NVMM buffers; compositing itself is covered by the
/// on-device runs in docs/validation.md.
TEST(compositor_is_aggregator_with_settable_output_size) {
    GstElement *comp = gst_element_factory_make("nvmmcompositor", "test-comp");
    ASSERT_NOT_NULL(comp);
    ASSERT_TRUE(GST_IS_AGGREGATOR(comp));

    gint w = 0, h = 0;
    g_object_get(comp, "width", &w, "height", &h, NULL);
    ASSERT_TRUE(w == 1280 && h == 720);

    g_object_set(comp, "width", 1920, "height", 1080, NULL);
    g_object_get(comp, "width", &w, "height", &h, NULL);
    ASSERT_TRUE(w == 1920 && h == 1080);

    gst_object_unref(comp);
}

TEST(request_pads) {
    GstElement *comp = gst_element_factory_make("nvmmcompositor", NULL);
    ASSERT_NOT_NULL(comp);

    GstPad *p0 = gst_element_request_pad_simple(comp, "sink_%u");
    GstPad *p1 = gst_element_request_pad_simple(comp, "sink_%u");
    ASSERT_NOT_NULL(p0);
    ASSERT_NOT_NULL(p1);
    ASSERT_TRUE(p0 != p1);

    gchar *n0 = gst_pad_get_name(p0);
    gchar *n1 = gst_pad_get_name(p1);
    ASSERT_TRUE(g_str_has_prefix(n0, "sink_"));
    ASSERT_TRUE(g_str_has_prefix(n1, "sink_"));
    ASSERT_TRUE(g_strcmp0(n0, n1) != 0);
    g_free(n0);
    g_free(n1);

    gst_element_release_request_pad(comp, p0);
    gst_element_release_request_pad(comp, p1);
    gst_object_unref(p0);
    gst_object_unref(p1);
    gst_object_unref(comp);
}

TEST(pad_placement_props) {
    GstElement *comp = gst_element_factory_make("nvmmcompositor", NULL);
    ASSERT_NOT_NULL(comp);

    GstPad *pad = gst_element_request_pad_simple(comp, "sink_%u");
    ASSERT_NOT_NULL(pad);

    g_object_set(pad, "xpos", 640, "ypos", 360, "width", 320, "height", 180, NULL);
    gint x = 0, y = 0, w = 0, h = 0;
    g_object_get(pad, "xpos", &x, "ypos", &y, "width", &w, "height", &h, NULL);
    ASSERT_TRUE(x == 640 && y == 360 && w == 320 && h == 180);

    gst_element_release_request_pad(comp, pad);
    gst_object_unref(pad);
    gst_object_unref(comp);
}

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
