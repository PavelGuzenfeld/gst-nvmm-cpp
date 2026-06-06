/// Unit tests for GstNvmmCompositor — VIC-composited multi-input NVMM mixer.
///
/// The element is loaded from the plugin registry (GST_PLUGIN_PATH), so these
/// tests link no compositor symbols directly — they exercise the GObject
/// surface: factory creation, request-pad allocation, and property round-trips
/// for both the element (output width/height) and the request pads (placement).
/// The compositing path itself (aggregate() → NvBufSurfTransform CROP_DST) is
/// covered on-device by the dual-input gst-launch runs recorded in
/// docs/validation.md, since aggregate() needs real NVMM buffers to run.

#include <gst/gst.h>
#include <gst/base/gstaggregator.h>

#include <cstdio>

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

static void test_compositor_creates() {
    GstElement *comp = gst_element_factory_make("nvmmcompositor", "test-comp");
    ASSERT_NOT_NULL(comp);
    ASSERT_TRUE(GST_IS_AGGREGATOR(comp));
    gst_object_unref(comp);
    PASS();
}

static void test_compositor_output_props() {
    GstElement *comp = gst_element_factory_make("nvmmcompositor", NULL);
    ASSERT_NOT_NULL(comp);

    /* Defaults documented as 1280x720. */
    gint w = 0, h = 0;
    g_object_get(comp, "width", &w, "height", &h, NULL);
    ASSERT_TRUE(w == 1280 && h == 720);

    g_object_set(comp, "width", 1920, "height", 1080, NULL);
    g_object_get(comp, "width", &w, "height", &h, NULL);
    ASSERT_TRUE(w == 1920 && h == 1080);

    gst_object_unref(comp);
    PASS();
}

static void test_request_pads() {
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
    PASS();
}

static void test_pad_placement_props() {
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
    PASS();
}

int main(int argc, char **argv) {
    gst_init(&argc, &argv);
    printf("Running GstNvmmCompositor tests:\n");

    RUN_TEST(compositor_creates);
    RUN_TEST(compositor_output_props);
    RUN_TEST(request_pads);
    RUN_TEST(pad_placement_props);

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
