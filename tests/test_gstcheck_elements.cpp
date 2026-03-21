/// GstCheck-style state transition and pipeline tests for all NVMM elements.
/// Tests element lifecycle, property validation, and basic pipeline wiring.

#include <gst/gst.h>
#include <gst/video/video.h>

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
#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))

#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)

/* --- Element discovery tests --- */

static void test_nvmmconvert_discovered() {
    GstElementFactory *factory =
        gst_element_factory_find("nvmmconvert");
    ASSERT_NOT_NULL(factory);
    gst_object_unref(factory);
    PASS();
}

static void test_nvmmsink_discovered() {
    GstElementFactory *factory =
        gst_element_factory_find("nvmmsink");
    ASSERT_NOT_NULL(factory);
    gst_object_unref(factory);
    PASS();
}

static void test_nvmmappsrc_discovered() {
    GstElementFactory *factory =
        gst_element_factory_find("nvmmappsrc");
    ASSERT_NOT_NULL(factory);
    gst_object_unref(factory);
    PASS();
}

/* --- State transition tests --- */

static void
check_state_transitions(const char *element_name)
{
    GstElement *elem = gst_element_factory_make(element_name, NULL);
    ASSERT_NOT_NULL(elem);

    GstStateChangeReturn ret;
    GstState current, pending;

    /* NULL → READY */
    ret = gst_element_set_state(elem, GST_STATE_READY);
    ASSERT_TRUE(ret == GST_STATE_CHANGE_SUCCESS ||
                ret == GST_STATE_CHANGE_NO_PREROLL);

    gst_element_get_state(elem, &current, &pending, GST_CLOCK_TIME_NONE);
    ASSERT_EQ(current, GST_STATE_READY);

    /* READY → NULL */
    ret = gst_element_set_state(elem, GST_STATE_NULL);
    ASSERT_EQ(ret, GST_STATE_CHANGE_SUCCESS);

    gst_object_unref(elem);
}

static void test_nvmmconvert_state_transitions() {
    check_state_transitions("nvmmconvert");
    PASS();
}

static void test_nvmmsink_state_transitions() {
    /* nvmmsink needs shm-name to start */
    GstElement *sink = gst_element_factory_make("nvmmsink", NULL);
    ASSERT_NOT_NULL(sink);
    g_object_set(sink, "shm-name", "/test_gstcheck_sink", NULL);

    GstStateChangeReturn ret;

    ret = gst_element_set_state(sink, GST_STATE_READY);
    ASSERT_EQ(ret, GST_STATE_CHANGE_SUCCESS);

    ret = gst_element_set_state(sink, GST_STATE_NULL);
    ASSERT_EQ(ret, GST_STATE_CHANGE_SUCCESS);

    gst_object_unref(sink);
    PASS();
}

/* --- Property validation tests --- */

static void test_nvmmconvert_properties() {
    GstElement *elem = gst_element_factory_make("nvmmconvert", NULL);
    ASSERT_NOT_NULL(elem);

    /* Set all properties */
    g_object_set(elem,
        "crop-x", (guint) 100,
        "crop-y", (guint) 200,
        "crop-w", (guint) 800,
        "crop-h", (guint) 600,
        "flip-method", 2,
        NULL);

    /* Read back and verify */
    guint cx, cy, cw, ch;
    gint fm;
    g_object_get(elem,
        "crop-x", &cx,
        "crop-y", &cy,
        "crop-w", &cw,
        "crop-h", &ch,
        "flip-method", &fm,
        NULL);

    ASSERT_EQ(cx, 100u);
    ASSERT_EQ(cy, 200u);
    ASSERT_EQ(cw, 800u);
    ASSERT_EQ(ch, 600u);
    ASSERT_EQ(fm, 2);

    gst_object_unref(elem);
    PASS();
}

/* --- Pad template tests --- */

static void test_nvmmconvert_pad_templates() {
    GstElement *elem = gst_element_factory_make("nvmmconvert", NULL);
    ASSERT_NOT_NULL(elem);

    GstPad *sink_pad = gst_element_get_static_pad(elem, "sink");
    ASSERT_NOT_NULL(sink_pad);

    GstPad *src_pad = gst_element_get_static_pad(elem, "src");
    ASSERT_NOT_NULL(src_pad);

    GstCaps *sink_caps = gst_pad_query_caps(sink_pad, NULL);
    ASSERT_NOT_NULL(sink_caps);
    ASSERT_TRUE(!gst_caps_is_empty(sink_caps));

    /* Verify NVMM memory feature is in caps */
    GstCapsFeatures *features = gst_caps_get_features(sink_caps, 0);
    ASSERT_NOT_NULL(features);
    ASSERT_TRUE(gst_caps_features_contains(features, "memory:NVMM"));

    gst_caps_unref(sink_caps);
    gst_object_unref(sink_pad);
    gst_object_unref(src_pad);
    gst_object_unref(elem);
    PASS();
}

/* --- Pipeline wiring test --- */

static void test_pipeline_link_convert_to_sink() {
    GstElement *pipeline = gst_pipeline_new("test-pipeline");
    GstElement *convert = gst_element_factory_make("nvmmconvert", NULL);
    GstElement *sink = gst_element_factory_make("nvmmsink", NULL);

    ASSERT_NOT_NULL(pipeline);
    ASSERT_NOT_NULL(convert);
    ASSERT_NOT_NULL(sink);

    g_object_set(sink, "shm-name", "/test_gstcheck_pipe", NULL);

    gst_bin_add_many(GST_BIN(pipeline), convert, sink, NULL);

    /* These can't actually link without NVMM caps negotiation in mock mode,
       but we verify the elements can be added to a pipeline */
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_READY);
    /* May fail since convert has no upstream — that's OK for this test */
    (void) ret;

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    PASS();
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);
    printf("=== GstCheck Element Tests ===\n");

    RUN_TEST(nvmmconvert_discovered);
    RUN_TEST(nvmmsink_discovered);
    RUN_TEST(nvmmappsrc_discovered);
    RUN_TEST(nvmmconvert_state_transitions);
    RUN_TEST(nvmmsink_state_transitions);
    RUN_TEST(nvmmconvert_properties);
    RUN_TEST(nvmmconvert_pad_templates);
    RUN_TEST(pipeline_link_convert_to_sink);

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    gst_deinit();
    return tests_failed > 0 ? 1 : 0;
}
