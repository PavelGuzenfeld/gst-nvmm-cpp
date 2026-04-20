/// Unit tests for GstNvmmAppSrc — shared memory source element.

#include <gst/gst.h>

#include "gstnvmmappsrc.h"

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

static void test_appsrc_creates() {
    GstElement *src = gst_element_factory_make("nvmmappsrc", "test-src");
    ASSERT_NOT_NULL(src);
    ASSERT_TRUE(GST_IS_NVMM_APP_SRC(src));
    gst_object_unref(src);
    PASS();
}

static void test_appsrc_properties() {
    GstElement *src = gst_element_factory_make("nvmmappsrc", NULL);
    ASSERT_NOT_NULL(src);

    g_object_set(src, "shm-name", "/test_appsrc_shm", NULL);

    gchar *name = NULL;
    g_object_get(src, "shm-name", &name, NULL);
    ASSERT_TRUE(g_strcmp0(name, "/test_appsrc_shm") == 0);
    g_free(name);

    gboolean is_live = FALSE;
    g_object_get(src, "is-live", &is_live, NULL);
    ASSERT_TRUE(is_live == TRUE);

    gst_object_unref(src);
    PASS();
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);
    printf("=== GstNvmmAppSrc Tests ===\n");

    RUN_TEST(appsrc_creates);
    RUN_TEST(appsrc_properties);

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    gst_deinit();
    return tests_failed > 0 ? 1 : 0;
}
