/// Unit tests for GstNvmmSink — shared memory sink element.

#include <gst/gst.h>

#include "gstnvmmsink.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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

/// Shared memory header must match the one in gstnvmmsink.cpp
#include "shm_protocol.h"
typedef NvmmShmHeader ShmHeader;

static void test_sink_creates() {
    GstElement *sink = gst_element_factory_make("nvmmsink", "test-sink");
    ASSERT_NOT_NULL(sink);
    ASSERT_TRUE(GST_IS_NVMM_SINK(sink));
    gst_object_unref(sink);
    PASS();
}

static void test_sink_properties() {
    GstElement *sink = gst_element_factory_make("nvmmsink", NULL);
    ASSERT_NOT_NULL(sink);

    g_object_set(sink, "shm-name", "/test_nvmm_sink", NULL);

    gchar *name = NULL;
    g_object_get(sink, "shm-name", &name, NULL);
    ASSERT_TRUE(g_strcmp0(name, "/test_nvmm_sink") == 0);
    g_free(name);

    gst_object_unref(sink);
    PASS();
}

static void test_sink_state_transition() {
    GstElement *sink = gst_element_factory_make("nvmmsink", NULL);
    ASSERT_NOT_NULL(sink);

    g_object_set(sink, "shm-name", "/test_nvmm_state", NULL);

    GstStateChangeReturn ret;

    ret = gst_element_set_state(sink, GST_STATE_READY);
    ASSERT_TRUE(ret == GST_STATE_CHANGE_SUCCESS);

    ret = gst_element_set_state(sink, GST_STATE_NULL);
    ASSERT_TRUE(ret == GST_STATE_CHANGE_SUCCESS);

    gst_object_unref(sink);
    PASS();
}

static void test_sink_shm_created() {
    const char *shm_name = "/test_nvmm_shm_check";
    GstElement *sink = gst_element_factory_make("nvmmsink", NULL);
    ASSERT_NOT_NULL(sink);

    g_object_set(sink, "shm-name", shm_name, NULL);

    /* Transition to READY → PAUSED triggers start() */
    GstStateChangeReturn ret = gst_element_set_state(sink, GST_STATE_READY);
    ASSERT_TRUE(ret == GST_STATE_CHANGE_SUCCESS);

    /* Verify shared memory exists */
    int fd = shm_open(shm_name, O_RDONLY, 0);
    ASSERT_TRUE(fd >= 0);

    struct stat st;
    fstat(fd, &st);
    ASSERT_TRUE(st.st_size > 0);

    close(fd);

    gst_element_set_state(sink, GST_STATE_NULL);
    gst_object_unref(sink);

    /* After NULL, shm should be unlinked */
    fd = shm_open(shm_name, O_RDONLY, 0);
    ASSERT_TRUE(fd < 0);  /* should fail */

    PASS();
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);
    printf("=== GstNvmmSink Tests ===\n");

    RUN_TEST(sink_creates);
    RUN_TEST(sink_properties);
    RUN_TEST(sink_state_transition);
    RUN_TEST(sink_shm_created);

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    gst_deinit();
    return tests_failed > 0 ? 1 : 0;
}
