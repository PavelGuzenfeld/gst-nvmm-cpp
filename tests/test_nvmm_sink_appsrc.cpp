#include <gst/gst.h>

#include "gstnvmmappsrc.h"
#include "gstnvmmsink.h"
#include "shm_protocol.h"

#include <cstdio>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "test_harness.h"

namespace {

/// Declared first: within one TU, static objects initialize in declaration order.
struct GstInit { GstInit() { gst_init(nullptr, nullptr); } } _gst_init;

TEST(sink_creates_and_round_trips_shm_name) {
    GstElement *sink = gst_element_factory_make("nvmmsink", "test-sink");
    ASSERT_NOT_NULL(sink);
    ASSERT_TRUE(GST_IS_NVMM_SINK(sink));

    g_object_set(sink, "shm-name", "/test_nvmm_sink", NULL);
    gchar *name = NULL;
    g_object_get(sink, "shm-name", &name, NULL);
    ASSERT_TRUE(g_strcmp0(name, "/test_nvmm_sink") == 0);
    g_free(name);

    gst_object_unref(sink);
}

/// The consumer holds up to RELEASE_DELAY (12) buffers in flight, so a pool under
/// 13 starves the producer.
TEST(sink_pool_size_guarded) {
    GstElement *sink = gst_element_factory_make("nvmmsink", NULL);
    ASSERT_NOT_NULL(sink);

    GParamSpec *pspec =
        g_object_class_find_property(G_OBJECT_GET_CLASS(sink), "pool-size");
    ASSERT_NOT_NULL(pspec);
    GParamSpecInt *ispec = G_PARAM_SPEC_INT(pspec);
    ASSERT_TRUE(ispec->minimum == NVMM_MIN_POOL_SIZE);
    ASSERT_TRUE(ispec->maximum == NVMM_POOL_SIZE);

    gint ps = 0;
    g_object_set(sink, "pool-size", 14, NULL);
    g_object_get(sink, "pool-size", &ps, NULL);
    ASSERT_TRUE(ps == 14);

    gst_object_unref(sink);
}

TEST(sink_ready_creates_shm_and_null_unlinks_it) {
    const char *shm_name = "/test_nvmm_shm_check";
    GstElement *sink = gst_element_factory_make("nvmmsink", NULL);
    ASSERT_NOT_NULL(sink);
    g_object_set(sink, "shm-name", shm_name, NULL);

    ASSERT_TRUE(gst_element_set_state(sink, GST_STATE_READY) == GST_STATE_CHANGE_SUCCESS);

    int fd = shm_open(shm_name, O_RDONLY, 0);
    ASSERT_TRUE(fd >= 0);
    struct stat st;
    fstat(fd, &st);
    ASSERT_TRUE(st.st_size > 0);
    close(fd);

    ASSERT_TRUE(gst_element_set_state(sink, GST_STATE_NULL) == GST_STATE_CHANGE_SUCCESS);
    gst_object_unref(sink);

    ASSERT_TRUE(shm_open(shm_name, O_RDONLY, 0) < 0);
}

TEST(appsrc_creates_live_and_round_trips_shm_name) {
    GstElement *src = gst_element_factory_make("nvmmappsrc", "test-src");
    ASSERT_NOT_NULL(src);
    ASSERT_TRUE(GST_IS_NVMM_APP_SRC(src));

    g_object_set(src, "shm-name", "/test_appsrc_shm", NULL);
    gchar *name = NULL;
    g_object_get(src, "shm-name", &name, NULL);
    ASSERT_TRUE(g_strcmp0(name, "/test_appsrc_shm") == 0);
    g_free(name);

    gboolean is_live = FALSE;
    g_object_get(src, "is-live", &is_live, NULL);
    ASSERT_TRUE(is_live == TRUE);

    gst_object_unref(src);
}

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
