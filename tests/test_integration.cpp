#include <gst/gst.h>
#include <gst/video/video.h>

#include "gstnvmmallocator.h"
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

TEST(nvmmconvert_state_transitions) {
    GstElement *elem = gst_element_factory_make("nvmmconvert", NULL);
    ASSERT_NOT_NULL(elem);

    GstStateChangeReturn ret;
    GstState current, pending;

    ret = gst_element_set_state(elem, GST_STATE_READY);
    ASSERT_TRUE(ret == GST_STATE_CHANGE_SUCCESS ||
                ret == GST_STATE_CHANGE_NO_PREROLL);

    gst_element_get_state(elem, &current, &pending, GST_CLOCK_TIME_NONE);
    ASSERT_EQ(current, GST_STATE_READY);

    ret = gst_element_set_state(elem, GST_STATE_NULL);
    ASSERT_EQ(ret, GST_STATE_CHANGE_SUCCESS);

    gst_object_unref(elem);
}

TEST(nvmmconvert_pad_templates) {
    GstElement *elem = gst_element_factory_make("nvmmconvert", NULL);
    ASSERT_NOT_NULL(elem);

    GstPad *sink_pad = gst_element_get_static_pad(elem, "sink");
    ASSERT_NOT_NULL(sink_pad);

    GstPad *src_pad = gst_element_get_static_pad(elem, "src");
    ASSERT_NOT_NULL(src_pad);

    GstCaps *sink_caps = gst_pad_query_caps(sink_pad, NULL);
    ASSERT_NOT_NULL(sink_caps);
    ASSERT_TRUE(!gst_caps_is_empty(sink_caps));

    GstCapsFeatures *features = gst_caps_get_features(sink_caps, 0);
    ASSERT_NOT_NULL(features);
    ASSERT_TRUE(gst_caps_features_contains(features, "memory:NVMM"));

    gst_caps_unref(sink_caps);
    gst_object_unref(sink_pad);
    gst_object_unref(src_pad);
    gst_object_unref(elem);
}

/// Sequential, because each segment is ~33MB and Docker's /dev/shm is small.
TEST(multiple_shm_segments) {
    const char *names[] = {"/test_int_multi_0", "/test_int_multi_1"};

    for (int i = 0; i < 2; i++) {
        GstElement *sink = gst_element_factory_make("nvmmsink", NULL);
        ASSERT_NOT_NULL(sink);
        g_object_set(sink, "shm-name", names[i], NULL);

        GstStateChangeReturn ret = gst_element_set_state(sink, GST_STATE_READY);
        ASSERT_EQ(ret, GST_STATE_CHANGE_SUCCESS);

        int fd = shm_open(names[i], O_RDONLY, 0);
        ASSERT_TRUE(fd >= 0);
        close(fd);

        gst_element_set_state(sink, GST_STATE_NULL);
        gst_object_unref(sink);

        fd = shm_open(names[i], O_RDONLY, 0);
        ASSERT_TRUE(fd < 0);
    }
}

TEST(convert_dynamic_properties) {
    GstElement *convert = gst_element_factory_make("nvmmconvert", NULL);
    ASSERT_NOT_NULL(convert);

    g_object_set(convert,
        "crop-x", (guint) 0, "crop-y", (guint) 0,
        "crop-w", (guint) 1920, "crop-h", (guint) 1080,
        "flip-method", 0, NULL);

    GstStateChangeReturn ret = gst_element_set_state(convert, GST_STATE_READY);
    ASSERT_TRUE(ret == GST_STATE_CHANGE_SUCCESS);

    g_object_set(convert,
        "crop-x", (guint) 100, "crop-y", (guint) 200,
        "crop-w", (guint) 800, "crop-h", (guint) 600,
        "flip-method", 2, NULL);

    guint cx, cy, cw, ch;
    gint fm;
    g_object_get(convert,
        "crop-x", &cx, "crop-y", &cy,
        "crop-w", &cw, "crop-h", &ch,
        "flip-method", &fm, NULL);

    ASSERT_EQ(cx, 100u);
    ASSERT_EQ(cy, 200u);
    ASSERT_EQ(cw, 800u);
    ASSERT_EQ(ch, 600u);
    ASSERT_EQ(fm, 2);

    gst_element_set_state(convert, GST_STATE_NULL);
    gst_object_unref(convert);
}

TEST(convert_in_pipeline_bin) {
    GstElement *pipeline = gst_pipeline_new("test");
    GstElement *convert = gst_element_factory_make("nvmmconvert", "conv");
    GstElement *sink = gst_element_factory_make("nvmmsink", "sink");

    ASSERT_NOT_NULL(pipeline);
    ASSERT_NOT_NULL(convert);
    ASSERT_NOT_NULL(sink);

    g_object_set(sink, "shm-name", "/test_int_bin", NULL);
    g_object_set(convert,
        "crop-w", (guint) 640, "crop-h", (guint) 480, NULL);

    gst_bin_add_many(GST_BIN(pipeline), convert, sink, NULL);

    GstElement *found = gst_bin_get_by_name(GST_BIN(pipeline), "conv");
    ASSERT_NOT_NULL(found);
    gst_object_unref(found);

    found = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    ASSERT_NOT_NULL(found);
    gst_object_unref(found);

    gst_element_set_state(pipeline, GST_STATE_READY);

    int fd = shm_open("/test_int_bin", O_RDONLY, 0);
    ASSERT_TRUE(fd >= 0);
    close(fd);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
}

TEST(allocator_no_leak_stress) {
    GstAllocator *alloc = gst_nvmm_allocator_new(0);
    ASSERT_NOT_NULL(alloc);

    for (int i = 0; i < 100; i++) {
        GstMemory *mem = gst_nvmm_allocator_alloc_video(alloc,
            GST_VIDEO_FORMAT_NV12, 640, 480);
        ASSERT_NOT_NULL(mem);
        gst_memory_unref(mem);
    }

    gst_object_unref(alloc);
}

TEST(shm_header_protocol) {
    const char *shm_name = "/test_int_protocol";

    GstElement *sink = gst_element_factory_make("nvmmsink", NULL);
    ASSERT_NOT_NULL(sink);
    g_object_set(sink, "shm-name", shm_name, NULL);
    gst_element_set_state(sink, GST_STATE_READY);

    int fd = shm_open(shm_name, O_RDONLY, 0);
    ASSERT_TRUE(fd >= 0);

    struct stat st;
    fstat(fd, &st);
    ASSERT_TRUE(st.st_size >= (off_t)sizeof(NvmmShmHeader));

    void *ptr = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    ASSERT_TRUE(ptr != MAP_FAILED);

    auto *header = static_cast<const NvmmShmHeader *>(ptr);

    ASSERT_EQ(header->ready, 0u);
    ASSERT_EQ(header->frame_number, 0u);

    munmap(ptr, st.st_size);
    close(fd);

    gst_element_set_state(sink, GST_STATE_NULL);
    gst_object_unref(sink);
}

TEST(source_missing_shm) {
    shm_unlink("/test_int_missing");

    GstElement *src = gst_element_factory_make("nvmmappsrc", NULL);
    ASSERT_NOT_NULL(src);
    g_object_set(src, "shm-name", "/test_int_missing", NULL);

    GstStateChangeReturn ret = gst_element_set_state(src, GST_STATE_PAUSED);
    ASSERT_TRUE(ret == GST_STATE_CHANGE_FAILURE ||
                ret == GST_STATE_CHANGE_NO_PREROLL);

    gst_element_set_state(src, GST_STATE_NULL);
    gst_object_unref(src);
}

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
