/// Unit tests for GstNvmmAppSrc — shared memory source element.
/// Also includes an integration test pairing nvmmsink → nvmmappsrc via shm.

#include <gst/gst.h>

#include "gstnvmmappsrc.h"
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

/// Integration test: sink writes a frame to shm, source reads it back.
struct ShmHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t format;
    uint32_t data_size;
    uint32_t num_planes;
    uint32_t pitches[4];
    uint32_t offsets[4];
    int32_t  dmabuf_fd;
    uint64_t frame_number;
    uint64_t timestamp_ns;
    uint32_t ready;
    uint32_t _reserved[8];
};

static void test_sink_to_source_integration() {
    const char *shm_name = "/test_nvmm_integration";

    /* Create sink, write a frame to shm */
    GstElement *sink = gst_element_factory_make("nvmmsink", NULL);
    ASSERT_NOT_NULL(sink);
    g_object_set(sink, "shm-name", shm_name, NULL);

    /* Start sink to create shm */
    GstStateChangeReturn ret = gst_element_set_state(sink, GST_STATE_READY);
    ASSERT_TRUE(ret == GST_STATE_CHANGE_SUCCESS);

    /* Manually write a test frame to the shm */
    int fd = shm_open(shm_name, O_RDWR, 0);
    ASSERT_TRUE(fd >= 0);

    struct stat st;
    fstat(fd, &st);
    void *ptr = mmap(NULL, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ASSERT_TRUE(ptr != MAP_FAILED);

    auto *header = static_cast<ShmHeader *>(ptr);
    header->magic = 0x4E564D4D;
    header->version = 1;
    header->width = 64;
    header->height = 64;
    header->format = 11;  /* RGBA in GstVideoFormat */
    header->data_size = 64 * 64 * 4;
    header->num_planes = 1;
    header->pitches[0] = 64 * 4;
    header->offsets[0] = 0;
    header->dmabuf_fd = -1;
    header->frame_number = 1;
    header->timestamp_ns = 42000;

    /* Fill frame data with a pattern */
    auto *frame_data = reinterpret_cast<uint8_t *>(ptr) + sizeof(ShmHeader);
    memset(frame_data, 0xAB, header->data_size);

    __sync_synchronize();
    header->ready = 1;

    munmap(ptr, st.st_size);
    close(fd);

    /* Now create source and verify it can read the frame */
    GstElement *src = gst_element_factory_make("nvmmappsrc", NULL);
    ASSERT_NOT_NULL(src);
    g_object_set(src, "shm-name", shm_name, "is-live", FALSE, NULL);

    /* Verify source can start (attaches to shm) */
    ret = gst_element_set_state(src, GST_STATE_READY);
    ASSERT_TRUE(ret == GST_STATE_CHANGE_SUCCESS);

    /* Clean up */
    gst_element_set_state(src, GST_STATE_NULL);
    gst_object_unref(src);
    gst_element_set_state(sink, GST_STATE_NULL);
    gst_object_unref(sink);

    PASS();
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);
    printf("=== GstNvmmAppSrc Tests ===\n");

    RUN_TEST(appsrc_creates);
    RUN_TEST(appsrc_properties);
    RUN_TEST(sink_to_source_integration);

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    gst_deinit();
    return tests_failed > 0 ? 1 : 0;
}
