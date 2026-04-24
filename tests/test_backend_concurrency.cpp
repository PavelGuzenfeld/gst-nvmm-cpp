/// Backend concurrency test — drives the IPC producer/consumer directly
/// (no GStreamer plugin loading), stresses the atomic access patterns
/// that matter under TSan.
///
/// Purpose: the GStreamer element factory tests can't run under TSan
/// because TSan can't be LD_PRELOADed the way ASan can, and GStreamer
/// rejects dlopen'ing TSan-instrumented .so files from an unsanitized
/// plugin scanner. This test bypasses the factory and talks to
/// ipc_backend.h directly, which is where the raciest code lives.
///
/// Covers:
///   - producer render() and consumer fetch() running on separate
///     threads, sharing ready / write_idx / frame_number / slots[].ref_count
///     atomically
///   - accept_thread (JP6 only) concurrent with the main render loop
///   - teardown ordering
///
/// Under TSan, any data race in the backend code will fail this test.

#include "ipc_backend.h"
#include "gstnvmmallocator.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>

#include <sys/mman.h>
#include <unistd.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(expr) do {                                         \
    if (!(expr)) {                                                     \
        std::printf("FAIL at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        tests_failed++; return;                                        \
    }                                                                  \
} while (0)

#define RUN(name) do {                       \
    std::printf("  TEST %s ... ", #name);    \
    std::fflush(stdout);                     \
    int failed_before = tests_failed;        \
    test_##name();                           \
    if (tests_failed == failed_before) {     \
        std::printf("PASS\n"); tests_passed++; \
    }                                        \
} while (0)

/// Build a GstBuffer carrying an NVMM-allocator-owned GstMemory for a
/// given video-info. The JP6 producer expects NVMM input.
static GstBuffer *
make_nvmm_buffer(GstAllocator *alloc, const GstVideoInfo *vi)
{
    GstMemory *mem = gst_nvmm_allocator_alloc_video(
        alloc,
        (int)GST_VIDEO_INFO_FORMAT(vi),
        (guint)GST_VIDEO_INFO_WIDTH(vi),
        (guint)GST_VIDEO_INFO_HEIGHT(vi));
    if (!mem) return nullptr;
    GstBuffer *buf = gst_buffer_new();
    gst_buffer_append_memory(buf, mem);
    return buf;
}

/// Producer and consumer on separate threads push ~200 frames through
/// the backend back-to-back. Under TSan this catches any missing
/// acquire/release on the header fields.
static void test_concurrent_producer_consumer()
{
    const char *shm = "/test_backend_concurrency";
    shm_unlink(shm);  /* any stale segment */

    GstElement *prod_owner = GST_ELEMENT(g_object_new(GST_TYPE_BIN, "name", "prod", NULL));
    GstElement *cons_owner = GST_ELEMENT(g_object_new(GST_TYPE_BIN, "name", "cons", NULL));

    GstAllocator *alloc = gst_nvmm_allocator_new(0);
    ASSERT_TRUE(alloc);

    NvmmIpcProducer *prod = nvmm_ipc_producer_new(shm, 8);
    ASSERT_TRUE(prod);
    ASSERT_TRUE(nvmm_ipc_producer_start(prod, prod_owner));

    GstVideoInfo vi;
    gst_video_info_set_format(&vi, GST_VIDEO_FORMAT_RGBA, 64, 64);
    ASSERT_TRUE(nvmm_ipc_producer_set_caps(prod, prod_owner, &vi, TRUE));

    std::atomic<int> frames_seen{0};
    std::atomic<bool> stop{false};

    std::thread consumer([&]() {
        NvmmIpcConsumer *cons = nvmm_ipc_consumer_new(shm);
        if (!nvmm_ipc_consumer_start(cons, cons_owner)) {
            nvmm_ipc_consumer_free(cons);
            return;
        }
        GstPad *pad = gst_pad_new("src", GST_PAD_SRC);
        while (!stop.load(std::memory_order_relaxed)) {
            GstBuffer *buf = nullptr;
            GstFlowReturn r = nvmm_ipc_consumer_fetch(cons, cons_owner, pad, &buf);
            if (r == GST_FLOW_OK && buf) {
                frames_seen.fetch_add(1, std::memory_order_relaxed);
                gst_buffer_unref(buf);
            } else if (r == GST_FLOW_EOS) {
                break;
            }
        }
        nvmm_ipc_consumer_stop(cons, cons_owner);
        nvmm_ipc_consumer_free(cons);
        gst_object_unref(pad);
    });

    /* Producer loop: 200 NVMM frames at ~1 kHz. */
    for (int i = 0; i < 200; i++) {
        GstBuffer *frame = make_nvmm_buffer(alloc, &vi);
        ASSERT_TRUE(frame);
        GST_BUFFER_PTS(frame) = i * 1000000;  /* 1 ms spacing */
        nvmm_ipc_producer_render(prod, prod_owner, frame);
        gst_buffer_unref(frame);
        g_usleep(1000);
    }

    g_usleep(50000);  /* let the consumer drain the last fetch */
    stop.store(true, std::memory_order_relaxed);
    consumer.join();

    nvmm_ipc_producer_stop(prod, prod_owner);
    nvmm_ipc_producer_free(prod);
    gst_object_unref(alloc);
    gst_object_unref(prod_owner);
    gst_object_unref(cons_owner);

    std::printf("[saw %d frames] ", frames_seen.load());
    ASSERT_TRUE(frames_seen.load() > 0);
}

/// Start + stop a producer 50 times; exercises the accept_thread
/// lifecycle (JP6) + shm cleanup (both backends).
static void test_producer_start_stop_churn()
{
    GstElement *owner = GST_ELEMENT(g_object_new(GST_TYPE_BIN, "name", "owner", NULL));
    for (int i = 0; i < 50; i++) {
        char name[64];
        std::snprintf(name, sizeof(name), "/churn_%d", i);
        shm_unlink(name);
        NvmmIpcProducer *p = nvmm_ipc_producer_new(name, 4);
        ASSERT_TRUE(p);
        ASSERT_TRUE(nvmm_ipc_producer_start(p, owner));
        nvmm_ipc_producer_stop(p, owner);
        nvmm_ipc_producer_free(p);
    }
    gst_object_unref(owner);
}

int main(int argc, char *argv[])
{
    gst_init(&argc, &argv);
    /* Initialize the backend's GST_DEBUG category. Without this, every
     * GST_INFO_OBJECT() call in the backend aborts on the NULL category
     * assertion because the test binary isn't a registered plugin. */
    nvmm_ipc_backend_init_debug();

    std::printf("=== Backend Concurrency Tests ===\n");
    RUN(concurrent_producer_consumer);
    RUN(producer_start_stop_churn);
    std::printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed ? 1 : 0;
}
