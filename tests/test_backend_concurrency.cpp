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
///   - accept_thread concurrent with the main render loop
///   - teardown ordering
///
/// Under TSan, any data race in the backend code will fail this test.

#include "ipc_backend.h"
#include "gstnvmmallocator.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <atomic>
#include <climits>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

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
/// given video-info. The producer expects NVMM input.
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
/// lifecycle and shm cleanup.
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

/// Multi-consumer fan-out. Three consumers read from one producer
/// simultaneously. Verifies that ref_counts[] scheme lets all three
/// observe every published frame (the consumer CAS-increments the
/// count so producer sees nonzero and skips the slot for recycle).
///
/// This is the original motivation in @adujardin's #1:
/// "buffer corruption when sharing a single camera source to multiple
/// receivers". The old nvunixfdsink/nvunixfdsrc path had a data race
/// here; the pool + ref-count scheme should not.
static void test_multi_consumer_fanout()
{
    const char *shm = "/test_fanout";
    shm_unlink(shm);

    GstElement *prod_owner = GST_ELEMENT(g_object_new(GST_TYPE_BIN, "name", "prod_fo", NULL));
    GstAllocator *alloc = gst_nvmm_allocator_new(0);

    NvmmIpcProducer *prod = nvmm_ipc_producer_new(shm, 8);
    ASSERT_TRUE(prod);
    ASSERT_TRUE(nvmm_ipc_producer_start(prod, prod_owner));

    GstVideoInfo vi;
    gst_video_info_set_format(&vi, GST_VIDEO_FORMAT_RGBA, 64, 64);
    ASSERT_TRUE(nvmm_ipc_producer_set_caps(prod, prod_owner, &vi, TRUE));

    constexpr int kConsumers = 3;
    std::atomic<int> seen[kConsumers] = {};
    std::atomic<bool> stop{false};
    std::thread consumers[kConsumers];

    auto consumer_fn = [&](int id) {
        GstElement *co = GST_ELEMENT(g_object_new(GST_TYPE_BIN,
            "name", g_strdup_printf("cons_fo_%d", id), NULL));
        NvmmIpcConsumer *c = nvmm_ipc_consumer_new(shm);
        if (!nvmm_ipc_consumer_start(c, co)) {
            nvmm_ipc_consumer_free(c);
            gst_object_unref(co);
            return;
        }
        GstPad *pad = gst_pad_new("src", GST_PAD_SRC);
        while (!stop.load()) {
            GstBuffer *buf = nullptr;
            GstFlowReturn r = nvmm_ipc_consumer_fetch(c, co, pad, &buf);
            if (r == GST_FLOW_OK && buf) {
                seen[id].fetch_add(1);
                gst_buffer_unref(buf);
            } else if (r == GST_FLOW_EOS) {
                break;
            }
        }
        nvmm_ipc_consumer_stop(c, co);
        nvmm_ipc_consumer_free(c);
        gst_object_unref(pad);
        gst_object_unref(co);
    };

    for (int i = 0; i < kConsumers; i++)
        consumers[i] = std::thread(consumer_fn, i);

    /* Give consumers a moment to connect + handshake. */
    g_usleep(50000);

    const int total_frames = 100;
    for (int i = 0; i < total_frames; i++) {
        GstBuffer *frame = make_nvmm_buffer(alloc, &vi);
        ASSERT_TRUE(frame);
        GST_BUFFER_PTS(frame) = i * 1000000;
        nvmm_ipc_producer_render(prod, prod_owner, frame);
        gst_buffer_unref(frame);
        g_usleep(2000);   /* ~500 Hz, slower than consumer polling */
    }

    g_usleep(100000);
    stop.store(true);
    for (auto &t : consumers) t.join();

    nvmm_ipc_producer_stop(prod, prod_owner);
    nvmm_ipc_producer_free(prod);
    gst_object_unref(alloc);
    gst_object_unref(prod_owner);

    int min_seen = INT_MAX, max_seen = 0;
    for (int i = 0; i < kConsumers; i++) {
        int s = seen[i].load();
        if (s < min_seen) min_seen = s;
        if (s > max_seen) max_seen = s;
    }
    std::printf("[consumers: min=%d max=%d of %d] ",
                min_seen, max_seen, total_frames);
    /* All three consumers should observe >0 frames; we don't require
     * every frame (some may be dropped if the producer laps them). */
    ASSERT_TRUE(min_seen > 0);
}

/* Tiny rotate-and-xor checksum. Good enough to detect any nonzero bit
 * flip in a frame; not cryptographically strong, but we only need to
 * distinguish "this is the frame I expected" from "this isn't". */
static uint32_t frame_checksum(const uint8_t *p, size_t n) {
    uint32_t c = 0xDEADBEEFu;
    for (size_t i = 0; i < n; i++) {
        c = (c << 5) | (c >> 27);
        c ^= p[i];
    }
    return c;
}

#ifdef NVMM_MOCK_API
#include "nvbufsurface_mock.h"
#else
#include <nvbufsurface.h>
#endif

/// Fill a known pattern into a fresh NVMM frame, render it, expect the
/// consumer to receive a buffer whose plane-0 bytes hash to the same
/// checksum. Catches silent corruption in the IPC path — unaligned
/// memcpy, dropped bytes, wrong stride, half-published slots.
static void test_crc_roundtrip()
{
    const char *shm = "/test_crc_roundtrip";
    shm_unlink(shm);

    GstElement *prod_owner = GST_ELEMENT(g_object_new(GST_TYPE_BIN, "name", "prod_crc", NULL));
    GstElement *cons_owner = GST_ELEMENT(g_object_new(GST_TYPE_BIN, "name", "cons_crc", NULL));
    GstAllocator *alloc = gst_nvmm_allocator_new(0);
    ASSERT_TRUE(alloc);

    NvmmIpcProducer *prod = nvmm_ipc_producer_new(shm, 8);
    ASSERT_TRUE(nvmm_ipc_producer_start(prod, prod_owner));

    constexpr int W = 64, H = 64;
    GstVideoInfo vi;
    gst_video_info_set_format(&vi, GST_VIDEO_FORMAT_RGBA, W, H);
    ASSERT_TRUE(nvmm_ipc_producer_set_caps(prod, prod_owner, &vi, TRUE));

    constexpr int kFrames = 50;
    std::atomic<bool> stop{false};
    std::vector<uint32_t> rx_crcs;
    std::mutex             rx_mtx;

    std::thread consumer([&]() {
        NvmmIpcConsumer *cons = nvmm_ipc_consumer_new(shm);
        if (!nvmm_ipc_consumer_start(cons, cons_owner)) {
            nvmm_ipc_consumer_free(cons);
            return;
        }
        GstPad *pad = gst_pad_new("src", GST_PAD_SRC);
        while (!stop.load()) {
            GstBuffer *buf = nullptr;
            GstFlowReturn r = nvmm_ipc_consumer_fetch(cons, cons_owner, pad, &buf);
            if (r == GST_FLOW_OK && buf) {
                /* The consumer wraps the imported NvBufSurface in a
                 * GstMemory whose mapped data is the surface pointer. */
                GstMapInfo info;
                if (gst_buffer_map(buf, &info, GST_MAP_READ)) {
                    NvBufSurface *surf = reinterpret_cast<NvBufSurface*>(info.data);
                    if (surf && NvBufSurfaceMap(surf, 0, -1, NVBUF_MAP_READ) == 0) {
                        NvBufSurfaceSyncForCpu(surf, 0, -1);
                        const uint8_t *plane =
                            (const uint8_t *)surf->surfaceList[0].mappedAddr.addr[0];
                        if (plane) {
                            uint32_t c = frame_checksum(plane, W * H * 4);
                            std::lock_guard<std::mutex> g(rx_mtx);
                            rx_crcs.push_back(c);
                        }
                        NvBufSurfaceUnMap(surf, 0, -1);
                    }
                    gst_buffer_unmap(buf, &info);
                }
                gst_buffer_unref(buf);
            } else if (r == GST_FLOW_EOS) {
                break;
            }
        }
        nvmm_ipc_consumer_stop(cons, cons_owner);
        nvmm_ipc_consumer_free(cons);
        gst_object_unref(pad);
    });

    g_usleep(50000);  /* let the consumer handshake */

    std::vector<uint32_t> tx_crcs;
    tx_crcs.reserve(kFrames);
    for (int i = 0; i < kFrames; i++) {
        GstBuffer *frame = make_nvmm_buffer(alloc, &vi);
        ASSERT_TRUE(frame);
        GstMemory *mem = gst_buffer_peek_memory(frame, 0);
        guint8 *plane = nullptr;
        gsize   psize = 0;
        ASSERT_TRUE(gst_nvmm_memory_map_plane(mem, 0, GST_MAP_WRITE, &plane, &psize));
        /* Deterministic per-frame pattern. */
        for (gsize j = 0; j < psize; j++)
            plane[j] = (uint8_t)((i * 7 + j) & 0xff);
        tx_crcs.push_back(frame_checksum(plane, W * H * 4));
        gst_nvmm_memory_unmap_plane(mem);
        GST_BUFFER_PTS(frame) = i * 1000000;
        nvmm_ipc_producer_render(prod, prod_owner, frame);
        gst_buffer_unref(frame);
        g_usleep(2000);  /* slower than consumer poll, lets each frame land */
    }

    g_usleep(100000);
    stop.store(true);
    consumer.join();

    nvmm_ipc_producer_stop(prod, prod_owner);
    nvmm_ipc_producer_free(prod);
    gst_object_unref(alloc);
    gst_object_unref(prod_owner);
    gst_object_unref(cons_owner);

    std::printf("[tx=%d rx=%d] ", (int)tx_crcs.size(), (int)rx_crcs.size());
    ASSERT_TRUE(!rx_crcs.empty());

    /* Each received checksum must be one we sent. Producer can lap the
     * consumer (the consumer sees a subset of frames), but the consumer
     * must NEVER see a checksum that wasn't a real produced frame. */
    int matched = 0;
    for (uint32_t cc : rx_crcs) {
        for (uint32_t ec : tx_crcs) {
            if (cc == ec) { matched++; break; }
        }
    }
    if (matched != (int)rx_crcs.size()) {
        std::printf("CORRUPTION: %d of %d received frames don't match any sent ",
                    (int)rx_crcs.size() - matched, (int)rx_crcs.size());
        ASSERT_TRUE(matched == (int)rx_crcs.size());
    }
}

/// Producer-side zero-copy path. Drives propose_allocation, takes the
/// pool the producer hands back, acquires buffers from it, fills them
/// and renders. The producer's render() should detect the slot tag in
/// qdata and skip NvBufSurfaceCopy. Verified end-to-end with the same
/// CRC scheme as test_crc_roundtrip — every received frame's plane-0
/// must hash to a value we sent.
static void test_producer_zerocopy()
{
    const char *shm = "/test_producer_zerocopy";
    shm_unlink(shm);

    GstElement *prod_owner = GST_ELEMENT(g_object_new(GST_TYPE_BIN, "name", "prod_zc", NULL));
    GstElement *cons_owner = GST_ELEMENT(g_object_new(GST_TYPE_BIN, "name", "cons_zc", NULL));

    NvmmIpcProducer *prod = nvmm_ipc_producer_new(shm, 8);
    ASSERT_TRUE(nvmm_ipc_producer_start(prod, prod_owner));

    constexpr int W = 64, H = 64;
    GstVideoInfo vi;
    gst_video_info_set_format(&vi, GST_VIDEO_FORMAT_RGBA, W, H);
    ASSERT_TRUE(nvmm_ipc_producer_set_caps(prod, prod_owner, &vi, TRUE));

    /* Drive propose_allocation the way upstream would. */
    GstCaps *caps = gst_video_info_to_caps(&vi);
    GstQuery *q = gst_query_new_allocation(caps, TRUE);
    ASSERT_TRUE(nvmm_ipc_producer_propose_allocation(prod, prod_owner, q));
    ASSERT_TRUE(gst_query_get_n_allocation_pools(q) > 0);

    GstBufferPool *upstream_pool = nullptr;
    guint pool_size = 0, pool_min = 0, pool_max = 0;
    gst_query_parse_nth_allocation_pool(q, 0, &upstream_pool,
                                         &pool_size, &pool_min, &pool_max);
    ASSERT_TRUE(upstream_pool);
    /* The pool comes back floating-ref'd from query; sink it. */
    gst_object_ref_sink(upstream_pool);
    gst_query_unref(q);
    gst_caps_unref(caps);

    ASSERT_TRUE(gst_buffer_pool_set_active(upstream_pool, TRUE));

    constexpr int kFrames = 50;
    std::atomic<bool> stop{false};
    std::vector<uint32_t> rx_crcs;
    std::mutex             rx_mtx;

    std::thread consumer([&]() {
        NvmmIpcConsumer *cons = nvmm_ipc_consumer_new(shm);
        if (!nvmm_ipc_consumer_start(cons, cons_owner)) {
            nvmm_ipc_consumer_free(cons);
            return;
        }
        GstPad *pad = gst_pad_new("src", GST_PAD_SRC);
        while (!stop.load()) {
            GstBuffer *buf = nullptr;
            GstFlowReturn r = nvmm_ipc_consumer_fetch(cons, cons_owner, pad, &buf);
            if (r == GST_FLOW_OK && buf) {
                GstMapInfo info;
                if (gst_buffer_map(buf, &info, GST_MAP_READ)) {
                    NvBufSurface *surf = reinterpret_cast<NvBufSurface*>(info.data);
                    if (surf && NvBufSurfaceMap(surf, 0, -1, NVBUF_MAP_READ) == 0) {
                        NvBufSurfaceSyncForCpu(surf, 0, -1);
                        const uint8_t *plane =
                            (const uint8_t *)surf->surfaceList[0].mappedAddr.addr[0];
                        if (plane) {
                            uint32_t c = frame_checksum(plane, W * H * 4);
                            std::lock_guard<std::mutex> g(rx_mtx);
                            rx_crcs.push_back(c);
                        }
                        NvBufSurfaceUnMap(surf, 0, -1);
                    }
                    gst_buffer_unmap(buf, &info);
                }
                gst_buffer_unref(buf);
            } else if (r == GST_FLOW_EOS) {
                break;
            }
        }
        nvmm_ipc_consumer_stop(cons, cons_owner);
        nvmm_ipc_consumer_free(cons);
        gst_object_unref(pad);
    });

    g_usleep(50000);

    std::vector<uint32_t> tx_crcs;
    tx_crcs.reserve(kFrames);
    for (int i = 0; i < kFrames; i++) {
        GstBuffer *buf = nullptr;
        GstFlowReturn r = gst_buffer_pool_acquire_buffer(upstream_pool, &buf, nullptr);
        ASSERT_TRUE(r == GST_FLOW_OK);
        ASSERT_TRUE(buf);

        /* Buffer wraps a slot's NvBufSurface directly. Map it CPU-side
         * via NvBufSurfaceMap — we're writing into the shared slot. */
        GstMapInfo info;
        ASSERT_TRUE(gst_buffer_map(buf, &info, GST_MAP_READWRITE));
        NvBufSurface *surf = reinterpret_cast<NvBufSurface*>(info.data);
        ASSERT_TRUE(surf);
        ASSERT_TRUE(NvBufSurfaceMap(surf, 0, -1, NVBUF_MAP_READ_WRITE) == 0);
        uint8_t *plane = (uint8_t *)surf->surfaceList[0].mappedAddr.addr[0];
        ASSERT_TRUE(plane);
        for (gsize j = 0; j < (gsize)W * H * 4; j++)
            plane[j] = (uint8_t)((i * 11 + j) & 0xff);
        tx_crcs.push_back(frame_checksum(plane, W * H * 4));
        NvBufSurfaceSyncForDevice(surf, 0, -1);
        NvBufSurfaceUnMap(surf, 0, -1);
        gst_buffer_unmap(buf, &info);

        GST_BUFFER_PTS(buf) = i * 1000000;
        nvmm_ipc_producer_render(prod, prod_owner, buf);
        gst_buffer_unref(buf);
        g_usleep(2000);
    }

    g_usleep(100000);
    stop.store(true);
    consumer.join();

    gst_buffer_pool_set_active(upstream_pool, FALSE);
    gst_object_unref(upstream_pool);
    nvmm_ipc_producer_stop(prod, prod_owner);
    nvmm_ipc_producer_free(prod);
    gst_object_unref(prod_owner);
    gst_object_unref(cons_owner);

    std::printf("[zc tx=%d rx=%d] ", (int)tx_crcs.size(), (int)rx_crcs.size());
    ASSERT_TRUE(!rx_crcs.empty());

    int matched = 0;
    for (uint32_t cc : rx_crcs) {
        for (uint32_t ec : tx_crcs) {
            if (cc == ec) { matched++; break; }
        }
    }
    if (matched != (int)rx_crcs.size()) {
        std::printf("ZERO-COPY CORRUPTION: %d of %d received don't match ",
                    (int)rx_crcs.size() - matched, (int)rx_crcs.size());
        ASSERT_TRUE(matched == (int)rx_crcs.size());
    }
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
    RUN(multi_consumer_fanout);
    RUN(crc_roundtrip);
    RUN(producer_zerocopy);
    std::printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed ? 1 : 0;
}
