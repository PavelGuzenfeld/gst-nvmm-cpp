/// IPC backend throughput / latency micro-bench.
///
/// What it measures (mock NVMM, host build):
///   - producer publish rate vs single consumer
///   - producer publish rate vs N consumers (1, 2, 4) — cache-line layout payoff
///   - end-to-end latency: producer publish timestamp -> consumer fetch return
///   - copy vs zero-copy producer paths side by side
///
/// Notes:
///   - On host this exercises the atomic + memcpy paths only; NvBufSurfaceCopy
///     is the mock-stub plane-0 memcpy. So fps numbers are upper-bound for the
///     synchronization machinery, NOT a Jetson VIC throughput claim.
///   - Frame size kept small (64x64 RGBA) so memcpy is not the bottleneck —
///     we want to see the atomic / accept-thread / refcount overhead.
///   - Producer is open-loop: pushes as fast as it can. Latency is wall-clock
///     between PTS-set on the producer side and gst_buffer_unref on the
///     consumer side.

#include "ipc_backend.h"
#include "gstnvmmallocator.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
using Ns    = std::chrono::nanoseconds;

enum BenchMode { MODE_COPY, MODE_ZEROCOPY };

static GstBuffer *make_nvmm_buffer(GstAllocator *alloc, const GstVideoInfo *vi)
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

struct ConsumerStats {
    int    frames    = 0;
    double sum_us    = 0;
    double min_us    = 1e18;
    double max_us    = 0;
    std::vector<double> samples;  /* for p50/p99 */
};

static void run_case(BenchMode mode, int n_consumers, int total_frames,
                     int width, int height)
{
    char shm_name[64];
    std::snprintf(shm_name, sizeof(shm_name), "/bench_ipc_%s_n%d",
                  mode == MODE_ZEROCOPY ? "zc" : "cp", n_consumers);
    shm_unlink(shm_name);

    GstElement *prod_owner = GST_ELEMENT(g_object_new(GST_TYPE_BIN,
        "name", "bench_prod", NULL));
    GstAllocator *alloc = (mode == MODE_COPY) ? gst_nvmm_allocator_new(0) : nullptr;

    NvmmIpcProducer *prod = nvmm_ipc_producer_new(shm_name, 16);
    if (!nvmm_ipc_producer_start(prod, prod_owner)) {
        std::fprintf(stderr, "producer start failed\n");
        return;
    }
    GstVideoInfo vi;
    gst_video_info_set_format(&vi, GST_VIDEO_FORMAT_RGBA, width, height);
    if (!nvmm_ipc_producer_set_caps(prod, prod_owner, &vi, TRUE)) {
        std::fprintf(stderr, "set_caps failed\n");
        return;
    }

    /* For zero-copy mode, get the upstream pool the same way a real
     * upstream element would: fire propose_allocation with a fresh
     * GstQuery, take the pool back. */
    GstBufferPool *upstream_pool = nullptr;
    if (mode == MODE_ZEROCOPY) {
        GstCaps *caps = gst_video_info_to_caps(&vi);
        GstQuery *q = gst_query_new_allocation(caps, TRUE);
        if (!nvmm_ipc_producer_propose_allocation(prod, prod_owner, q) ||
            gst_query_get_n_allocation_pools(q) == 0) {
            std::fprintf(stderr, "propose_allocation didn't return a pool\n");
            return;
        }
        gst_query_parse_nth_allocation_pool(q, 0, &upstream_pool, nullptr, nullptr, nullptr);
        gst_object_ref_sink(upstream_pool);
        gst_query_unref(q);
        gst_caps_unref(caps);
        if (!gst_buffer_pool_set_active(upstream_pool, TRUE)) {
            std::fprintf(stderr, "upstream pool activate failed\n");
            return;
        }
    }

    std::vector<ConsumerStats> stats(n_consumers);
    std::atomic<bool> stop{false};
    std::atomic<int> ready_consumers{0};
    std::vector<std::thread> consumers;

    for (int i = 0; i < n_consumers; i++) {
        consumers.emplace_back([&, i]() {
            GstElement *co = GST_ELEMENT(g_object_new(GST_TYPE_BIN,
                "name", g_strdup_printf("bench_cons_%d", i), NULL));
            NvmmIpcConsumer *c = nvmm_ipc_consumer_new(shm_name);
            if (!nvmm_ipc_consumer_start(c, co)) {
                nvmm_ipc_consumer_free(c);
                gst_object_unref(co);
                return;
            }
            GstPad *pad = gst_pad_new("src", GST_PAD_SRC);
            ready_consumers.fetch_add(1);

            stats[i].samples.reserve(total_frames);

            while (!stop.load(std::memory_order_relaxed)) {
                GstBuffer *buf = nullptr;
                GstFlowReturn r = nvmm_ipc_consumer_fetch(c, co, pad, &buf);
                if (r == GST_FLOW_OK && buf) {
                    auto recv_ns = Clock::now().time_since_epoch().count();
                    auto pts_ns  = (int64_t)GST_BUFFER_PTS(buf);
                    double us    = double(recv_ns - pts_ns) / 1000.0;
                    if (us < 0) us = 0;          /* clock skew within thread */
                    stats[i].frames++;
                    stats[i].sum_us += us;
                    if (us < stats[i].min_us) stats[i].min_us = us;
                    if (us > stats[i].max_us) stats[i].max_us = us;
                    stats[i].samples.push_back(us);
                    gst_buffer_unref(buf);
                } else if (r == GST_FLOW_EOS) {
                    break;
                }
            }
            nvmm_ipc_consumer_stop(c, co);
            nvmm_ipc_consumer_free(c);
            gst_object_unref(pad);
            gst_object_unref(co);
        });
    }

    /* Wait for all consumers to handshake. */
    while (ready_consumers.load() < n_consumers) g_usleep(1000);
    g_usleep(20000);  /* small settle */

    auto t0 = Clock::now();
    for (int i = 0; i < total_frames; i++) {
        GstBuffer *frame = nullptr;
        if (mode == MODE_ZEROCOPY) {
            if (gst_buffer_pool_acquire_buffer(upstream_pool, &frame, nullptr)
                != GST_FLOW_OK || !frame) {
                std::fprintf(stderr, "pool acquire failed at frame %d\n", i);
                break;
            }
        } else {
            frame = make_nvmm_buffer(alloc, &vi);
            if (!frame) { std::fprintf(stderr, "frame alloc failed\n"); break; }
        }
        GST_BUFFER_PTS(frame) = (GstClockTime)Clock::now().time_since_epoch().count();
        nvmm_ipc_producer_render(prod, prod_owner, frame);
        gst_buffer_unref(frame);
        /* No sleep — open-loop. The consumer poll-rate (1ms) caps obs fps. */
    }
    auto t1 = Clock::now();
    double producer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /* Drain. */
    g_usleep(100000);
    stop.store(true);
    for (auto &t : consumers) t.join();

    if (upstream_pool) {
        gst_buffer_pool_set_active(upstream_pool, FALSE);
        gst_object_unref(upstream_pool);
    }
    nvmm_ipc_producer_stop(prod, prod_owner);
    nvmm_ipc_producer_free(prod);
    if (alloc) gst_object_unref(alloc);
    gst_object_unref(prod_owner);

    /* Aggregate. */
    int    cons_min_frames = INT32_MAX;
    int    cons_max_frames = 0;
    double cons_min_p50_us = 1e18;
    double cons_max_p99_us = 0;
    double cons_avg_us_sum = 0;
    int    cons_with_data  = 0;

    for (auto &s : stats) {
        if (s.frames < cons_min_frames) cons_min_frames = s.frames;
        if (s.frames > cons_max_frames) cons_max_frames = s.frames;
        if (s.frames == 0) continue;
        std::sort(s.samples.begin(), s.samples.end());
        double p50 = s.samples[s.samples.size() * 50 / 100];
        double p99 = s.samples[std::min<size_t>(s.samples.size() - 1,
                                                s.samples.size() * 99 / 100)];
        if (p50 < cons_min_p50_us) cons_min_p50_us = p50;
        if (p99 > cons_max_p99_us) cons_max_p99_us = p99;
        cons_avg_us_sum += s.sum_us / s.frames;
        cons_with_data++;
    }

    double prod_fps = (total_frames * 1000.0) / producer_ms;
    double avg_p50  = cons_with_data ? cons_min_p50_us : 0;
    double avg_lat  = cons_with_data ? cons_avg_us_sum / cons_with_data : 0;

    std::printf(
        "%-10s n_cons=%d  prod_fps=%8.0f  cons_frames=[min=%d,max=%d]/%d  "
        "lat_avg=%6.0f us  p50=%6.0f us  p99=%6.0f us\n",
        mode == MODE_ZEROCOPY ? "zero-copy" : "copy",
        n_consumers, prod_fps,
        cons_min_frames, cons_max_frames, total_frames,
        avg_lat, avg_p50, cons_max_p99_us);
}

int main(int argc, char *argv[])
{
    gst_init(&argc, &argv);
    nvmm_ipc_backend_init_debug();

    int total_frames = 5000;
    int width = 64, height = 64;
    if (argc >= 2) total_frames = std::atoi(argv[1]);

#ifdef NVMM_MOCK_API
    const char *backend_kind = "mock NVMM";
    const char *note =
        "Note: mock backend, host x86. Numbers reflect synchronization\n"
        "      overhead, not Jetson VIC throughput.";
#else
    const char *backend_kind = "real NVMM (NvBufSurfaceImport)";
    const char *note =
        "Note: real Jetson NVMM. Two producer paths benchmarked side by\n"
        "      side: 'copy' = upstream allocates separately, render() does\n"
        "      one NvBufSurfaceCopy GPU->GPU into the slot.\n"
        "      'zero-copy' = upstream acquires from our propose_allocation\n"
        "      pool, render() publishes the slot index without any copy.\n"
        "      Frame size kept small to expose the IPC sync path rather than\n"
        "      VIC bandwidth.";
#endif
    std::printf("=== IPC backend bench (%s, %dx%d RGBA, %d frames) ===\n",
                backend_kind, width, height, total_frames);
    std::printf("%s\n\n", note);

    for (int n : {1, 2, 4}) {
        run_case(MODE_COPY,     n, total_frames, width, height);
        run_case(MODE_ZEROCOPY, n, total_frames, width, height);
    }
    return 0;
}
