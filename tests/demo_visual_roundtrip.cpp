/// Visual roundtrip demo — proves frames physically traverse the IPC.
///
/// Producer thread renders a recognizable per-frame pattern (color
/// gradient + a frame-index bar across the top) into RGBA NVMM buffers.
/// Consumer thread fetches them, NvBufSurfaceMap's the result, dumps
/// each frame as a PPM (P6 binary) to /tmp/nvmm-visual/. Convert the
/// PPMs to PNGs on the host with PIL or ImageMagick to view.
///
/// Bypasses GStreamer's nvvidconv on consumer side because that path
/// requires a kernel-userspace match (R35.2.1 host kernel + R35.6.4
/// userspace doesn't satisfy it). The IPC layer itself is independent
/// of nvvidconv.

#include "ipc_backend.h"
#include "gstnvmmallocator.h"
#include "shm_protocol.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef NVMM_MOCK_API
#include "nvbufsurface_mock.h"
#else
#include <nvbufsurface.h>
#endif

static const char *kShm = "/visual_demo";
static const char *kOutDir = "/tmp/nvmm-visual";

constexpr int W = 320;
constexpr int H = 240;
constexpr int kFrames = 8;

/* Render a recognizable pattern: vertical color gradient + a black bar
 * across the top whose width encodes the frame index (1/N of the way
 * across for frame N). RGBA8 layout. */
static void render_pattern(uint8_t *rgba, int frame_idx, int total_frames)
{
    /* Maximum possible value of the blue numerator — scale to this so the
     * channel never exceeds 255 and the uint8_t cast never wraps. */
    const int b_max = (W - 1) + (H - 1) + (total_frames - 1) * 30;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint8_t r = (uint8_t)((x * 255) / (W - 1));
            uint8_t g = (uint8_t)((y * 255) / (H - 1));
            uint8_t b = (uint8_t)(((x + y + frame_idx * 30) * 255) / b_max);
            uint8_t *px = rgba + (y * W + x) * 4;
            px[0] = r; px[1] = g; px[2] = b; px[3] = 0xff;
        }
    }
    /* Top progress bar: black up to (idx+1)/N of width, red after. */
    int bar_h = 20;
    int progress_x = ((frame_idx + 1) * W) / total_frames;
    for (int y = 0; y < bar_h; y++) {
        for (int x = 0; x < W; x++) {
            uint8_t *px = rgba + (y * W + x) * 4;
            if (x < progress_x) { px[0] = 0;    px[1] = 0;    px[2] = 0;    }
            else                { px[0] = 0xff; px[1] = 0x40; px[2] = 0x40; }
            px[3] = 0xff;
        }
    }
    /* Big white digit in center spelling out the frame index 0..7
     * (5x7 font, scaled 8x). Skipping for brevity — gradient + bar
     * already make the frame index visually obvious. */
}

/* Write a P6 (binary) PPM. RGBA -> RGB drop alpha. */
static bool write_ppm(const char *path, const uint8_t *rgba, int w, int h)
{
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int i = 0; i < w * h; i++) {
        const uint8_t *p = rgba + i * 4;
        std::fputc(p[0], f);
        std::fputc(p[1], f);
        std::fputc(p[2], f);
    }
    fclose(f);
    return true;
}

int main(int argc, char *argv[])
{
    gst_init(&argc, &argv);
    nvmm_ipc_backend_init_debug();
    shm_unlink(kShm);
    mkdir(kOutDir, 0777);

    GstElement *prod_owner = GST_ELEMENT(g_object_new(GST_TYPE_BIN, "name", "vd_p", NULL));
    GstElement *cons_owner = GST_ELEMENT(g_object_new(GST_TYPE_BIN, "name", "vd_c", NULL));
    GstAllocator *alloc = gst_nvmm_allocator_new(0);

    NvmmIpcProducer *prod = nvmm_ipc_producer_new(kShm, 8);
    if (!nvmm_ipc_producer_start(prod, prod_owner)) {
        std::fprintf(stderr, "producer start failed\n"); return 1;
    }
    GstVideoInfo vi;
    gst_video_info_set_format(&vi, GST_VIDEO_FORMAT_RGBA, W, H);
    if (!nvmm_ipc_producer_set_caps(prod, prod_owner, &vi, TRUE)) {
        std::fprintf(stderr, "set_caps failed\n"); return 1;
    }

    std::atomic<bool> stop{false};
    std::atomic<int> rx_count{0};

    std::thread consumer([&]() {
        NvmmIpcConsumer *c = nvmm_ipc_consumer_new(kShm);
        if (!nvmm_ipc_consumer_start(c, cons_owner)) {
            nvmm_ipc_consumer_free(c); return;
        }
        GstPad *pad = gst_pad_new("src", GST_PAD_SRC);
        while (!stop.load()) {
            GstBuffer *buf = nullptr;
            GstFlowReturn r = nvmm_ipc_consumer_fetch(c, cons_owner, pad, &buf);
            if (r == GST_FLOW_OK && buf) {
                GstMapInfo info;
                if (gst_buffer_map(buf, &info, GST_MAP_READ)) {
                    NvBufSurface *surf = reinterpret_cast<NvBufSurface*>(info.data);
                    if (surf && NvBufSurfaceMap(surf, 0, -1, NVBUF_MAP_READ) == 0) {
                        NvBufSurfaceSyncForCpu(surf, 0, -1);
                        const uint8_t *rgba =
                            (const uint8_t *)surf->surfaceList[0].mappedAddr.addr[0];
                        if (rgba) {
                            int idx = rx_count.fetch_add(1);
                            char path[256];
                            std::snprintf(path, sizeof(path),
                                          "%s/rx_frame_%02d.ppm", kOutDir, idx);
                            write_ppm(path, rgba, W, H);
                            std::fprintf(stderr, "[consumer] wrote %s\n", path);
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
        nvmm_ipc_consumer_stop(c, cons_owner);
        nvmm_ipc_consumer_free(c);
        gst_object_unref(pad);
    });

    g_usleep(100000);  /* let consumer connect */

    for (int i = 0; i < kFrames; i++) {
        GstMemory *mem = gst_nvmm_allocator_alloc_video(
            alloc, GST_VIDEO_FORMAT_RGBA, W, H);
        if (!mem) { std::fprintf(stderr, "alloc failed\n"); break; }
        GstBuffer *frame = gst_buffer_new();
        gst_buffer_append_memory(frame, mem);

        guint8 *plane = nullptr;
        gsize psize = 0;
        if (gst_nvmm_memory_map_plane(mem, 0, GST_MAP_WRITE, &plane, &psize) &&
            psize >= (gsize)W * H * 4) {
            render_pattern(plane, i, kFrames);
            /* Also dump what the producer rendered, for side-by-side diff. */
            char path[256];
            std::snprintf(path, sizeof(path), "%s/tx_frame_%02d.ppm", kOutDir, i);
            write_ppm(path, plane, W, H);
            std::fprintf(stderr, "[producer] wrote %s, rendering...\n", path);
        }
        gst_nvmm_memory_unmap_plane(mem);

        GST_BUFFER_PTS(frame) = i * 100000000;  /* 100 ms apart */
        nvmm_ipc_producer_render(prod, prod_owner, frame);
        gst_buffer_unref(frame);
        g_usleep(100000);  /* slow it so consumer keeps up */
    }

    /* Drain. */
    g_usleep(500000);
    stop.store(true);
    consumer.join();

    nvmm_ipc_producer_stop(prod, prod_owner);
    nvmm_ipc_producer_free(prod);
    gst_object_unref(alloc);
    gst_object_unref(prod_owner);
    gst_object_unref(cons_owner);

    std::fprintf(stderr, "\nDone. %d frames produced, %d frames received & dumped.\n",
                 kFrames, rx_count.load());
    std::fprintf(stderr, "PPMs at %s/tx_frame_NN.ppm and rx_frame_NN.ppm\n", kOutDir);
    return 0;
}
