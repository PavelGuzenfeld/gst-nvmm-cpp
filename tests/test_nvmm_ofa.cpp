#include <gst/gst.h>
#include <gst/check/gstharness.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nvmm_optical_flow_meta.h"
#include "nvmm_frame.h"
#include "test_harness.h"

namespace {

struct GstInit { GstInit() { gst_init(nullptr, nullptr); } } _gst_init;

std::string printed;
void capture(const gchar *s) { printed += s; }

void fill(GstMemory *m, const std::vector<int16_t> &cells)
{
    GstMapInfo map;
    gst_memory_map(m, &map, GST_MAP_WRITE);
    std::memcpy(map.data, cells.data(), cells.size() * sizeof(int16_t));
    gst_memory_unmap(m, &map);
}

GstBuffer *with_flow(std::vector<int16_t> cells, gint mv_w, gint mv_h)
{
    GstBuffer *buf = gst_buffer_new_allocate(nullptr, 16, nullptr);
    const gsize bytes = cells.size() * sizeof(int16_t);
    GstMemory *mv = gst_allocator_alloc(nullptr, bytes, nullptr);
    fill(mv, cells);
    gst_buffer_add_nvmm_optical_flow_meta(buf, mv, mv_w, mv_h, 4, mv_w * 4, mv_h * 4);
    return buf;
}

std::string flowstats_output(bool silent, std::vector<GstBuffer *> frames)
{
    GstElement *e = gst_element_factory_make("nvmmflowstats", nullptr);
    if (!e) throw std::runtime_error("nvmmflowstats not found");
    g_object_set(e, "silent", silent, "sync", FALSE, NULL);
    GstHarness *h = gst_harness_new_with_element(e, "sink", nullptr);
    gst_harness_set_src_caps_str(h, "video/x-raw(memory:NVMM),format=NV12,width=8,height=8");
    printed.clear();
    GPrintFunc old = g_set_print_handler(capture);
    for (GstBuffer *b : frames) gst_harness_push(h, b);
    gst_harness_teardown(h);
    g_set_print_handler(old);
    gst_object_unref(e);
    return printed;
}

/// S10.5: (96, 128) is (3, 4) px, magnitude 5.
TEST(flowstats_reports_mean_and_max_vector_magnitude_in_pixels_and_skips_frames_without_flow) {
    const std::string out = flowstats_output(false, {
        with_flow({96, 128, 0, 0, 96, 0, 0, 128}, 2, 2),
        gst_buffer_new_allocate(nullptr, 16, nullptr),
        with_flow({0, 0, 0, 0, 0, 0, 0, 0}, 2, 2),
    });
    ASSERT_EQ(out,
        std::string("[nvmmflowstats] frame 1: 2x2 grid=4 mean=3.00 px max=5.00 px\n"
                    "[nvmmflowstats] frame 3: 2x2 grid=4 mean=0.00 px max=0.00 px\n"
                    "[nvmmflowstats] summary: 3 frames, 2 with flow, "
                    "avg mean magnitude 1.50 px\n"));
}

TEST(silent_flowstats_prints_only_the_summary) {
    const std::string out = flowstats_output(true, {
        with_flow({96, 128, 96, 128, 96, 128, 96, 128}, 2, 2),
    });
    ASSERT_EQ(out, std::string("[nvmmflowstats] summary: 1 frames, 1 with flow, "
                               "avg mean magnitude 5.00 px\n"));
}

uint8_t texture(int x, int y)
{
    uint32_t v = (uint32_t)(x / 4) * 2654435761u ^ (uint32_t)(y / 4) * 40503u;
    v ^= v >> 13;
    return (uint8_t)(32 + (v * 2246822519u >> 24) % 192);
}

Nv12Painter shifted_right(int dx)
{
    return [dx](int x, int y) { return Yuv{texture(x - dx + 64, y), 128, 128}; };
}

struct OfaRun {
    std::vector<GstBuffer *> out;
    OfaRun(int w, int h, int grid, std::vector<int> shifts)
    {
        GstElement *e = gst_element_factory_make("nvmmofa", nullptr);
        if (!e) throw std::runtime_error("nvmmofa not found");
        gst_util_set_object_arg(G_OBJECT(e), "grid-size", std::to_string(grid).c_str());
        GstHarness *hn = gst_harness_new_with_element(e, "sink", "src");
        const std::string caps = "video/x-raw(memory:NVMM),format=NV12,width=" +
            std::to_string(w) + ",height=" + std::to_string(h) + ",framerate=30/1";
        gst_harness_set_src_caps_str(hn, caps.c_str());
        for (int dx : shifts) {
            if (gst_harness_push(hn, block_linear_nv12_buffer(w, h, shifted_right(dx))) !=
                GST_FLOW_OK)
                throw std::runtime_error("push failed");
            out.push_back(gst_harness_pull(hn));
        }
        gst_harness_teardown(hn);
        gst_object_unref(e);
    }
    ~OfaRun() { for (GstBuffer *b : out) gst_buffer_unref(b); }
};

TEST(first_ofa_frame_has_no_predecessor_and_carries_no_flow_meta) {
    OfaRun r(128, 128, 4, {0});
    ASSERT_TRUE(gst_buffer_get_nvmm_optical_flow_meta(r.out[0]) == nullptr);
}

TEST(ofa_flow_field_is_ceil_of_frame_over_grid_cells) {
    OfaRun r(136, 128, 8, {0, 0});
    NvmmOpticalFlowMeta *m = gst_buffer_get_nvmm_optical_flow_meta(r.out[1]);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->mv_width, 17);
    ASSERT_EQ(m->mv_height, 16);
    ASSERT_EQ(m->grid_size, 8);
    ASSERT_EQ(m->frame_width, 136);
    ASSERT_EQ(m->frame_height, 128);
    ASSERT_EQ(gst_memory_get_sizes(m->mv, nullptr, nullptr), (gsize)17 * 16 * 4);
}

/// Interior cells only: the shifted-in border has no match. OFA refines to sub-pixel,
/// measured 129/32 px on this texture, so 1/8 px (4 S10.5 units) of slack.
TEST(ofa_reports_a_four_pixel_rightward_shift_as_the_median_interior_vector) {
    OfaRun r(128, 128, 4, {0, 4});
    NvmmOpticalFlowMeta *m = gst_buffer_get_nvmm_optical_flow_meta(r.out[1]);
    ASSERT_NOT_NULL(m);
    GstMapInfo map;
    ASSERT_TRUE(gst_memory_map(m->mv, &map, GST_MAP_READ));
    const auto *v = reinterpret_cast<const int16_t *>(map.data);
    std::vector<int16_t> dx, dy;
    for (int y = 4; y < m->mv_height - 4; y++)
        for (int x = 4; x < m->mv_width - 4; x++) {
            dx.push_back(v[2 * (y * m->mv_width + x)]);
            dy.push_back(v[2 * (y * m->mv_width + x) + 1]);
        }
    gst_memory_unmap(m->mv, &map);
    std::nth_element(dx.begin(), dx.begin() + dx.size() / 2, dx.end());
    std::nth_element(dy.begin(), dy.begin() + dy.size() / 2, dy.end());
    ASSERT_TRUE(std::abs(dx[dx.size() / 2] - 4 * 32) <= 4);
    ASSERT_TRUE(std::abs(dy[dy.size() / 2]) <= 4);
}

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
