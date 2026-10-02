#include <gst/gst.h>
#include <gst/check/gstharness.h>

#include <cstring>

#include "nvmm_det_meta.h"
#include "nvmm_motion_meta.h"
#include "nvmm_track_meta.h"
#include "nvmm_frame.h"
#include "test_harness.h"

namespace {

struct GstInit { GstInit() { gst_init(nullptr, nullptr); } } _gst_init;

constexpr int kW = 64, kH = 64;
constexpr Rgba kClass0{255, 64, 64, 255};
constexpr Rgba kClass2{64, 160, 255, 255};
constexpr Rgba kTrackCyan{0, 255, 255, 255};

struct Box { int x, y, w, h, t; };

bool on_border(const Box &b, int x, int y)
{
    if (x < b.x || x >= b.x + b.w || y < b.y || y >= b.y + b.h) return false;
    return y < b.y + b.t || y >= b.y + b.h - b.t || x < b.x + b.t || x >= b.x + b.w - b.t;
}

NvmmDetObject det(float l, float t, float w, float h, int cls)
{
    NvmmDetObject o{};
    o.left = l; o.top = t; o.width = w; o.height = h;
    o.class_id = cls; o.confidence = 0.9f;
    return o;
}

void add_dets(GstBuffer *buf, guint32 infer_w, guint32 infer_h,
              std::initializer_list<NvmmDetObject> objs)
{
    static NvmmFrameMeta fm;
    std::memset(&fm, 0, sizeof fm);
    fm.infer_width = infer_w;
    fm.infer_height = infer_h;
    for (const auto &o : objs) fm.objects[fm.num_objects++] = o;
    gst_buffer_add_nvmm_det_meta(buf, &fm);
}

/// Pulls the one RGBA frame drawdet emits for `buf`.
struct Drawn {
    GstBuffer *out = nullptr;
    GstMapInfo map{};
    explicit Drawn(GstBuffer *buf, bool draw_track = false, bool draw_det = true)
    {
        GstElement *e = gst_element_factory_make("nvmmdrawdet", nullptr);
        if (!e) throw std::runtime_error("nvmmdrawdet not found");
        g_object_set(e, "thickness", 2, "draw-labels", FALSE, "draw-track", draw_track,
                     "draw-det", draw_det, NULL);
        GstHarness *h = gst_harness_new_with_element(e, "sink", "src");
        gst_harness_set_src_caps_str(h,
            "video/x-raw(memory:NVMM),format=NV12,width=64,height=64,framerate=30/1");
        if (gst_harness_push(h, buf) != GST_FLOW_OK) throw std::runtime_error("push failed");
        out = gst_harness_pull(h);
        gst_harness_teardown(h);
        gst_object_unref(e);
        if (!out || !gst_buffer_map(out, &map, GST_MAP_READ))
            throw std::runtime_error("no RGBA output");
    }
    ~Drawn() { gst_buffer_unmap(out, &map); gst_buffer_unref(out); }
    Rgba at(int x, int y) const
    {
        const guint8 *p = map.data + ((size_t)y * kW + x) * 4;
        return Rgba{p[0], p[1], p[2], p[3]};
    }
};

int pixels_off_expectation(const Drawn &d, const Box &b, const Rgba &color, int from_row = 0)
{
    const Rgba bg = d.at(kW - 1, kH - 1);
    int wrong = 0;
    for (int y = from_row; y < kH; y++)
        for (int x = 0; x < kW; x++)
            if (d.at(x, y) != (on_border(b, x, y) ? color : bg)) wrong++;
    return wrong;
}

TEST(det_box_border_is_class_colour_exactly_thickness_px_and_nothing_else_changes) {
    GstBuffer *buf = nvmm_nv12_buffer(kW, kH, solid(kBlack));
    add_dets(buf, kW, kH, {det(10, 8, 20, 12, 0)});
    Drawn d(buf);
    ASSERT_TRUE(d.at(kW - 1, kH - 1) != kClass0);
    ASSERT_EQ(pixels_off_expectation(d, Box{10, 8, 20, 12, 2}, kClass0), 0);
}

TEST(det_box_in_half_size_infer_space_is_drawn_at_double_frame_coordinates) {
    GstBuffer *buf = nvmm_nv12_buffer(kW, kH, solid(kBlack));
    add_dets(buf, kW / 2, kH / 2, {det(5, 4, 10, 6, 2)});
    Drawn d(buf);
    ASSERT_EQ(pixels_off_expectation(d, Box{10, 8, 20, 12, 2}, kClass2), 0);
}

TEST(moving_det_box_is_drawn_at_double_thickness) {
    GstBuffer *buf = nvmm_nv12_buffer(kW, kH, solid(kBlack));
    add_dets(buf, kW, kH, {det(10, 8, 20, 12, 0)});
    const nvmm::MotionEntry moving{3.f, 1u};
    gst_buffer_add_nvmm_motion_meta(buf, &moving, 1);
    Drawn d(buf);
    ASSERT_EQ(pixels_off_expectation(d, Box{10, 8, 20, 12, 4}, kClass0), 0);
}

TEST(draw_det_false_leaves_the_frame_free_of_det_boxes) {
    GstBuffer *buf = nvmm_nv12_buffer(kW, kH, solid(kBlack));
    add_dets(buf, kW, kH, {det(10, 8, 20, 12, 0)});
    Drawn d(buf, false, false);
    ASSERT_EQ(pixels_off_expectation(d, Box{0, 0, 0, 0, 0}, kClass0), 0);
}

/// Rows above 30 hold the HUD and the track label; the box itself starts at row 30.
TEST(valid_track_meta_is_drawn_as_a_cyan_box_at_track_coordinates) {
    GstBuffer *buf = nvmm_nv12_buffer(kW, kH, solid(kBlack));
    GstNvmmTrackMeta *tm = gst_buffer_add_nvmm_track_meta(buf);
    tm->valid = TRUE;
    tm->left = 10; tm->top = 30; tm->width = 20; tm->height = 12;
    Drawn d(buf, true, false);
    ASSERT_EQ(pixels_off_expectation(d, Box{10, 30, 20, 12, 2}, kTrackCyan, 30), 0);
}

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
