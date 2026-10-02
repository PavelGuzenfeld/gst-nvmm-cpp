#include <gst/gst.h>
#include <gst/check/gstharness.h>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <glib.h>

#include "../gst/nvmmdetgate/detgate.hpp"
#include "nvmm_det_meta.h"
#include "nvmm_frame.h"
#include "test_harness.h"

namespace {

constexpr int kSkipExitCode = 77;

/// The engines come from tools/build_test_engines.sh; without them there is nothing to drive.
struct EngineDirOrSkip {
    std::string dir;
    EngineDirOrSkip()
    {
        const char *d = std::getenv("NVMM_TEST_ENGINE_DIR");
        dir = d ? d : "";
        for (const char *f : {"xfeat.engine", "lightglue.engine"}) {
            if (!dir.empty() && g_file_test((dir + "/" + f).c_str(), G_FILE_TEST_IS_REGULAR))
                continue;
            std::printf("SKIP: NVMM_TEST_ENGINE_DIR has no %s\n", f);
            std::exit(kSkipExitCode);
        }
        gst_init(nullptr, nullptr);
    }
} engines;

/// 480x270 is XfeatMatcher's registration space, so residuals are in surface px.
constexpr int kW = 480, kH = 270;
constexpr int kMoverSide = 48;
constexpr int kMoverX0 = 200, kMoverY = 150;
/// 20 px over the default dlt of 5 frames: above rmin 12, below the dist 45 association gate.
constexpr int kMoverStep = 4;
constexpr float kStaticLeft = 60.f, kStaticTop = 100.f;
constexpr int kDlt = 5, kAmin = 6;
constexpr int kFirstConfirmedFrame = 2 * kDlt + kAmin - 1;
constexpr int kFrames = kFirstConfirmedFrame + 8;
/// 24 px over dlt 2: above rmin 12 and the motion-blob residual of 16, below the dist 45 gate.
constexpr int kFastDlt = 2, kFastStep = 12;
constexpr int kFastFirstConfirmedFrame = 2 * kFastDlt + kAmin - 1;
constexpr int kFastFrames = 11;
/// XFeat keypoints sit on a stride-8 grid, so a blob edge can fall one cell off the mover.
constexpr float kKeypointStridePx = 8.f;

uint32_t hash2(uint32_t x, uint32_t y, uint32_t salt)
{
    uint32_t h = x * 374761393u + y * 668265263u + salt * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}

Nv12Painter moving_scene(int mover_x, int mover_y)
{
    return [mover_x, mover_y](int x, int y) -> Yuv {
        if (x >= mover_x && x < mover_x + kMoverSide && y >= mover_y && y < mover_y + kMoverSide) {
            const bool on = hash2((uint32_t)(x - mover_x) / 4, (uint32_t)(y - mover_y) / 4, 7) & 1u;
            return on ? Yuv{235, 118, 138} : Yuv{16, 138, 118};
        }
        const uint32_t cell = hash2((uint32_t)x / 6, (uint32_t)y / 6, 3);
        return Yuv{(uint8_t)(70 + cell % 110), (uint8_t)(112 + (cell >> 8) % 32),
                   (uint8_t)(112 + (cell >> 16) % 32)};
    };
}

Nv12Painter scene(int frame)
{
    return moving_scene(kMoverX0 + kMoverStep * frame, kMoverY);
}

NvmmDetObject det_box(float left, float top, float width, float height, float conf,
                      const char *label, int class_id = 0)
{
    NvmmDetObject o{};
    o.left = left; o.top = top; o.width = width; o.height = height;
    o.class_id = class_id;
    o.confidence = conf;
    g_strlcpy(o.label, label, NVMM_META_LABEL_LEN);
    return o;
}

NvmmDetObject det(float left, float top, float conf, const char *label)
{
    return det_box(left, top, kMoverSide, kMoverSide, conf, label);
}

NvmmDetObject mover_det(int frame)
{
    return det((float)(kMoverX0 + kMoverStep * frame), (float)kMoverY, 0.6f, "mover");
}

NvmmDetObject fast_mover_det(int frame, float conf = 0.6f, int class_id = 0)
{
    return det_box((float)(kMoverX0 + kFastStep * frame), (float)kMoverY, kMoverSide, kMoverSide,
                   conf, "mover", class_id);
}

GstBuffer *with_dets(GstBuffer *buf, const std::vector<NvmmDetObject> &dets,
                     guint32 infer_w = kW, guint32 infer_h = kH)
{
    static NvmmFrameMeta fm;
    std::memset(&fm, 0, sizeof fm);
    fm.infer_width = infer_w;
    fm.infer_height = infer_h;
    for (const NvmmDetObject &d : dets) fm.objects[fm.num_objects++] = d;
    gst_buffer_add_nvmm_det_meta(buf, &fm);
    return buf;
}

/// The static det goes first and scores higher, so neither "first wins" nor
/// "most confident wins" can pass for the motion gate.
GstBuffer *frame_with_two_dets(int frame)
{
    return with_dets(nvmm_nv12_buffer(kW, kH, scene(frame)),
                     {det(kStaticLeft, kStaticTop, 0.9f, "static"), mover_det(frame)});
}

GstBuffer *fast_frame(int frame, const std::vector<NvmmDetObject> &dets,
                      guint32 infer_w = kW, guint32 infer_h = kH)
{
    return with_dets(nvmm_nv12_buffer(kW, kH, moving_scene(kMoverX0 + kFastStep * frame, kMoverY)),
                     dets, infer_w, infer_h);
}

using Configure = std::function<void(GstElement *)>;

void fast_gate(GstElement *gate)
{
    g_object_set(gate, "dlt", kFastDlt, NULL);
}

class GateRig {
public:
    explicit GateRig(const Configure &configure)
    {
        gate_ = gst_element_factory_make("nvmmdetgate", nullptr);
        ASSERT_NOT_NULL(gate_);
        g_object_set(gate_, "engine-dir", engines.dir.c_str(), NULL);
        configure(gate_);
        h_ = gst_harness_new_with_element(gate_, "sink", "src");
        gst_harness_set_src_caps_str(h_, "video/x-raw(memory:NVMM),format=NV12,width=480,"
                                         "height=270,framerate=30/1");
    }
    GateRig(const GateRig &) = delete;
    GateRig &operator=(const GateRig &) = delete;
    ~GateRig()
    {
        gst_harness_teardown(h_);
        gst_object_unref(gate_);
    }

    std::vector<NvmmDetObject> push(GstBuffer *buf)
    {
        ASSERT_EQ(gst_harness_push(h_, buf), GST_FLOW_OK);
        GstBuffer *out = gst_harness_pull(h_);
        const GstNvmmDetMeta *m = gst_buffer_get_nvmm_det_meta(out);
        ASSERT_NOT_NULL(m);
        std::vector<NvmmDetObject> kept(m->objects, m->objects + m->num_objects);
        gst_buffer_unref(out);
        return kept;
    }

    GstHarness *harness() const { return h_; }

private:
    GstElement *gate_ = nullptr;
    GstHarness *h_ = nullptr;
};

using Kept = std::vector<std::vector<NvmmDetObject>>;
using DetsOf = std::function<std::vector<NvmmDetObject>(int)>;

Kept run_fast(const DetsOf &dets_of, const Configure &configure,
              guint32 infer_w = kW, guint32 infer_h = kH, int frames = kFastFrames)
{
    GateRig rig(configure);
    Kept kept;
    for (int f = 0; f < frames; f++) kept.push_back(rig.push(fast_frame(f, dets_of(f), infer_w, infer_h)));
    return kept;
}

int first_kept_frame(const Kept &kept)
{
    for (size_t f = 0; f < kept.size(); f++)
        if (!kept[f].empty()) return (int)f;
    return -1;
}

std::vector<NvmmDetObject> just_the_mover(int frame)
{
    return {fast_mover_det(frame)};
}

using nvmm::DetGate;
using nvmm::GateCfg;
using nvmm::GateDet;
using nvmm::MotionSample;

constexpr int kUnitFrame = 400;
constexpr float kMovedResidual = 20.f;
constexpr int kSynthConfirm = -2;

GateCfg unit_cfg(int amin, int ksup)
{
    GateCfg c;
    c.amin = amin;
    c.ksup = ksup;
    c.borderfrac = 0.f;
    return c;
}

GateCfg blob_cfg(int silent)
{
    GateCfg c = unit_cfg(1, 1);
    c.seed_on_motion = true;
    c.motion_silent = silent;
    return c;
}

GateDet unit_det(float cx, float cy, float conf, int src)
{
    return GateDet{cx, cy, 20.f, 20.f, conf, src};
}

std::vector<MotionSample> moving_at(float x, float y, float resid = kMovedResidual)
{
    return {MotionSample{x, y, resid}};
}

int feed(DetGate &g, const std::vector<MotionSample> &m, const std::vector<GateDet> &d,
         int w = kUnitFrame, int h = kUnitFrame)
{
    return g.update(m, d, w, h);
}

int feed_det(DetGate &g, float cx, float cy, int src, float resid = kMovedResidual)
{
    return feed(g, moving_at(cx, cy, resid), {unit_det(cx, cy, 0.9f, src)});
}

void lock_on(DetGate &g, float cx, float cy, int src)
{
    ASSERT_EQ(feed_det(g, cx, cy, src), src);
    ASSERT_TRUE(g.locked());
}

bool confirms_at(float cx, float cy, float borderfrac, int w, int h)
{
    GateCfg c = unit_cfg(1, 1);
    c.borderfrac = borderfrac;
    DetGate g(c);
    return g.update(moving_at(cx, cy), {unit_det(cx, cy, 0.9f, 3)}, w, h) == 3;
}

std::vector<MotionSample> cluster(std::initializer_list<std::pair<float, float>> pts,
                                  float resid = kMovedResidual)
{
    std::vector<MotionSample> m;
    for (const auto &p : pts) m.push_back(MotionSample{p.first, p.second, resid});
    return m;
}

TEST(gate_config_defaults_are_the_documented_tuning) {
    const GateCfg c;
    ASSERT_EQ(c.rmin, 12.f);
    ASSERT_EQ(c.dist, 45.f);
    ASSERT_EQ(c.amin, 6);
    ASSERT_EQ(c.ksup, 4);
    ASSERT_EQ(c.maxlost, 2);
    ASSERT_EQ(c.borderfrac, 0.02f);
    ASSERT_EQ(c.sample_rad, 32.f);
    ASSERT_EQ(c.seed_on_motion, false);
    ASSERT_EQ(c.motion_silent, 12);
    ASSERT_EQ(c.motion_rmin, 16.f);
    ASSERT_EQ(c.motion_minpts, 4);
    ASSERT_EQ(c.motion_cell, 48.f);
    ASSERT_EQ(c.confsky, 0.55f);
}

TEST(synth_confirm_code_is_the_minus_two_the_element_switches_on) {
    ASSERT_EQ(DetGate::kSynthConfirm, kSynthConfirm);
}

TEST(a_detection_confirms_on_the_ksup_th_supported_frame) {
    DetGate g(unit_cfg(1, 4));
    for (int f = 1; f < 4; f++) ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), -1);
    ASSERT_TRUE(!g.locked());
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), 7);
    ASSERT_TRUE(g.locked());
}

TEST(a_detection_confirms_on_the_amin_th_frame_even_when_support_came_first) {
    DetGate g(unit_cfg(5, 1));
    for (int f = 1; f < 5; f++) ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), -1);
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), 7);
}

TEST(residual_equal_to_rmin_counts_as_moved) {
    DetGate g(unit_cfg(1, 1));
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7, 12.f), 7);
}

TEST(residual_just_below_rmin_never_confirms) {
    DetGate g(unit_cfg(1, 1));
    for (int f = 0; f < 6; f++) ASSERT_EQ(feed_det(g, 100.f, 100.f, 7, 11.75f), -1);
}

TEST(a_motion_sample_exactly_sample_rad_along_x_supports_the_detection) {
    DetGate g(unit_cfg(1, 1));
    ASSERT_EQ(feed(g, moving_at(132.f, 100.f), {unit_det(100.f, 100.f, 0.9f, 7)}), 7);
}

TEST(a_motion_sample_exactly_sample_rad_along_y_supports_the_detection) {
    DetGate g(unit_cfg(1, 1));
    ASSERT_EQ(feed(g, moving_at(100.f, 68.f), {unit_det(100.f, 100.f, 0.9f, 7)}), 7);
}

TEST(a_motion_sample_a_pixel_beyond_sample_rad_does_not_support_the_detection) {
    DetGate g(unit_cfg(1, 1));
    ASSERT_EQ(feed(g, moving_at(133.f, 100.f), {unit_det(100.f, 100.f, 0.9f, 7)}), -1);
}

TEST(sample_rad_is_a_euclidean_disc_not_a_square) {
    DetGate corner(unit_cfg(1, 1));
    ASSERT_EQ(feed(corner, moving_at(124.f, 124.f), {unit_det(100.f, 100.f, 0.9f, 7)}), -1);
    DetGate inside(unit_cfg(1, 1));
    ASSERT_EQ(feed(inside, moving_at(119.f, 119.f), {unit_det(100.f, 100.f, 0.9f, 7)}), 7);
}

TEST(the_strongest_residual_in_range_decides_whatever_the_sample_order) {
    std::vector<MotionSample> strong_first = {{100.f, 100.f, 20.f}, {101.f, 100.f, 5.f}};
    DetGate g(unit_cfg(1, 1));
    ASSERT_EQ(feed(g, strong_first, {unit_det(100.f, 100.f, 0.9f, 7)}), 7);
    std::vector<MotionSample> weak_then_strong = {{100.f, 100.f, 5.f}, {101.f, 100.f, 20.f}};
    DetGate h(unit_cfg(1, 1));
    ASSERT_EQ(feed(h, weak_then_strong, {unit_det(100.f, 100.f, 0.9f, 7)}), 7);
}

TEST(an_unsupported_frame_restarts_the_support_count) {
    DetGate g(unit_cfg(1, 3));
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), -1);
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), -1);
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7, 0.f), -1);
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), -1);
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), -1);
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), 7);
}

TEST(a_detection_44_px_from_its_track_continues_it) {
    DetGate g(unit_cfg(3, 1));
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), -1);
    ASSERT_EQ(feed_det(g, 144.f, 100.f, 7), -1);
    ASSERT_EQ(feed_det(g, 188.f, 100.f, 7), 7);
}

TEST(a_detection_exactly_dist_from_its_track_starts_a_new_one) {
    DetGate g(unit_cfg(2, 1));
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), -1);
    ASSERT_EQ(feed_det(g, 145.f, 100.f, 7), -1);
    ASSERT_EQ(feed_det(g, 190.f, 100.f, 7), -1);
}

TEST(association_distance_is_euclidean_over_both_axes) {
    DetGate inside(unit_cfg(2, 1));
    ASSERT_EQ(feed_det(inside, 100.f, 100.f, 7), -1);
    ASSERT_EQ(feed_det(inside, 130.f, 130.f, 7), 7);
    DetGate outside(unit_cfg(2, 1));
    ASSERT_EQ(feed_det(outside, 100.f, 100.f, 7), -1);
    ASSERT_EQ(feed_det(outside, 133.f, 133.f, 7), -1);
}

TEST(two_detections_never_share_one_track) {
    DetGate g(unit_cfg(2, 1));
    ASSERT_EQ(feed(g, cluster({{100.f, 100.f}, {110.f, 100.f}}),
                   {unit_det(100.f, 100.f, 0.9f, 1), unit_det(110.f, 100.f, 0.8f, 2)}), -1);
    ASSERT_EQ(feed(g, cluster({{100.f, 100.f}, {110.f, 100.f}}),
                   {unit_det(100.f, 100.f, 0.9f, 1), unit_det(110.f, 100.f, 0.8f, 2)}), 1);
}

TEST(a_track_survives_maxlost_missing_frames) {
    DetGate g(unit_cfg(2, 1));
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), -1);
    ASSERT_EQ(feed(g, {}, {}), -1);
    ASSERT_EQ(feed(g, {}, {}), -1);
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), 7);
}

TEST(a_track_is_reaped_after_maxlost_plus_one_missing_frames) {
    DetGate g(unit_cfg(2, 1));
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), -1);
    for (int f = 0; f < 3; f++) ASSERT_EQ(feed(g, {}, {}), -1);
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 7), -1);
}

TEST(more_support_beats_more_confidence) {
    DetGate g(unit_cfg(3, 2));
    const std::vector<MotionSample> both = cluster({{100.f, 100.f}, {300.f, 300.f}});
    const std::vector<GateDet> dets = {unit_det(100.f, 100.f, 0.5f, 3),
                                       unit_det(300.f, 300.f, 0.9f, 5)};
    ASSERT_EQ(feed(g, moving_at(100.f, 100.f), dets), -1);
    ASSERT_EQ(feed(g, both, dets), -1);
    ASSERT_EQ(feed(g, both, dets), 3);
}

TEST(equal_support_is_decided_by_the_higher_confidence_whatever_the_list_order) {
    const std::vector<MotionSample> both = cluster({{100.f, 100.f}, {300.f, 300.f}});
    DetGate high_second(unit_cfg(1, 1));
    ASSERT_EQ(feed(high_second, both, {unit_det(100.f, 100.f, 0.5f, 3), unit_det(300.f, 300.f, 0.9f, 5)}), 5);
    DetGate high_first(unit_cfg(1, 1));
    ASSERT_EQ(feed(high_first, both, {unit_det(100.f, 100.f, 0.9f, 3), unit_det(300.f, 300.f, 0.5f, 5)}), 3);
}

TEST(a_track_keeps_the_best_confidence_it_ever_had) {
    DetGate g(unit_cfg(2, 2));
    const std::vector<MotionSample> both = cluster({{100.f, 100.f}, {300.f, 300.f}});
    ASSERT_EQ(feed(g, both, {unit_det(100.f, 100.f, 0.9f, 3), unit_det(300.f, 300.f, 0.6f, 5)}), -1);
    ASSERT_EQ(feed(g, both, {unit_det(100.f, 100.f, 0.3f, 3), unit_det(300.f, 300.f, 0.6f, 5)}), 3);
}

TEST(a_detection_centred_on_the_left_border_line_is_active_and_just_outside_is_not) {
    ASSERT_TRUE(confirms_at(50.f, 50.f, 0.25f, 200, 100));
    ASSERT_TRUE(!confirms_at(49.5f, 50.f, 0.25f, 200, 100));
}

TEST(a_detection_centred_on_the_right_border_line_is_active_and_just_outside_is_not) {
    ASSERT_TRUE(confirms_at(150.f, 50.f, 0.25f, 200, 100));
    ASSERT_TRUE(!confirms_at(150.5f, 50.f, 0.25f, 200, 100));
}

TEST(a_detection_centred_on_the_top_border_line_is_active_and_just_outside_is_not) {
    ASSERT_TRUE(confirms_at(100.f, 25.f, 0.25f, 200, 100));
    ASSERT_TRUE(!confirms_at(100.f, 24.5f, 0.25f, 200, 100));
}

TEST(a_detection_centred_on_the_bottom_border_line_is_active_and_just_outside_is_not) {
    ASSERT_TRUE(confirms_at(100.f, 75.f, 0.25f, 200, 100));
    ASSERT_TRUE(!confirms_at(100.f, 75.5f, 0.25f, 200, 100));
}

TEST(border_fraction_zero_accepts_a_detection_in_the_corner) {
    ASSERT_TRUE(!confirms_at(1.f, 1.f, 0.25f, 200, 100));
    ASSERT_TRUE(confirms_at(1.f, 1.f, 0.f, 200, 100));
}

TEST(an_unknown_frame_width_disables_the_border_check) {
    ASSERT_TRUE(confirms_at(1.f, 50.f, 0.25f, 0, 100));
}

TEST(an_unknown_frame_height_disables_the_border_check) {
    ASSERT_TRUE(confirms_at(100.f, 1.f, 0.25f, 200, 0));
}

TEST(a_locked_gate_returns_the_detection_nearest_the_lock_whatever_the_list_order) {
    DetGate far_first(unit_cfg(1, 1));
    lock_on(far_first, 100.f, 100.f, 1);
    ASSERT_EQ(feed(far_first, {}, {unit_det(130.f, 100.f, 0.9f, 4), unit_det(110.f, 100.f, 0.2f, 5)}), 5);
    DetGate near_first(unit_cfg(1, 1));
    lock_on(near_first, 100.f, 100.f, 1);
    ASSERT_EQ(feed(near_first, {}, {unit_det(110.f, 100.f, 0.2f, 5), unit_det(130.f, 100.f, 0.9f, 4)}), 5);
}

TEST(a_locked_gate_ignores_motion_evidence) {
    DetGate g(unit_cfg(1, 1));
    lock_on(g, 100.f, 100.f, 1);
    ASSERT_EQ(feed(g, {}, {unit_det(105.f, 100.f, 0.9f, 4)}), 4);
}

TEST(a_lock_follows_its_target_beyond_dist_from_the_first_position) {
    DetGate g(unit_cfg(1, 1));
    lock_on(g, 100.f, 100.f, 1);
    for (int f = 1; f <= 5; f++)
        ASSERT_EQ(feed(g, {}, {unit_det(100.f + 40.f * f, 100.f, 0.9f, 2)}), 2);
}

TEST(a_locked_gate_takes_a_detection_44_px_from_the_lock) {
    DetGate g(unit_cfg(1, 1));
    lock_on(g, 100.f, 100.f, 1);
    ASSERT_EQ(feed(g, {}, {unit_det(144.f, 100.f, 0.9f, 4)}), 4);
}

TEST(a_locked_gate_drops_a_detection_exactly_dist_from_the_lock) {
    DetGate g(unit_cfg(1, 1));
    lock_on(g, 100.f, 100.f, 1);
    ASSERT_EQ(feed(g, {}, {unit_det(145.f, 100.f, 0.9f, 4)}), -1);
}

TEST(a_lock_survives_maxlost_missing_frames) {
    DetGate g(unit_cfg(1, 1));
    lock_on(g, 100.f, 100.f, 1);
    ASSERT_EQ(feed(g, {}, {}), -1);
    ASSERT_EQ(feed(g, {}, {}), -1);
    ASSERT_TRUE(g.locked());
    ASSERT_EQ(feed(g, {}, {unit_det(100.f, 100.f, 0.9f, 4)}), 4);
}

TEST(a_lock_is_released_after_maxlost_plus_one_missing_frames) {
    DetGate g(unit_cfg(1, 1));
    lock_on(g, 100.f, 100.f, 1);
    for (int f = 0; f < 3; f++) ASSERT_EQ(feed(g, {}, {}), -1);
    ASSERT_TRUE(!g.locked());
}

TEST(a_released_lock_forgets_its_track_and_must_earn_amin_again) {
    DetGate g(unit_cfg(3, 1));
    for (int f = 0; f < 2; f++) ASSERT_EQ(feed_det(g, 100.f, 100.f, 1), -1);
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 1), 1);
    for (int f = 0; f < 3; f++) ASSERT_EQ(feed(g, {}, {}), -1);
    ASSERT_TRUE(!g.locked());
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 1), -1);
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 1), -1);
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 1), 1);
}

TEST(a_matched_frame_resets_the_lock_miss_count) {
    DetGate g(unit_cfg(1, 1));
    lock_on(g, 100.f, 100.f, 1);
    for (int round = 0; round < 3; round++) {
        ASSERT_EQ(feed(g, {}, {}), -1);
        ASSERT_EQ(feed(g, {}, {}), -1);
        ASSERT_EQ(feed(g, {}, {unit_det(100.f, 100.f, 0.9f, 4)}), 4);
    }
    ASSERT_TRUE(g.locked());
}

TEST(a_lock_that_reached_the_border_is_released_even_with_its_detection_present) {
    GateCfg c = unit_cfg(1, 1);
    c.borderfrac = 0.25f;
    DetGate g(c);
    ASSERT_EQ(g.update(moving_at(100.f, 50.f), {unit_det(100.f, 50.f, 0.9f, 3)}, 200, 100), 3);
    ASSERT_EQ(g.update({}, {unit_det(70.f, 50.f, 0.9f, 3)}, 200, 100), 3);
    ASSERT_EQ(g.update({}, {unit_det(49.f, 50.f, 0.9f, 3)}, 200, 100), 3);
    ASSERT_EQ(g.update({}, {unit_det(49.f, 50.f, 0.9f, 3)}, 200, 100), -1);
    ASSERT_TRUE(!g.locked());
}

TEST(reset_unlatches_the_gate_and_drops_its_tracks) {
    DetGate g(unit_cfg(2, 1));
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 1), -1);
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 1), 1);
    g.reset();
    ASSERT_TRUE(!g.locked());
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 1), -1);
    ASSERT_EQ(feed_det(g, 100.f, 100.f, 1), 1);
}

TEST(a_motion_blob_confirms_with_its_exact_bounding_box_once) {
    DetGate g(blob_cfg(1));
    const auto m = cluster({{100.f, 40.f}, {110.f, 40.f}, {100.f, 50.f}, {120.f, 60.f}});
    ASSERT_EQ(feed(g, m, {}), kSynthConfirm);
    ASSERT_TRUE(g.locked());
    float cx = 0, cy = 0, w = 0, h = 0;
    ASSERT_TRUE(g.synth_seed(cx, cy, w, h));
    ASSERT_EQ(cx, 110.f);
    ASSERT_EQ(cy, 50.f);
    ASSERT_EQ(w, 20.f);
    ASSERT_EQ(h, 20.f);
    ASSERT_TRUE(!g.synth_seed(cx, cy, w, h));
}

TEST(three_moving_points_do_not_make_a_motion_blob) {
    DetGate g(blob_cfg(1));
    ASSERT_EQ(feed(g, cluster({{100.f, 40.f}, {110.f, 40.f}, {100.f, 50.f}}), {}), -1);
}

TEST(a_point_just_below_motion_rmin_is_not_part_of_a_blob) {
    DetGate g(blob_cfg(1));
    std::vector<MotionSample> m = cluster({{100.f, 40.f}, {110.f, 40.f}, {100.f, 50.f}});
    m.push_back(MotionSample{120.f, 60.f, 15.75f});
    ASSERT_EQ(feed(g, m, {}), -1);
}

TEST(a_point_exactly_at_motion_rmin_is_part_of_a_blob) {
    DetGate g(blob_cfg(1));
    std::vector<MotionSample> m = cluster({{100.f, 40.f}, {110.f, 40.f}, {100.f, 50.f}});
    m.push_back(MotionSample{120.f, 60.f, 16.f});
    ASSERT_EQ(feed(g, m, {}), kSynthConfirm);
}

TEST(points_exactly_motion_cell_apart_along_x_join_one_blob) {
    DetGate g(blob_cfg(1));
    const auto m = cluster({{60.f, 50.f}, {60.f, 60.f}, {100.f, 50.f}, {148.f, 50.f}});
    ASSERT_EQ(feed(g, m, {}), kSynthConfirm);
    float cx, cy, w, h;
    ASSERT_TRUE(g.synth_seed(cx, cy, w, h));
    ASSERT_EQ(cx, 104.f);
    ASSERT_EQ(cy, 55.f);
    ASSERT_EQ(w, 88.f);
    ASSERT_EQ(h, 10.f);
}

TEST(points_a_pixel_beyond_motion_cell_apart_along_x_split_into_two_clusters) {
    DetGate g(blob_cfg(1));
    ASSERT_EQ(feed(g, cluster({{60.f, 50.f}, {60.f, 60.f}, {100.f, 50.f}, {149.f, 50.f}}), {}), -1);
}

TEST(points_exactly_motion_cell_apart_along_y_join_one_blob) {
    DetGate g(blob_cfg(1));
    const auto m = cluster({{50.f, 60.f}, {60.f, 60.f}, {50.f, 100.f}, {50.f, 148.f}});
    ASSERT_EQ(feed(g, m, {}), kSynthConfirm);
    float cx, cy, w, h;
    ASSERT_TRUE(g.synth_seed(cx, cy, w, h));
    ASSERT_EQ(cx, 55.f);
    ASSERT_EQ(cy, 104.f);
    ASSERT_EQ(w, 10.f);
    ASSERT_EQ(h, 88.f);
}

TEST(points_a_pixel_beyond_motion_cell_apart_along_y_split_into_two_clusters) {
    DetGate g(blob_cfg(1));
    ASSERT_EQ(feed(g, cluster({{50.f, 60.f}, {60.f, 60.f}, {50.f, 100.f}, {50.f, 149.f}}), {}), -1);
}

TEST(points_close_in_one_axis_and_far_in_the_other_are_not_neighbours) {
    DetGate g(blob_cfg(1));
    ASSERT_EQ(feed(g, cluster({{100.f, 40.f}, {110.f, 40.f}, {100.f, 50.f}, {105.f, 250.f}}), {}), -1);
}

TEST(diagonal_neighbours_within_motion_cell_on_both_axes_join_a_blob) {
    DetGate g(blob_cfg(1));
    ASSERT_EQ(feed(g, cluster({{100.f, 40.f}, {110.f, 40.f}, {100.f, 50.f}, {140.f, 90.f}}), {}),
               kSynthConfirm);
}

TEST(the_largest_cluster_becomes_the_blob) {
    DetGate g(blob_cfg(1));
    const auto m = cluster({{50.f, 50.f}, {60.f, 50.f}, {50.f, 60.f}, {60.f, 60.f},
                            {300.f, 300.f}, {310.f, 300.f}, {300.f, 310.f}, {310.f, 310.f},
                            {320.f, 320.f}});
    ASSERT_EQ(feed(g, m, {}), kSynthConfirm);
    float cx, cy, w, h;
    ASSERT_TRUE(g.synth_seed(cx, cy, w, h));
    ASSERT_EQ(cx, 310.f);
    ASSERT_EQ(cy, 310.f);
    ASSERT_EQ(w, 20.f);
    ASSERT_EQ(h, 20.f);
}

TEST(a_motion_blob_waits_for_motion_silent_frames) {
    DetGate g(blob_cfg(3));
    const auto m = cluster({{100.f, 40.f}, {110.f, 40.f}, {100.f, 50.f}, {120.f, 60.f}});
    ASSERT_EQ(feed(g, m, {}), -1);
    ASSERT_EQ(feed(g, m, {}), -1);
    ASSERT_EQ(feed(g, m, {}), kSynthConfirm);
}

TEST(no_motion_blob_is_seeded_unless_seed_on_motion_is_on) {
    GateCfg c = blob_cfg(1);
    c.seed_on_motion = false;
    DetGate g(c);
    const auto m = cluster({{100.f, 40.f}, {110.f, 40.f}, {100.f, 50.f}, {120.f, 60.f}});
    for (int f = 0; f < 4; f++) ASSERT_EQ(feed(g, m, {}), -1);
}

TEST(a_motion_blob_competes_with_a_yolo_detection_at_confsky) {
    const auto blob = cluster({{100.f, 40.f}, {110.f, 40.f}, {100.f, 50.f}, {120.f, 60.f}});
    std::vector<MotionSample> both = blob;
    both.push_back(MotionSample{300.f, 300.f, kMovedResidual});
    DetGate weaker_yolo(blob_cfg(1));
    ASSERT_EQ(feed(weaker_yolo, both, {unit_det(300.f, 300.f, 0.4f, 5)}), kSynthConfirm);
    DetGate stronger_yolo(blob_cfg(1));
    ASSERT_EQ(feed(stronger_yolo, both, {unit_det(300.f, 300.f, 0.7f, 5)}), 5);
}

TEST(a_motion_blob_is_ranked_at_the_configured_confsky) {
    const auto blob = cluster({{100.f, 40.f}, {110.f, 40.f}, {100.f, 50.f}, {120.f, 60.f}});
    std::vector<MotionSample> both = blob;
    both.push_back(MotionSample{300.f, 300.f, kMovedResidual});
    GateCfg c = blob_cfg(1);
    c.confsky = 0.8f;
    DetGate g(c);
    ASSERT_EQ(feed(g, both, {unit_det(300.f, 300.f, 0.7f, 5)}), kSynthConfirm);
}

TEST(only_the_detection_on_the_independently_moving_patch_passes_the_gate) {
    GateRig rig([](GstElement *) {});
    for (int f = 0; f < kFrames; f++) {
        const auto kept = rig.push(frame_with_two_dets(f));
        if (f < kFirstConfirmedFrame) {
            ASSERT_EQ(kept.size(), (size_t)0);
            continue;
        }
        ASSERT_EQ(kept.size(), (size_t)1);
        const NvmmDetObject want = mover_det(f);
        ASSERT_EQ(std::string(kept[0].label), std::string("mover"));
        ASSERT_EQ(kept[0].left, want.left);
        ASSERT_EQ(kept[0].top, want.top);
        ASSERT_EQ(kept[0].width, want.width);
        ASSERT_EQ(kept[0].height, want.height);
        ASSERT_EQ(kept[0].confidence, want.confidence);
    }
}

TEST(dlt_two_confirms_at_frame_nine_while_the_default_dlt_has_not_yet_gated) {
    ASSERT_EQ(first_kept_frame(run_fast(just_the_mover, fast_gate)), kFastFirstConfirmedFrame);
    const Configure default_dlt = [](GstElement *) {};
    ASSERT_EQ(first_kept_frame(run_fast(just_the_mover, default_dlt)), -1);
}

TEST(detections_in_a_half_by_third_coordinate_space_are_gated_at_frame_pixels) {
    constexpr float kSx = 2.f, kSy = 3.f;
    GateRig rig(fast_gate);
    for (int f = 0; f < kFastFrames; f++) {
        const NvmmDetObject src = fast_mover_det(f);
        const NvmmDetObject mover = det_box(src.left / kSx, src.top / kSy, kMoverSide / kSx,
                                            kMoverSide / kSy, 0.6f, "mover");
        const NvmmDetObject still = det_box(kStaticLeft / kSx, kStaticTop / kSy, kMoverSide / kSx,
                                            kMoverSide / kSy, 0.9f, "static");
        const auto kept = rig.push(fast_frame(f, {still, mover}, kW / 2, kH / 3));
        if (f < kFastFirstConfirmedFrame) {
            ASSERT_EQ(kept.size(), (size_t)0);
            continue;
        }
        ASSERT_EQ(kept.size(), (size_t)1);
        ASSERT_EQ(std::string(kept[0].label), std::string("mover"));
        ASSERT_EQ(kept[0].left, mover.left);
        ASSERT_EQ(kept[0].top, mover.top);
        ASSERT_EQ(kept[0].width, mover.width);
        ASSERT_EQ(kept[0].height, mover.height);
    }
}

TEST(a_zero_infer_width_leaves_x_in_frame_pixels_while_y_is_scaled) {
    const DetsOf dets_of = [](int f) {
        const NvmmDetObject src = fast_mover_det(f);
        return std::vector<NvmmDetObject>{
            det_box(src.left, src.top / 2.f, kMoverSide, kMoverSide / 2.f, 0.6f, "mover")};
    };
    ASSERT_EQ(first_kept_frame(run_fast(dets_of, fast_gate, 0, kH / 2)), kFastFirstConfirmedFrame);
}

TEST(a_zero_infer_height_leaves_y_in_frame_pixels_while_x_is_scaled) {
    const DetsOf dets_of = [](int f) {
        const NvmmDetObject src = fast_mover_det(f);
        return std::vector<NvmmDetObject>{
            det_box(src.left / 2.f, src.top, kMoverSide / 2.f, kMoverSide, 0.6f, "mover")};
    };
    ASSERT_EQ(first_kept_frame(run_fast(dets_of, fast_gate, kW / 2, 0)), kFastFirstConfirmedFrame);
}

TEST(a_detection_of_another_class_is_never_confirmed) {
    const DetsOf dets_of = [](int f) { return std::vector<NvmmDetObject>{fast_mover_det(f, 0.6f, 1)}; };
    ASSERT_EQ(first_kept_frame(run_fast(dets_of, fast_gate)), -1);
}

TEST(target_class_selects_which_class_is_confirmed) {
    const DetsOf dets_of = [](int f) { return std::vector<NvmmDetObject>{fast_mover_det(f, 0.6f, 1)}; };
    const Configure class1 = [](GstElement *g) { fast_gate(g); g_object_set(g, "target-class", 1, NULL); };
    ASSERT_EQ(first_kept_frame(run_fast(dets_of, class1)), kFastFirstConfirmedFrame);
}

TEST(a_detection_below_the_default_min_conf_is_never_confirmed) {
    const DetsOf dets_of = [](int f) { return std::vector<NvmmDetObject>{fast_mover_det(f, 0.2f)}; };
    ASSERT_EQ(first_kept_frame(run_fast(dets_of, fast_gate)), -1);
}

TEST(a_detection_exactly_at_the_default_min_conf_is_confirmed) {
    const DetsOf dets_of = [](int f) { return std::vector<NvmmDetObject>{fast_mover_det(f, 0.25f)}; };
    ASSERT_EQ(first_kept_frame(run_fast(dets_of, fast_gate)), kFastFirstConfirmedFrame);
}

TEST(a_residual_below_rmin_is_never_confirmed) {
    const Configure strict = [](GstElement *g) { fast_gate(g); g_object_set(g, "rmin", 40.0, NULL); };
    ASSERT_EQ(first_kept_frame(run_fast(just_the_mover, strict)), -1);
}

TEST(amin_and_ksup_each_delay_confirmation) {
    const Configure amin8 = [](GstElement *g) { fast_gate(g); g_object_set(g, "amin", 8, "ksup", 2, NULL); };
    ASSERT_EQ(first_kept_frame(run_fast(just_the_mover, amin8, kW, kH, 2 * kFastDlt + 8)), 2 * kFastDlt + 8 - 1);
    const Configure ksup8 = [](GstElement *g) { fast_gate(g); g_object_set(g, "amin", 2, "ksup", 8, NULL); };
    ASSERT_EQ(first_kept_frame(run_fast(just_the_mover, ksup8, kW, kH, 2 * kFastDlt + 8)), 2 * kFastDlt + 8 - 1);
}

TEST(border_frac_rejects_a_mover_outside_the_active_area) {
    const Configure wide = [](GstElement *g) { fast_gate(g); g_object_set(g, "border-frac", 0.4, NULL); };
    ASSERT_EQ(first_kept_frame(run_fast(just_the_mover, wide)), -1);
}

TEST(a_disabled_gate_passes_every_detection_untouched) {
    const Configure off = [](GstElement *g) { g_object_set(g, "enabled", FALSE, NULL); };
    const DetsOf dets_of = [](int f) {
        return std::vector<NvmmDetObject>{det(kStaticLeft, kStaticTop, 0.9f, "static"), fast_mover_det(f)};
    };
    const Kept kept = run_fast(dets_of, off, kW, kH, 3);
    for (const auto &frame : kept) {
        ASSERT_EQ(frame.size(), (size_t)2);
        ASSERT_EQ(std::string(frame[0].label), std::string("static"));
        ASSERT_EQ(std::string(frame[1].label), std::string("mover"));
    }
}

TEST(seed_on_motion_replaces_the_detections_with_one_target_on_the_mover) {
    const Configure seed = [](GstElement *g) {
        fast_gate(g);
        g_object_set(g, "seed-on-motion", TRUE, "motion-silent", 1, NULL);
    };
    const DetsOf silent = [](int) { return std::vector<NvmmDetObject>{}; };
    const Kept kept = run_fast(silent, seed);
    ASSERT_EQ(first_kept_frame(kept), kFastFirstConfirmedFrame);
    const std::vector<NvmmDetObject> &seeded = kept[kFastFirstConfirmedFrame];
    ASSERT_EQ(seeded.size(), (size_t)1);
    const NvmmDetObject &o = seeded[0];
    ASSERT_EQ(std::string(o.label), std::string("target"));
    ASSERT_EQ(o.class_id, 0);
    ASSERT_EQ(o.confidence, 0.90f);
    ASSERT_EQ(o.tracker_id, (guint)0);
    const float left = (float)(kMoverX0 + kFastStep * kFastFirstConfirmedFrame);
    ASSERT_TRUE(o.left >= left - kKeypointStridePx);
    ASSERT_TRUE(o.top >= kMoverY - kKeypointStridePx);
    ASSERT_TRUE(o.left + o.width <= left + kMoverSide + kKeypointStridePx);
    ASSERT_TRUE(o.top + o.height <= kMoverY + kMoverSide + kKeypointStridePx);
    ASSERT_TRUE(o.width > 0.f && o.height > 0.f);
}

TEST(seed_on_motion_stays_silent_unless_enabled) {
    const Configure plain = [](GstElement *g) { fast_gate(g); g_object_set(g, "motion-silent", 1, NULL); };
    const DetsOf silent = [](int) { return std::vector<NvmmDetObject>{}; };
    ASSERT_EQ(first_kept_frame(run_fast(silent, plain)), -1);
}

TEST(a_buffer_the_allocator_did_not_make_is_gated_through_its_surface) {
    GateRig rig(fast_gate);
    int first = -1;
    for (int f = 0; f < kFastFrames && first < 0; f++) {
        NvBufSurface *s = pitch_nv12(kW, kH, moving_scene(kMoverX0 + kFastStep * f, kMoverY));
        const auto kept = rig.push(with_dets(wrapped_surface_buffer(s), {fast_mover_det(f)}));
        if (!kept.empty()) first = f;
    }
    ASSERT_EQ(first, kFastFirstConfirmedFrame);
}

GstEvent *upstream_event(const char *name)
{
    return gst_event_new_custom(GST_EVENT_CUSTOM_UPSTREAM, gst_structure_new_empty(name));
}

TEST(nvmm_reset_unlatches_a_confirmed_gate) {
    GateRig rig(fast_gate);
    for (int f = 0; f <= kFastFirstConfirmedFrame + 1; f++) rig.push(fast_frame(f, {fast_mover_det(f)}));
    const int next = kFastFirstConfirmedFrame + 2;
    ASSERT_TRUE(gst_harness_push_upstream_event(rig.harness(), upstream_event("nvmm-reset")));
    ASSERT_EQ(rig.push(fast_frame(next, {fast_mover_det(next)})).size(), (size_t)0);
}

TEST(an_upstream_event_with_another_name_leaves_the_gate_latched) {
    GateRig rig(fast_gate);
    for (int f = 0; f <= kFastFirstConfirmedFrame + 1; f++) rig.push(fast_frame(f, {fast_mover_det(f)}));
    const int next = kFastFirstConfirmedFrame + 2;
    ASSERT_TRUE(gst_harness_push_upstream_event(rig.harness(), upstream_event("nvmm-other")));
    ASSERT_EQ(rig.push(fast_frame(next, {fast_mover_det(next)})).size(), (size_t)1);
}

TEST(nvmm_reset_is_still_delivered_upstream) {
    GateRig rig(fast_gate);
    rig.push(fast_frame(0, {fast_mover_det(0)}));
    ASSERT_TRUE(gst_harness_push_upstream_event(rig.harness(), upstream_event("nvmm-reset")));
    bool delivered = false;
    while (GstEvent *seen = gst_harness_try_pull_upstream_event(rig.harness())) {
        const GstStructure *s = gst_event_get_structure(seen);
        delivered = delivered || (s && gst_structure_has_name(s, "nvmm-reset"));
        gst_event_unref(seen);
    }
    ASSERT_TRUE(delivered);
}

TEST(nvmm_reset_before_the_first_frame_is_harmless) {
    GateRig rig(fast_gate);
    ASSERT_TRUE(gst_harness_push_upstream_event(rig.harness(), upstream_event("nvmm-reset")));
    int first = -1;
    for (int f = 0; f < kFastFrames && first < 0; f++)
        if (!rig.push(fast_frame(f, {fast_mover_det(f)})).empty()) first = f;
    ASSERT_EQ(first, kFastFirstConfirmedFrame);
}

TEST(a_fresh_gate_has_the_documented_property_defaults) {
    GstElement *g = gst_element_factory_make("nvmmdetgate", nullptr);
    ASSERT_NOT_NULL(g);
    gint target_class, ds, dlt, amin, ksup, maxlost, motion_silent;
    gdouble min_conf, rmin, rminsky, confsky, dist, border, motion_rmin;
    gboolean enabled, seed_on_motion;
    gchar *engine_dir = nullptr;
    g_object_get(g, "engine-dir", &engine_dir, "target-class", &target_class, "min-conf", &min_conf,
                 "ds", &ds, "dlt", &dlt, "rmin", &rmin, "rminsky", &rminsky, "confsky", &confsky,
                 "dist", &dist, "amin", &amin, "ksup", &ksup, "maxlost", &maxlost,
                 "border-frac", &border, "enabled", &enabled, "seed-on-motion", &seed_on_motion,
                 "motion-silent", &motion_silent, "motion-rmin", &motion_rmin, NULL);
    ASSERT_TRUE(engine_dir == nullptr);
    ASSERT_EQ(target_class, 0);
    ASSERT_EQ(min_conf, 0.25);
    ASSERT_EQ(ds, 2);
    ASSERT_EQ(dlt, 5);
    ASSERT_EQ(rmin, 12.0);
    ASSERT_EQ(rminsky, 8.0);
    ASSERT_EQ(confsky, 0.55);
    ASSERT_EQ(dist, 45.0);
    ASSERT_EQ(amin, 6);
    ASSERT_EQ(ksup, 4);
    ASSERT_EQ(maxlost, 2);
    ASSERT_EQ(border, 0.02);
    ASSERT_EQ(enabled, TRUE);
    ASSERT_EQ(seed_on_motion, FALSE);
    ASSERT_EQ(motion_silent, 12);
    ASSERT_EQ(motion_rmin, 16.0);
    gst_object_unref(g);
}

struct IntSpec { const char *name; gint min, max, def; };
struct DoubleSpec { const char *name; gdouble min, max, def; };

TEST(int_properties_advertise_the_documented_range_and_default) {
    GstElement *g = gst_element_factory_make("nvmmdetgate", nullptr);
    ASSERT_NOT_NULL(g);
    const IntSpec specs[] = {
        {"target-class", 0, 9999, 0}, {"ds", 1, 8, 2}, {"dlt", 1, 30, 5}, {"amin", 1, 1000, 6},
        {"ksup", 1, 1000, 4}, {"maxlost", 0, 1000, 2}, {"motion-silent", 1, 1000, 12},
    };
    for (const IntSpec &s : specs) {
        GParamSpecInt *p = G_PARAM_SPEC_INT(g_object_class_find_property(G_OBJECT_GET_CLASS(g), s.name));
        ASSERT_NOT_NULL(p);
        ASSERT_EQ(p->minimum, s.min);
        ASSERT_EQ(p->maximum, s.max);
        ASSERT_EQ(p->default_value, s.def);
    }
    gst_object_unref(g);
}

TEST(double_properties_advertise_the_documented_range_and_default) {
    GstElement *g = gst_element_factory_make("nvmmdetgate", nullptr);
    ASSERT_NOT_NULL(g);
    const DoubleSpec specs[] = {
        {"min-conf", 0, 1, 0.25}, {"rmin", 0, 255, 12}, {"rminsky", 0, 255, 8},
        {"confsky", 0, 1, 0.55}, {"dist", 1, 1000, 45}, {"border-frac", 0, 0.4, 0.02},
        {"motion-rmin", 0, 255, 16},
    };
    for (const DoubleSpec &s : specs) {
        GParamSpecDouble *p = G_PARAM_SPEC_DOUBLE(g_object_class_find_property(G_OBJECT_GET_CLASS(g), s.name));
        ASSERT_NOT_NULL(p);
        ASSERT_EQ(p->minimum, s.min);
        ASSERT_EQ(p->maximum, s.max);
        ASSERT_EQ(p->default_value, s.def);
    }
    gst_object_unref(g);
}

TEST(boolean_properties_advertise_the_documented_defaults) {
    GstElement *g = gst_element_factory_make("nvmmdetgate", nullptr);
    ASSERT_NOT_NULL(g);
    GObjectClass *k = G_OBJECT_GET_CLASS(g);
    ASSERT_EQ(G_PARAM_SPEC_BOOLEAN(g_object_class_find_property(k, "enabled"))->default_value, TRUE);
    ASSERT_EQ(G_PARAM_SPEC_BOOLEAN(g_object_class_find_property(k, "seed-on-motion"))->default_value, FALSE);
    gst_object_unref(g);
}

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
