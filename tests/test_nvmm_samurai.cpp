#include <gst/gst.h>
#include <gst/check/gstharness.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <glib.h>
#include <glib/gstdio.h>

#include "nvmm_det_meta.h"
#include "nvmm_frame.h"
#include "nvmm_track_meta.h"
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
        for (const char *f : {"image_encoder_bplus_512.engine", "prompt_encoder.engine",
                              "mask_decoder.engine", "memory_encoder.engine",
                              "memory_attention.engine", "samurai_consts.bin"}) {
            if (!dir.empty() && g_file_test((dir + "/" + f).c_str(), G_FILE_TEST_IS_REGULAR))
                continue;
            std::printf("SKIP: NVMM_TEST_ENGINE_DIR has no %s\n", f);
            std::exit(kSkipExitCode);
        }
        gst_init(nullptr, nullptr);
    }
} engines;

/// Both sides at least the 512 crop, so the encoder sees frame pixels 1:1.
constexpr int kW = 768, kH = 512;
constexpr int kSide = 64;
constexpr int kX0 = 160, kY0 = 160;
constexpr int kStepX = 6, kStepY = 3;
constexpr int kFrames = 20;
/// One low-res mask cell: the decoder's mask is crop/4 = 128 px for a 512 px crop.
constexpr float kEdgeTolPx = 512.f / 128.f;
constexpr Yuv kGrey{128, 128, 128};
constexpr int kCrop = 512;

Nv12Painter square_at(int frame)
{
    const int x0 = kX0 + kStepX * frame, y0 = kY0 + kStepY * frame;
    return [x0, y0](int x, int y) {
        return (x >= x0 && x < x0 + kSide && y >= y0 && y < y0 + kSide) ? kRed : kGrey;
    };
}

uint32_t hash2(uint32_t x, uint32_t y, uint32_t salt)
{
    uint32_t h = x * 374761393u + y * 668265263u + salt * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}

Yuv texture(int x, int y)
{
    const uint32_t c = hash2((uint32_t)x / 8, (uint32_t)y / 8, 5);
    return Yuv{(uint8_t)(90 + c % 80), (uint8_t)(112 + (c >> 8) % 32), (uint8_t)(112 + (c >> 16) % 32)};
}

constexpr int kNestedOuter = 96, kNestedInner = 48;
constexpr int kNestedMargin = (kNestedOuter - kNestedInner) / 2;

Nv12Painter nested_squares_at(int frame)
{
    const int x0 = kX0 + kStepX * frame, y0 = kY0 + kStepY * frame;
    return [x0, y0](int x, int y) {
        const bool outer = x >= x0 && x < x0 + kNestedOuter && y >= y0 && y < y0 + kNestedOuter;
        const bool inner = x >= x0 + kNestedMargin && x < x0 + kNestedMargin + kNestedInner &&
                           y >= y0 + kNestedMargin && y < y0 + kNestedMargin + kNestedInner;
        return inner ? kRed : outer ? kBlue : texture(x, y);
    };
}

constexpr int kPanX = 8, kPanY = 4;
constexpr int kPanFrames = 6;
/// FFT sub-pixel peak jitter measured at about 0.1 px per frame; six frames stay under 1 px.
constexpr float kFftJitterPx = 1.f;
/// One 8-bit code step in normalised units: 1/255 over the smallest std, 0.224.
constexpr float kColourConversionTol = 0.02f;
/// BT.601 limited-range VIC output for Y=128: 1.164 * (128 - 16) = 130.
constexpr float kGreyRgbCode = 130.f;

Nv12Painter panned_textured_square_at(int frame)
{
    const int dx = kPanX * frame, dy = kPanY * frame;
    return [dx, dy](int x, int y) {
        const int wx = x - dx, wy = y - dy;
        const bool in = wx >= kX0 && wx < kX0 + kSide && wy >= kY0 && wy < kY0 + kSide;
        return in ? kRed : texture(wx + 4000, wy + 4000);
    };
}

std::string roi_of(int x, int y, int w, int h)
{
    return std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(w) + "," +
           std::to_string(h);
}

using Configure = std::function<void(GstElement *)>;
using Frame = std::function<GstBuffer *(int)>;

void on_square(GstElement *e)
{
    g_object_set(e, "seed-roi", roi_of(kX0, kY0, kSide, kSide).c_str(), NULL);
}

class TrackerRig {
public:
    explicit TrackerRig(const Configure &configure)
    {
        element_ = gst_element_factory_make("nvmmsamurai", nullptr);
        ASSERT_NOT_NULL(element_);
        const std::string consts = engines.dir + "/samurai_consts.bin";
        g_object_set(element_, "engine-dir", engines.dir.c_str(), "consts-file", consts.c_str(), NULL);
        configure(element_);
        harness_ = gst_harness_new_with_element(element_, "sink", "src");
        gst_harness_set_src_caps_str(harness_, "video/x-raw(memory:NVMM),format=NV12,width=768,"
                                               "height=512,framerate=30/1");
    }
    TrackerRig(const TrackerRig &) = delete;
    TrackerRig &operator=(const TrackerRig &) = delete;
    ~TrackerRig()
    {
        gst_harness_teardown(harness_);
        gst_object_unref(element_);
    }

    GstNvmmTrackMeta push(GstBuffer *buf)
    {
        ASSERT_EQ(gst_harness_push(harness_, buf), GST_FLOW_OK);
        GstBuffer *out = gst_harness_pull(harness_);
        const GstNvmmTrackMeta *m = gst_buffer_get_nvmm_track_meta(out);
        ASSERT_NOT_NULL(m);
        const GstNvmmTrackMeta copy = *m;
        gst_buffer_unref(out);
        return copy;
    }

    GstHarness *harness() const { return harness_; }

private:
    GstElement *element_ = nullptr;
    GstHarness *harness_ = nullptr;
};

std::vector<GstNvmmTrackMeta> track(const Configure &configure, int frames, const Frame &frame)
{
    TrackerRig rig(configure);
    std::vector<GstNvmmTrackMeta> out;
    for (int f = 0; f < frames; f++) out.push_back(rig.push(frame(f)));
    return out;
}

std::vector<GstNvmmTrackMeta> track_square(int max_kf)
{
    const Configure configure = [max_kf](GstElement *e) {
        on_square(e);
        g_object_set(e, "max-kf", max_kf, NULL);
    };
    return track(configure, kFrames, [](int f) { return nvmm_nv12_buffer(kW, kH, square_at(f)); });
}

void assert_box_on_square(const GstNvmmTrackMeta &m, int f)
{
    ASSERT_NEAR(m.left, kX0 + kStepX * f, kEdgeTolPx);
    ASSERT_NEAR(m.top, kY0 + kStepY * f, kEdgeTolPx);
    ASSERT_NEAR(m.width, kSide, kEdgeTolPx);
    ASSERT_NEAR(m.height, kSide, kEdgeTolPx);
}

void assert_box_on(const GstNvmmTrackMeta &m, int left, int top, int side)
{
    ASSERT_NEAR(m.left, left, kEdgeTolPx);
    ASSERT_NEAR(m.top, top, kEdgeTolPx);
    ASSERT_NEAR(m.width, side, kEdgeTolPx);
    ASSERT_NEAR(m.height, side, kEdgeTolPx);
}

TEST(track_box_follows_a_square_moving_six_right_three_down_per_frame) {
    const auto t = track_square(0);
    for (int f = 0; f < kFrames; f++) {
        const GstNvmmTrackMeta &m = t[f];
        ASSERT_EQ(m.frame_number, (guint64)f);
        ASSERT_EQ(m.frame_width, (guint32)kW);
        ASSERT_EQ(m.frame_height, (guint32)kH);
        ASSERT_TRUE(m.valid);
        ASSERT_EQ(m.target_id, (guint64)1);
        ASSERT_TRUE(!m.is_kf_only);
        ASSERT_TRUE(m.object_score > 0.f);
        ASSERT_EQ(m.stable_frames, (guint32)(f + 2));
        assert_box_on_square(m, f);
    }
}

/// Default max-kf=2: two Kalman-only frames, then one model frame, starting on the seed frame.
TEST(default_max_kf_runs_the_model_on_every_third_frame) {
    const auto t = track_square(2);
    guint32 model_frames = 1;
    for (int f = 0; f < kFrames; f++) {
        const GstNvmmTrackMeta &m = t[f];
        const bool kf_only = f % 3 != 2;
        ASSERT_EQ((bool)m.is_kf_only, kf_only);
        ASSERT_TRUE(m.valid);
        ASSERT_EQ(m.target_id, (guint64)1);
        if (kf_only) {
            ASSERT_EQ(m.object_score, 1.f);
            ASSERT_EQ(m.kf_left, m.left);
            ASSERT_EQ(m.kf_width, m.width);
        } else {
            model_frames++;
            assert_box_on_square(m, f);
        }
        ASSERT_EQ(m.stable_frames, model_frames);
    }
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

NvmmDetObject det_box(float left, float top, float w, float h, float conf, int class_id = 0)
{
    NvmmDetObject o{};
    o.left = left; o.top = top; o.width = w; o.height = h;
    o.class_id = class_id;
    o.confidence = conf;
    return o;
}

NvmmDetObject det_on_square(float conf = 0.9f, int class_id = 0)
{
    return det_box(kX0, kY0, kSide, kSide, conf, class_id);
}

void no_roi_max_kf_zero(GstElement *e)
{
    g_object_set(e, "max-kf", 0, NULL);
}

std::vector<GstNvmmTrackMeta> track_with_dets(const Configure &configure,
                                              const std::vector<NvmmDetObject> &dets, int frames,
                                              guint32 infer_w = kW, guint32 infer_h = kH)
{
    return track(configure, frames, [&](int) {
        return with_dets(nvmm_nv12_buffer(kW, kH, square_at(0)), dets, infer_w, infer_h);
    });
}

TEST(a_detection_in_a_quarter_by_eighth_infer_space_seeds_the_scaled_box) {
    constexpr guint32 kInferW = kW / 2, kInferH = kH / 4;
    const auto t = track_with_dets(no_roi_max_kf_zero,
                                   {det_box(kX0 / 2.f, kY0 / 4.f, kSide / 2.f, kSide / 4.f, 0.9f)},
                                   3, kInferW, kInferH);
    for (const auto &m : t) {
        ASSERT_TRUE(m.valid);
        assert_box_on(m, kX0, kY0, kSide);
    }
}

constexpr int kDecoyX = 100, kDecoyY = 100;
constexpr int kNearCenterX = 400, kNearCenterY = 230;

Nv12Painter decoy_and_near_center_squares()
{
    return [](int x, int y) {
        const bool decoy = x >= kDecoyX && x < kDecoyX + kSide && y >= kDecoyY && y < kDecoyY + kSide;
        const bool near_center = x >= kNearCenterX && x < kNearCenterX + kSide &&
                                 y >= kNearCenterY && y < kNearCenterY + kSide;
        return decoy ? kRed : near_center ? kBlue : kGrey;
    };
}

std::vector<GstNvmmTrackMeta> seed_between_two_squares(const Configure &configure)
{
    const std::vector<NvmmDetObject> dets = {
        det_box(kDecoyX, kDecoyY, kSide, kSide, 0.9f),
        det_box(kNearCenterX, kNearCenterY, kSide, kSide, 0.5f),
    };
    return track(configure, 2, [&](int) {
        return with_dets(nvmm_nv12_buffer(kW, kH, decoy_and_near_center_squares()), dets);
    });
}

TEST(seed_takes_the_most_confident_detection_by_default) {
    const auto t = seed_between_two_squares(no_roi_max_kf_zero);
    assert_box_on(t[0], kDecoyX, kDecoyY, kSide);
}

TEST(seed_prefer_center_takes_the_detection_nearest_the_frame_center) {
    const Configure center = [](GstElement *e) {
        no_roi_max_kf_zero(e);
        g_object_set(e, "seed-prefer-center", TRUE, NULL);
    };
    const auto t = seed_between_two_squares(center);
    assert_box_on(t[0], kNearCenterX, kNearCenterY, kSide);
}

TEST(a_detection_below_seed_conf_never_seeds) {
    const auto t = track_with_dets(no_roi_max_kf_zero, {det_on_square(0.2f)}, 3);
    for (const auto &m : t) {
        ASSERT_TRUE(!m.valid);
        ASSERT_EQ(m.target_id, (guint64)0);
    }
}

TEST(a_detection_exactly_at_seed_conf_seeds) {
    const Configure strict = [](GstElement *e) {
        no_roi_max_kf_zero(e);
        g_object_set(e, "seed-conf", 0.5, NULL);
    };
    const auto t = track_with_dets(strict, {det_on_square(0.5f)}, 1);
    ASSERT_TRUE(t[0].valid);
}

TEST(a_detection_of_another_class_never_seeds) {
    const auto t = track_with_dets(no_roi_max_kf_zero, {det_on_square(0.9f, 1)}, 2);
    for (const auto &m : t) ASSERT_TRUE(!m.valid);
}

TEST(target_class_selects_the_class_that_seeds) {
    const Configure class1 = [](GstElement *e) {
        no_roi_max_kf_zero(e);
        g_object_set(e, "target-class", 1, NULL);
    };
    const auto t = track_with_dets(class1, {det_on_square(0.9f, 1)}, 1);
    ASSERT_TRUE(t[0].valid);
}

TEST(seed_delay_holds_off_auto_seeding_for_that_many_frames) {
    const Configure delayed = [](GstElement *e) {
        no_roi_max_kf_zero(e);
        g_object_set(e, "seed-delay", 2u, NULL);
    };
    const auto t = track_with_dets(delayed, {det_on_square()}, 3);
    ASSERT_TRUE(!t[0].valid);
    ASSERT_TRUE(!t[1].valid);
    ASSERT_TRUE(t[2].valid);
}

TEST(seed_roi_waits_for_seed_delay_too) {
    const Configure delayed = [](GstElement *e) {
        on_square(e);
        g_object_set(e, "max-kf", 0, "seed-delay", 2u, NULL);
    };
    const auto t = track(delayed, 3, [](int) { return nvmm_nv12_buffer(kW, kH, square_at(0)); });
    ASSERT_TRUE(!t[0].valid);
    ASSERT_TRUE(!t[1].valid);
    ASSERT_TRUE(t[2].valid);
}

TEST(a_seed_roi_of_zero_width_never_seeds) {
    const Configure flat = [](GstElement *e) {
        g_object_set(e, "seed-roi", roi_of(kX0, kY0, 0, kSide).c_str(), "max-kf", 0, NULL);
    };
    const auto t = track(flat, 2, [](int) { return nvmm_nv12_buffer(kW, kH, square_at(0)); });
    for (const auto &m : t) ASSERT_TRUE(!m.valid);
}

TEST(a_seed_roi_of_zero_height_never_seeds) {
    const Configure flat = [](GstElement *e) {
        g_object_set(e, "seed-roi", roi_of(kX0, kY0, kSide, 0).c_str(), "max-kf", 0, NULL);
    };
    const auto t = track(flat, 2, [](int) { return nvmm_nv12_buffer(kW, kH, square_at(0)); });
    for (const auto &m : t) ASSERT_TRUE(!m.valid);
}

GstEvent *upstream_event(const char *name, const std::function<void(GstStructure *)> &fill = {})
{
    GstStructure *s = gst_structure_new_empty(name);
    if (fill) fill(s);
    return gst_event_new_custom(GST_EVENT_CUSTOM_UPSTREAM, s);
}

bool delivered_upstream(GstHarness *h, const char *name)
{
    bool seen = false;
    while (GstEvent *ev = gst_harness_try_pull_upstream_event(h)) {
        const GstStructure *s = gst_event_get_structure(ev);
        seen = seen || (s && gst_structure_has_name(s, name));
        gst_event_unref(ev);
    }
    return seen;
}

TEST(nvmm_reseed_moves_the_track_to_the_box_on_the_next_frame) {
    constexpr int kEmptyX = 500, kEmptyY = 50;
    const Configure on_empty = [](GstElement *e) {
        g_object_set(e, "seed-roi", roi_of(kEmptyX, kEmptyY, kSide, kSide).c_str(), "max-kf", 0, NULL);
    };
    TrackerRig rig(on_empty);
    const GstNvmmTrackMeta before = rig.push(nvmm_nv12_buffer(kW, kH, square_at(0)));
    ASSERT_TRUE(std::fabs(before.left - kX0) > 4 * kEdgeTolPx);
    const auto fill = [](GstStructure *s) {
        gst_structure_set(s, "x", G_TYPE_DOUBLE, (double)kX0, "y", G_TYPE_DOUBLE, (double)kY0,
                          "w", G_TYPE_DOUBLE, (double)kSide, "h", G_TYPE_DOUBLE, (double)kSide, NULL);
    };
    ASSERT_TRUE(gst_harness_push_upstream_event(rig.harness(), upstream_event("nvmm-reseed", fill)));
    const GstNvmmTrackMeta after = rig.push(nvmm_nv12_buffer(kW, kH, square_at(0)));
    ASSERT_TRUE(after.valid);
    assert_box_on(after, kX0, kY0, kSide);
}

TEST(nvmm_reseed_is_consumed_and_other_upstream_events_are_forwarded) {
    TrackerRig rig(on_square);
    ASSERT_TRUE(gst_harness_push_upstream_event(rig.harness(), upstream_event("nvmm-reseed")));
    ASSERT_TRUE(!delivered_upstream(rig.harness(), "nvmm-reseed"));
    ASSERT_TRUE(gst_harness_push_upstream_event(rig.harness(), upstream_event("nvmm-other")));
    ASSERT_TRUE(delivered_upstream(rig.harness(), "nvmm-other"));
}

const char *default_backend_nick(GstElement *e, gint *instance_value)
{
    GParamSpecEnum *p = G_PARAM_SPEC_ENUM(g_object_class_find_property(G_OBJECT_GET_CLASS(e), "gmc-backend"));
    g_object_get(e, "gmc-backend", instance_value, NULL);
    return g_enum_get_value(p->enum_class, p->default_value)->value_nick;
}

Configure kalman_only_pan(bool gmc, const char *backend)
{
    return [gmc, backend](GstElement *e) {
        g_object_set(e, "seed-roi", roi_of(kX0, kY0, kSide, kSide).c_str(), "max-kf", 30, NULL);
        if (!gmc) return;
        g_object_set(e, "gmc", TRUE, NULL);
        GParamSpec *p = g_object_class_find_property(G_OBJECT_GET_CLASS(e), "gmc-backend");
        GValue v = G_VALUE_INIT;
        g_value_init(&v, p->value_type);
        ASSERT_TRUE(gst_value_deserialize(&v, backend));
        g_object_set_property(G_OBJECT(e), "gmc-backend", &v);
    };
}

std::vector<GstNvmmTrackMeta> pan_kalman_only(bool gmc, const char *backend)
{
    return track(kalman_only_pan(gmc, backend), kPanFrames,
                 [](int f) { return nvmm_nv12_buffer(kW, kH, panned_textured_square_at(f)); });
}

void assert_kalman_box_follows_the_pan(const std::vector<GstNvmmTrackMeta> &t, float tolerance)
{
    for (int f = 0; f < kPanFrames; f++) {
        ASSERT_TRUE(t[f].is_kf_only);
        ASSERT_NEAR(t[f].left - t[0].left, kPanX * f, tolerance);
        ASSERT_NEAR(t[f].top - t[0].top, kPanY * f, tolerance);
    }
}

TEST(a_kalman_only_box_stays_put_under_a_camera_pan_without_gmc) {
    const auto t = pan_kalman_only(false, "ncc");
    for (int f = 0; f < kPanFrames; f++) {
        ASSERT_EQ(t[f].left, t[0].left);
        ASSERT_EQ(t[f].top, t[0].top);
    }
}

TEST(ncc_gmc_carries_the_kalman_box_along_an_exact_camera_pan) {
    assert_kalman_box_follows_the_pan(pan_kalman_only(true, "ncc"), 0.f);
}

TEST(fft_cpu_gmc_carries_the_kalman_box_along_a_camera_pan) {
    assert_kalman_box_follows_the_pan(pan_kalman_only(true, "fft-cpu"), kFftJitterPx);
}

TEST(fft_cuda_gmc_carries_the_kalman_box_along_a_camera_pan) {
    assert_kalman_box_follows_the_pan(pan_kalman_only(true, "fft-cuda"), kFftJitterPx);
}

std::vector<guint32> stable_run_under(const Configure &extra)
{
    const Configure configure = [&](GstElement *e) {
        on_square(e);
        g_object_set(e, "max-kf", 0, NULL);
        extra(e);
    };
    std::vector<guint32> stable;
    for (const auto &m : track(configure, 6, [](int f) { return nvmm_nv12_buffer(kW, kH, square_at(f)); }))
        stable.push_back(m.stable_frames);
    return stable;
}

TEST(an_iou_threshold_of_one_rejects_every_kalman_update) {
    const auto stable = stable_run_under([](GstElement *e) { g_object_set(e, "iou-threshold", 1.0, NULL); });
    for (size_t f = 0; f < stable.size(); f++) ASSERT_EQ(stable[f], (guint32)(f % 2 == 0 ? 0 : 1));
}

TEST(a_kf_min_area_above_the_target_area_rejects_every_kalman_update) {
    const auto stable = stable_run_under([](GstElement *e) { g_object_set(e, "kf-min-area", 1e6, NULL); });
    for (size_t f = 0; f < stable.size(); f++) ASSERT_EQ(stable[f], (guint32)(f % 2 == 0 ? 0 : 1));
}

constexpr int kRegimeFrames = 5;
constexpr int kRegimeThreshold = 5;

std::vector<GstNvmmTrackMeta> track_nested(int threshold, double weight)
{
    const Configure configure = [=](GstElement *e) {
        g_object_set(e, "seed-roi", roi_of(kX0 + kNestedMargin, kY0 + kNestedMargin, kNestedInner, kNestedInner).c_str(),
                     "max-kf", 0, "stable-frames-threshold", threshold, "kf-score-weight", weight, NULL);
    };
    return track(configure, kRegimeFrames, [](int f) { return nvmm_nv12_buffer(kW, kH, nested_squares_at(f)); });
}

bool same_box(const GstNvmmTrackMeta &a, const GstNvmmTrackMeta &b)
{
    return a.left == b.left && a.top == b.top && a.width == b.width && a.height == b.height;
}

const std::vector<GstNvmmTrackMeta> &never_stable_reference()
{
    static const auto reference = track_nested(1000, 1.0);
    return reference;
}

TEST(selection_follows_the_kalman_box_exactly_once_stable_frames_reach_the_threshold) {
    const auto &reference = never_stable_reference();
    const auto t = track_nested(kRegimeThreshold, 1.0);
    for (int f = 0; f < kRegimeFrames; f++) {
        const bool stable_regime = t[f].stable_frames > (guint32)kRegimeThreshold;
        ASSERT_EQ(same_box(t[f], reference[f]), !stable_regime);
    }
}

TEST(kf_score_weight_picks_a_different_candidate_only_in_the_stable_regime) {
    const auto &reference = never_stable_reference();
    const auto kalman_led = track_nested(0, 1.0);
    const auto mask_led = track_nested(0, 0.0);
    bool differs = false;
    for (int f = 0; f < kRegimeFrames; f++) {
        ASSERT_TRUE(same_box(mask_led[f], reference[f]));
        differs = differs || !same_box(kalman_led[f], reference[f]);
    }
    ASSERT_TRUE(differs);
}

std::string make_dump_dir()
{
    gchar *dir = g_dir_make_tmp("samurai-dump-XXXXXX", nullptr);
    ASSERT_NOT_NULL(dir);
    const std::string path = dir;
    g_free(dir);
    return path;
}

gsize file_size(const std::string &dir, const char *name)
{
    GStatBuf st;
    return g_stat((dir + "/" + name).c_str(), &st) == 0 ? (gsize)st.st_size : (gsize)-1;
}

std::vector<float> read_floats(const std::string &path)
{
    gchar *data = nullptr;
    gsize len = 0;
    ASSERT_TRUE(g_file_get_contents(path.c_str(), &data, &len, nullptr));
    std::vector<float> v(len / sizeof(float));
    std::memcpy(v.data(), data, v.size() * sizeof(float));
    g_free(data);
    return v;
}

void remove_dump_dir(const std::string &dir)
{
    GDir *d = g_dir_open(dir.c_str(), 0, nullptr);
    while (const gchar *name = d ? g_dir_read_name(d) : nullptr) g_remove((dir + "/" + name).c_str());
    if (d) g_dir_close(d);
    g_rmdir(dir.c_str());
}

constexpr gsize kFloat = sizeof(float);
constexpr gsize kCropFloats = (gsize)3 * kCrop * kCrop;
constexpr gsize kTokens = (kCrop / 16) * (kCrop / 16);

TEST(the_dump_dir_receives_the_crop_the_encoder_outputs_and_the_seed_state_once) {
    const std::string dir = make_dump_dir();
    g_setenv("SAMURAI_DUMP_DIR", dir.c_str(), TRUE);
    {
        TrackerRig rig([](GstElement *e) { on_square(e); no_roi_max_kf_zero(e); });
        g_unsetenv("SAMURAI_DUMP_DIR");
        rig.push(nvmm_nv12_buffer(kW, kH, square_at(0)));
        ASSERT_EQ(file_size(dir, "crop_input.bin"), kCropFloats * kFloat);
        ASSERT_EQ(file_size(dir, "out6.bin"), (gsize)256 * kTokens * kFloat);
        ASSERT_EQ(file_size(dir, "out3.bin"), (gsize)256 * kTokens * kFloat);
        ASSERT_EQ(file_size(dir, "out5.bin"), (gsize)256 * 64 * 64 * kFloat);
        ASSERT_EQ(file_size(dir, "out4.bin"), (gsize)256 * 128 * 128 * kFloat);
        ASSERT_EQ(file_size(dir, "seed_token.bin"), (gsize)256 * kFloat);
        ASSERT_EQ(file_size(dir, "seed_obj_ptr.bin"), (gsize)256 * kFloat);
        ASSERT_EQ(file_size(dir, "seed_high.bin"), (gsize)kCrop * kCrop * kFloat);
        ASSERT_EQ(file_size(dir, "seed_maskmem_feat.bin"), (gsize)64 * kTokens * kFloat);
        g_remove((dir + "/crop_input.bin").c_str());
        rig.push(nvmm_nv12_buffer(kW, kH, square_at(1)));
        ASSERT_EQ(file_size(dir, "crop_input.bin"), (gsize)-1);
    }
    remove_dump_dir(dir);
}

TEST(the_dumped_crop_carries_red_green_blue_planes_normalised_for_sam2) {
    const std::string dir = make_dump_dir();
    g_setenv("SAMURAI_DUMP_DIR", dir.c_str(), TRUE);
    {
        TrackerRig rig([](GstElement *e) { on_square(e); no_roi_max_kf_zero(e); });
        g_unsetenv("SAMURAI_DUMP_DIR");
        rig.push(nvmm_nv12_buffer(kW, kH, square_at(0)));
    }
    const std::vector<float> crop = read_floats(dir + "/crop_input.bin");
    remove_dump_dir(dir);
    ASSERT_EQ(crop.size(), kCropFloats);
    const size_t at = (size_t)(kY0 + kSide / 2) * kCrop + (kX0 + kSide / 2);
    const size_t grey_at = (size_t)400 * kCrop + 400;
    const float red_rgb[3] = {(1.f - 0.485f) / 0.229f, (0.f - 0.456f) / 0.224f, (0.f - 0.406f) / 0.225f};
    const float grey = kGreyRgbCode / 255.f;
    const float grey_rgb[3] = {(grey - 0.485f) / 0.229f, (grey - 0.456f) / 0.224f, (grey - 0.406f) / 0.225f};
    for (int c = 0; c < 3; c++) {
        ASSERT_NEAR(crop[(size_t)c * kCrop * kCrop + at], red_rgb[c], kColourConversionTol);
        ASSERT_NEAR(crop[(size_t)c * kCrop * kCrop + grey_at], grey_rgb[c], kColourConversionTol);
    }
}

TEST(a_fresh_tracker_has_the_documented_property_defaults) {
    GstElement *e = gst_element_factory_make("nvmmsamurai", nullptr);
    ASSERT_NOT_NULL(e);
    gint crop, max_kf, stable_threshold, target_class;
    gdouble kf_weight, iou, kf_min_area, seed_conf;
    gboolean prefer_center, gmc;
    guint seed_delay;
    gchar *roi = nullptr, *engine_dir = nullptr, *consts = nullptr;
    g_object_get(e, "crop-size", &crop, "max-kf", &max_kf, "kf-score-weight", &kf_weight,
                 "stable-frames-threshold", &stable_threshold, "iou-threshold", &iou,
                 "kf-min-area", &kf_min_area, "target-class", &target_class, "seed-conf", &seed_conf,
                 "seed-prefer-center", &prefer_center, "seed-roi", &roi, "seed-delay", &seed_delay,
                 "gmc", &gmc, "engine-dir", &engine_dir, "consts-file", &consts, NULL);
    ASSERT_EQ(crop, 512);
    ASSERT_EQ(max_kf, 2);
    ASSERT_EQ(kf_weight, 0.25);
    ASSERT_EQ(stable_threshold, 10);
    ASSERT_EQ(iou, 0.5);
    ASSERT_EQ(kf_min_area, 25.0);
    ASSERT_EQ(target_class, 0);
    ASSERT_EQ(seed_conf, 0.25);
    ASSERT_EQ(prefer_center, FALSE);
    ASSERT_TRUE(roi == nullptr);
    ASSERT_EQ(seed_delay, 0u);
    ASSERT_EQ(gmc, FALSE);
    ASSERT_TRUE(engine_dir == nullptr);
    ASSERT_TRUE(consts == nullptr);
    gint backend_value = -1;
    const char *nick = default_backend_nick(e, &backend_value);
    ASSERT_EQ(std::string(nick), std::string("ncc"));
    GParamSpecEnum *backend = G_PARAM_SPEC_ENUM(g_object_class_find_property(G_OBJECT_GET_CLASS(e), "gmc-backend"));
    ASSERT_EQ(backend_value, backend->default_value);
    gst_object_unref(e);
}

struct IntSpec { const char *name; gint min, max, def; };
struct DoubleSpec { const char *name; gdouble min, max, def; };

TEST(property_ranges_and_defaults_match_the_documented_table) {
    GstElement *e = gst_element_factory_make("nvmmsamurai", nullptr);
    ASSERT_NOT_NULL(e);
    GObjectClass *k = G_OBJECT_GET_CLASS(e);
    const IntSpec ints[] = {{"crop-size", 64, 2048, 512}, {"max-kf", 0, 30, 2},
                            {"stable-frames-threshold", 0, 1000, 10}, {"target-class", 0, 1000, 0}};
    for (const IntSpec &s : ints) {
        GParamSpecInt *p = G_PARAM_SPEC_INT(g_object_class_find_property(k, s.name));
        ASSERT_NOT_NULL(p);
        ASSERT_EQ(p->minimum, s.min);
        ASSERT_EQ(p->maximum, s.max);
        ASSERT_EQ(p->default_value, s.def);
    }
    const DoubleSpec doubles[] = {{"kf-score-weight", 0, 1, 0.25}, {"iou-threshold", 0, 1, 0.5},
                                  {"kf-min-area", 0, 1e8, 25}, {"seed-conf", 0, 1, 0.25}};
    for (const DoubleSpec &s : doubles) {
        GParamSpecDouble *p = G_PARAM_SPEC_DOUBLE(g_object_class_find_property(k, s.name));
        ASSERT_NOT_NULL(p);
        ASSERT_EQ(p->minimum, s.min);
        ASSERT_EQ(p->maximum, s.max);
        ASSERT_EQ(p->default_value, s.def);
    }
    GParamSpecUInt *delay = G_PARAM_SPEC_UINT(g_object_class_find_property(k, "seed-delay"));
    ASSERT_EQ(delay->minimum, 0u);
    ASSERT_EQ(delay->maximum, 100000u);
    ASSERT_EQ(delay->default_value, 0u);
    gst_object_unref(e);
}

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
