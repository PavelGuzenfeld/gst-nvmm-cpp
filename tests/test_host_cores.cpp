#include "yolo_parser.hpp"
#include "tracker.hpp"
#include "nvmm_motion.hpp"
#include "secondary_cache.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "test_harness.h"

using nvmm::LetterboxInfo;
using nvmm::YoloParams;

namespace {
constexpr int N = 8400;
constexpr int C = 80;

/// Output is channels-first: output[ch * N + i], ch 0..3 = cx, cy, w, h, 4 + cls = score.
void set_prop(std::vector<float> &o, int i, float cx, float cy, float w, float h,
              int cls, float score) {
    o[0 * N + i] = cx; o[1 * N + i] = cy; o[2 * N + i] = w; o[3 * N + i] = h;
    o[(4 + cls) * N + i] = score;
}

std::vector<float> blank() { return std::vector<float>((size_t)(4 + C) * N, 0.f); }
}

TEST(decode_single) {
    auto o = blank();
    set_prop(o, 0, 320, 320, 100, 200, 5, 0.9f);
    YoloParams p;
    LetterboxInfo lb{1.f, 0.f, 0.f, 640, 640};
    NvmmDetObject out[NVMM_META_MAX_OBJECTS];
    bool trunc = false;
    uint32_t n = nvmm::yolo_parse(o.data(), p, lb, out, &trunc);
    ASSERT_EQ(n, 1u);
    ASSERT_EQ(out[0].class_id, 5);
    ASSERT_TRUE(std::strcmp(out[0].label, "bus") == 0);
    ASSERT_NEAR(out[0].confidence, 0.9f, 1e-5);
    ASSERT_NEAR(out[0].left, 270.f, 0.5);
    ASSERT_NEAR(out[0].top, 220.f, 0.5);
    ASSERT_NEAR(out[0].width, 100.f, 0.5);
    ASSERT_NEAR(out[0].height, 200.f, 0.5);
    ASSERT_TRUE(!trunc);
}

TEST(letterbox_unmap) {
    auto o = blank();
    set_prop(o, 0, 400, 240, 100, 80, 2, 0.8f);
    YoloParams p;
    LetterboxInfo lb{0.5f, 80.f, 40.f, 4096, 4096};
    NvmmDetObject out[NVMM_META_MAX_OBJECTS];
    uint32_t n = nvmm::yolo_parse(o.data(), p, lb, out, nullptr);
    ASSERT_EQ(n, 1u);
    ASSERT_NEAR(out[0].left, 540.f, 0.5);
    ASSERT_NEAR(out[0].top, 320.f, 0.5);
    ASSERT_NEAR(out[0].width, 200.f, 0.5);
    ASSERT_NEAR(out[0].height, 160.f, 0.5);
}

TEST(conf_threshold) {
    auto o = blank();
    set_prop(o, 0, 320, 320, 100, 100, 5, 0.20f);
    YoloParams p;
    LetterboxInfo lb{1.f, 0.f, 0.f, 640, 640};
    NvmmDetObject out[NVMM_META_MAX_OBJECTS];
    ASSERT_EQ(nvmm::yolo_parse(o.data(), p, lb, out, nullptr), 0u);
}

TEST(nms_same_class_suppresses) {
    auto o = blank();
    set_prop(o, 0, 320, 320, 100, 200, 5, 0.90f);
    set_prop(o, 1, 325, 322, 100, 200, 5, 0.80f);
    YoloParams p;
    LetterboxInfo lb{1.f, 0.f, 0.f, 640, 640};
    NvmmDetObject out[NVMM_META_MAX_OBJECTS];
    uint32_t n = nvmm::yolo_parse(o.data(), p, lb, out, nullptr);
    ASSERT_EQ(n, 1u);
    ASSERT_NEAR(out[0].confidence, 0.90f, 1e-5);
}

TEST(nms_cross_class_keeps_both) {
    auto o = blank();
    set_prop(o, 0, 320, 320, 100, 200, 5, 0.90f);
    set_prop(o, 1, 322, 321, 100, 200, 2, 0.85f);
    YoloParams p;
    LetterboxInfo lb{1.f, 0.f, 0.f, 640, 640};
    NvmmDetObject out[NVMM_META_MAX_OBJECTS];
    ASSERT_EQ(nvmm::yolo_parse(o.data(), p, lb, out, nullptr), 2u);
}

TEST(truncation_flag) {
    auto o = blank();
    const int M = NVMM_META_MAX_OBJECTS + 40;
    for (int i = 0; i < M; i++) {
        float cx = (float)(i % 40) * 20 + 5, cy = (float)(i / 40) * 20 + 5;
        set_prop(o, i, cx, cy, 5, 5, i % C, 0.9f);
    }
    YoloParams p;
    LetterboxInfo lb{1.f, 0.f, 0.f, 4096, 4096};
    NvmmDetObject out[NVMM_META_MAX_OBJECTS];
    bool trunc = false;
    uint32_t n = nvmm::yolo_parse(o.data(), p, lb, out, &trunc);
    ASSERT_EQ(n, (uint32_t)NVMM_META_MAX_OBJECTS);
    ASSERT_TRUE(trunc);
}

using nvmm::Tracker;
using nvmm::TrackerParams;

namespace {
NvmmDetObject det(float x, float y, float w, float h, int cls = 0) {
    NvmmDetObject o{};
    o.left = x; o.top = y; o.width = w; o.height = h;
    o.class_id = cls; o.confidence = 0.9f; o.tracker_id = 0;
    return o;
}
}

TEST(first_frame_assigns_unique_ids) {
    Tracker t;
    std::vector<NvmmDetObject> f = {det(0, 0, 10, 10), det(100, 100, 10, 10)};
    t.update(f.data(), f.size());
    ASSERT_TRUE(f[0].tracker_id != 0);
    ASSERT_TRUE(f[1].tracker_id != 0);
    ASSERT_TRUE(f[0].tracker_id != f[1].tracker_id);
}

TEST(same_object_keeps_id) {
    Tracker t;
    std::vector<NvmmDetObject> f1 = {det(50, 50, 20, 20)};
    t.update(f1.data(), f1.size());
    uint64_t id = f1[0].tracker_id;

    std::vector<NvmmDetObject> f2 = {det(52, 51, 20, 20)};
    t.update(f2.data(), f2.size());
    ASSERT_EQ(f2[0].tracker_id, id);
}

TEST(new_object_gets_new_id) {
    Tracker t;
    std::vector<NvmmDetObject> f1 = {det(0, 0, 20, 20)};
    t.update(f1.data(), f1.size());
    uint64_t id1 = f1[0].tracker_id;

    std::vector<NvmmDetObject> f2 = {det(0, 0, 20, 20), det(500, 500, 20, 20)};
    t.update(f2.data(), f2.size());
    ASSERT_EQ(f2[0].tracker_id, id1);
    ASSERT_TRUE(f2[1].tracker_id != 0);
    ASSERT_TRUE(f2[1].tracker_id != id1);
}

TEST(different_class_not_matched) {
    Tracker t;
    std::vector<NvmmDetObject> f1 = {det(50, 50, 20, 20, 1)};
    t.update(f1.data(), f1.size());
    uint64_t id = f1[0].tracker_id;

    std::vector<NvmmDetObject> f2 = {det(50, 50, 20, 20, 2)};
    t.update(f2.data(), f2.size());
    ASSERT_TRUE(f2[0].tracker_id != id);
}

TEST(low_overlap_not_matched) {
    Tracker t{TrackerParams{0.5f, 30}};
    std::vector<NvmmDetObject> f1 = {det(0, 0, 20, 20)};
    t.update(f1.data(), f1.size());
    uint64_t id = f1[0].tracker_id;

    std::vector<NvmmDetObject> f2 = {det(15, 15, 20, 20)};
    t.update(f2.data(), f2.size());
    ASSERT_TRUE(f2[0].tracker_id != id);
}

TEST(track_expires_after_max_age) {
    Tracker t{TrackerParams{0.3f, 2}};
    std::vector<NvmmDetObject> f1 = {det(0, 0, 20, 20)};
    t.update(f1.data(), f1.size());
    uint64_t id = f1[0].tracker_id;

    for (int i = 0; i < 3; i++) t.update(nullptr, 0);
    ASSERT_EQ(t.live_tracks(), (std::size_t)0);

    std::vector<NvmmDetObject> f2 = {det(0, 0, 20, 20)};
    t.update(f2.data(), f2.size());
    ASSERT_TRUE(f2[0].tracker_id != id);
}

TEST(reset_clears_tracks) {
    Tracker t;
    std::vector<NvmmDetObject> f1 = {det(0, 0, 20, 20)};
    t.update(f1.data(), f1.size());
    ASSERT_TRUE(t.live_tracks() > 0);
    t.reset();
    ASSERT_EQ(t.live_tracks(), (std::size_t)0);
}

using nvmm::MotionEntry;

namespace {
constexpr int GRID = 4, MVW = 8, MVH = 8;
constexpr int FW = MVW * GRID, FH = MVH * GRID;

/// Flow is int16 S10.5 (px = raw / 32). Cells x >= 4 move 2 px right.
std::vector<int16_t> half_moving_field() {
    std::vector<int16_t> f(MVW * MVH * 2, 0);
    for (int y = 0; y < MVH; y++)
        for (int x = MVW / 2; x < MVW; x++)
            f[(y * MVW + x) * 2 + 0] = 64;
    return f;
}

NvmmDetObject box(float x, float y, float w, float h) {
    NvmmDetObject o{};
    o.left = x; o.top = y; o.width = w; o.height = h;
    return o;
}
}

TEST(static_vs_moving_box) {
    auto f = half_moving_field();
    NvmmDetObject objs[2] = { box(0, 0, 12, 12), box(20, 0, 12, 12) };
    MotionEntry out[2];
    ASSERT_EQ(nvmm::compute_box_motion(f.data(), MVW, MVH, GRID, FW, FH,
                                       objs, 2, 1.0f, out), 2u);
    ASSERT_NEAR(out[0].mean_px, 0.0f, 1e-4);
    ASSERT_EQ(out[0].moving, 0u);
    ASSERT_NEAR(out[1].mean_px, 2.0f, 1e-4);
    ASSERT_EQ(out[1].moving, 1u);
}

TEST(magnitude_is_euclidean) {
    std::vector<int16_t> f(MVW * MVH * 2, 0);
    f[0] = 96; f[1] = 128;
    NvmmDetObject o = box(0, 0, GRID, GRID);
    MotionEntry e;
    nvmm::compute_box_motion(f.data(), MVW, MVH, GRID, FW, FH, &o, 1, 1.0f, &e);
    ASSERT_NEAR(e.mean_px, 5.0f, 1e-4);
    ASSERT_EQ(e.moving, 1u);
}

TEST(straddling_box_averages) {
    auto f = half_moving_field();
    NvmmDetObject o = box(8, 8, 16, 8);
    MotionEntry e;
    nvmm::compute_box_motion(f.data(), MVW, MVH, GRID, FW, FH, &o, 1, 0.5f, &e);
    ASSERT_NEAR(e.mean_px, 1.0f, 1e-4);
    ASSERT_EQ(e.moving, 1u);
}

TEST(out_of_frame_clamped) {
    auto f = half_moving_field();
    NvmmDetObject o = box(-10, -10, 14, 14);
    MotionEntry e;
    ASSERT_EQ(nvmm::compute_box_motion(f.data(), MVW, MVH, GRID, FW, FH,
                                       &o, 1, 1.0f, &e), 1u);
    ASSERT_NEAR(e.mean_px, 0.0f, 1e-4);
}

TEST(tiny_box_reads_a_cell) {
    auto f = half_moving_field();
    NvmmDetObject o = box(29, 29, 1, 1);
    MotionEntry e;
    nvmm::compute_box_motion(f.data(), MVW, MVH, GRID, FW, FH, &o, 1, 1.0f, &e);
    ASSERT_NEAR(e.mean_px, 2.0f, 1e-4);
    ASSERT_EQ(e.moving, 1u);
}

TEST(zero_objects) {
    auto f = half_moving_field();
    ASSERT_EQ(nvmm::compute_box_motion(f.data(), MVW, MVH, GRID, FW, FH,
                                       nullptr, 0, 1.0f, nullptr), 0u);
}

using nvmm::ClassResult;
using nvmm::SecondaryCache;
using nvmm::SecondaryCacheParams;

namespace {
ClassResult result(int32_t id, float conf, const char *label) {
    ClassResult r;
    r.class_id = id;
    r.confidence = conf;
    snprintf(r.label, sizeof r.label, "%s", label);
    return r;
}
}

TEST(unknown_track_is_due) {
    SecondaryCache c({10, 60});
    ASSERT_TRUE(c.due(7, 100));
    c.store(7, result(3, 0.9f, "cat"), 100);
    ASSERT_TRUE(!c.due(7, 100));
    ASSERT_TRUE(!c.due(7, 109));
    ASSERT_TRUE(c.due(7, 110));
}

TEST(interval_one_reinfers_every_frame) {
    SecondaryCache c({1, 60});
    c.store(1, result(0, 0.5f, "a"), 5);
    ASSERT_TRUE(!c.due(1, 5));
    ASSERT_TRUE(c.due(1, 6));
}

TEST(lookup_returns_stored_result) {
    SecondaryCache c({10, 60});
    c.store(42, result(2, 0.8f, "dog"), 0);
    const ClassResult *r = c.lookup(42, 3);
    ASSERT_NOT_NULL(r);
    ASSERT_EQ(r->class_id, 2);
    ASSERT_NEAR(r->confidence, 0.8f, 1e-6);
    ASSERT_TRUE(strcmp(r->label, "dog") == 0);
    ASSERT_TRUE(c.lookup(43, 3) == nullptr);
}

TEST(expiry_drops_unseen_tracks) {
    SecondaryCache c({10, 20});
    c.store(1, result(0, 0.5f, "a"), 0);
    c.store(2, result(1, 0.6f, "b"), 0);
    ASSERT_EQ(c.size(), (size_t)2);

    c.lookup(1, 15);
    c.expire(25);
    ASSERT_EQ(c.size(), (size_t)1);
    ASSERT_NOT_NULL(c.lookup(1, 25));
    ASSERT_TRUE(c.lookup(2, 25) == nullptr);
}

TEST(infer_cadence_does_not_block_expiry) {
    SecondaryCache c({10, 20});
    c.store(1, result(0, 0.5f, "a"), 0);
    c.expire(21);
    ASSERT_EQ(c.size(), (size_t)0);
    ASSERT_TRUE(c.due(1, 21));
}

TEST(reset_clears_cache) {
    SecondaryCache c({10, 60});
    c.store(1, result(0, 0.5f, "a"), 0);
    c.reset();
    ASSERT_EQ(c.size(), (size_t)0);
    ASSERT_TRUE(c.due(1, 0));
}

TEST(store_overwrites) {
    SecondaryCache c({10, 60});
    c.store(1, result(0, 0.5f, "cat"), 0);
    c.store(1, result(4, 0.9f, "dog"), 10);
    const ClassResult *r = c.lookup(1, 10);
    ASSERT_NOT_NULL(r);
    ASSERT_EQ(r->class_id, 4);
    ASSERT_TRUE(strcmp(r->label, "dog") == 0);
    ASSERT_TRUE(!c.due(1, 19));
}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
