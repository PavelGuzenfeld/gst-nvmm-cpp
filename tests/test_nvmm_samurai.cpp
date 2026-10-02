#include <gst/gst.h>
#include <gst/check/gstharness.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <glib.h>

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

Nv12Painter square_at(int frame)
{
    const int x0 = kX0 + kStepX * frame, y0 = kY0 + kStepY * frame;
    return [x0, y0](int x, int y) {
        return (x >= x0 && x < x0 + kSide && y >= y0 && y < y0 + kSide) ? kRed : kGrey;
    };
}

std::vector<GstNvmmTrackMeta> track_square(int max_kf)
{
    GstElement *e = gst_element_factory_make("nvmmsamurai", nullptr);
    if (!e) throw std::runtime_error("nvmmsamurai not found");
    const std::string consts = engines.dir + "/samurai_consts.bin";
    const std::string roi = std::to_string(kX0) + "," + std::to_string(kY0) + "," +
                            std::to_string(kSide) + "," + std::to_string(kSide);
    g_object_set(e, "engine-dir", engines.dir.c_str(), "consts-file", consts.c_str(),
                 "seed-roi", roi.c_str(), "max-kf", max_kf, NULL);
    GstHarness *h = gst_harness_new_with_element(e, "sink", "src");
    gst_harness_set_src_caps_str(h, "video/x-raw(memory:NVMM),format=NV12,width=768,"
                                    "height=512,framerate=30/1");
    std::vector<GstNvmmTrackMeta> out;
    for (int f = 0; f < kFrames; f++) {
        if (gst_harness_push(h, nvmm_nv12_buffer(kW, kH, square_at(f))) != GST_FLOW_OK)
            throw std::runtime_error("push failed");
        GstBuffer *b = gst_harness_pull(h);
        const GstNvmmTrackMeta *m = gst_buffer_get_nvmm_track_meta(b);
        if (!m) throw std::runtime_error("no track meta");
        out.push_back(*m);
        gst_buffer_unref(b);
    }
    gst_harness_teardown(h);
    gst_object_unref(e);
    return out;
}

void assert_box_on_square(const GstNvmmTrackMeta &m, int f)
{
    ASSERT_NEAR(m.left, kX0 + kStepX * f, kEdgeTolPx);
    ASSERT_NEAR(m.top, kY0 + kStepY * f, kEdgeTolPx);
    ASSERT_NEAR(m.width, kSide, kEdgeTolPx);
    ASSERT_NEAR(m.height, kSide, kEdgeTolPx);
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

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
