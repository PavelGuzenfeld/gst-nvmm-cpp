#include <gst/gst.h>
#include <gst/check/gstharness.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <glib.h>

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

uint32_t hash2(uint32_t x, uint32_t y, uint32_t salt)
{
    uint32_t h = x * 374761393u + y * 668265263u + salt * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}

Nv12Painter scene(int frame)
{
    const int mx = kMoverX0 + kMoverStep * frame;
    return [mx](int x, int y) -> Yuv {
        if (x >= mx && x < mx + kMoverSide && y >= kMoverY && y < kMoverY + kMoverSide) {
            const bool on = hash2((uint32_t)(x - mx) / 4, (uint32_t)(y - kMoverY) / 4, 7) & 1u;
            return Yuv{(uint8_t)(on ? 235 : 16), 128, 128};
        }
        return Yuv{(uint8_t)(70 + hash2((uint32_t)x / 6, (uint32_t)y / 6, 3) % 110), 128, 128};
    };
}

NvmmDetObject det(float left, float top, float conf, const char *label)
{
    NvmmDetObject o{};
    o.left = left; o.top = top; o.width = kMoverSide; o.height = kMoverSide;
    o.class_id = 0;
    o.confidence = conf;
    g_strlcpy(o.label, label, NVMM_META_LABEL_LEN);
    return o;
}

NvmmDetObject mover_det(int frame)
{
    return det((float)(kMoverX0 + kMoverStep * frame), (float)kMoverY, 0.6f, "mover");
}

/// The static det goes first and scores higher, so neither "first wins" nor
/// "most confident wins" can pass for the motion gate.
GstBuffer *frame_with_two_dets(int frame)
{
    GstBuffer *buf = nvmm_nv12_buffer(kW, kH, scene(frame));
    static NvmmFrameMeta fm;
    std::memset(&fm, 0, sizeof fm);
    fm.infer_width = kW;
    fm.infer_height = kH;
    fm.objects[fm.num_objects++] = det(kStaticLeft, kStaticTop, 0.9f, "static");
    fm.objects[fm.num_objects++] = mover_det(frame);
    gst_buffer_add_nvmm_det_meta(buf, &fm);
    return buf;
}

TEST(only_the_detection_on_the_independently_moving_patch_passes_the_gate) {
    GstElement *gate = gst_element_factory_make("nvmmdetgate", nullptr);
    ASSERT_NOT_NULL(gate);
    g_object_set(gate, "engine-dir", engines.dir.c_str(), NULL);
    GstHarness *h = gst_harness_new_with_element(gate, "sink", "src");
    gst_harness_set_src_caps_str(h, "video/x-raw(memory:NVMM),format=NV12,width=480,"
                                    "height=270,framerate=30/1");
    std::vector<guint32> counts;
    std::vector<NvmmDetObject> kept;
    for (int f = 0; f < kFrames; f++) {
        ASSERT_EQ(gst_harness_push(h, frame_with_two_dets(f)), GST_FLOW_OK);
        GstBuffer *out = gst_harness_pull(h);
        const GstNvmmDetMeta *m = gst_buffer_get_nvmm_det_meta(out);
        ASSERT_NOT_NULL(m);
        counts.push_back(m->num_objects);
        kept.push_back(m->num_objects ? m->objects[0] : NvmmDetObject{});
        gst_buffer_unref(out);
    }
    gst_harness_teardown(h);
    gst_object_unref(gate);

    for (int f = 0; f < kFirstConfirmedFrame; f++) ASSERT_EQ(counts[f], 0u);
    for (int f = kFirstConfirmedFrame; f < kFrames; f++) {
        ASSERT_EQ(counts[f], 1u);
        const NvmmDetObject want = mover_det(f);
        ASSERT_EQ(std::string(kept[f].label), std::string("mover"));
        ASSERT_EQ(kept[f].left, want.left);
        ASSERT_EQ(kept[f].top, want.top);
        ASSERT_EQ(kept[f].width, want.width);
        ASSERT_EQ(kept[f].height, want.height);
        ASSERT_EQ(kept[f].confidence, want.confidence);
    }
}

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
