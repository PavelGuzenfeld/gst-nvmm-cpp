/// Unit test for get_view_around_bbox (gst/nvmmsamurai/samurai_view.hpp).
/// Golden values produced by the real Python samurai.tracker.get_view_around_bbox
/// on a 1920x1080 frame, crop=512 (view_golden.py). Header-only geometry, so
/// this builds off-target with no CUDA/GStreamer.
#include <cmath>

#include "test_harness.h"

#include "samurai_view.hpp"

using nvmm::get_view_around_bbox;
using nvmm::SamuraiView;

namespace {
constexpr int W = 1920, H = 1080, CROP = 512;

struct Case {
    float bx, by, bw, bh;          // box top-left + size
    float ex, ey;                  // expected view x,y (w,h always == CROP)
};

// (x1,y1,w,h) -> expected (x,y); see view_golden.py.
const Case kCases[] = {
    {1015, 446, 18, 12, 768, 196},     // real seed box, mid-frame
    {0,    0,   10, 10, 0,   0},       // top-left corner (clamped)
    {1910, 1070,10, 10, 1408,568},     // bottom-right corner (clamped)
    {940,  520, 40, 40, 704, 284},     // near center
    {5,    500, 10, 20, 0,   254},     // left edge
    {1900, 500, 15, 20, 1408,254},     // right edge
    {500,  2,   20, 10, 254, 0},       // top edge
    {500,  1065,20, 13, 254, 568},     // bottom edge
};
}  // namespace

TEST(view_matches_python_golden) {
    for (const Case &c : kCases) {
        SamuraiView v = get_view_around_bbox(c.bx, c.by, c.bw, c.bh, CROP, W, H);
        ASSERT_NEAR(v.x, c.ex, 1e-4);
        ASSERT_NEAR(v.y, c.ey, 1e-4);
        ASSERT_NEAR(v.width, (float)CROP, 1e-4);
        ASSERT_NEAR(v.height, (float)CROP, 1e-4);
    }
}

TEST(view_stays_in_frame) {
    for (const Case &c : kCases) {
        SamuraiView v = get_view_around_bbox(c.bx, c.by, c.bw, c.bh, CROP, W, H);
        ASSERT_TRUE(v.x >= 0.f && v.y >= 0.f);
        ASSERT_TRUE(v.x + v.width <= (float)W);
        ASSERT_TRUE(v.y + v.height <= (float)H);
    }
}

TEST(view_clamps_oversized_crop) {
    // crop larger than frame -> clamp to frame dimensions.
    SamuraiView v = get_view_around_bbox(100, 100, 10, 10, 4096, W, H);
    ASSERT_NEAR(v.width, (float)W, 1e-4);
    ASSERT_NEAR(v.height, (float)H, 1e-4);
}

int main() {
    printf("=== get_view_around_bbox Tests (vs SAMURAI Python reference) ===\n");
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
