/// Detector ∩ independent-motion gate — confirm the one detection that is moving.
///
/// A detector on a panning camera fires on the target AND on static structure. This
/// composes the pieces in this folder into the full confirmation:
///
///   detector boxes ──┐
///                     ├─► is each box on INDEPENDENT motion? (dual-homography residual
///   current + 2 refs ─┘     for textured backgrounds, OR low-texture frame-diff for
///                           open-sky/water backgrounds)
///                                     │ per-box "supported" flag
///                                     ▼
///                           persistence gate (must persist + stay supported K frames)
///                                     │
///                                     ▼  index of the confirmed, independently-moving box
///
/// Self-latching: the expensive motion step runs only while SEARCHING; once a box is
/// confirmed the persistence gate tracks it by association until lost, then re-engages.
///
/// OpenCV-only, header-only. Composes dual_homography.hpp + low_texture_motion.hpp +
/// persistence_gate.hpp.
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

#include "dual_homography.hpp"
#include "low_texture_motion.hpp"
#include "persistence_gate.hpp"

namespace nvmm {
namespace motion {

struct MotionGateParams {
    DualHomographyParams      dualh;
    LowTextureMotionParams    lowtex;
    track::PersistenceParams  persist;
    float dualh_thresh = 12.f;   // dual-homography residual at a box >= this => supported
    float lowtex_thresh = 8.f;   // low-texture motion at a box >= this => supported
    bool  use_dualh = true;      // textured-background movers
    bool  use_lowtex = true;     // open-sky / low-texture-background movers
    int   sample_radius = 4;     // half-window (px) for sampling a motion map at a box centre
};

class MovingObjectGate {
public:
    explicit MovingObjectGate(const MotionGateParams &p = {}) : p_(p), gate_(p.persist) {}

    bool locked() const { return gate_.locked(); }

    /// One frame. `boxes` are detector centres (cx,cy,conf); `cur/ref_a/ref_b` are
    /// single-channel CV_8U frames (e.g. current and two past frames). Returns the
    /// index into `boxes` of the confirmed independently-moving detection, or -1.
    int update(const std::vector<track::Detection> &boxes,
               const cv::Mat &cur, const cv::Mat &ref_a, const cv::Mat &ref_b)
    {
        std::vector<track::Detection> dets = boxes;   // copy: we set the `supported` field

        if (!gate_.locked()) {                        // search phase: compute motion evidence
            cv::Mat dh = p_.use_dualh
                ? independent_motion_residual(cur, ref_a, ref_b, p_.dualh) : cv::Mat();
            cv::Mat lt = p_.use_lowtex
                ? low_texture_motion(cur, ref_a, ref_b, p_.lowtex) : cv::Mat();
            for (auto &d : dets) {
                bool moved = (sample(dh, d.cx, d.cy) >= p_.dualh_thresh) ||
                             (sample(lt, d.cx, d.cy) >= p_.lowtex_thresh);
                d.supported = moved;
            }
        }
        return gate_.update(dets);                    // persistence + latch
    }

    void reset() { gate_.reset(); }

private:
    float sample(const cv::Mat &m, float cx, float cy) const {
        if (m.empty()) return 0.f;
        int r = p_.sample_radius;
        int x0 = std::max(0, (int)cx - r), y0 = std::max(0, (int)cy - r);
        int x1 = std::min(m.cols, (int)cx + r), y1 = std::min(m.rows, (int)cy + r);
        if (x1 <= x0 || y1 <= y0) return 0.f;
        double mx = 0; cv::minMaxLoc(m(cv::Rect(x0, y0, x1 - x0, y1 - y0)), nullptr, &mx);
        return (float)mx;
    }

    MotionGateParams p_;
    track::PersistenceGate gate_;
};

}  // namespace motion
}  // namespace nvmm
