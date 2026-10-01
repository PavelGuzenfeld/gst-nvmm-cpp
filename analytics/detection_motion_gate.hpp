#pragma once
#include <cstdint>
#include <vector>

#include "dual_homography.hpp"
#include "image.hpp"
#include "image_ops.hpp"
#include "low_texture_motion.hpp"
#include "persistence_gate.hpp"

namespace nvmm {
namespace motion {

struct MotionGateParams {
    DualHomographyParams      dualh;
    LowTextureMotionParams    lowtex;
    track::PersistenceParams  persist;
    float dualh_thresh = 12.f;
    float lowtex_thresh = 8.f;
    bool  use_dualh = true;
    bool  use_lowtex = true;
    /// Half-window in pixels for sampling a motion map at a box centre.
    int   sample_radius = 4;
};

/// Detector boxes confirmed by independent motion: dual_homography for textured
/// backgrounds, low_texture_motion for sky or water. Motion runs only while searching.
class MovingObjectGate {
public:
    explicit MovingObjectGate(const MotionGateParams &p = {}) : p_(p), gate_(p.persist) {}

    bool locked() const { return gate_.locked(); }

    /// `cur`, `ref_a`, `ref_b` are single-channel u8 frames: current and two past.
    /// Returns the index into `boxes` of the confirmed moving detection, or -1.
    int update(const std::vector<track::Detection> &boxes,
               img::View<const uint8_t> cur, img::View<const uint8_t> ref_a,
               img::View<const uint8_t> ref_b)
    {
        std::vector<track::Detection> dets = boxes;

        if (!gate_.locked()) {
            img::Image<float> dh, lt;
            if (p_.use_dualh) dh = independent_motion_residual(cur, ref_a, ref_b, p_.dualh);
            if (p_.use_lowtex) lt = low_texture_motion(cur, ref_a, ref_b, p_.lowtex);
            for (auto &d : dets) {
                const bool moved =
                    img::window_max(dh.view(), d.cx, d.cy, p_.sample_radius) >= p_.dualh_thresh ||
                    img::window_max(lt.view(), d.cx, d.cy, p_.sample_radius) >= p_.lowtex_thresh;
                d.supported = moved;
            }
        }
        return gate_.update(dets);
    }

    void reset() { gate_.reset(); }

private:
    MotionGateParams p_;
    track::PersistenceGate gate_;
};

}
}
