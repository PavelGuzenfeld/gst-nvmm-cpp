/// KalmanBox — 8D constant-velocity Kalman filter for a single bounding box,
/// a faithful C++ port of SAMURAI's samurai/utils/kalman_filter.py (itself the
/// SORT/ByteTrack filter). State = (cx, cy, w, h, vx, vy, vw, vh); the box
/// (cx,cy,w,h) is observed directly (H = [I4 | 0]).
///
/// Pure host C++ — no Eigen, no CUDA. Used by nvmmsamurai (SAMURAI kf_score /
/// kf_box) and by nvmmfusekf (the master fusion KF). All measurements/boxes are
/// center-form (cx, cy, w, h) in frame pixel coords.
#pragma once

#include <array>

namespace nvmm {

class KalmanBox {
public:
    using Vec8 = std::array<double, 8>;
    using Mat8 = std::array<std::array<double, 8>, 8>;

    /// Initialize a fresh track from a center-form measurement; velocities 0.
    void initiate(double cx, double cy, double w, double h);

    /// Constant-velocity prediction by `dt` (frames or seconds).
    void predict(double dt);

    /// Measurement correction with a center-form box.
    void update(double cx, double cy, double w, double h);

    /// Squared Mahalanobis distance from the current (projected) state to a
    /// center-form measurement — for gating (compare to chi2inv95[4]=9.4877).
    double gating_distance(double cx, double cy, double w, double h) const;

    bool   initiated() const { return initiated_; }
    /// Shift the tracked center by (dx,dy) — global/camera-motion compensation
    /// (GMC): the target's image position moved with the camera this frame.
    void   shift(double dx, double dy) { mean_[0] += dx; mean_[1] += dy; }
    /// Current box estimate (center form).
    void   box(double &cx, double &cy, double &w, double &h) const {
        cx = mean_[0]; cy = mean_[1]; w = mean_[2]; h = mean_[3];
    }
    const Vec8 &mean() const { return mean_; }
    const Mat8 &covariance() const { return cov_; }

private:
    void project(std::array<double, 4> &pmean,
                 std::array<std::array<double, 4>, 4> &pcov) const;

    Vec8 mean_{};
    Mat8 cov_{};
    bool initiated_ = false;
    static constexpr double kStdWPos = 1.0 / 20.0;
    static constexpr double kStdWVel = 1.0 / 160.0;
};

}  // namespace nvmm
