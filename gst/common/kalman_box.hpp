#pragma once

#include <array>

namespace nvmm {

/// Port of SAMURAI's kalman_filter.py (the SORT/ByteTrack filter). State is
/// (cx, cy, w, h, vx, vy, vw, vh); boxes are center-form in frame pixels.
class KalmanBox {
public:
    using Vec8 = std::array<double, 8>;
    using Mat8 = std::array<std::array<double, 8>, 8>;

    void initiate(double cx, double cy, double w, double h);

    /// `dt` in frames or seconds.
    void predict(double dt);

    void update(double cx, double cy, double w, double h);

    /// Squared Mahalanobis distance; gate against chi2inv95[4] = 9.4877.
    double gating_distance(double cx, double cy, double w, double h) const;

    bool   initiated() const { return initiated_; }
    /// Camera-motion compensation: the target moved with the camera this frame.
    void   shift(double dx, double dy) { mean_[0] += dx; mean_[1] += dy; }
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

}
