/// KalmanBox implementation — see kalman_box.hpp. Port of kalman_filter.py.

#include "kalman_box.hpp"

#include <cmath>

namespace nvmm {

namespace {

/// Cholesky factor L (lower) of a symmetric positive-definite 4x4 A: A = L L^T.
static void chol4(const std::array<std::array<double, 4>, 4> &A,
                  std::array<std::array<double, 4>, 4> &L)
{
    for (auto &row : L) row.fill(0.0);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[i][j];
            for (int k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
            if (i == j) L[i][j] = std::sqrt(s > 0.0 ? s : 0.0);
            else        L[i][j] = (L[j][j] != 0.0) ? s / L[j][j] : 0.0;
        }
    }
}

/// Solve A x = b for a single 4-vector via the precomputed Cholesky factor L.
static std::array<double, 4>
chol_solve_vec(const std::array<std::array<double, 4>, 4> &L,
               const std::array<double, 4> &b)
{
    std::array<double, 4> y{}, x{};
    for (int i = 0; i < 4; ++i) {            // forward: L y = b
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= L[i][k] * y[k];
        y[i] = (L[i][i] != 0.0) ? s / L[i][i] : 0.0;
    }
    for (int i = 3; i >= 0; --i) {           // back: L^T x = y
        double s = y[i];
        for (int k = i + 1; k < 4; ++k) s -= L[k][i] * x[k];
        x[i] = (L[i][i] != 0.0) ? s / L[i][i] : 0.0;
    }
    return x;
}

}  // namespace

void KalmanBox::initiate(double cx, double cy, double w, double h)
{
    mean_ = {cx, cy, w, h, 0, 0, 0, 0};
    const double sp = std_w_pos_, sv = std_w_vel_;
    const double std[8] = {
        2 * sp * w, 2 * sp * h, 2 * sp * w, 2 * sp * h,
        10 * sv * w, 10 * sv * h, 10 * sv * w, 10 * sv * h};
    for (auto &row : cov_) row.fill(0.0);
    for (int i = 0; i < 8; ++i) cov_[i][i] = std[i] * std[i];
    initiated_ = true;
}

void KalmanBox::predict(double dt)
{
    // Process noise Q uses the PRE-prediction w,h (matches kalman_filter.py,
    // which computes std from mean[2],mean[3] before advancing the mean).
    const double sp = std_w_pos_, sv = std_w_vel_;
    const double q[8] = {
        sp * mean_[2], sp * mean_[3], sp * mean_[2], sp * mean_[3],
        sv * mean_[2], sv * mean_[3], sv * mean_[2], sv * mean_[3]};

    // mean = F mean : cx,cy,w,h += dt * (vx,vy,vw,vh)
    for (int i = 0; i < 4; ++i) mean_[i] += dt * mean_[4 + i];

    // cov = F cov F^T + Q.  F = I with F[i][4+i]=dt (i<4).
    // (F cov)[i][j] = cov[i][j] + (i<4 ? dt*cov[4+i][j] : 0)
    Mat8 fc{};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            fc[i][j] = cov_[i][j] + (i < 4 ? dt * cov_[4 + i][j] : 0.0);
    // (fc F^T)[i][j] = fc[i][j] + (j<4 ? dt*fc[i][4+j] : 0)
    Mat8 nc{};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            nc[i][j] = fc[i][j] + (j < 4 ? dt * fc[i][4 + j] : 0.0);

    for (int i = 0; i < 8; ++i) nc[i][i] += q[i] * q[i];
    cov_ = nc;
}

void KalmanBox::project(std::array<double, 4> &pmean,
                        std::array<std::array<double, 4>, 4> &pcov) const
{
    for (int i = 0; i < 4; ++i) pmean[i] = mean_[i];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) pcov[i][j] = cov_[i][j];
    const double sp = std_w_pos_;
    const double r[4] = {sp * mean_[2], sp * mean_[3], sp * mean_[2], sp * mean_[3]};
    for (int i = 0; i < 4; ++i) pcov[i][i] += r[i] * r[i];
}

void KalmanBox::update(double cx, double cy, double w, double h)
{
    std::array<double, 4> pmean;
    std::array<std::array<double, 4>, 4> pcov;
    project(pmean, pcov);

    std::array<std::array<double, 4>, 4> L;
    chol4(pcov, L);

    // K (8x4) = cov H^T pcov^{-1}, where cov H^T = cov[:, :4].
    // Solve pcov K^T = (cov[:, :4])^T : per state-row r, solve pcov * x = B_r
    // where B_r = cov[r][0..3]; then K[r] = x.
    std::array<std::array<double, 4>, 8> K;
    for (int r = 0; r < 8; ++r) {
        std::array<double, 4> b = {cov_[r][0], cov_[r][1], cov_[r][2], cov_[r][3]};
        K[r] = chol_solve_vec(L, b);
    }

    const double innov[4] = {cx - pmean[0], cy - pmean[1], w - pmean[2], h - pmean[3]};
    // new_mean = mean + K innov
    for (int r = 0; r < 8; ++r) {
        double s = 0.0;
        for (int j = 0; j < 4; ++j) s += K[r][j] * innov[j];
        mean_[r] += s;
    }
    // new_cov = cov - K pcov K^T
    // M = pcov K^T  (4x8): M[a][c] = sum_b pcov[a][b] K[c][b]
    std::array<std::array<double, 8>, 4> M{};
    for (int a = 0; a < 4; ++a)
        for (int c = 0; c < 8; ++c) {
            double s = 0.0;
            for (int b = 0; b < 4; ++b) s += pcov[a][b] * K[c][b];
            M[a][c] = s;
        }
    // (K M)[i][j] = sum_a K[i][a] M[a][j]
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            double s = 0.0;
            for (int a = 0; a < 4; ++a) s += K[i][a] * M[a][j];
            cov_[i][j] -= s;
        }
}

double KalmanBox::gating_distance(double cx, double cy, double w, double h) const
{
    std::array<double, 4> pmean;
    std::array<std::array<double, 4>, 4> pcov;
    project(pmean, pcov);
    std::array<std::array<double, 4>, 4> L;
    chol4(pcov, L);
    const std::array<double, 4> d = {cx - pmean[0], cy - pmean[1], w - pmean[2], h - pmean[3]};
    // squared Mahalanobis = || L^{-1} d ||^2 (forward solve only)
    std::array<double, 4> z{};
    for (int i = 0; i < 4; ++i) {
        double s = d[i];
        for (int k = 0; k < i; ++k) s -= L[i][k] * z[k];
        z[i] = (L[i][i] != 0.0) ? s / L[i][i] : 0.0;
    }
    double m = 0.0;
    for (int i = 0; i < 4; ++i) m += z[i] * z[i];
    return m;
}

}  // namespace nvmm
