/// Robust homography estimation for the dual-homography residual — the part of
/// cv::findHomography(..., RANSAC) this project actually uses.
///
/// Hartley-normalised 4-point DLT, with the null vector taken from a cyclic
/// Jacobi eigen-decomposition of the 9x9 normal matrix AᵀA (no general SVD
/// needed at this size), inside a fixed-seed RANSAC loop with the adaptive
/// iteration bound. Deterministic by construction — unlike OpenCV's RANSAC —
/// so tests can assert on exact behaviour. No Levenberg-Marquardt polish; the
/// final model is a least-squares DLT refit on the consensus set, which is
/// enough for residual gating (validated against OpenCV in the golden tests).
///
/// Pure C++14, header-only, no dependencies.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace nvmm {
namespace motion {
namespace detail {

struct Pt {
    float x = 0.f, y = 0.f;
};

struct Mat3 {
    // row-major
    double m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    static Mat3 identity() { return Mat3(); }

    double det() const {
        return m[0] * (m[4] * m[8] - m[5] * m[7]) -
               m[1] * (m[3] * m[8] - m[5] * m[6]) +
               m[2] * (m[3] * m[7] - m[4] * m[6]);
    }

    bool invert(Mat3 &out) const {
        const double d = det();
        if (std::abs(d) < 1e-12) return false;
        const double r = 1.0 / d;
        out.m[0] = (m[4] * m[8] - m[5] * m[7]) * r;
        out.m[1] = (m[2] * m[7] - m[1] * m[8]) * r;
        out.m[2] = (m[1] * m[5] - m[2] * m[4]) * r;
        out.m[3] = (m[5] * m[6] - m[3] * m[8]) * r;
        out.m[4] = (m[0] * m[8] - m[2] * m[6]) * r;
        out.m[5] = (m[2] * m[3] - m[0] * m[5]) * r;
        out.m[6] = (m[3] * m[7] - m[4] * m[6]) * r;
        out.m[7] = (m[1] * m[6] - m[0] * m[7]) * r;
        out.m[8] = (m[0] * m[4] - m[1] * m[3]) * r;
        return true;
    }

    Mat3 operator*(const Mat3 &b) const {
        Mat3 r;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                r.m[i * 3 + j] = m[i * 3] * b.m[j] + m[i * 3 + 1] * b.m[3 + j] +
                                 m[i * 3 + 2] * b.m[6 + j];
        return r;
    }

    /// Project (x, y, 1); false when the point maps to infinity.
    bool project(double x, double y, double &u, double &v) const {
        const double w = m[6] * x + m[7] * y + m[8];
        if (std::abs(w) < 1e-12) return false;
        u = (m[0] * x + m[1] * y + m[2]) / w;
        v = (m[3] * x + m[4] * y + m[5]) / w;
        return true;
    }
};

/// Smallest-eigenvalue eigenvector of a symmetric 9x9 matrix (cyclic Jacobi).
inline void min_eigenvector9(double a[9][9], double v_out[9])
{
    double v[9][9] = {};
    for (int i = 0; i < 9; i++) v[i][i] = 1.0;
    for (int sweep = 0; sweep < 64; sweep++) {
        double off = 0.0;
        for (int p = 0; p < 9; p++)
            for (int q = p + 1; q < 9; q++) off += a[p][q] * a[p][q];
        if (off < 1e-24) break;
        for (int p = 0; p < 9; p++)
            for (int q = p + 1; q < 9; q++) {
                if (std::abs(a[p][q]) < 1e-30) continue;
                const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                const double t = (theta >= 0 ? 1.0 : -1.0) /
                                 (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int i = 0; i < 9; i++) {
                    const double aip = a[i][p], aiq = a[i][q];
                    a[i][p] = c * aip - s * aiq;
                    a[i][q] = s * aip + c * aiq;
                }
                for (int i = 0; i < 9; i++) {
                    const double api = a[p][i], aqi = a[q][i];
                    a[p][i] = c * api - s * aqi;
                    a[q][i] = s * api + c * aqi;
                }
                for (int i = 0; i < 9; i++) {
                    const double vip = v[i][p], viq = v[i][q];
                    v[i][p] = c * vip - s * viq;
                    v[i][q] = s * vip + c * viq;
                }
            }
    }
    int best = 0;
    for (int i = 1; i < 9; i++)
        if (a[i][i] < a[best][best]) best = i;
    for (int i = 0; i < 9; i++) v_out[i] = v[i][best];
}

/// Hartley normalisation: translate centroid to origin, scale mean distance to sqrt(2).
inline Mat3 normalize_points(const std::vector<Pt> &pts, const std::vector<int> &idx,
                             std::vector<Pt> &out)
{
    double cx = 0, cy = 0;
    for (int i : idx) { cx += pts[(size_t)i].x; cy += pts[(size_t)i].y; }
    cx /= (double)idx.size(); cy /= (double)idx.size();
    double md = 0;
    for (int i : idx) {
        const double dx = pts[(size_t)i].x - cx, dy = pts[(size_t)i].y - cy;
        md += std::sqrt(dx * dx + dy * dy);
    }
    md /= (double)idx.size();
    const double s = md > 1e-12 ? std::sqrt(2.0) / md : 1.0;
    out.resize(idx.size());
    for (size_t k = 0; k < idx.size(); k++) {
        out[k].x = (float)((pts[(size_t)idx[k]].x - cx) * s);
        out[k].y = (float)((pts[(size_t)idx[k]].y - cy) * s);
    }
    Mat3 t;
    t.m[0] = s; t.m[1] = 0; t.m[2] = -cx * s;
    t.m[3] = 0; t.m[4] = s; t.m[5] = -cy * s;
    t.m[6] = 0; t.m[7] = 0; t.m[8] = 1;
    return t;
}

/// Least-squares DLT over the selected correspondences. False on degenerate input.
inline bool fit_homography_dlt(const std::vector<Pt> &p1, const std::vector<Pt> &p2,
                               const std::vector<int> &idx, Mat3 &H)
{
    if (idx.size() < 4) return false;
    std::vector<Pt> n1, n2;
    const Mat3 T1 = normalize_points(p1, idx, n1);
    const Mat3 T2 = normalize_points(p2, idx, n2);

    double ata[9][9] = {};
    for (size_t k = 0; k < idx.size(); k++) {
        const double x = n1[k].x, y = n1[k].y, u = n2[k].x, v = n2[k].y;
        const double r1[9] = {-x, -y, -1, 0, 0, 0, u * x, u * y, u};
        const double r2[9] = {0, 0, 0, -x, -y, -1, v * x, v * y, v};
        for (int i = 0; i < 9; i++)
            for (int j = i; j < 9; j++)
                ata[i][j] += r1[i] * r1[j] + r2[i] * r2[j];
    }
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < i; j++) ata[i][j] = ata[j][i];

    double h[9];
    min_eigenvector9(ata, h);
    Mat3 Hn;
    std::copy(h, h + 9, Hn.m);
    Mat3 T2i;
    if (!T2.invert(T2i)) return false;
    H = T2i * Hn * T1;
    if (std::abs(H.m[8]) < 1e-12) return false;
    for (int i = 0; i < 9; i++) H.m[i] /= H.m[8] == 0 ? 1.0 : H.m[8];
    return true;
}

/// Any 3 of the 4 sample points (nearly) collinear => degenerate sample.
inline bool sample_degenerate(const std::vector<Pt> &p, const int s[4])
{
    for (int a = 0; a < 2; a++)
        for (int b = a + 1; b < 3; b++)
            for (int c = b + 1; c < 4; c++) {
                const Pt &A = p[(size_t)s[a]], &B = p[(size_t)s[b]], &C = p[(size_t)s[c]];
                const double cr = (double)(B.x - A.x) * (C.y - A.y) -
                                  (double)(B.y - A.y) * (C.x - A.x);
                if (std::abs(cr) < 1.0) return true;
            }
    return false;
}

/// Deterministic LCG (numerical-recipes constants) for RANSAC sampling.
class Lcg {
public:
    explicit Lcg(uint32_t seed) : s_(seed ? seed : 1u) {}
    uint32_t next() { s_ = s_ * 1664525u + 1013904223u; return s_; }
    int below(int n) { return (int)((uint64_t)next() * (uint32_t)n >> 32); }

private:
    uint32_t s_;
};

/// RANSAC homography p1 -> p2 (forward reprojection error, adaptive iteration
/// bound, confidence 0.995, <= 2000 iterations, fixed seed). On success the
/// final H is a least-squares refit on the consensus set and `inliers` is
/// recomputed from it.
inline bool find_homography_ransac(const std::vector<Pt> &p1, const std::vector<Pt> &p2,
                                   double thresh, Mat3 &H, std::vector<uint8_t> &inliers,
                                   uint32_t seed = 0x5A17u)
{
    const int n = (int)p1.size();
    inliers.assign((size_t)n, 0);
    if (n < 4) return false;

    const double t2 = thresh * thresh;
    auto count_inliers = [&](const Mat3 &M, std::vector<uint8_t> &mask) {
        int cnt = 0;
        for (int i = 0; i < n; i++) {
            double u, v;
            const bool ok = M.project(p1[(size_t)i].x, p1[(size_t)i].y, u, v);
            const double dx = ok ? u - p2[(size_t)i].x : 1e9;
            const double dy = ok ? v - p2[(size_t)i].y : 1e9;
            mask[(size_t)i] = (ok && dx * dx + dy * dy <= t2) ? 1 : 0;
            cnt += mask[(size_t)i];
        }
        return cnt;
    };

    Lcg rng(seed);
    std::vector<uint8_t> mask((size_t)n, 0);
    std::vector<int> quad(4);
    int best_cnt = 0;
    Mat3 best_H;
    int iters = 2000;
    for (int it = 0; it < iters; it++) {
        int s[4];
        s[0] = rng.below(n);
        do { s[1] = rng.below(n); } while (s[1] == s[0]);
        do { s[2] = rng.below(n); } while (s[2] == s[0] || s[2] == s[1]);
        do { s[3] = rng.below(n); } while (s[3] == s[0] || s[3] == s[1] || s[3] == s[2]);
        if (sample_degenerate(p1, s) || sample_degenerate(p2, s)) continue;
        quad.assign(s, s + 4);
        Mat3 M;
        if (!fit_homography_dlt(p1, p2, quad, M)) continue;
        const int cnt = count_inliers(M, mask);
        if (cnt > best_cnt) {
            best_cnt = cnt;
            best_H = M;
            // adaptive bound: enough iterations to hit an all-inlier sample
            // with 99.5% confidence given the observed inlier ratio
            const double w = (double)cnt / n;
            const double p_good = w * w * w * w;
            if (p_good > 1e-9) {
                const double need = std::log(1 - 0.995) / std::log(1 - p_good);
                iters = std::min(iters, it + 1 + (int)std::min(2000.0, std::ceil(need)));
            }
        }
    }
    if (best_cnt < 4) return false;

    count_inliers(best_H, inliers);
    std::vector<int> in_idx;
    for (int i = 0; i < n; i++)
        if (inliers[(size_t)i]) in_idx.push_back(i);
    Mat3 refined;
    if (fit_homography_dlt(p1, p2, in_idx, refined)) {
        std::vector<uint8_t> refined_mask((size_t)n, 0);
        if (count_inliers(refined, refined_mask) >= best_cnt) {
            H = refined;
            inliers.swap(refined_mask);
            return true;
        }
    }
    H = best_H;
    return true;
}

}  // namespace detail
}  // namespace motion
}  // namespace nvmm
