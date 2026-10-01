#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstddef>

namespace nvmm { namespace xfeat {

struct Pt2 { double x, y; };

inline void normalize_keypoints(const std::vector<Pt2>& kpts, double W, double H, std::vector<Pt2>& out) {
    double sx = W / 2.0, sy = H / 2.0, scale = std::max(W, H) / 2.0;
    out.resize(kpts.size());
    for (size_t i = 0; i < kpts.size(); ++i) out[i] = { (kpts[i].x - sx) / scale, (kpts[i].y - sy) / scale };
}

inline double softplus(double x) { return std::max(x, 0.0) + std::log1p(std::exp(-std::fabs(x))); }
inline double logsigmoid(double x) { return -softplus(-x); }

struct Match { int i, j; };

/// kornia sigmoid_log_double_softmax + filter_matches, fused over the m x n
/// core. Returns mutual matches with mscore0 > th.
inline std::vector<Match> filter_matches(const float* sim, const float* z0, const float* z1,
                                         int m, int n, double th = 0.1) {
    std::vector<double> lse_row(m), lse_col(n, 0.0), colmax(n, -1e300);
    std::vector<double> lz0(m), lz1(n);
    for (int i = 0; i < m; ++i) lz0[i] = logsigmoid((double)z0[i]);
    for (int j = 0; j < n; ++j) lz1[j] = logsigmoid((double)z1[j]);
    for (int j = 0; j < n; ++j) { double mx = -1e300; for (int i = 0; i < m; ++i) mx = std::max(mx, (double)sim[(size_t)i*n+j]); colmax[j]=mx; }
    for (int i = 0; i < m; ++i) {
        const float* row = sim + (size_t)i*n;
        double mx = -1e300; for (int j = 0; j < n; ++j) mx = std::max(mx, (double)row[j]);
        double s = 0; for (int j = 0; j < n; ++j) s += std::exp((double)row[j] - mx);
        lse_row[i] = mx + std::log(s);
    }
    for (int j = 0; j < n; ++j) { double s=0; for (int i=0;i<m;++i) s += std::exp((double)sim[(size_t)i*n+j]-colmax[j]); lse_col[j]=colmax[j]+std::log(s); }

    std::vector<int> m0(m, -1), m1(n, -1);
    std::vector<double> v0(m, -1e300), vcol(n, -1e300);
    for (int i = 0; i < m; ++i) {
        const float* row = sim + (size_t)i*n;
        double bestv = -1e300; int bestj = -1;
        for (int j = 0; j < n; ++j) {
            double c = 2.0*(double)row[j] - lse_row[i] - lse_col[j] + lz0[i] + lz1[j];
            if (c > bestv) { bestv = c; bestj = j; }
            if (c > vcol[j]) { vcol[j] = c; m1[j] = i; }
        }
        m0[i] = bestj; v0[i] = bestv;
    }
    std::vector<Match> out;
    for (int i = 0; i < m; ++i) {
        int j = m0[i];
        if (j < 0 || m1[j] != i) continue;
        if (std::exp(v0[i]) <= th) continue;
        out.push_back({i, j});
    }
    return out;
}

inline bool affine_from_3pts(const std::array<Pt2,3>& src, const std::array<Pt2,3>& dst,
                             double M[6]) {
    double a[3][3];
    for (int i = 0; i < 3; ++i) { a[i][0] = src[i].x; a[i][1] = src[i].y; a[i][2] = 1.0; }
    double det = a[0][0]*(a[1][1]*a[2][2] - a[1][2]*a[2][1])
               - a[0][1]*(a[1][0]*a[2][2] - a[1][2]*a[2][0])
               + a[0][2]*(a[1][0]*a[2][1] - a[1][1]*a[2][0]);
    if (std::fabs(det) < 1e-12) return false;
    auto solve3 = [&](double r0, double r1, double r2, double out[3]) {
        double rhs[3] = {r0, r1, r2};
        for (int col = 0; col < 3; ++col) {
            double m[3][3];
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) m[i][j] = (j == col) ? rhs[i] : a[i][j];
            double d = m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
                     - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
                     + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
            out[col] = d / det;
        }
    };
    double row0[3], row1[3];
    solve3(dst[0].x, dst[1].x, dst[2].x, row0);
    solve3(dst[0].y, dst[1].y, dst[2].y, row1);
    M[0]=row0[0]; M[1]=row0[1]; M[2]=row0[2];
    M[3]=row1[0]; M[4]=row1[1]; M[5]=row1[2];
    return true;
}

}}
