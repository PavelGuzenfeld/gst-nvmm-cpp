// XFeat registration projector port (host C++), faithful to utils/registration.py
// PointToPointProjector (rocx v2.0.0.0-rc): given matched (ref,query) keypoint coords
// and a point in the ref image, project it into the query image via a 3-point affine.
//
//   near = 9 ref-coords nearest to `point`
//   tri  = triplet of those 9 maximizing triangle area (most non-collinear)
//   M    = affine mapping ref_coords[tri] -> query_coords[tri]  (cv2.getAffineTransform)
//   proj = M @ [point, 1]
//
// Vendored from ../gst-nvmm-ostrack/src/ostrack_xfeat_register.hpp (namespace
// ostrack::xfeat -> nvmm::xfeat). Pure host, std-only, OpenCV-free.
#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstddef>

namespace nvmm { namespace xfeat {

struct Pt2 { double x, y; };

// kornia normalize_keypoints: shift=size/2, scale=max(W,H)/2, nk=(kp-shift)/scale.
inline void normalize_keypoints(const std::vector<Pt2>& kpts, double W, double H, std::vector<Pt2>& out) {
    double sx = W / 2.0, sy = H / 2.0, scale = std::max(W, H) / 2.0;
    out.resize(kpts.size());
    for (size_t i = 0; i < kpts.size(); ++i) out[i] = { (kpts[i].x - sx) / scale, (kpts[i].y - sy) / scale };
}

// ---- LightGlue host post-processing: log-double-softmax + mutual-NN filter ----
// faithful to kornia sigmoid_log_double_softmax + filter_matches, but fused: we only
// need the m x n CORE (filter_matches argmaxes over scores[:, :-1, :-1]). One O(m*n)
// pass computes row-argmax (m0) and col-argmax (m1) without materializing the matrix.
inline double softplus(double x) { return std::max(x, 0.0) + std::log1p(std::exp(-std::fabs(x))); }
inline double logsigmoid(double x) { return -softplus(-x); }

struct Match { int i, j; };  // ref index i -> query index j

// sim: row-major m x n; z0: m; z1: n. Returns mutual matches with mscore0 > th.
inline std::vector<Match> filter_matches(const float* sim, const float* z0, const float* z1,
                                         int m, int n, double th = 0.1) {
    // per-row / per-col log-sum-exp of sim (numerically stabilized)
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

    // single pass: row-argmax (best j per i) and col-argmax (best i per j) of core
    // core[i][j] = 2*sim - lse_row[i] - lse_col[j] + lz0[i] + lz1[j]
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
        if (j < 0 || m1[j] != i) continue;              // mutual
        if (std::exp(v0[i]) <= th) continue;            // mscore0 > th
        out.push_back({i, j});
    }
    return out;
}

// indices of the k nearest ref-coords to `point` (Euclidean). Matches
// np.argpartition(d,k)[:k] as a SET (order irrelevant downstream).
inline std::vector<int> k_nearest(const std::vector<Pt2>& pts, Pt2 p, int k) {
    const int n = (int)pts.size();
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    auto d2 = [&](int i){ double dx = pts[i].x - p.x, dy = pts[i].y - p.y; return dx*dx + dy*dy; };
    if (k >= n) return idx;
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                     [&](int a, int b){ return d2(a) < d2(b); });
    idx.resize(k);
    return idx;
}

// triplet (3 indices into `sub`) with the largest triangle area (shoelace).
inline std::array<int,3> most_non_collinear_triplet(const std::vector<Pt2>& sub) {
    const int n = (int)sub.size();
    double best = -1; std::array<int,3> tri{0,1,2};
    for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b)
            for (int c = b + 1; c < n; ++c) {
                double area = 0.5 * std::fabs(
                    sub[a].x*(sub[b].y - sub[c].y) +
                    sub[b].x*(sub[c].y - sub[a].y) +
                    sub[c].x*(sub[a].y - sub[b].y));
                if (area > best) { best = area; tri = {a, b, c}; }
            }
    return tri;
}

// cv2.getAffineTransform: exact 2x3 affine from 3 correspondences src->dst.
// Solves a*sx+b*sy+c=dx and d*sx+e*sy+f=dy (two 3x3 systems, Cramer's rule).
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
    solve3(dst[0].x, dst[1].x, dst[2].x, row0);   // a,b,c
    solve3(dst[0].y, dst[1].y, dst[2].y, row1);   // d,e,f
    M[0]=row0[0]; M[1]=row0[1]; M[2]=row0[2];
    M[3]=row1[0]; M[4]=row1[1]; M[5]=row1[2];
    return true;
}

// full projector: matched ref/query coords + ref-image point -> query-image point.
// Returns false if fewer than 9 matches (compute_affine bails like the reference).
inline bool project_point(const std::vector<Pt2>& mref, const std::vector<Pt2>& mqry,
                          Pt2 point, Pt2& out) {
    if (mref.size() < 9) return false;
    std::vector<int> near = k_nearest(mref, point, 9);   // returns all indices in order when size()==9
    std::vector<Pt2> subR(9), subQ(9);
    for (int i = 0; i < 9; ++i) { subR[i] = mref[near[i]]; subQ[i] = mqry[near[i]]; }
    auto tri = most_non_collinear_triplet(subR);
    std::array<Pt2,3> sr{ subR[tri[0]], subR[tri[1]], subR[tri[2]] };
    std::array<Pt2,3> dq{ subQ[tri[0]], subQ[tri[1]], subQ[tri[2]] };
    double M[6];
    if (!affine_from_3pts(sr, dq, M)) return false;
    out.x = M[0]*point.x + M[1]*point.y + M[2];
    out.y = M[3]*point.x + M[4]*point.y + M[5];
    return true;
}

// ---- registration mode glue (registration() + _build_bbox_from_registered_point) ----
// faithful to tracker_engine.py: center*scale -> project -> /scale (int trunc) -> mode size.
struct BBox { long x, y, w, h; };  // top-left + dims (ints, as the tracker uses)
enum RegMode { REG_FREEZE_VL, REG_FLIP };

// returns false (caller keeps prior bbox) if projection fails (<9 matches).
inline bool registration_bbox(const std::vector<Pt2>& mref, const std::vector<Pt2>& mqry,
                              BBox in, double scale, RegMode mode, BBox& out) {
    long cx = (long)((double)in.x + (double)in.w / 2.0);   // get_center(as_int): int(x+w/2)
    long cy = (long)((double)in.y + (double)in.h / 2.0);
    Pt2 pt{ cx * scale, cy * scale }, proj;
    if (!project_point(mref, mqry, pt, proj)) return false;
    long nx = (long)(proj.x / scale);                       // int() truncation
    long ny = (long)(proj.y / scale);
    long nw = (mode == REG_FLIP) ? (in.w / 2) : in.w;       // // floor div (positive)
    long nh = (mode == REG_FLIP) ? (in.h / 2) : in.h;
    out = { nx - nw / 2, ny - nh / 2, nw, nh };
    return true;
}

}} // namespace nvmm::xfeat
