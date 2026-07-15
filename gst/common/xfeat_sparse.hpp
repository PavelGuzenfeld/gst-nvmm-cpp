// XFeat sparse detectAndCompute port (host C++), faithful to
// externals/xfeat/modules/xfeat.py + interpolator.py (rocx v2.0.0.0-rc).
//
// Pipeline (post-net): normalize(feats) -> get_kpts_heatmap (softmax65 + pixel-shuffle)
//  -> NMS (5x5 maxpool + thr + nonzero) -> reliability score (nearest(K1h)*bilinear(H1))
//  -> argsort top_k -> bicubic grid_sample descriptors -> L2-norm -> scale by (rw,rh).
//
// Vendored from ../gst-nvmm-ostrack/src/ostrack_xfeat_sparse.hpp (namespace
// ostrack::xfeat -> nvmm::xfeat). Pure host, std-only, OpenCV-free.
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>

namespace nvmm { namespace xfeat {

// ---- Stage A: get_kpts_heatmap ----------------------------------------------
// kpts: (65, Hc, Wc) row-major (channel-major: kpts[c*Hc*Wc + h*Wc + w]).
// out:  (Hc*8, Wc*8) row-major. softmax over all 65 channels, keep 64, then the
// (Hc,Wc,8,8)->(Hc,8,Wc,8) pixel-shuffle: out[h*8+a, w*8+d] = softmax_c[a*8+d].
inline std::vector<float> get_kpts_heatmap(const float* kpts, int Hc, int Wc,
                                           float softmax_temp = 1.0f) {
    const int C = 65;
    const int Ho = Hc * 8, Wo = Wc * 8;
    std::vector<float> out((size_t)Ho * Wo);
    const int HW = Hc * Wc;
    for (int h = 0; h < Hc; ++h) {
        for (int w = 0; w < Wc; ++w) {
            const int base = h * Wc + w;
            // softmax denominator over all 65 channels (numerically stabilized)
            float mx = -1e30f;
            for (int c = 0; c < C; ++c) mx = std::max(mx, kpts[(size_t)c * HW + base] * softmax_temp);
            float denom = 0.f;
            for (int c = 0; c < C; ++c) denom += std::exp(kpts[(size_t)c * HW + base] * softmax_temp - mx);
            // first 64 channels -> 8x8 block
            for (int a = 0; a < 8; ++a) {
                for (int d = 0; d < 8; ++d) {
                    const int c = a * 8 + d;
                    const float v = std::exp(kpts[(size_t)c * HW + base] * softmax_temp - mx) / denom;
                    out[(size_t)(h * 8 + a) * Wo + (w * 8 + d)] = v;
                }
            }
        }
    }
    return out;
}

// ---- Stage B: NMS (5x5 maxpool stride1 pad2 + threshold + nonzero) ----------
// x: (H,W) row-major. Returns keypoints as (x,y) int pairs in raster order
// (torch nonzero yields row-major (row,col); .flip(-1) -> (col,row)=(x,y)).
struct KptI { int x, y; };
inline std::vector<KptI> nms(const float* x, int H, int W,
                             float threshold = 0.05f, int kernel = 5) {
    const int pad = kernel / 2;
    std::vector<KptI> kpts;
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            const float v = x[(size_t)r * W + c];
            if (!(v > threshold)) continue;
            // local max over kxk window (clamped, matching MaxPool2d zero-pad: pad
            // adds zeros but since v>thr>0 and we compare equality to the max of the
            // real window, zero-pad never raises the max above an interior value).
            float lm = -1e30f;
            for (int dr = -pad; dr <= pad; ++dr) {
                const int rr = r + dr; if (rr < 0 || rr >= H) continue;
                for (int dc = -pad; dc <= pad; ++dc) {
                    const int cc = c + dc; if (cc < 0 || cc >= W) continue;
                    lm = std::max(lm, x[(size_t)rr * W + cc]);
                }
            }
            if (v == lm) kpts.push_back({c, r});  // (x=col, y=row)
        }
    }
    return kpts;
}

// ---- grid_sample helpers (align_corners=False, padding_mode='zeros') --------
// Sample a (C,Ht,Wt) tensor at keypoint (px,py), where positions are normalized by
// (normH,normW): combined coord ix = px*Wt/(normW-1) - 0.5, iy = py*Ht/(normH-1) - 0.5.
inline void gs_coord(int px, int py, int Ht, int Wt, int normH, int normW,
                     double& ix, double& iy) {
    ix = (double)px * Wt / (normW - 1) - 0.5;
    iy = (double)py * Ht / (normH - 1) - 0.5;
}

inline void grid_sample_nearest(const float* t, int C, int Ht, int Wt,
                                 int px, int py, int normH, int normW, float* out) {
    double ix, iy; gs_coord(px, py, Ht, Wt, normH, normW, ix, iy);
    long xn = (long)std::nearbyint(ix), yn = (long)std::nearbyint(iy);
    for (int c = 0; c < C; ++c) {
        float v = 0.f;
        if (xn >= 0 && xn < Wt && yn >= 0 && yn < Ht) v = t[((size_t)c*Ht + yn)*Wt + xn];
        out[c] = v;
    }
}

inline void grid_sample_bilinear(const float* t, int C, int Ht, int Wt,
                                  int px, int py, int normH, int normW, float* out) {
    double ix, iy; gs_coord(px, py, Ht, Wt, normH, normW, ix, iy);
    long x0 = (long)std::floor(ix), y0 = (long)std::floor(iy);
    long x1 = x0 + 1, y1 = y0 + 1;
    double wx1 = ix - x0, wy1 = iy - y0, wx0 = 1.0 - wx1, wy0 = 1.0 - wy1;
    auto in = [&](long x, long y){ return x >= 0 && x < Wt && y >= 0 && y < Ht; };
    for (int c = 0; c < C; ++c) {
        const float* tc = t + (size_t)c*Ht*Wt;
        double v = 0.0;
        if (in(x0,y0)) v += wx0*wy0 * tc[y0*Wt + x0];
        if (in(x1,y0)) v += wx1*wy0 * tc[y0*Wt + x1];
        if (in(x0,y1)) v += wx0*wy1 * tc[y1*Wt + x0];
        if (in(x1,y1)) v += wx1*wy1 * tc[y1*Wt + x1];
        out[c] = (float)v;
    }
}

// PyTorch bicubic (cubic convolution, A=-0.75)
inline double cubic1(double x, double A){ return ((A+2)*x - (A+3))*x*x + 1; }
inline double cubic2(double x, double A){ return (((A)*x - 5*A)*x + 8*A)*x - 4*A; }
inline void cubic_coeffs(double t, double c[4]) {
    const double A = -0.75;
    c[0] = cubic2(t + 1, A);
    c[1] = cubic1(t, A);
    c[2] = cubic1(1 - t, A);
    c[3] = cubic2(2 - t, A);
}
inline void grid_sample_bicubic(const float* t, int C, int Ht, int Wt,
                                 int px, int py, int normH, int normW, float* out) {
    double ix, iy; gs_coord(px, py, Ht, Wt, normH, normW, ix, iy);
    long x0 = (long)std::floor(ix), y0 = (long)std::floor(iy);
    double tx = ix - x0, ty = iy - y0;
    double cx[4], cy[4]; cubic_coeffs(tx, cx); cubic_coeffs(ty, cy);
    auto in = [&](long x, long y){ return x >= 0 && x < Wt && y >= 0 && y < Ht; };
    for (int c = 0; c < C; ++c) {
        const float* tc = t + (size_t)c*Ht*Wt;
        double v = 0.0;
        for (int i = 0; i < 4; ++i) {
            long yy = y0 - 1 + i;
            for (int j = 0; j < 4; ++j) {
                long xx = x0 - 1 + j;
                double s = in(xx, yy) ? (double)tc[yy*Wt + xx] : 0.0;  // zeros padding
                v += cy[i] * cx[j] * s;
            }
        }
        out[c] = (float)v;
    }
}

// ---- Stage C: reliability scores + top-k ------------------------------------
// score(kp) = nearest(K1h@256x480, kp) * bilinear(H1@32x60, kp), both normalized by (256,480).
// Returns indices sorted by descending score (stable, matching torch.argsort default
// which is NOT stable -- but ties are vanishingly unlikely on float scores).
struct ScoredKpt { int x, y; float score; };
inline std::vector<ScoredKpt> score_and_sort(
        const std::vector<KptI>& kpts,
        const float* K1h, int Ho, int Wo,    // 256x480
        const float* H1, int Hh, int Wh,     // 32x60
        int normH, int normW) {
    std::vector<ScoredKpt> sk(kpts.size());
    for (size_t i = 0; i < kpts.size(); ++i) {
        float sn, sb;
        grid_sample_nearest(K1h, 1, Ho, Wo, kpts[i].x, kpts[i].y, normH, normW, &sn);
        grid_sample_bilinear(H1, 1, Hh, Wh, kpts[i].x, kpts[i].y, normH, normW, &sb);
        sk[i] = {kpts[i].x, kpts[i].y, sn * sb};
    }
    // torch.argsort(-scores): descending. Use stable sort by descending score.
    std::vector<int> idx(sk.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b){ return sk[a].score > sk[b].score; });
    std::vector<ScoredKpt> out(sk.size());
    for (size_t i = 0; i < idx.size(); ++i) out[i] = sk[idx[i]];
    return out;
}

}} // namespace nvmm::xfeat
