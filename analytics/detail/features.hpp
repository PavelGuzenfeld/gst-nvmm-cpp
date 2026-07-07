/// Feature detection + matching for dual_homography.hpp — two pipelines behind
/// one correspondence-list interface:
///
///   small_motion — FAST-9/16 corners on the current frame + ZNCC patch match
///     into the reference within a bounded search radius (coarse SAD grid, ZNCC
///     refinement). Sized to this project's actual registration problem:
///     near-consecutive frames of a panning camera — displacements are small
///     and rotation is negligible, so descriptor rotation invariance buys
///     nothing. Deterministic, no descriptors, no cross-frame detection.
///
///   orb — from-scratch ORB as in the OpenCV path it replaces: image pyramid,
///     FAST per level, intensity-centroid orientation, rotated-BRIEF 256-bit
///     descriptors on the blurred level, Hamming KNN(2) + Lowe ratio. The
///     BRIEF sampling pattern is generated from a fixed seed (not OpenCV's
///     learned table), so descriptors differ from cv::ORB bit-for-bit while
///     the geometry they produce is validated to the same quality bar.
///
/// Both are exposed so the golden tests + benchmarks can decide which one the
/// gate should default to. Pure C++14, header-only, no dependencies.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../image.hpp"
#include "../image_ops.hpp"
#include "homography.hpp"

namespace nvmm {
namespace motion {
namespace detail {

struct Corner {
    int x = 0, y = 0;
    float score = 0.f;
};

/// FAST-9/16 with 3x3 non-max suppression on a SAD-over-arc score, strongest
/// first, capped at max_corners. `margin` excludes a border band (>= 3).
inline std::vector<Corner> fast_corners(img::View<const uint8_t> im, int thresh,
                                        int max_corners, int margin)
{
    static const int CX[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
    static const int CY[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};
    const int w = im.width, h = im.height, t = thresh;
    const int m = std::max(margin, 3);
    if (w <= 2 * m || h <= 2 * m) return {};

    std::ptrdiff_t off[16];
    for (int i = 0; i < 16; i++) off[i] = (std::ptrdiff_t)CY[i] * im.stride + CX[i];

    img::Image<float> score(w, h, 0.f);
    for (int y = m; y < h - m; y++) {
        const uint8_t *row = im.row(y);
        float *sc = score.row(y);
        for (int x = m; x < w - m; x++) {
            const uint8_t *p = row + x;
            const int v = *p;
            // an arc of 9 must contain one of each antipodal pair: cheap reject
            const int d0 = p[off[0]] - v, d8 = p[off[8]] - v;
            const int d4 = p[off[4]] - v, d12 = p[off[12]] - v;
            const bool may_b = (d0 > t || d8 > t) && (d4 > t || d12 > t);
            const bool may_d = (d0 < -t || d8 < -t) && (d4 < -t || d12 < -t);
            if (!may_b && !may_d) continue;

            uint32_t bright = 0, dark = 0;
            int sad = 0;
            for (int i = 0; i < 16; i++) {
                const int d = p[off[i]] - v;
                if (d > t) bright |= 1u << i;
                else if (d < -t) dark |= 1u << i;
                const int a = d < 0 ? -d : d;
                if (a > t) sad += a - t;
            }
            auto has_run9 = [](uint32_t bits) {
                uint32_t c = bits | (bits << 16);
                for (int i = 1; i < 9; i++) c &= c >> 1;
                return c != 0;
            };
            if ((may_b && has_run9(bright)) || (may_d && has_run9(dark)))
                sc[x] = (float)sad;
        }
    }

    std::vector<Corner> out;
    for (int y = m; y < h - m; y++) {
        const float *sm = score.row(y - 1), *s0 = score.row(y), *sp = score.row(y + 1);
        for (int x = m; x < w - m; x++) {
            const float s = s0[x];
            if (s <= 0.f) continue;
            if (s < sm[x - 1] || s < sm[x] || s < sm[x + 1] || s < s0[x - 1] ||
                s <= s0[x + 1] || s <= sp[x - 1] || s <= sp[x] || s <= sp[x + 1])
                continue;
            out.push_back(Corner{x, y, s});
        }
    }
    std::sort(out.begin(), out.end(),
              [](const Corner &a, const Corner &b) { return a.score > b.score; });
    if ((int)out.size() > max_corners) out.resize((size_t)max_corners);
    return out;
}

// ---------------------------------------------------------------------------
// small_motion pipeline
// ---------------------------------------------------------------------------

struct SmallMotionParams {
    int fast_thresh = 20;
    int max_corners = 800;
    int search_radius = 32;   // max displacement searched (px)
    int patch_r = 5;          // ZNCC patch half-size (patch is 2r+1 square)
    int coarse_step = 3;      // SAD grid step of the coarse search
    float zncc_min = 0.6f;    // acceptance threshold on the refined ZNCC score
    float sad_ratio = 0.8f;   // ambiguity: best coarse SAD must be < ratio * runner-up
                              // (runner-up outside the refinement neighbourhood) —
                              // repetitive texture otherwise yields aliased matches
                              // coherent enough to fake a parallax plane downstream
};

/// ZNCC of the (2r+1)^2 patch at (cx,cy) in `a` vs (mx,my) in `b`; -1 on flat patches.
inline float zncc_at(img::View<const uint8_t> a, int cx, int cy,
                     img::View<const uint8_t> b, int mx, int my, int r)
{
    const int n = (2 * r + 1) * (2 * r + 1);
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (int dy = -r; dy <= r; dy++) {
        const uint8_t *ra = a.row(cy + dy) + cx;
        const uint8_t *rb = b.row(my + dy) + mx;
        for (int dx = -r; dx <= r; dx++) {
            const double va = ra[dx], vb = rb[dx];
            sa += va; sb += vb;
            saa += va * va; sbb += vb * vb; sab += va * vb;
        }
    }
    const double ca = saa - sa * sa / n, cb = sbb - sb * sb / n;
    if (ca < 1e-6 || cb < 1e-6) return -1.f;
    return (float)((sab - sa * sb / n) / std::sqrt(ca * cb));
}

/// FAST corners of `cur` matched into `ref` by bounded-radius patch search.
/// Appends integer-pixel correspondences to p1 (cur) / p2 (ref).
inline void small_motion_matches(img::View<const uint8_t> cur, img::View<const uint8_t> ref,
                                 const std::vector<Corner> &corners,
                                 const SmallMotionParams &p,
                                 std::vector<Pt> &p1, std::vector<Pt> &p2)
{
    const int r = p.patch_r, R = p.search_radius, w = cur.width, h = cur.height;
    for (const Corner &c : corners) {
        if (c.x < r || c.y < r || c.x >= w - r || c.y >= h - r) continue;
        // coarse: SAD on a coarse displacement grid, tracking the runner-up
        // outside the refinement neighbourhood for the ambiguity test
        int best_dx = 0, best_dy = 0;
        long best_sad = -1, second_sad = -1;
        for (int dy = -R; dy <= R; dy += p.coarse_step) {
            const int my = c.y + dy;
            if (my < r || my >= h - r) continue;
            for (int dx = -R; dx <= R; dx += p.coarse_step) {
                const int mx = c.x + dx;
                if (mx < r || mx >= w - r) continue;
                long sad = 0;
                for (int py = -r; py <= r; py++) {
                    const uint8_t *ra = cur.row(c.y + py) + c.x;
                    const uint8_t *rb = ref.row(my + py) + mx;
                    for (int px = -r; px <= r; px++) {
                        const int d = (int)ra[px] - rb[px];
                        sad += d < 0 ? -d : d;
                    }
                }
                if (best_sad < 0 || sad < best_sad) {
                    const bool near_best = std::abs(dx - best_dx) <= p.coarse_step &&
                                           std::abs(dy - best_dy) <= p.coarse_step;
                    if (best_sad >= 0 && !near_best &&
                        (second_sad < 0 || best_sad < second_sad))
                        second_sad = best_sad;
                    best_sad = sad; best_dx = dx; best_dy = dy;
                } else if (std::abs(dx - best_dx) > p.coarse_step ||
                           std::abs(dy - best_dy) > p.coarse_step) {
                    if (second_sad < 0 || sad < second_sad) second_sad = sad;
                }
            }
        }
        if (best_sad < 0) continue;
        // ambiguous over repetitive texture: a far-away displacement explains the
        // patch almost as well — a wrong-but-coherent match downstream, so drop it
        if (second_sad >= 0 && (float)best_sad >= p.sad_ratio * (float)second_sad) continue;
        // refine: exhaustive ZNCC around the coarse winner
        const int rr = p.coarse_step;
        float best_z = -2.f;
        int zx = 0, zy = 0;
        for (int dy = best_dy - rr; dy <= best_dy + rr; dy++) {
            const int my = c.y + dy;
            if (my < r || my >= h - r) continue;
            for (int dx = best_dx - rr; dx <= best_dx + rr; dx++) {
                const int mx = c.x + dx;
                if (mx < r || mx >= w - r) continue;
                const float z = zncc_at(cur, c.x, c.y, ref, mx, my, r);
                if (z > best_z) { best_z = z; zx = mx; zy = my; }
            }
        }
        if (best_z < p.zncc_min) continue;
        p1.push_back(Pt{(float)c.x, (float)c.y});
        p2.push_back(Pt{(float)zx, (float)zy});
    }
}

// ---------------------------------------------------------------------------
// orb pipeline
// ---------------------------------------------------------------------------

struct OrbParams {
    int nfeatures = 4000;
    int nlevels = 8;
    float scale_factor = 1.2f;
    int fast_thresh = 20;
    float ratio = 0.75f;  // Lowe ratio for KNN(2) matching
};

struct OrbFeature {
    float x = 0.f, y = 0.f;  // level-0 coordinates
    uint64_t desc[4] = {0, 0, 0, 0};
};

/// Bilinear resize (shrink) — pyramid construction.
inline img::Image<uint8_t> resize_bilinear(img::View<const uint8_t> src, int dw, int dh)
{
    img::Image<uint8_t> dst(dw, dh);
    const float sx = (float)src.width / dw, sy = (float)src.height / dh;
    for (int y = 0; y < dh; y++) {
        const float fy = ((float)y + 0.5f) * sy - 0.5f;
        const int y0 = std::max(0, std::min(src.height - 1, (int)std::floor(fy)));
        const int y1 = std::min(src.height - 1, y0 + 1);
        const float wy = fy - (float)y0;
        const uint8_t *r0 = src.row(y0), *r1 = src.row(y1);
        uint8_t *d = dst.row(y);
        for (int x = 0; x < dw; x++) {
            const float fx = ((float)x + 0.5f) * sx - 0.5f;
            const int x0 = std::max(0, std::min(src.width - 1, (int)std::floor(fx)));
            const int x1 = std::min(src.width - 1, x0 + 1);
            const float wx = fx - (float)x0;
            const float v = (1.f - wy) * ((1.f - wx) * r0[x0] + wx * r0[x1]) +
                            wy * ((1.f - wx) * r1[x0] + wx * r1[x1]);
            d[x] = (uint8_t)(v + 0.5f);
        }
    }
    return dst;
}

/// 256 BRIEF point pairs inside the 31x31 patch (radius <= 15 after rotation),
/// from a fixed-seed gaussian sampler — generated once, deterministic.
inline const int8_t *brief_pattern()
{
    static const std::vector<int8_t> pat = [] {
        std::vector<int8_t> v;
        v.reserve(256 * 4);
        Lcg rng(0xB51EFu);
        auto coord = [&]() {
            // sum of three uniforms ~ gaussian(sigma ~ 31/5), clipped to the patch
            for (;;) {
                const int c = (int)(rng.below(13) + rng.below(13) + rng.below(13)) - 18;
                if (c >= -13 && c <= 13) return (int8_t)c;
            }
        };
        while (v.size() < (size_t)256 * 4) {
            const int8_t x0 = coord(), y0 = coord(), x1 = coord(), y1 = coord();
            if (x0 * x0 + y0 * y0 > 15 * 15 || x1 * x1 + y1 * y1 > 15 * 15) continue;
            if (x0 == x1 && y0 == y1) continue;
            v.push_back(x0); v.push_back(y0); v.push_back(x1); v.push_back(y1);
        }
        return v;
    }();
    return pat.data();
}

/// Intensity-centroid orientation over the radius-15 disc (ORB's moment method).
inline void orb_orientation(img::View<const uint8_t> im, int cx, int cy,
                            float &cosA, float &sinA)
{
    static const int R = 15;
    long m10 = 0, m01 = 0;
    for (int dy = -R; dy <= R; dy++) {
        const int span = (int)std::sqrt((float)(R * R - dy * dy));
        const uint8_t *row = im.row(cy + dy) + cx;
        for (int dx = -span; dx <= span; dx++) {
            m10 += (long)dx * row[dx];
            m01 += (long)dy * row[dx];
        }
    }
    const float norm = std::sqrt((float)m10 * m10 + (float)m01 * m01);
    if (norm < 1e-6f) { cosA = 1.f; sinA = 0.f; return; }
    cosA = (float)m10 / norm;
    sinA = (float)m01 / norm;
}

/// ORB features of one frame: pyramid, per-level FAST, orientation, rotated
/// BRIEF on the blurred level. Coordinates are mapped back to level 0.
inline std::vector<OrbFeature> orb_detect(img::View<const uint8_t> im, const OrbParams &p)
{
    // Rotated pattern reach is <= 15, +1 for rounding; FAST ring adds nothing beyond it.
    const int margin = 17;
    std::vector<OrbFeature> out;
    img::Image<uint8_t> level_store;
    img::Image<float> blur_tmp;

    // per-level feature budget shrinking with level area (factor 1/scale^2)
    std::vector<int> budget((size_t)p.nlevels);
    {
        const double f = 1.0 / ((double)p.scale_factor * p.scale_factor);
        double sum = 0, cur = 1;
        for (int i = 0; i < p.nlevels; i++, cur *= f) sum += cur;
        cur = 1;
        for (int i = 0; i < p.nlevels; i++, cur *= f)
            budget[(size_t)i] = (int)(p.nfeatures * cur / sum + 0.5);
    }

    double level_scale = 1.0;
    for (int lvl = 0; lvl < p.nlevels; lvl++) {
        if (lvl > 0) {
            level_scale *= p.scale_factor;
            const int lw = (int)(im.width / level_scale + 0.5);
            const int lh = (int)(im.height / level_scale + 0.5);
            if (lw <= 2 * margin || lh <= 2 * margin) break;
            level_store = resize_bilinear(im, lw, lh);
        }
        img::View<const uint8_t> lv = lvl == 0 ? im : level_store.view();

        std::vector<Corner> corners =
            fast_corners(lv, p.fast_thresh, budget[(size_t)lvl], margin);
        if (corners.empty()) continue;

        // descriptors sample a smoothed image, as in ORB (blur once per level)
        img::Image<float> smooth(lv.width, lv.height);
        img::gaussian_blur<uint8_t>(lv, blur_tmp, smooth.view(), 7);

        const int8_t *pat = brief_pattern();
        for (const Corner &c : corners) {
            float ca, sa;
            orb_orientation(lv, c.x, c.y, ca, sa);
            OrbFeature f;
            f.x = (float)(c.x * level_scale);
            f.y = (float)(c.y * level_scale);
            for (int bit = 0; bit < 256; bit++) {
                const int8_t *q = pat + bit * 4;
                const int x0 = c.x + (int)std::lround(ca * q[0] - sa * q[1]);
                const int y0 = c.y + (int)std::lround(sa * q[0] + ca * q[1]);
                const int x1 = c.x + (int)std::lround(ca * q[2] - sa * q[3]);
                const int y1 = c.y + (int)std::lround(sa * q[2] + ca * q[3]);
                if (smooth.at(y0, x0) < smooth.at(y1, x1))
                    f.desc[bit >> 6] |= 1ull << (bit & 63);
            }
            out.push_back(f);
        }
    }
    return out;
}

inline int hamming256(const uint64_t a[4], const uint64_t b[4])
{
    return __builtin_popcountll(a[0] ^ b[0]) + __builtin_popcountll(a[1] ^ b[1]) +
           __builtin_popcountll(a[2] ^ b[2]) + __builtin_popcountll(a[3] ^ b[3]);
}

/// Brute-force Hamming KNN(2) + Lowe ratio; appends correspondences to p1/p2.
inline void orb_match(const std::vector<OrbFeature> &a, const std::vector<OrbFeature> &b,
                      float ratio, std::vector<Pt> &p1, std::vector<Pt> &p2)
{
    if (b.size() < 2) return;
    for (const OrbFeature &fa : a) {
        int d1 = 257, d2 = 257, best = -1;
        for (size_t j = 0; j < b.size(); j++) {
            const int d = hamming256(fa.desc, b[j].desc);
            if (d < d1) { d2 = d1; d1 = d; best = (int)j; }
            else if (d < d2) { d2 = d; }
        }
        if (best >= 0 && (float)d1 < ratio * (float)d2) {
            p1.push_back(Pt{fa.x, fa.y});
            p2.push_back(Pt{b[(size_t)best].x, b[(size_t)best].y});
        }
    }
}

}  // namespace detail
}  // namespace motion
}  // namespace nvmm
