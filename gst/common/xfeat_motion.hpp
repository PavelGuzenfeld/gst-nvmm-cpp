#pragma once
#include "xfeat_register.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace nvmm { namespace motion {

using nvmm::xfeat::Pt2;

struct MatchPair { int idx; Pt2 a; Pt2 b; };

struct Box {
    double x = 0, y = 0, w = 0, h = 0;
    bool contains(const Pt2& p) const {
        return p.x >= x && p.x <= x + w && p.y >= y && p.y <= y + h;
    }
};

inline Pt2 apply_affine(const double M[6], const Pt2& p) {
    return { M[0]*p.x + M[1]*p.y + M[2], M[3]*p.x + M[4]*p.y + M[5] };
}

inline double residual(const double M[6], const MatchPair& mp) {
    Pt2 q = apply_affine(M, mp.a);
    double dx = q.x - mp.b.x, dy = q.y - mp.b.y;
    return std::sqrt(dx*dx + dy*dy);
}

struct GmcEstimate {
    double dx = 0, dy = 0;
    double inlier_frac = 0;
    int    n = 0;
    bool   ok = false;
};

inline GmcEstimate global_translation_median(const std::vector<MatchPair>& m,
                                             const Box* exclude = nullptr,
                                             double tol = 2.0, int min_n = 8) {
    std::vector<double> dxs, dys;
    dxs.reserve(m.size()); dys.reserve(m.size());
    for (const auto& mp : m) {
        if (exclude && exclude->contains(mp.a)) continue;
        dxs.push_back(mp.b.x - mp.a.x);
        dys.push_back(mp.b.y - mp.a.y);
    }
    GmcEstimate e;
    e.n = (int)dxs.size();
    if (e.n < min_n) return e;
    auto median = [](std::vector<double>& v) {
        std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
        double m1 = v[v.size()/2];
        if (v.size() % 2) return m1;
        double m0 = *std::max_element(v.begin(), v.begin() + v.size()/2);
        return 0.5 * (m0 + m1);
    };
    e.dx = median(dxs);
    e.dy = median(dys);
    int inl = 0;
    for (const auto& mp : m) {
        if (exclude && exclude->contains(mp.a)) continue;
        double ex = (mp.b.x - mp.a.x) - e.dx, ey = (mp.b.y - mp.a.y) - e.dy;
        if (std::sqrt(ex*ex + ey*ey) <= tol) inl++;
    }
    e.inlier_frac = (double)inl / (double)e.n;
    e.ok = true;
    return e;
}

struct AffineFit {
    double M[6] = {1,0,0, 0,1,0};
    int    inliers = 0;
    int    n = 0;
    bool   ok = false;
};

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed = 0x9e3779b9u) : s(seed ? seed : 0x9e3779b9u) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    int below(int n) { return n <= 0 ? 0 : (int)(next() % (uint32_t)n); }
};

inline AffineFit ransac_affine(const std::vector<MatchPair>& m,
                               const Box* exclude = nullptr,
                               int iters = 200, double tol = 2.0,
                               int min_inliers = 12, uint32_t seed = 0x9e3779b9u) {
    AffineFit fit;
    std::vector<int> cand;
    cand.reserve(m.size());
    for (int i = 0; i < (int)m.size(); ++i)
        if (!(exclude && exclude->contains(m[i].a))) cand.push_back(i);
    fit.n = (int)cand.size();
    if (fit.n < 3) return fit;

    Rng rng(seed);
    int best_inl = -1;
    double bestM[6];
    for (int it = 0; it < iters; ++it) {
        int i0 = cand[rng.below(fit.n)];
        int i1 = cand[rng.below(fit.n)];
        int i2 = cand[rng.below(fit.n)];
        if (i1 == i0 || i2 == i0 || i2 == i1) continue;
        std::array<Pt2,3> src{ m[i0].a, m[i1].a, m[i2].a };
        std::array<Pt2,3> dst{ m[i0].b, m[i1].b, m[i2].b };
        double M[6];
        if (!nvmm::xfeat::affine_from_3pts(src, dst, M)) continue;
        int inl = 0;
        for (int c : cand) if (residual(M, m[c]) <= tol) inl++;
        if (inl > best_inl) { best_inl = inl; std::copy(M, M+6, bestM); }
    }
    if (best_inl >= min_inliers) {
        std::copy(bestM, bestM+6, fit.M);
        fit.inliers = best_inl;
        fit.ok = true;
    }
    return fit;
}

struct RegionResidual {
    double max_resid = 0;
    int    n = 0;
    bool   ok = false;
};

inline RegionResidual region_max_residual(const double M[6],
                                          const std::vector<MatchPair>& m,
                                          const Box& box, int min_in_box = 3) {
    RegionResidual r;
    for (const auto& mp : m) {
        if (!box.contains(mp.a)) continue;
        r.max_resid = std::max(r.max_resid, residual(M, mp));
        r.n++;
    }
    r.ok = r.n >= min_in_box;
    return r;
}

inline RegionResidual region_max_residual_2ref(const double Ma[6],
                                               const std::vector<MatchPair>& ma,
                                               const double Mb[6],
                                               const std::vector<MatchPair>& mb,
                                               const Box& box, int min_in_box = 3) {
    std::vector<std::pair<int,double>> rb;
    rb.reserve(mb.size());
    for (const auto& mp : mb) if (box.contains(mp.a)) rb.push_back({mp.idx, residual(Mb, mp)});
    std::sort(rb.begin(), rb.end());
    auto find_b = [&](int idx) -> double {
        auto lo = std::lower_bound(rb.begin(), rb.end(), std::make_pair(idx, -1e300));
        if (lo != rb.end() && lo->first == idx) return lo->second;
        return -1.0;
    };
    RegionResidual r;
    for (const auto& mp : ma) {
        if (!box.contains(mp.a)) continue;
        double rbv = find_b(mp.idx);
        if (rbv < 0) continue;
        double comb = std::min(residual(Ma, mp), rbv);
        r.max_resid = std::max(r.max_resid, comb);
        r.n++;
    }
    r.ok = r.n >= min_in_box;
    return r;
}

struct ResidPt { Pt2 pt; double resid; };

inline std::vector<ResidPt> combined_residuals_2ref(const double Ma[6],
                                                    const std::vector<MatchPair>& ma,
                                                    const double Mb[6],
                                                    const std::vector<MatchPair>& mb) {
    std::vector<std::pair<int,double>> rb;
    rb.reserve(mb.size());
    for (const auto& mp : mb) rb.push_back({mp.idx, residual(Mb, mp)});
    std::sort(rb.begin(), rb.end());
    std::vector<ResidPt> out;
    out.reserve(ma.size());
    for (const auto& mp : ma) {
        auto lo = std::lower_bound(rb.begin(), rb.end(), std::make_pair(mp.idx, -1e300));
        if (lo == rb.end() || lo->first != mp.idx) continue;
        out.push_back({ mp.a, std::min(residual(Ma, mp), lo->second) });
    }
    return out;
}

struct MotionBlob {
    Box  box;
    int  n = 0;
    bool ok = false;
};

inline MotionBlob cluster_moving(const std::vector<MatchPair>& m, const double M[6],
                                 double resid_thresh, int min_pts = 4, double cell = 16.0) {
    MotionBlob out;
    std::vector<Pt2> pts;
    for (const auto& mp : m)
        if (residual(M, mp) >= resid_thresh) pts.push_back(mp.a);
    const int n = (int)pts.size();
    if (n < min_pts) return out;

    std::vector<int> parent(n);
    for (int i = 0; i < n; ++i) parent[i] = i;
    auto find = [&](int i){ while (parent[i]!=i){ parent[i]=parent[parent[i]]; i=parent[i]; } return i; };
    auto unite = [&](int a, int b){ a=find(a); b=find(b); if (a!=b) parent[a]=b; };
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            if (std::fabs(pts[i].x - pts[j].x) <= cell && std::fabs(pts[i].y - pts[j].y) <= cell)
                unite(i, j);
        }
    std::vector<int> cnt(n, 0);
    for (int i = 0; i < n; ++i) cnt[find(i)]++;
    int root = -1, best = 0;
    for (int i = 0; i < n; ++i) if (cnt[i] > best) { best = cnt[i]; root = i; }
    if (root < 0 || best < min_pts) return out;

    double x0=1e300, y0=1e300, x1=-1e300, y1=-1e300;
    for (int i = 0; i < n; ++i) if (find(i) == root) {
        x0 = std::min(x0, pts[i].x); y0 = std::min(y0, pts[i].y);
        x1 = std::max(x1, pts[i].x); y1 = std::max(y1, pts[i].y);
    }
    out.box = { x0, y0, x1 - x0, y1 - y0 };
    out.n = best;
    out.ok = true;
    return out;
}

}}
