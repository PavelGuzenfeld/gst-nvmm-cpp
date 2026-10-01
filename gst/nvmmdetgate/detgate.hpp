#pragma once
#include "xfeat_motion.hpp"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace nvmm {

struct GateDet {
    float cx, cy, w, h, conf;
    int   src_index;
};

struct MotionSample { float x, y, resid; };

struct GateCfg {
    int   dlt      = 5;
    float rmin     = 12.f;
    float dist     = 45.f;
    int   amin     = 6;
    int   ksup     = 4;
    int   maxlost  = 2;
    float borderfrac = 0.02f;
    float sample_rad = 32.f;

    bool  seed_on_motion = false;
    int   motion_silent  = 12;
    float motion_rmin    = 16.f;
    int   motion_minpts  = 4;
    float motion_cell    = 48.f;

    float ds = 2.f, rminsky = 8.f, confsky = 0.55f, cleanconf = 0.6f, skydom = 0.95f;
    int   cleanmax = 2; float motion_minarea = 4.f; int stride = 3;
};

class DetGate {
public:
    explicit DetGate(const GateCfg &c) : cfg_(c) {
        dbg_ = (std::getenv("DETGATE_DEBUG") != nullptr);
        fprintf(stderr, "[DD] gate constructed (XFeat): debug=%d rmin=%.1f amin=%d ksup=%d dist=%.0f\n",
                (int)dbg_, cfg_.rmin, cfg_.amin, cfg_.ksup, cfg_.dist);
    }

    bool locked() const { return locked_; }

    void reset() { locked_ = false; tracks_.clear(); lock_lost_ = 0; }

    static constexpr int kMotionSentinel = -100;
    static constexpr int kSynthConfirm   = -2;

    int update(const std::vector<MotionSample> &motion, const std::vector<GateDet> &dets,
               int frameW, int frameH)
    {
        fcount_++;
        fw_ = frameW; fh_ = frameH;
        for (auto &t : tracks_) { t.lost++; t.matched = false; }

        if (locked_) return update_locked(dets);

        std::vector<GateDet> aug = dets;
        if (cfg_.seed_on_motion && fcount_ >= cfg_.motion_silent) {
            GateDet mb;
            if (motion_blob(motion, mb)) aug.push_back(mb);
        }
        last_ndets_ = (int)aug.size();

        std::vector<size_t> order(aug.size());
        for (size_t i = 0; i < aug.size(); i++) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b){ return aug[a].conf > aug[b].conf; });

        for (size_t oi : order) {
            const GateDet &d = aug[oi];
            Trk *best = nullptr; float bd = cfg_.dist;
            for (auto &t : tracks_) {
                if (t.matched) continue;
                float dd = std::hypot(t.cx - d.cx, t.cy - d.cy);
                if (dd < bd) { bd = dd; best = &t; }
            }
            const float rd = sample(motion, d.cx, d.cy);
            const bool moved = rd >= cfg_.rmin;
            if (best) {
                best->cx = d.cx; best->cy = d.cy; best->w = d.w; best->h = d.h;
                best->age++; best->lost = 0; best->matched = true;
                best->maxc = std::max(best->maxc, d.conf);
                best->sup = moved ? best->sup + 1 : 0;
                best->src_index = d.src_index;
            } else {
                tracks_.push_back(Trk{d.cx, d.cy, d.w, d.h, d.conf,
                                      1, 0, moved ? 1 : 0, true, d.src_index});
            }
            if (dbg_)
                fprintf(stderr, "[DD] f%ld det(%.0f,%.0f) c=%.2f resid=%.1f age=%d sup=%d\n",
                        fcount_, d.cx, d.cy, d.conf, rd,
                        best ? best->age : 1, best ? best->sup : (moved ? 1 : 0));
        }
        reap();

        int best_idx = -1, best_sup = -1; float best_conf = -1.f;
        float sel_w = 0.f, sel_h = 0.f;
        for (auto &t : tracks_) {
            if (t.lost != 0 || t.age < cfg_.amin) continue;
            if (!in_active(t.cx, t.cy)) continue;
            if (t.sup < cfg_.ksup) continue;
            if (t.sup > best_sup || (t.sup == best_sup && t.maxc > best_conf)) {
                best_sup = t.sup; best_conf = t.maxc; best_idx = t.src_index;
                lock_cx_ = t.cx; lock_cy_ = t.cy; sel_w = t.w; sel_h = t.h;
            }
        }
        if (best_idx == kMotionSentinel) {
            locked_ = true; lock_lost_ = 0;
            synth_valid_ = true;
            synth_cx_ = lock_cx_; synth_cy_ = lock_cy_; synth_w_ = sel_w; synth_h_ = sel_h;
            if (dbg_) fprintf(stderr, "[DD] MOTION-SEED f%ld (%.0f,%.0f %.0fx%.0f) — synth det\n",
                              fcount_, synth_cx_, synth_cy_, synth_w_, synth_h_);
            return kSynthConfirm;
        }
        if (best_idx >= 0) {
            locked_ = true; lock_lost_ = 0;
            if (dbg_) fprintf(stderr, "[DD] CONFIRM f%ld (%.0f,%.0f) sup=%d conf=%.2f\n",
                              fcount_, lock_cx_, lock_cy_, best_sup, best_conf);
        }
        return best_idx;
    }

    bool synth_seed(float &cx, float &cy, float &w, float &h) {
        if (!synth_valid_) return false;
        cx = synth_cx_; cy = synth_cy_; w = synth_w_; h = synth_h_;
        synth_valid_ = false; return true;
    }

private:
    struct Trk {
        float cx, cy, w, h, maxc;
        int age, lost, sup; bool matched; int src_index;
    };

    int update_locked(const std::vector<GateDet> &dets)
    {
        if (!in_active(lock_cx_, lock_cy_)) {
            locked_ = false; tracks_.clear();
            if (dbg_) fprintf(stderr, "[DD] UNLOCK f%ld (%.0f,%.0f) — drifted to border\n",
                              fcount_, lock_cx_, lock_cy_);
            return -1;
        }
        int best = -1; float bd = cfg_.dist;
        for (const auto &d : dets) {
            float dd = std::hypot(lock_cx_ - d.cx, lock_cy_ - d.cy);
            if (dd < bd) { bd = dd; best = d.src_index; lock_cx_ = d.cx; lock_cy_ = d.cy; }
        }
        if (best >= 0) { lock_lost_ = 0; return best; }
        if (++lock_lost_ > cfg_.maxlost) { locked_ = false; tracks_.clear(); }
        return -1;
    }

    bool in_active(float cx, float cy) const {
        if (cfg_.borderfrac <= 0.f || fw_ <= 0 || fh_ <= 0) return true;
        const float mx = cfg_.borderfrac * fw_, my = cfg_.borderfrac * fh_;
        return cx >= mx && cx <= fw_ - mx && cy >= my && cy <= fh_ - my;
    }

    void reap() {
        std::vector<Trk> keep;
        for (auto &t : tracks_) if (t.lost <= cfg_.maxlost) keep.push_back(t);
        tracks_.swap(keep);
    }

    float sample(const std::vector<MotionSample> &m, float cx, float cy) const {
        const float r2 = cfg_.sample_rad * cfg_.sample_rad;
        float best = 0.f;
        for (const auto &s : m) {
            const float dx = s.x - cx, dy = s.y - cy;
            if (dx * dx + dy * dy <= r2) best = std::max(best, s.resid);
        }
        return best;
    }

    bool motion_blob(const std::vector<MotionSample> &m, GateDet &out) {
        std::vector<int> ix;
        for (int i = 0; i < (int)m.size(); ++i) if (m[i].resid >= cfg_.motion_rmin) ix.push_back(i);
        const int n = (int)ix.size();
        if (n < cfg_.motion_minpts) return false;
        std::vector<int> parent(n);
        for (int i = 0; i < n; ++i) parent[i] = i;
        auto find = [&](int i){ while (parent[i]!=i){ parent[i]=parent[parent[i]]; i=parent[i]; } return i; };
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (std::fabs(m[ix[i]].x - m[ix[j]].x) <= cfg_.motion_cell &&
                    std::fabs(m[ix[i]].y - m[ix[j]].y) <= cfg_.motion_cell) {
                    int a = find(i), b = find(j); if (a != b) parent[a] = b;
                }
        std::vector<int> cnt(n, 0); for (int i = 0; i < n; ++i) cnt[find(i)]++;
        int root = -1, best = 0; for (int i = 0; i < n; ++i) if (cnt[i] > best) { best = cnt[i]; root = i; }
        if (root < 0 || best < cfg_.motion_minpts) return false;
        float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
        for (int i = 0; i < n; ++i) if (find(i) == root) {
            const auto &s = m[ix[i]];
            x0 = std::min(x0, s.x); y0 = std::min(y0, s.y);
            x1 = std::max(x1, s.x); y1 = std::max(y1, s.y);
        }
        out.w = x1 - x0; out.h = y1 - y0;
        out.cx = x0 + out.w / 2.f; out.cy = y0 + out.h / 2.f;
        out.conf = cfg_.confsky; out.src_index = kMotionSentinel;
        return true;
    }

    GateCfg cfg_;
    std::vector<Trk> tracks_;
    bool  locked_ = false;
    int   lock_lost_ = 0;
    bool  dbg_ = false;
    long  fcount_ = 0;
    float lock_cx_ = 0.f, lock_cy_ = 0.f;
    int   last_ndets_ = 0;
    int   fw_ = 0, fh_ = 0;

    bool  synth_valid_ = false;
    float synth_cx_ = 0.f, synth_cy_ = 0.f, synth_w_ = 0.f, synth_h_ = 0.f;
};

}
