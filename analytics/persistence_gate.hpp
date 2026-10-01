#pragma once
#include <vector>
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace nvmm {
namespace track {

struct Detection {
    float cx, cy;
    float conf;
    bool  supported;
};

struct PersistenceParams {
    float assoc_dist = 45.f;
    int   min_age = 6;
    int   min_support = 4;
    int   max_lost = 2;
};

class PersistenceGate {
public:
    explicit PersistenceGate(const PersistenceParams &p = {}) : p_(p) {}

    bool locked() const { return locked_; }
    float lock_x() const { return lx_; }
    float lock_y() const { return ly_; }

    int update(const std::vector<Detection> &dets)
    {
        if (locked_) return update_locked(dets);

        for (auto &t : tracks_) { t.lost++; t.matched = false; }

        std::vector<size_t> order(dets.size());
        for (size_t i = 0; i < dets.size(); i++) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b){ return dets[a].conf > dets[b].conf; });

        for (size_t oi : order) {
            const Detection &d = dets[oi];
            Track *best = nullptr; float bd = p_.assoc_dist;
            for (auto &t : tracks_) {
                if (t.matched) continue;
                float dd = std::hypot(t.cx - d.cx, t.cy - d.cy);
                if (dd < bd) { bd = dd; best = &t; }
            }
            if (best) {
                best->cx = d.cx; best->cy = d.cy; best->age++; best->lost = 0;
                best->matched = true; best->src = (int)oi;
                best->sup = d.supported ? best->sup + 1 : 0;
            } else {
                tracks_.push_back(Track{d.cx, d.cy, 1, 0, d.supported ? 1 : 0, true, (int)oi});
            }
        }
        std::vector<Track> keep;
        for (auto &t : tracks_) if (t.lost <= p_.max_lost) keep.push_back(t);
        tracks_.swap(keep);

        for (auto &t : tracks_)
            if (t.lost == 0 && t.age >= p_.min_age && t.sup >= p_.min_support) {
                locked_ = true; lock_lost_ = 0; lx_ = t.cx; ly_ = t.cy;
                return t.src;
            }
        return -1;
    }

    void reset() { tracks_.clear(); locked_ = false; lock_lost_ = 0; }

private:
    struct Track { float cx, cy; int age, lost, sup; bool matched; int src; };

    int update_locked(const std::vector<Detection> &dets)
    {
        int best = -1; float bd = p_.assoc_dist;
        for (size_t i = 0; i < dets.size(); i++) {
            float dd = std::hypot(lx_ - dets[i].cx, ly_ - dets[i].cy);
            if (dd < bd) { bd = dd; best = (int)i; lx_ = dets[i].cx; ly_ = dets[i].cy; }
        }
        if (best >= 0) { lock_lost_ = 0; return best; }
        if (++lock_lost_ > p_.max_lost) { locked_ = false; tracks_.clear(); }
        return -1;
    }

    PersistenceParams p_;
    std::vector<Track> tracks_;
    bool  locked_ = false;
    int   lock_lost_ = 0;
    float lx_ = 0.f, ly_ = 0.f;
};

}
}
