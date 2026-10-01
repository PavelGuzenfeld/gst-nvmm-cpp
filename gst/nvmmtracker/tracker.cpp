#include "tracker.hpp"

#include <algorithm>

namespace nvmm {

namespace {
float iou(float ax, float ay, float aw, float ah,
          float bx, float by, float bw, float bh) {
    const float ix1 = std::max(ax, bx), iy1 = std::max(ay, by);
    const float ix2 = std::min(ax + aw, bx + bw), iy2 = std::min(ay + ah, by + bh);
    const float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
    const float inter = iw * ih;
    const float uni = aw * ah + bw * bh - inter;
    return uni > 0.f ? inter / uni : 0.f;
}
}

void Tracker::reset() {
    tracks_.clear();
    next_id_ = 1;
}

void Tracker::update(NvmmDetObject* objects, uint32_t num_objects) {
    std::vector<char> matched(tracks_.size(), 0);

    for (uint32_t i = 0; i < num_objects; i++) {
        NvmmDetObject& o = objects[i];

        int best = -1;
        float best_iou = params_.iou_threshold;
        for (std::size_t k = 0; k < tracks_.size(); k++) {
            if (matched[k] || tracks_[k].class_id != o.class_id) continue;
            const Track& t = tracks_[k];
            const float v = iou(o.left, o.top, o.width, o.height,
                                t.left, t.top, t.width, t.height);
            if (v >= best_iou) { best = static_cast<int>(k); best_iou = v; }
        }

        if (best >= 0) {
            Track& tr = tracks_[best];
            tr.left = o.left; tr.top = o.top; tr.width = o.width; tr.height = o.height;
            tr.age = 0;
            matched[best] = 1;
            o.tracker_id = tr.id;
        } else {
            tracks_.push_back(Track{next_id_, o.left, o.top, o.width, o.height,
                                    o.class_id, 0});
            matched.push_back(1);
            o.tracker_id = next_id_++;
        }
    }

    for (std::size_t k = 0; k < tracks_.size(); k++)
        if (!matched[k]) tracks_[k].age++;
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                      [&](const Track& t) { return t.age > params_.max_age; }),
                  tracks_.end());
}

}
