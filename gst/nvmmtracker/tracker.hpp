#pragma once

#include <cstdint>
#include <vector>

#include "shm_protocol.h"

namespace nvmm {

struct TrackerParams {
    float iou_threshold = 0.3f;
    /// Frames a track survives with no match.
    int   max_age       = 30;
};

/// Greedy per-class IOU matching against prior-frame tracks.
class Tracker {
public:
    explicit Tracker(const TrackerParams& params = {}) : params_(params) {}

    /// Writes `objects[i].tracker_id` in place: 1-based, stable across frames. Call once
    /// per frame in arrival order.
    void update(NvmmDetObject* objects, uint32_t num_objects);

    void reset();

    std::size_t live_tracks() const { return tracks_.size(); }

private:
    struct Track {
        uint64_t id;
        float    left, top, width, height;
        int32_t  class_id;
        /// Frames since last match; 0 means matched this frame.
        int      age;
    };

    TrackerParams       params_;
    std::vector<Track>  tracks_;
    uint64_t            next_id_ = 1;
};

}
