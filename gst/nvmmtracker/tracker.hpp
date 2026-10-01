#pragma once

#include <cstdint>
#include <vector>

#include "shm_protocol.h"

namespace nvmm {

struct TrackerParams {
    float iou_threshold = 0.3f;
    int   max_age       = 30;
};

class Tracker {
public:
    explicit Tracker(const TrackerParams& params = {}) : params_(params) {}

    void update(NvmmDetObject* objects, uint32_t num_objects);

    void reset();

    std::size_t live_tracks() const { return tracks_.size(); }

private:
    struct Track {
        uint64_t id;
        float    left, top, width, height;
        int32_t  class_id;
        int      age;
    };

    TrackerParams       params_;
    std::vector<Track>  tracks_;
    uint64_t            next_id_ = 1;
};

}
