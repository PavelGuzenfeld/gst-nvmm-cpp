/// Multi-object tracker (pure host, no CUDA/GStreamer).
///
/// Assigns a stable `tracker_id` to each detection across frames by greedy IOU
/// matching against the previous frames' tracks (per class). Unmatched
/// detections start new tracks; tracks unseen for `max_age` frames expire. This
/// is the algorithm core behind the `nvmmtracker` element, kept dependency-free
/// so it is unit-tested on the host CI build.
#pragma once

#include <cstdint>
#include <vector>

#include "shm_protocol.h"  // NvmmDetObject

namespace nvmm {

struct TrackerParams {
    float iou_threshold = 0.3f;  // min IOU (same class) to continue a track
    int   max_age       = 30;    // frames a track survives with no match
};

class Tracker {
public:
    explicit Tracker(const TrackerParams& params = {}) : params_(params) {}

    /// Assign `objects[i].tracker_id` in place (1-based; stable across frames).
    /// Call once per frame in arrival order.
    void update(NvmmDetObject* objects, uint32_t num_objects);

    /// Forget all tracks (e.g. on stream restart / flush).
    void reset();

    /// Number of currently-live tracks (for tests/diagnostics).
    std::size_t live_tracks() const { return tracks_.size(); }

private:
    struct Track {
        uint64_t id;
        float    left, top, width, height;  // last matched box
        int32_t  class_id;
        int      age;  // frames since last match (0 = matched this frame)
    };

    TrackerParams       params_;
    std::vector<Track>  tracks_;
    uint64_t            next_id_ = 1;
};

}  // namespace nvmm
