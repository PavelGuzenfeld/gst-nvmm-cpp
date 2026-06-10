/// Per-track classification cache (pure host, no CUDA/GStreamer).
///
/// The multi-rate half of `nvmmsecondaryinfer`: the classifier engine runs on a
/// track only every `infer_interval` frames; in between, the last result is
/// served from this cache. Tracks unseen for `max_age` frames are dropped.
/// Dependency-free so it is unit-tested on the host CI build.
#pragma once

#include <cstdint>
#include <unordered_map>

#include "shm_protocol.h"  // NVMM_META_LABEL_LEN

namespace nvmm {

struct ClassResult {
    int32_t class_id = -1;
    float   confidence = 0.f;
    char    label[NVMM_META_LABEL_LEN] = {};
};

struct SecondaryCacheParams {
    uint32_t infer_interval = 10;  // re-infer a track every N frames (1 = every frame)
    uint32_t max_age        = 60;  // drop a track unseen for this many frames
};

class SecondaryCache {
public:
    explicit SecondaryCache(const SecondaryCacheParams& params = {}) : params_(params) {}

    /// True if `tracker_id` should be (re-)inferred at `frame_no`: unknown
    /// track, or `infer_interval` frames elapsed since its last inference.
    bool due(uint64_t tracker_id, uint64_t frame_no) const;

    /// Record a fresh inference result for `tracker_id` at `frame_no`.
    void store(uint64_t tracker_id, const ClassResult& result, uint64_t frame_no);

    /// Last stored result for `tracker_id` (nullptr if none). Marks the track
    /// as seen at `frame_no` so expiry is driven by detector visibility,
    /// not by inference cadence.
    const ClassResult* lookup(uint64_t tracker_id, uint64_t frame_no);

    /// Drop tracks not seen for `max_age` frames. Call once per frame.
    void expire(uint64_t frame_no);

    /// Forget everything (stream restart / flush).
    void reset() { entries_.clear(); }

    std::size_t size() const { return entries_.size(); }

private:
    struct Entry {
        ClassResult result;
        uint64_t    last_infer = 0;
        uint64_t    last_seen  = 0;
    };

    SecondaryCacheParams                 params_;
    std::unordered_map<uint64_t, Entry>  entries_;
};

}  // namespace nvmm
