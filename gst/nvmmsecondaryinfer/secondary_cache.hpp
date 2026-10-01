#pragma once

#include <cstdint>
#include <unordered_map>

#include "shm_protocol.h"

namespace nvmm {

struct ClassResult {
    int32_t class_id = -1;
    float   confidence = 0.f;
    char    label[NVMM_META_LABEL_LEN] = {};
};

struct SecondaryCacheParams {
    uint32_t infer_interval = 10;
    uint32_t max_age        = 60;
};

class SecondaryCache {
public:
    explicit SecondaryCache(const SecondaryCacheParams& params = {}) : params_(params) {}

    bool due(uint64_t tracker_id, uint64_t frame_no) const;

    void store(uint64_t tracker_id, const ClassResult& result, uint64_t frame_no);

    /// Marks the track seen, so expiry follows detector visibility, not inference cadence.
    const ClassResult* lookup(uint64_t tracker_id, uint64_t frame_no);

    void expire(uint64_t frame_no);

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

}
