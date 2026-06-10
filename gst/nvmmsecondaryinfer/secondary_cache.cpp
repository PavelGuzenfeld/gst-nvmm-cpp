#include "secondary_cache.hpp"

namespace nvmm {

bool SecondaryCache::due(uint64_t tracker_id, uint64_t frame_no) const
{
    auto it = entries_.find(tracker_id);
    if (it == entries_.end())
        return true;
    return frame_no - it->second.last_infer >= params_.infer_interval;
}

void SecondaryCache::store(uint64_t tracker_id, const ClassResult& result,
                           uint64_t frame_no)
{
    Entry& e = entries_[tracker_id];
    e.result = result;
    e.last_infer = frame_no;
    e.last_seen = frame_no;
}

const ClassResult* SecondaryCache::lookup(uint64_t tracker_id, uint64_t frame_no)
{
    auto it = entries_.find(tracker_id);
    if (it == entries_.end())
        return nullptr;
    it->second.last_seen = frame_no;
    return &it->second.result;
}

void SecondaryCache::expire(uint64_t frame_no)
{
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (frame_no - it->second.last_seen > params_.max_age)
            it = entries_.erase(it);
        else
            ++it;
    }
}

}  // namespace nvmm
