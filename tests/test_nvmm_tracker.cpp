#include "tracker.hpp"
#include "test_harness.h"

#include <vector>

using nvmm::Tracker;
using nvmm::TrackerParams;

namespace {
NvmmDetObject det(float x, float y, float w, float h, int cls = 0) {
    NvmmDetObject o{};
    o.left = x; o.top = y; o.width = w; o.height = h;
    o.class_id = cls; o.confidence = 0.9f; o.tracker_id = 0;
    return o;
}
}

TEST(first_frame_assigns_unique_ids) {
    Tracker t;
    std::vector<NvmmDetObject> f = {det(0, 0, 10, 10), det(100, 100, 10, 10)};
    t.update(f.data(), f.size());
    ASSERT_TRUE(f[0].tracker_id != 0);
    ASSERT_TRUE(f[1].tracker_id != 0);
    ASSERT_TRUE(f[0].tracker_id != f[1].tracker_id);
}

TEST(same_object_keeps_id) {
    Tracker t;
    std::vector<NvmmDetObject> f1 = {det(50, 50, 20, 20)};
    t.update(f1.data(), f1.size());
    uint64_t id = f1[0].tracker_id;

    std::vector<NvmmDetObject> f2 = {det(52, 51, 20, 20)};
    t.update(f2.data(), f2.size());
    ASSERT_EQ(f2[0].tracker_id, id);
}

TEST(new_object_gets_new_id) {
    Tracker t;
    std::vector<NvmmDetObject> f1 = {det(0, 0, 20, 20)};
    t.update(f1.data(), f1.size());
    uint64_t id1 = f1[0].tracker_id;

    std::vector<NvmmDetObject> f2 = {det(0, 0, 20, 20), det(500, 500, 20, 20)};
    t.update(f2.data(), f2.size());
    ASSERT_EQ(f2[0].tracker_id, id1);
    ASSERT_TRUE(f2[1].tracker_id != 0);
    ASSERT_TRUE(f2[1].tracker_id != id1);
}

TEST(different_class_not_matched) {
    Tracker t;
    std::vector<NvmmDetObject> f1 = {det(50, 50, 20, 20, 1)};
    t.update(f1.data(), f1.size());
    uint64_t id = f1[0].tracker_id;

    std::vector<NvmmDetObject> f2 = {det(50, 50, 20, 20, 2)};
    t.update(f2.data(), f2.size());
    ASSERT_TRUE(f2[0].tracker_id != id);
}

TEST(low_overlap_not_matched) {
    Tracker t{TrackerParams{0.5f, 30}};
    std::vector<NvmmDetObject> f1 = {det(0, 0, 20, 20)};
    t.update(f1.data(), f1.size());
    uint64_t id = f1[0].tracker_id;

    std::vector<NvmmDetObject> f2 = {det(15, 15, 20, 20)};
    t.update(f2.data(), f2.size());
    ASSERT_TRUE(f2[0].tracker_id != id);
}

TEST(track_expires_after_max_age) {
    Tracker t{TrackerParams{0.3f, 2}};
    std::vector<NvmmDetObject> f1 = {det(0, 0, 20, 20)};
    t.update(f1.data(), f1.size());
    uint64_t id = f1[0].tracker_id;

    for (int i = 0; i < 3; i++) t.update(nullptr, 0);
    ASSERT_EQ(t.live_tracks(), (std::size_t)0);

    std::vector<NvmmDetObject> f2 = {det(0, 0, 20, 20)};
    t.update(f2.data(), f2.size());
    ASSERT_TRUE(f2[0].tracker_id != id);
}

TEST(reset_clears_tracks) {
    Tracker t;
    std::vector<NvmmDetObject> f1 = {det(0, 0, 20, 20)};
    t.update(f1.data(), f1.size());
    ASSERT_TRUE(t.live_tracks() > 0);
    t.reset();
    ASSERT_EQ(t.live_tracks(), (std::size_t)0);
}

int main() {
    printf("=== nvmm::Tracker tests ===\n");
    return tests_failed > 0 ? 1 : 0;
}
