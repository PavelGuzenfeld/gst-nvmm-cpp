// Host unit tests for the per-track classification cache (interval policy +
// expiry). No CUDA/GStreamer — runs on x86 CI.
#include "secondary_cache.hpp"
#include "test_harness.h"

#include <cmath>
#include <cstdio>
#include <cstring>

using nvmm::ClassResult;
using nvmm::SecondaryCache;
using nvmm::SecondaryCacheParams;

namespace {
ClassResult result(int32_t id, float conf, const char *label) {
    ClassResult r;
    r.class_id = id;
    r.confidence = conf;
    snprintf(r.label, sizeof r.label, "%s", label);
    return r;
}
}  // namespace

// An unknown track is always due; a freshly-stored one is not.
TEST(unknown_track_is_due) {
    SecondaryCache c({/*infer_interval=*/10, /*max_age=*/60});
    ASSERT_TRUE(c.due(7, 100));
    c.store(7, result(3, 0.9f, "cat"), 100);
    ASSERT_TRUE(!c.due(7, 100));
    ASSERT_TRUE(!c.due(7, 109));
    ASSERT_TRUE(c.due(7, 110));  // interval elapsed
}

// interval=1 means re-infer every frame.
TEST(interval_one_reinfers_every_frame) {
    SecondaryCache c({1, 60});
    c.store(1, result(0, 0.5f, "a"), 5);
    ASSERT_TRUE(!c.due(1, 5));
    ASSERT_TRUE(c.due(1, 6));
}

// lookup returns the stored result and keeps the track alive.
TEST(lookup_returns_stored_result) {
    SecondaryCache c({10, 60});
    c.store(42, result(2, 0.8f, "dog"), 0);
    const ClassResult *r = c.lookup(42, 3);
    ASSERT_NOT_NULL(r);
    ASSERT_EQ(r->class_id, 2);
    ASSERT_NEAR(r->confidence, 0.8f, 1e-6);
    ASSERT_TRUE(strcmp(r->label, "dog") == 0);
    ASSERT_TRUE(c.lookup(43, 3) == nullptr);  // unknown track
}

// A track not seen for max_age frames expires; a looked-up one survives.
TEST(expiry_drops_unseen_tracks) {
    SecondaryCache c({10, /*max_age=*/20});
    c.store(1, result(0, 0.5f, "a"), 0);
    c.store(2, result(1, 0.6f, "b"), 0);
    ASSERT_EQ(c.size(), (size_t)2);

    /* Track 1 is still seen by the detector (lookup at frame 15); 2 is not. */
    c.lookup(1, 15);
    c.expire(25);
    ASSERT_EQ(c.size(), (size_t)1);
    ASSERT_NOT_NULL(c.lookup(1, 25));
    ASSERT_TRUE(c.lookup(2, 25) == nullptr);
}

// Inference cadence does NOT extend a track's life — only being seen does.
TEST(infer_cadence_does_not_block_expiry) {
    SecondaryCache c({10, 20});
    c.store(1, result(0, 0.5f, "a"), 0);
    c.expire(21);  // never seen since store
    ASSERT_EQ(c.size(), (size_t)0);
    ASSERT_TRUE(c.due(1, 21));  // expired -> due again
}

// reset() forgets everything.
TEST(reset_clears_cache) {
    SecondaryCache c({10, 60});
    c.store(1, result(0, 0.5f, "a"), 0);
    c.reset();
    ASSERT_EQ(c.size(), (size_t)0);
    ASSERT_TRUE(c.due(1, 0));
}

// Re-storing overwrites the previous result.
TEST(store_overwrites) {
    SecondaryCache c({10, 60});
    c.store(1, result(0, 0.5f, "cat"), 0);
    c.store(1, result(4, 0.9f, "dog"), 10);
    const ClassResult *r = c.lookup(1, 10);
    ASSERT_NOT_NULL(r);
    ASSERT_EQ(r->class_id, 4);
    ASSERT_TRUE(strcmp(r->label, "dog") == 0);
    ASSERT_TRUE(!c.due(1, 19));
}

int main()
{
    printf("secondary_cache unit tests\n");
    printf("%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
