/// Unit tests for GstNvmmTrackMeta (nvmm_track_meta.h): the single-target
/// SAMURAI track record. Pure GstMeta round-trips — no NvBufSurface needed.

#include "nvmm_track_meta.h"

#include <gst/gst.h>

#include <cmath>
#include <cstdio>

#include "test_harness.h"

namespace {

/* gst_init must run before the static-registered TEST constructors below. */
struct GstInit { GstInit() { gst_init(nullptr, nullptr); } } _gst_init;

static void fill(GstNvmmTrackMeta *m)
{
    m->frame_number = 42;
    m->frame_width = 1920;
    m->frame_height = 1080;
    m->valid = TRUE;
    m->target_id = 7;
    m->left = 100.f; m->top = 200.f; m->width = 14.f; m->height = 8.f;
    m->object_score = 12.5f;
    m->kf_left = 101.f; m->kf_top = 201.f; m->kf_width = 14.f; m->kf_height = 8.f;
    m->kf_score = 0.8f;
    m->is_kf_only = FALSE;
    m->stable_frames = 11;
}

TEST(add_get_roundtrip) {
    GstBuffer *buf = gst_buffer_new();
    GstNvmmTrackMeta *m = gst_buffer_add_nvmm_track_meta(buf);
    ASSERT_TRUE(m != nullptr);
    fill(m);

    GstNvmmTrackMeta *got = gst_buffer_get_nvmm_track_meta(buf);
    ASSERT_TRUE(got != nullptr);
    ASSERT_EQ(got->frame_number, 42u);
    ASSERT_EQ(got->frame_width, 1920u);
    ASSERT_EQ(got->target_id, 7u);
    ASSERT_TRUE(got->valid);
    ASSERT_TRUE(got->width == 14.f && got->height == 8.f);
    ASSERT_NEAR(got->object_score, 12.5f, 1e-6f);
    ASSERT_EQ(got->stable_frames, 11u);
    ASSERT_TRUE(!got->is_kf_only);
    gst_buffer_unref(buf);
}

TEST(zero_initialized_on_add) {
    GstBuffer *buf = gst_buffer_new();
    GstNvmmTrackMeta *m = gst_buffer_add_nvmm_track_meta(buf);
    ASSERT_TRUE(m != nullptr);
    /* Fresh meta is a lost/unseeded track. */
    ASSERT_TRUE(!m->valid);
    ASSERT_EQ(m->target_id, 0u);
    ASSERT_TRUE(m->left == 0.f && m->width == 0.f);
    ASSERT_EQ(m->stable_frames, 0u);
    gst_buffer_unref(buf);
}

TEST(fetch_or_create_is_idempotent) {
    GstBuffer *buf = gst_buffer_new();
    GstNvmmTrackMeta *a = gst_buffer_add_nvmm_track_meta(buf);
    a->target_id = 99;
    /* Second add must return the SAME meta (so nvmmfusekf overwrites in place). */
    GstNvmmTrackMeta *b = gst_buffer_add_nvmm_track_meta(buf);
    ASSERT_TRUE(a == b);
    ASSERT_EQ(b->target_id, 99u);
    gst_buffer_unref(buf);
}

TEST(survives_buffer_copy) {
    GstBuffer *buf = gst_buffer_new();
    fill(gst_buffer_add_nvmm_track_meta(buf));

    GstBuffer *copy = gst_buffer_copy(buf);  /* triggers meta transform (copy) */
    GstNvmmTrackMeta *got = gst_buffer_get_nvmm_track_meta(copy);
    ASSERT_TRUE(got != nullptr);
    ASSERT_EQ(got->frame_number, 42u);
    ASSERT_EQ(got->target_id, 7u);
    ASSERT_TRUE(got->valid);
    ASSERT_TRUE(got->width == 14.f);
    ASSERT_EQ(got->stable_frames, 11u);

    gst_buffer_unref(buf);
    gst_buffer_unref(copy);
}

}  // namespace

int main() {
    printf("=== NVMM Track Metadata Tests ===\n");
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
