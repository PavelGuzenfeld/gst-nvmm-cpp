/// GstNvmmTrackMeta — a DeepStream-free GstMeta carrying the single tracked
/// target produced by the SAMURAI tracker (nvmmsamurai) and corrected by the
/// master Kalman fusion (nvmmfusekf).
///
/// SAMURAI tracks one target per stream (B=1 — the reference raises on >1 in
/// _kalman_predict), so unlike GstNvmmDetMeta (an array of detections) this
/// meta is a single fixed-size record: no heap, trivial copy. Coordinates are
/// in `frame_width` x `frame_height` pixel space (the decoded frame).
///
/// Producers: nvmmsamurai attaches the SAM/SAMURAI track; nvmmfusekf overwrites
/// `box` with the KF-fused result. Consumers: nvmmdrawdet overlays `box`.
#pragma once

#include <gst/gst.h>

G_BEGIN_DECLS

typedef struct _GstNvmmTrackMeta {
    GstMeta  meta;
    guint64  frame_number;
    guint32  frame_width;    /* coordinate space of box/kf_box */
    guint32  frame_height;

    gboolean valid;          /* FALSE when the target is lost / not yet seeded */
    guint64  target_id;      /* stable id of the tracked target */

    /* Tracked bounding box in frame pixel coords (the element's primary output;
       nvmmfusekf replaces this with the fused box). */
    float    left, top, width, height;
    float    object_score;   /* SAM object-score logit used for gating */

    /* Kalman-predicted box (frame coords) + its SAMURAI motion score. */
    float    kf_left, kf_top, kf_width, kf_height;
    float    kf_score;

    gboolean is_kf_only;     /* this frame was KF-only (no engine inference) */
    guint32  stable_frames;  /* SAMURAI consecutive-stable-frame counter */
} GstNvmmTrackMeta;

GType              gst_nvmm_track_meta_api_get_type(void);
const GstMetaInfo *gst_nvmm_track_meta_get_info(void);

#define GST_NVMM_TRACK_META_API_TYPE  (gst_nvmm_track_meta_api_get_type())
#define gst_buffer_get_nvmm_track_meta(b) \
    ((GstNvmmTrackMeta *)gst_buffer_get_meta((b), GST_NVMM_TRACK_META_API_TYPE))

/// Add a zero-initialized GstNvmmTrackMeta to `buffer` (valid=FALSE).
/// Returns the meta (owned by the buffer) for the caller to populate; NULL on
/// failure. If the buffer already carries one, the existing meta is returned
/// (so nvmmfusekf can fetch-or-create then overwrite in place).
GstNvmmTrackMeta *gst_buffer_add_nvmm_track_meta(GstBuffer *buffer);

G_END_DECLS
