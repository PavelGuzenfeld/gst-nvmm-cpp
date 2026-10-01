#pragma once

#include <gst/gst.h>

G_BEGIN_DECLS

typedef struct _GstNvmmTrackMeta {
    GstMeta  meta;
    guint64  frame_number;
    /// Coordinate space of the boxes.
    guint32  frame_width;
    guint32  frame_height;

    gboolean valid;
    guint64  target_id;

    float    left, top, width, height;
    /// SAM object-score logit.
    float    object_score;

    float    kf_left, kf_top, kf_width, kf_height;
    float    kf_score;

    gboolean is_kf_only;
    guint32  stable_frames;
} GstNvmmTrackMeta;

GType              gst_nvmm_track_meta_api_get_type(void);
const GstMetaInfo *gst_nvmm_track_meta_get_info(void);

#define GST_NVMM_TRACK_META_API_TYPE  (gst_nvmm_track_meta_api_get_type())
#define gst_buffer_get_nvmm_track_meta(b) \
    ((GstNvmmTrackMeta *)gst_buffer_get_meta((b), GST_NVMM_TRACK_META_API_TYPE))

/// Returns the existing meta when the buffer already has one, so nvmmfusekf
/// can overwrite the box in place.
GstNvmmTrackMeta *gst_buffer_add_nvmm_track_meta(GstBuffer *buffer);

G_END_DECLS
