/// GstNvmmMotionMeta — the fusion-result sibling meta (Phase 3).
///
/// Per-detection motion computed by `nvmmfusion` from the optical-flow field
/// under each `GstNvmmDetMeta` box: entries align by index with the det meta's
/// objects array on the SAME buffer. Consumers (e.g. `nvmmdrawdet`) use it to
/// mark which detected objects are actually moving.
///
/// In-process only, like the flow meta it derives from.
#pragma once

#include <gst/gst.h>
#include "nvmm_motion.hpp"  // nvmm::MotionEntry

G_BEGIN_DECLS

typedef struct _GstNvmmMotionMeta {
    GstMeta            meta;
    guint32            num_objects;  /* matches the det meta's count/order */
    nvmm::MotionEntry *objects;      /* heap array of num_objects (NULL if 0) */
} GstNvmmMotionMeta;

GType              gst_nvmm_motion_meta_api_get_type(void);
const GstMetaInfo *gst_nvmm_motion_meta_get_info(void);

#define GST_NVMM_MOTION_META_API_TYPE (gst_nvmm_motion_meta_api_get_type())
#define gst_buffer_get_nvmm_motion_meta(b) \
    ((GstNvmmMotionMeta *)gst_buffer_get_meta((b), GST_NVMM_MOTION_META_API_TYPE))

/// Attach a copy of `n` entries to `buffer`. Returns the meta (owned by buffer).
GstNvmmMotionMeta *gst_buffer_add_nvmm_motion_meta(GstBuffer *buffer,
                                                   const nvmm::MotionEntry *entries,
                                                   guint32 n);

G_END_DECLS
