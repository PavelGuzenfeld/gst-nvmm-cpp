#pragma once

#include <gst/gst.h>
#include "nvmm_motion.hpp"

G_BEGIN_DECLS

/// Entry i belongs to GstNvmmDetMeta object i on the same buffer.
typedef struct _GstNvmmMotionMeta {
    GstMeta            meta;
    guint32            num_objects;
    nvmm::MotionEntry *objects;
} GstNvmmMotionMeta;

GType              gst_nvmm_motion_meta_api_get_type(void);
const GstMetaInfo *gst_nvmm_motion_meta_get_info(void);

#define GST_NVMM_MOTION_META_API_TYPE (gst_nvmm_motion_meta_api_get_type())
#define gst_buffer_get_nvmm_motion_meta(b) \
    ((GstNvmmMotionMeta *)gst_buffer_get_meta((b), GST_NVMM_MOTION_META_API_TYPE))

GstNvmmMotionMeta *gst_buffer_add_nvmm_motion_meta(GstBuffer *buffer,
                                                   const nvmm::MotionEntry *entries,
                                                   guint32 n);

G_END_DECLS
