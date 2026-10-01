#pragma once

#include <gst/gst.h>
#include "shm_protocol.h"

G_BEGIN_DECLS

/// Entry i belongs to GstNvmmDetMeta object i on the same buffer.
typedef struct _NvmmClassEntry {
    /// -1 when the classifier skipped the object.
    gint32  class_id;
    /// Top-1 score after activation.
    gfloat  confidence;
    /// 1 when inferred on this frame, 0 when served from the cache.
    guint32 fresh;
    gchar   label[NVMM_META_LABEL_LEN];
} NvmmClassEntry;

typedef struct _GstNvmmClassMeta {
    GstMeta         meta;
    guint32         num_objects;
    NvmmClassEntry *objects;
} GstNvmmClassMeta;

GType              gst_nvmm_class_meta_api_get_type(void);
const GstMetaInfo *gst_nvmm_class_meta_get_info(void);

#define GST_NVMM_CLASS_META_API_TYPE (gst_nvmm_class_meta_api_get_type())
#define gst_buffer_get_nvmm_class_meta(b) \
    ((GstNvmmClassMeta *)gst_buffer_get_meta((b), GST_NVMM_CLASS_META_API_TYPE))

GstNvmmClassMeta *gst_buffer_add_nvmm_class_meta(GstBuffer *buffer,
                                                 const NvmmClassEntry *entries,
                                                 guint32 n);

G_END_DECLS
