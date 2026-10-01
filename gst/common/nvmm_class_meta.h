#pragma once

#include <gst/gst.h>
#include "shm_protocol.h"

G_BEGIN_DECLS

typedef struct _NvmmClassEntry {
    gint32  class_id;
    gfloat  confidence;
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
