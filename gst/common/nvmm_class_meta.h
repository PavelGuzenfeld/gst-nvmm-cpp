/// GstNvmmClassMeta — the secondary-classifier sibling meta (Phase 3).
///
/// Per-detection classification produced by `nvmmsecondaryinfer`: entries align
/// by index with the `GstNvmmDetMeta` objects array on the SAME buffer, exactly
/// like `GstNvmmMotionMeta`. An object the classifier skipped (ROI too small,
/// no result yet) carries class_id == -1.
///
/// In-process only, like the motion meta it is a sibling of.
#pragma once

#include <gst/gst.h>
#include "shm_protocol.h"  // NVMM_META_LABEL_LEN

G_BEGIN_DECLS

typedef struct _NvmmClassEntry {
    gint32  class_id;                    /* -1 = not classified */
    gfloat  confidence;                  /* top-1 score (post-activation) */
    guint32 fresh;                       /* 1 = inferred on this frame, 0 = cached */
    gchar   label[NVMM_META_LABEL_LEN];  /* NUL-terminated class label */
} NvmmClassEntry;

typedef struct _GstNvmmClassMeta {
    GstMeta         meta;
    guint32         num_objects;  /* matches the det meta's count/order */
    NvmmClassEntry *objects;      /* heap array of num_objects (NULL if 0) */
} GstNvmmClassMeta;

GType              gst_nvmm_class_meta_api_get_type(void);
const GstMetaInfo *gst_nvmm_class_meta_get_info(void);

#define GST_NVMM_CLASS_META_API_TYPE (gst_nvmm_class_meta_api_get_type())
#define gst_buffer_get_nvmm_class_meta(b) \
    ((GstNvmmClassMeta *)gst_buffer_get_meta((b), GST_NVMM_CLASS_META_API_TYPE))

/// Attach a copy of `n` entries to `buffer`. Returns the meta (owned by buffer).
GstNvmmClassMeta *gst_buffer_add_nvmm_class_meta(GstBuffer *buffer,
                                                 const NvmmClassEntry *entries,
                                                 guint32 n);

G_END_DECLS
