#pragma once

#include <gst/gst.h>
#include "shm_protocol.h"

G_BEGIN_DECLS

typedef struct _GstNvmmDetMeta {
    GstMeta        meta;
    guint64        frame_number;
    /// Bbox coordinate space.
    guint32        infer_width;
    guint32        infer_height;
    guint32        flags;
    guint32        num_objects;
    NvmmDetObject *objects;
} GstNvmmDetMeta;

GType                 gst_nvmm_det_meta_api_get_type(void);
const GstMetaInfo    *gst_nvmm_det_meta_get_info(void);

#define GST_NVMM_DET_META_API_TYPE  (gst_nvmm_det_meta_api_get_type())
#define gst_buffer_get_nvmm_det_meta(b) \
    ((GstNvmmDetMeta *)gst_buffer_get_meta((b), GST_NVMM_DET_META_API_TYPE))

/// Copies the detections; num_objects is clamped to NVMM_META_MAX_OBJECTS.
GstNvmmDetMeta *gst_buffer_add_nvmm_det_meta(GstBuffer *buffer,
                                             const NvmmFrameMeta *frame);

#ifdef NVMM_DEEPSTREAM_META
/// Built only with -Denable_deepstream_meta. `batch` is an NvDsBatchMeta*;
/// `infer_w`/`infer_h` record the bbox coordinate space. Returns objects written.
guint nvmm_frame_meta_from_nvds(void *batch, guint frame_index,
                                guint32 infer_w, guint32 infer_h,
                                guint64 frame_number, NvmmFrameMeta *out);
#endif

G_END_DECLS
