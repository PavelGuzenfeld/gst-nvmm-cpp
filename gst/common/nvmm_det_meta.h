/// GstNvmmDetMeta — a DeepStream-free GstMeta carrying per-frame object
/// detections across the NVMM IPC boundary.
///
/// nvmmsink serializes detections into the shared-memory side-channel (the flat
/// NvmmFrameMeta records in shm_protocol.h); nvmmappsrc reads them back and
/// attaches this GstMeta to the output buffer so downstream GStreamer elements
/// get structured detections — with no dependency on DeepStream.
///
/// A consumer that is itself a DeepStream pipeline can convert this meta into
/// NvDsBatchMeta from a pad probe; the optional bridge helpers
/// (nvmm_frame_meta_from_nvds / nvmm_det_meta_to_nvds) are compiled only when the
/// build is configured with -Denable_deepstream_meta=true (NVMM_DEEPSTREAM_META).
#pragma once

#include <gst/gst.h>
#include "shm_protocol.h"

G_BEGIN_DECLS

typedef struct _GstNvmmDetMeta {
    GstMeta        meta;
    guint64        frame_number;
    guint32        infer_width;   /* coordinate space of the bboxes */
    guint32        infer_height;
    guint32        flags;         /* NVMM_FRAME_META_FLAG_* */
    guint32        num_objects;
    NvmmDetObject *objects;       /* heap array of num_objects (NULL if 0) */
} GstNvmmDetMeta;

GType                 gst_nvmm_det_meta_api_get_type(void);
const GstMetaInfo    *gst_nvmm_det_meta_get_info(void);

#define GST_NVMM_DET_META_API_TYPE  (gst_nvmm_det_meta_api_get_type())
#define gst_buffer_get_nvmm_det_meta(b) \
    ((GstNvmmDetMeta *)gst_buffer_get_meta((b), GST_NVMM_DET_META_API_TYPE))

/// Attach a copy of `frame`'s detections to `buffer`. `frame->num_objects` is
/// clamped to NVMM_META_MAX_OBJECTS. Returns the attached meta (owned by buffer).
GstNvmmDetMeta *gst_buffer_add_nvmm_det_meta(GstBuffer *buffer,
                                             const NvmmFrameMeta *frame);

#ifdef NVMM_DEEPSTREAM_META
/// Optional DeepStream bridge (only built with -Denable_deepstream_meta).
/// Serialize the object detections of `batch`'s frame at `frame_index` into the
/// flat `out` record. `infer_w`/`infer_h` record the bbox coordinate space.
/// Returns the number of objects written. `batch` is an NvDsBatchMeta*.
guint nvmm_frame_meta_from_nvds(void *batch, guint frame_index,
                                guint32 infer_w, guint32 infer_h,
                                guint64 frame_number, NvmmFrameMeta *out);
#endif

G_END_DECLS
