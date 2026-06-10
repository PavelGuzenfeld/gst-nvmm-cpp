/// GstNvmmTracker — assign a stable tracker_id to each detection across frames.
///
/// In-place passthrough on video/x-raw(memory:NVMM): reads the GstNvmmDetMeta
/// (e.g. from nvmminfer), greedy-IOU-matches detections to prior-frame tracks
/// per class, and writes NvmmDetObject.tracker_id. Pixels are untouched (no
/// CUDA), so it builds and is tested on the host CI too.
#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_TRACKER (gst_nvmm_tracker_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmTracker, gst_nvmm_tracker, GST, NVMM_TRACKER,
                     GstBaseTransform)

G_END_DECLS
