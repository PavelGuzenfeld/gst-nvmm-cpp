/// GstNvmmDrawDet — draw NvmmDetMeta detections onto the video for viewing.
///
/// Sink: video/x-raw(memory:NVMM), NV12 (+ GstNvmmDetMeta, e.g. from nvmminfer).
/// Src:  video/x-raw, RGBA (system memory) with class-colored boxes drawn.
///
/// Converts the NVMM frame to RGBA on the VIC, pulls it to host via EGL/CUDA,
/// and draws detection boxes on the CPU. Intended for demo/visualization +
/// end-to-end tests (encode + stream the result) — not the zero-copy data path.
#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_DRAWDET (gst_nvmm_drawdet_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmDrawDet, gst_nvmm_drawdet, GST, NVMM_DRAWDET, GstBaseTransform)

G_END_DECLS
