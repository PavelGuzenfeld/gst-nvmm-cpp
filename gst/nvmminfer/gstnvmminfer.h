/// GstNvmmInfer — DeepStream-free TensorRT inference on NVMM frames.
///
/// Passthrough transform: the NV12 NVMM frame flows downstream unchanged
/// (zero-copy); each frame is preprocessed (VIC resize/convert + NPP
/// normalize/planarize), run through a TensorRT engine on DLA/GPU, and the
/// parsed detections are attached as a GstNvmmDetMeta. No DeepStream dependency.
///
/// Phase 1: single detector engine -> det_meta. See docs/B5_NVMMINFER_DESIGN.md.
#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_INFER (gst_nvmm_infer_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmInfer, gst_nvmm_infer, GST, NVMM_INFER, GstBaseTransform)

G_END_DECLS
