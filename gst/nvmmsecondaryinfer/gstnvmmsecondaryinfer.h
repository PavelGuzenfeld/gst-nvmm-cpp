/// GstNvmmSecondaryInfer — TensorRT cascade classifier on detected objects.
///
/// Passthrough transform (the Phase-3 cascade node): reads the upstream
/// GstNvmmDetMeta, VIC-crops each detection's ROI out of the NV12 NVMM frame,
/// classifies it with a TensorRT engine, and attaches the results as a
/// GstNvmmClassMeta sibling (index-aligned with the det meta). Multi-rate:
/// a track is re-inferred only every `infer-interval` frames; in between the
/// per-tracker_id cache serves the last result. No DeepStream dependency.
///
/// Phase 3.2: see docs/B5_NVMMINFER_DESIGN.md.
#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_SECONDARY_INFER (gst_nvmm_secondary_infer_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmSecondaryInfer, gst_nvmm_secondary_infer,
                     GST, NVMM_SECONDARY_INFER, GstBaseTransform)

G_END_DECLS
