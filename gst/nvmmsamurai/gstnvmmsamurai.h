/// GstNvmmSamurai — SAMURAI (SAM2.1) visual tracker as a zero-copy NVMM element.
///
/// In-place passthrough transform on video/x-raw(memory:NVMM),NV12. Reads the
/// upstream GstNvmmDetMeta (YOLO detections), runs the SAMURAI tracker (5 TRT
/// engines on one CUDA stream) on the NV12 surface, and attaches a
/// GstNvmmTrackMeta with the tracked box. Seeds autonomously from the first
/// confident target-class detection; later re-seeds are driven by a downstream
/// nvmmfusekf event (Phase C). The frame surface is never copied.
///
/// Phase B: see PLAN.md / docs.
#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_SAMURAI (gst_nvmm_samurai_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmSamurai, gst_nvmm_samurai,
                     GST, NVMM_SAMURAI, GstBaseTransform)

G_END_DECLS
