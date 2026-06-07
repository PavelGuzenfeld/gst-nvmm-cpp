/// GstNvmmOfa — dense optical flow on the Orin OFA engine via VPI.
///
/// Passthrough transform: the NV12 NVMM frame flows downstream unchanged
/// (zero-copy); for every consecutive pair of frames it runs VPI dense optical
/// flow on the OFA hardware engine and attaches the motion-vector field as an
/// NvmmOpticalFlowMeta. Orin-only (Xavier has no OFA).
#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_OFA (gst_nvmm_ofa_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmOfa, gst_nvmm_ofa, GST, NVMM_OFA, GstBaseTransform)

G_END_DECLS
