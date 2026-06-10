/// GstNvmmFusion — join the detector and optical-flow branches by PTS.
///
/// A GstAggregator with two sink pads:
///   - "detection": NVMM NV12 carrying GstNvmmDetMeta (nvmminfer[->nvmmtracker])
///   - "flow":      NVMM NV12 carrying NvmmOpticalFlowMeta (nvmmofa->flowstats)
/// Per output frame it takes the detection buffer as the carrier (zero-copy —
/// same NvBufSurface, made writable via the share-capable allocator) and unions
/// the flow meta onto it, so a single buffer downstream holds BOTH metas at the
/// same PTS. Phase-2 is a structural join only (co-locate); the cross-modal
/// "mark moving objects" payoff lands in Phase 3. No CUDA — pixels untouched.
#pragma once

#include <gst/gst.h>
#include <gst/base/gstaggregator.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_FUSION (gst_nvmm_fusion_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmFusion, gst_nvmm_fusion, GST, NVMM_FUSION, GstAggregator)

G_END_DECLS
