/// GstNvmmFlowStats — example consumer of NvmmOpticalFlowMeta.
///
/// A sink that reads the per-frame optical-flow field attached by `nvmmofa`,
/// computes mean/max motion-vector magnitude (in pixels), and logs it —
/// demonstrating that the flow metadata travels and is consumable downstream.
/// No VPI dependency: it reads the host flow buffer in the meta directly.
#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_FLOWSTATS (gst_nvmm_flowstats_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmFlowStats, gst_nvmm_flowstats, GST, NVMM_FLOWSTATS, GstBaseSink)

G_END_DECLS
