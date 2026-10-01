#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_FLOWSTATS (gst_nvmm_flowstats_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmFlowStats, gst_nvmm_flowstats, GST, NVMM_FLOWSTATS, GstBaseSink)

G_END_DECLS
