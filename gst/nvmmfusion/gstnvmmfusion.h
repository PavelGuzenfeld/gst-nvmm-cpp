#pragma once

#include <gst/gst.h>
#include <gst/base/gstaggregator.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_FUSION (gst_nvmm_fusion_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmFusion, gst_nvmm_fusion, GST, NVMM_FUSION, GstAggregator)

G_END_DECLS
