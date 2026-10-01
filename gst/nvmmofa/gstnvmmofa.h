#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_OFA (gst_nvmm_ofa_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmOfa, gst_nvmm_ofa, GST, NVMM_OFA, GstBaseTransform)

G_END_DECLS
