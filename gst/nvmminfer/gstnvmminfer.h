#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_INFER (gst_nvmm_infer_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmInfer, gst_nvmm_infer, GST, NVMM_INFER, GstBaseTransform)

G_END_DECLS
