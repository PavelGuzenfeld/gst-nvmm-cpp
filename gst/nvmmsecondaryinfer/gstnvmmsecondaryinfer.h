#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_SECONDARY_INFER (gst_nvmm_secondary_infer_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmSecondaryInfer, gst_nvmm_secondary_infer,
                     GST, NVMM_SECONDARY_INFER, GstBaseTransform)

G_END_DECLS
