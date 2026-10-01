#pragma once

#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_FUSEKF (gst_nvmm_fusekf_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmFuseKf, gst_nvmm_fusekf, GST, NVMM_FUSEKF,
                     GstBaseTransform)

G_END_DECLS
