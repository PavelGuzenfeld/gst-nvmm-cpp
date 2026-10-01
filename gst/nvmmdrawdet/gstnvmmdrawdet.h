#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_DRAWDET (gst_nvmm_drawdet_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmDrawDet, gst_nvmm_drawdet, GST, NVMM_DRAWDET, GstBaseTransform)

G_END_DECLS
