#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_SAMURAI (gst_nvmm_samurai_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmSamurai, gst_nvmm_samurai,
                     GST, NVMM_SAMURAI, GstBaseTransform)

G_END_DECLS
