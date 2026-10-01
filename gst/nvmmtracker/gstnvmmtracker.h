#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_TRACKER (gst_nvmm_tracker_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmTracker, gst_nvmm_tracker, GST, NVMM_TRACKER,
                     GstBaseTransform)

G_END_DECLS
