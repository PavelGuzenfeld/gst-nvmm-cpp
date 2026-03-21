/// GstNvmmConvert — GStreamer element for NVMM crop/scale/format conversion.
/// Wraps NvBufSurfTransform (Tegra VIC hardware engine).
#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_CONVERT (gst_nvmm_convert_get_type())
#define GST_NVMM_CONVERT(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVMM_CONVERT, GstNvmmConvert))

typedef struct _GstNvmmConvert GstNvmmConvert;
typedef struct _GstNvmmConvertClass GstNvmmConvertClass;
typedef struct _GstNvmmConvertPrivate GstNvmmConvertPrivate;

struct _GstNvmmConvert {
    GstBaseTransform parent;
    GstNvmmConvertPrivate* priv;
};

struct _GstNvmmConvertClass {
    GstBaseTransformClass parent_class;
};

GType gst_nvmm_convert_get_type(void);

G_END_DECLS
