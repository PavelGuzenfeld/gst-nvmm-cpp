/// GstNvmmAppSrc — GStreamer source that reads NVMM frames from shared memory.
///
/// Connects to a named POSIX shared memory segment (written by nvmmsink or
/// an external producer) and pushes frames into a GStreamer pipeline as
/// video/x-raw(memory:NVMM) buffers.
#pragma once

#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_APP_SRC (gst_nvmm_app_src_get_type())
#define GST_NVMM_APP_SRC(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVMM_APP_SRC, GstNvmmAppSrc))
#define GST_IS_NVMM_APP_SRC(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVMM_APP_SRC))

typedef struct _GstNvmmAppSrc GstNvmmAppSrc;
typedef struct _GstNvmmAppSrcClass GstNvmmAppSrcClass;
typedef struct _GstNvmmAppSrcPrivate GstNvmmAppSrcPrivate;

struct _GstNvmmAppSrc {
    GstPushSrc parent;
    GstNvmmAppSrcPrivate *priv;
};

struct _GstNvmmAppSrcClass {
    GstPushSrcClass parent_class;
};

GType gst_nvmm_app_src_get_type(void);

G_END_DECLS
