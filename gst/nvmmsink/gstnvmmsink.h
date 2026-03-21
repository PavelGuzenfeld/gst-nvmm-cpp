/// GstNvmmSink — GStreamer sink that exports NVMM buffers via shared memory.
///
/// Receives video/x-raw(memory:NVMM) buffers and publishes them to a named
/// POSIX shared memory segment. Downstream consumers (e.g., ROS2 nodes,
/// inference engines) can attach to the segment and access the DMA-buf fd
/// for zero-copy interop.
#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_SINK (gst_nvmm_sink_get_type())
#define GST_NVMM_SINK(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVMM_SINK, GstNvmmSink))
#define GST_IS_NVMM_SINK(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVMM_SINK))

typedef struct _GstNvmmSink GstNvmmSink;
typedef struct _GstNvmmSinkClass GstNvmmSinkClass;
typedef struct _GstNvmmSinkPrivate GstNvmmSinkPrivate;

struct _GstNvmmSink {
    GstBaseSink parent;
    GstNvmmSinkPrivate *priv;
};

struct _GstNvmmSinkClass {
    GstBaseSinkClass parent_class;
};

GType gst_nvmm_sink_get_type(void);

G_END_DECLS
