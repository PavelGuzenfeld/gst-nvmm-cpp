#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstaggregator.h>

G_BEGIN_DECLS

/// GstAggregator, not GstVideoAggregator: its GstVideoFrame mapping does not
/// understand NVMM memory.
#define GST_TYPE_NVMM_COMPOSITOR (gst_nvmm_compositor_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmCompositor, gst_nvmm_compositor,
                     GST, NVMM_COMPOSITOR, GstAggregator)

#define GST_TYPE_NVMM_COMPOSITOR_PAD (gst_nvmm_compositor_pad_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmCompositorPad, gst_nvmm_compositor_pad,
                     GST, NVMM_COMPOSITOR_PAD, GstAggregatorPad)

G_END_DECLS
