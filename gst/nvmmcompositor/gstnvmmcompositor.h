/// GstNvmmCompositor — VIC-composited multi-input NVMM mixer.
///
/// Each request sink pad places its NVMM frame into a rectangle of the output
/// NVMM frame via NvBufSurfTransform (VIC) — no CPU copy. Multi-camera mosaic /
/// overlay without DeepStream. Built on GstAggregator (not GstVideoAggregator,
/// whose GstVideoFrame mapping does not understand NVMM memory).
#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstaggregator.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_COMPOSITOR (gst_nvmm_compositor_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmCompositor, gst_nvmm_compositor,
                     GST, NVMM_COMPOSITOR, GstAggregator)

#define GST_TYPE_NVMM_COMPOSITOR_PAD (gst_nvmm_compositor_pad_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmCompositorPad, gst_nvmm_compositor_pad,
                     GST, NVMM_COMPOSITOR_PAD, GstAggregatorPad)

G_END_DECLS
