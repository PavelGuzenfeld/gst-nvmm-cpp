/// NvmmOpticalFlowMeta — per-frame dense optical-flow field, carried as GstMeta.
///
/// `nvmmofa` attaches this to each (passthrough) NV12 NVMM buffer: the original
/// frame travels downstream unchanged/zero-copy, and the motion-vector field
/// rides along as metadata for in-process analytics consumers (e.g.
/// `nvmmflowstats`). OFA only writes a VPI-native `2S16_BL` image (a wrapped NVMM
/// surface is rejected), so the field is locked to host and copied — tightly
/// packed — into a host `GstMemory`: `mv_width * mv_height` cells, each two
/// int16 (dx, dy) in S10.5 fixed point (divide by 32 for pixels). Small
/// (e.g. 160x120 -> ~75 KB), so the copy-out is cheap; the video frame itself is
/// still passed through zero-copy.
///
/// In-process only: GstMeta does not cross the nvmmsink->nvmmappsrc IPC boundary.
#pragma once

#include <gst/gst.h>

G_BEGIN_DECLS

typedef struct _NvmmOpticalFlowMeta {
    GstMeta meta;

    GstMemory *mv;        /* NVMM SIGNED_R16G16 motion-vector field (owns a ref) */
    gint mv_width;        /* motion-vector grid width  = ceil(frame_width  / grid_size) */
    gint mv_height;       /* motion-vector grid height = ceil(frame_height / grid_size) */
    gint grid_size;       /* OFA grid size (1/2/4/8): pixels per vector cell */
    gint frame_width;     /* source frame width */
    gint frame_height;    /* source frame height */
} NvmmOpticalFlowMeta;

GType nvmm_optical_flow_meta_api_get_type(void);
#define NVMM_OPTICAL_FLOW_META_API_TYPE (nvmm_optical_flow_meta_api_get_type())

const GstMetaInfo *nvmm_optical_flow_meta_get_info(void);

#define gst_buffer_get_nvmm_optical_flow_meta(b) \
    ((NvmmOpticalFlowMeta *)gst_buffer_get_meta((b), NVMM_OPTICAL_FLOW_META_API_TYPE))

/// Attach a flow meta to @buffer. Takes ownership of a ref on @mv (the meta
/// unrefs it on free). Returns the meta (owned by the buffer) or NULL.
NvmmOpticalFlowMeta *
gst_buffer_add_nvmm_optical_flow_meta(GstBuffer *buffer, GstMemory *mv,
                                      gint mv_width, gint mv_height, gint grid_size,
                                      gint frame_width, gint frame_height);

G_END_DECLS
