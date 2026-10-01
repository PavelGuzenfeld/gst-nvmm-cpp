#pragma once

#include <gst/gst.h>

G_BEGIN_DECLS

typedef struct _NvmmOpticalFlowMeta {
    GstMeta meta;

    GstMemory *mv;
    gint mv_width;
    gint mv_height;
    gint grid_size;
    gint frame_width;
    gint frame_height;
} NvmmOpticalFlowMeta;

GType nvmm_optical_flow_meta_api_get_type(void);
#define NVMM_OPTICAL_FLOW_META_API_TYPE (nvmm_optical_flow_meta_api_get_type())

const GstMetaInfo *nvmm_optical_flow_meta_get_info(void);

#define gst_buffer_get_nvmm_optical_flow_meta(b) \
    ((NvmmOpticalFlowMeta *)gst_buffer_get_meta((b), NVMM_OPTICAL_FLOW_META_API_TYPE))

NvmmOpticalFlowMeta *
gst_buffer_add_nvmm_optical_flow_meta(GstBuffer *buffer, GstMemory *mv,
                                      gint mv_width, gint mv_height, gint grid_size,
                                      gint frame_width, gint frame_height);

G_END_DECLS
