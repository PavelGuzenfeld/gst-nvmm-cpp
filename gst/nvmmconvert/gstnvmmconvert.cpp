#include "gstnvmmconvert.h"

#include "gstnvmmallocator.h"
#include "nvmm_buffer.hpp"
#include "nvmm_transform.hpp"
#include "nvmm_types.hpp"

namespace {

// Caps template for NVMM video
const char* kSinkCapsStr =
    "video/x-raw(memory:NVMM), "
    "format=(string){NV12, RGBA, I420, BGRA}, "
    "width=(int)[1, 8192], height=(int)[1, 8192], "
    "framerate=(fraction)[0/1, 240/1]";

const char* kSrcCapsStr =
    "video/x-raw(memory:NVMM), "
    "format=(string){NV12, RGBA, I420, BGRA}, "
    "width=(int)[1, 8192], height=(int)[1, 8192], "
    "framerate=(fraction)[0/1, 240/1]";

nvmm::ColorFormat gst_format_to_nvmm(GstVideoFormat fmt) {
    switch (fmt) {
        case GST_VIDEO_FORMAT_NV12:  return nvmm::ColorFormat::kNV12;
        case GST_VIDEO_FORMAT_RGBA:  return nvmm::ColorFormat::kRGBA;
        case GST_VIDEO_FORMAT_BGRA:  return nvmm::ColorFormat::kBGRA;
        case GST_VIDEO_FORMAT_I420:  return nvmm::ColorFormat::kI420;
        default:                     return nvmm::ColorFormat::kNV12;
    }
}

}  // namespace

enum {
    PROP_0,
    PROP_CROP_X,
    PROP_CROP_Y,
    PROP_CROP_W,
    PROP_CROP_H,
    PROP_FLIP_METHOD,
};

struct _GstNvmmConvertPrivate {
    nvmm::CropRect crop;
    nvmm::FlipMethod flip;
    GstVideoInfo sink_info;
    GstVideoInfo src_info;
};

G_DEFINE_TYPE_WITH_PRIVATE(GstNvmmConvert, gst_nvmm_convert, GST_TYPE_BASE_TRANSFORM)

static void gst_nvmm_convert_set_property(GObject* object, guint prop_id,
                                            const GValue* value, GParamSpec* pspec) {
    auto* self = GST_NVMM_CONVERT(object);
    switch (prop_id) {
        case PROP_CROP_X: self->priv->crop.x = g_value_get_uint(value); break;
        case PROP_CROP_Y: self->priv->crop.y = g_value_get_uint(value); break;
        case PROP_CROP_W: self->priv->crop.width = g_value_get_uint(value); break;
        case PROP_CROP_H: self->priv->crop.height = g_value_get_uint(value); break;
        case PROP_FLIP_METHOD:
            self->priv->flip = static_cast<nvmm::FlipMethod>(g_value_get_int(value));
            break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec); break;
    }
}

static void gst_nvmm_convert_get_property(GObject* object, guint prop_id,
                                            GValue* value, GParamSpec* pspec) {
    auto* self = GST_NVMM_CONVERT(object);
    switch (prop_id) {
        case PROP_CROP_X: g_value_set_uint(value, self->priv->crop.x); break;
        case PROP_CROP_Y: g_value_set_uint(value, self->priv->crop.y); break;
        case PROP_CROP_W: g_value_set_uint(value, self->priv->crop.width); break;
        case PROP_CROP_H: g_value_set_uint(value, self->priv->crop.height); break;
        case PROP_FLIP_METHOD:
            g_value_set_int(value, static_cast<int>(self->priv->flip));
            break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec); break;
    }
}

static GstCaps* gst_nvmm_convert_transform_caps(GstBaseTransform* trans,
                                                  GstPadDirection direction,
                                                  GstCaps* caps,
                                                  GstCaps* filter) {
    (void)trans;
    (void)direction;

    /* We can transform to any supported format/resolution within NVMM */
    GstCaps* result = gst_caps_from_string(kSrcCapsStr);

    if (filter) {
        GstCaps* tmp = gst_caps_intersect_full(result, filter,
                                                GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(result);
        result = tmp;
    }

    (void)caps;
    return result;
}

static gboolean gst_nvmm_convert_set_caps(GstBaseTransform* trans,
                                            GstCaps* incaps, GstCaps* outcaps) {
    auto* self = GST_NVMM_CONVERT(trans);

    if (!gst_video_info_from_caps(&self->priv->sink_info, incaps)) {
        GST_ERROR_OBJECT(self, "Failed to parse sink caps");
        return FALSE;
    }
    if (!gst_video_info_from_caps(&self->priv->src_info, outcaps)) {
        GST_ERROR_OBJECT(self, "Failed to parse src caps");
        return FALSE;
    }

    GST_INFO_OBJECT(self, "Configured: %dx%d -> %dx%d",
                    GST_VIDEO_INFO_WIDTH(&self->priv->sink_info),
                    GST_VIDEO_INFO_HEIGHT(&self->priv->sink_info),
                    GST_VIDEO_INFO_WIDTH(&self->priv->src_info),
                    GST_VIDEO_INFO_HEIGHT(&self->priv->src_info));
    return TRUE;
}

static GstFlowReturn gst_nvmm_convert_transform(GstBaseTransform* trans,
                                                  GstBuffer* inbuf,
                                                  GstBuffer* outbuf) {
    auto* self = GST_NVMM_CONVERT(trans);

    GstMemory* in_mem = gst_buffer_peek_memory(inbuf, 0);
    GstMemory* out_mem = gst_buffer_peek_memory(outbuf, 0);

    if (!gst_is_nvmm_memory(in_mem) || !gst_is_nvmm_memory(out_mem)) {
        GST_ERROR_OBJECT(self, "Input/output must be NVMM memory");
        return GST_FLOW_ERROR;
    }

    auto* src_surface = static_cast<NvBufSurface*>(gst_nvmm_memory_get_surface(in_mem));
    auto* dst_surface = static_cast<NvBufSurface*>(gst_nvmm_memory_get_surface(out_mem));

    if (!src_surface || !dst_surface) {
        GST_ERROR_OBJECT(self, "Failed to get NvBufSurface from memory");
        return GST_FLOW_ERROR;
    }

    /* Wrap raw surfaces (non-owning, so we use a temporary) */
    nvmm::NvmmBuffer src_buf{src_surface};
    nvmm::NvmmBuffer dst_buf{dst_surface};

    nvmm::TransformParams params;
    params.src_crop = self->priv->crop;
    params.flip = self->priv->flip;

    auto result = nvmm::NvmmTransform::transform(src_buf, dst_buf, params);

    /* Release without destroying — we don't own these surfaces */
    /* This is a design limitation of wrapping raw pointers; in production,
       we'd use a separate non-owning view type. For now, we leak the raw
       pointer out to prevent double-free. */

    if (!result) {
        GST_ERROR_OBJECT(self, "Transform failed: %s", result.error().detail.c_str());
        return GST_FLOW_ERROR;
    }

    return GST_FLOW_OK;
}

static void gst_nvmm_convert_class_init(GstNvmmConvertClass* klass) {
    auto* gobject_class = G_OBJECT_CLASS(klass);
    auto* element_class = GST_ELEMENT_CLASS(klass);
    auto* transform_class = GST_BASE_TRANSFORM_CLASS(klass);

    gobject_class->set_property = gst_nvmm_convert_set_property;
    gobject_class->get_property = gst_nvmm_convert_get_property;

    g_object_class_install_property(gobject_class, PROP_CROP_X,
        g_param_spec_uint("crop-x", "Crop X", "Source crop X offset",
                          0, 8192, 0, G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_CROP_Y,
        g_param_spec_uint("crop-y", "Crop Y", "Source crop Y offset",
                          0, 8192, 0, G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_CROP_W,
        g_param_spec_uint("crop-w", "Crop Width", "Source crop width (0 = full)",
                          0, 8192, 0, G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_CROP_H,
        g_param_spec_uint("crop-h", "Crop Height", "Source crop height (0 = full)",
                          0, 8192, 0, G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_FLIP_METHOD,
        g_param_spec_int("flip-method", "Flip Method",
                         "Video flip method (0=none, 1=90CW, 2=180, 3=90CCW, 4=flipH, 6=flipV)",
                         0, 7, 0, G_PARAM_READWRITE));

    gst_element_class_set_static_metadata(element_class,
        "NVMM Video Converter",
        "Filter/Converter/Video",
        "Crop, scale, and convert video using Tegra VIC (NvBufSurfTransform)",
        "Pavel Guzenfeld");

    GstCaps* sink_caps = gst_caps_from_string(kSinkCapsStr);
    GstCaps* src_caps = gst_caps_from_string(kSrcCapsStr);
    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, sink_caps));
    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, src_caps));
    gst_caps_unref(sink_caps);
    gst_caps_unref(src_caps);

    transform_class->transform_caps = gst_nvmm_convert_transform_caps;
    transform_class->set_caps = gst_nvmm_convert_set_caps;
    transform_class->transform = gst_nvmm_convert_transform;
}

static void gst_nvmm_convert_init(GstNvmmConvert* self) {
    self->priv = static_cast<GstNvmmConvertPrivate*>(
        gst_nvmm_convert_get_instance_private(self));
    self->priv->crop = {};
    self->priv->flip = nvmm::FlipMethod::kNone;
}
