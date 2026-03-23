#include "gstnvmmconvert.h"

#include "gstnvmmallocator.h"
#include "gstnvmmbufferpool.h"
#include "nvmm_buffer.hpp"
#include "nvmm_transform.hpp"
#include "nvmm_types.hpp"

namespace {

// Caps template for NVMM video
const char* kCapsStr =
    "video/x-raw(memory:NVMM), "
    "format=(string){NV12, RGBA, I420, BGRA}, "
    "width=(int)[1, 8192], height=(int)[1, 8192], "
    "framerate=(fraction)[0/1, 240/1]";

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
    GstBufferPool* pool;
};

G_DEFINE_TYPE_WITH_PRIVATE(GstNvmmConvert, gst_nvmm_convert, GST_TYPE_BASE_TRANSFORM)

/* Forward declarations */
static gboolean gst_nvmm_convert_stop(GstBaseTransform* trans);
static void gst_nvmm_convert_finalize(GObject* object);

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

/* Remove format info and rangify size — allows format conversion + scaling */
static GstCaps*
remove_format_and_rangify(GstCaps* caps)
{
    GstCaps* result = gst_caps_new_empty();

    for (guint i = 0; i < gst_caps_get_size(caps); i++) {
        GstStructure* s = gst_structure_copy(gst_caps_get_structure(caps, i));
        GstCapsFeatures* f = gst_caps_features_copy(
            gst_caps_get_features(caps, i));

        /* Remove format — we can convert between any supported format */
        gst_structure_remove_fields(s, "format", "colorimetry", "chroma-site",
                                    NULL);
        /* Rangify size — we can scale to any dimension */
        gst_structure_set(s,
            "width", GST_TYPE_INT_RANGE, 1, 8192,
            "height", GST_TYPE_INT_RANGE, 1, 8192,
            NULL);

        gst_caps_append_structure_full(result, s, f);
    }
    return result;
}

static GstCaps*
gst_nvmm_convert_transform_caps(GstBaseTransform* trans,
                                 GstPadDirection direction,
                                 GstCaps* caps,
                                 GstCaps* filter)
{
    (void)trans;
    (void)direction;

    GstCaps* result = remove_format_and_rangify(caps);

    if (filter) {
        GstCaps* tmp = gst_caps_intersect_full(result, filter,
                                                GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(result);
        result = tmp;
    }

    return result;
}

static GstCaps*
gst_nvmm_convert_fixate_caps(GstBaseTransform* trans,
                              GstPadDirection direction,
                              GstCaps* caps,
                              GstCaps* othercaps)
{
    auto* self = GST_NVMM_CONVERT(trans);
    GstStructure* ins = gst_caps_get_structure(caps, 0);
    GstStructure* outs;

    othercaps = gst_caps_truncate(othercaps);
    othercaps = gst_caps_make_writable(othercaps);
    outs = gst_caps_get_structure(othercaps, 0);

    /* Prefer input format if output doesn't specify one */
    if (direction == GST_PAD_SINK) {
        const gchar* in_fmt = gst_structure_get_string(ins, "format");
        if (in_fmt && !gst_structure_has_field(outs, "format")) {
            gst_structure_set(outs, "format", G_TYPE_STRING, in_fmt, NULL);
        }
    }

    /* Fixate dimensions: prefer crop size if set, otherwise input size */
    gint in_w = 0, in_h = 0;
    gst_structure_get_int(ins, "width", &in_w);
    gst_structure_get_int(ins, "height", &in_h);

    if (direction == GST_PAD_SINK) {
        /* If crop is configured, output dimensions = crop dimensions */
        if (self->priv->crop.is_valid()) {
            gst_structure_fixate_field_nearest_int(outs, "width",
                static_cast<int>(self->priv->crop.width));
            gst_structure_fixate_field_nearest_int(outs, "height",
                static_cast<int>(self->priv->crop.height));
        } else if (in_w > 0 && in_h > 0) {
            /* No crop: prefer input dimensions */
            gst_structure_fixate_field_nearest_int(outs, "width", in_w);
            gst_structure_fixate_field_nearest_int(outs, "height", in_h);
        }
    } else {
        /* SRC→SINK: prefer output dimensions as input dimensions */
        if (in_w > 0 && in_h > 0) {
            gst_structure_fixate_field_nearest_int(outs, "width", in_w);
            gst_structure_fixate_field_nearest_int(outs, "height", in_h);
        }
    }

    othercaps = gst_caps_fixate(othercaps);
    return othercaps;
}

static gboolean
gst_nvmm_convert_get_unit_size(GstBaseTransform* trans, GstCaps* caps,
                                gsize* size)
{
    (void)trans;
    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, caps)) {
        return FALSE;
    }
    *size = GST_VIDEO_INFO_SIZE(&info);
    return TRUE;
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

    GST_INFO_OBJECT(self, "Configured: %dx%d %s -> %dx%d %s",
                    GST_VIDEO_INFO_WIDTH(&self->priv->sink_info),
                    GST_VIDEO_INFO_HEIGHT(&self->priv->sink_info),
                    gst_video_format_to_string(
                        GST_VIDEO_INFO_FORMAT(&self->priv->sink_info)),
                    GST_VIDEO_INFO_WIDTH(&self->priv->src_info),
                    GST_VIDEO_INFO_HEIGHT(&self->priv->src_info),
                    gst_video_format_to_string(
                        GST_VIDEO_INFO_FORMAT(&self->priv->src_info)));

    /* Enable passthrough when caps match and no crop/flip */
    gboolean same = gst_caps_is_equal(incaps, outcaps);
    gboolean no_transform = !self->priv->crop.is_valid() &&
                            self->priv->flip == nvmm::FlipMethod::kNone;
    gst_base_transform_set_passthrough(trans, same && no_transform);

    /* Set up output buffer pool when not passthrough */
    if (!(same && no_transform)) {
        if (self->priv->pool) {
            gst_buffer_pool_set_active(self->priv->pool, FALSE);
            gst_object_unref(self->priv->pool);
            self->priv->pool = NULL;
        }

        self->priv->pool = gst_nvmm_buffer_pool_new();
        GstStructure* config = gst_buffer_pool_get_config(self->priv->pool);
        gst_buffer_pool_config_set_params(config, outcaps,
            GST_VIDEO_INFO_SIZE(&self->priv->src_info), 2, 8);
        if (!gst_buffer_pool_set_config(self->priv->pool, config)) {
            GST_ERROR_OBJECT(self, "Failed to configure output buffer pool");
            gst_object_unref(self->priv->pool);
            self->priv->pool = NULL;
            return FALSE;
        }
        gst_buffer_pool_set_active(self->priv->pool, TRUE);
    }

    return TRUE;
}

static GstFlowReturn
gst_nvmm_convert_prepare_output_buffer(GstBaseTransform* trans,
                                        GstBuffer* inbuf,
                                        GstBuffer** outbuf)
{
    auto* self = GST_NVMM_CONVERT(trans);

    /* Passthrough: reuse input buffer */
    if (gst_base_transform_is_passthrough(trans)) {
        *outbuf = inbuf;
        return GST_FLOW_OK;
    }

    /* Acquire buffer from the NVMM pool */
    if (!self->priv->pool) {
        GST_ERROR_OBJECT(self, "No output buffer pool");
        return GST_FLOW_ERROR;
    }

    GstFlowReturn ret = gst_buffer_pool_acquire_buffer(self->priv->pool,
                                                        outbuf, NULL);
    if (ret != GST_FLOW_OK || !*outbuf) {
        GST_ERROR_OBJECT(self, "Failed to acquire buffer from pool");
        return ret;
    }

    /* Copy timestamps */
    gst_buffer_copy_into(*outbuf, inbuf,
        static_cast<GstBufferCopyFlags>(GST_BUFFER_COPY_TIMESTAMPS | GST_BUFFER_COPY_FLAGS),
        0, static_cast<gsize>(-1));

    return GST_FLOW_OK;
}

/* Extract NvBufSurface from a GstBuffer — works with both our allocator
   and NVIDIA's nvvidconv/nvv4l2 allocator. NVIDIA's convention: the mapped
   data pointer IS the NvBufSurface*. */
static NvBufSurface*
get_nvbuf_surface(GstBuffer* buf)
{
    GstMemory* mem = gst_buffer_peek_memory(buf, 0);

    /* Try our allocator first */
    if (gst_is_nvmm_memory(mem)) {
        return static_cast<NvBufSurface*>(gst_nvmm_memory_get_surface(mem));
    }

    /* NVIDIA convention: map the buffer, data pointer = NvBufSurface* */
    GstMapInfo map;
    if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
        auto* surface = reinterpret_cast<NvBufSurface*>(map.data);
        gst_buffer_unmap(buf, &map);
        return surface;
    }

    return nullptr;
}

static GstFlowReturn gst_nvmm_convert_transform(GstBaseTransform* trans,
                                                  GstBuffer* inbuf,
                                                  GstBuffer* outbuf) {
    auto* self = GST_NVMM_CONVERT(trans);

    auto* src_surface = get_nvbuf_surface(inbuf);
    auto* dst_surface = get_nvbuf_surface(outbuf);

    if (!src_surface || !dst_surface) {
        GST_ERROR_OBJECT(self, "Failed to get NvBufSurface from buffers");
        return GST_FLOW_ERROR;
    }

    /* Wrap raw surfaces — we don't own these, so release() after use */
    nvmm::NvmmBuffer src_buf{src_surface};
    nvmm::NvmmBuffer dst_buf{dst_surface};

    nvmm::TransformParams params;
    params.src_crop = self->priv->crop;
    params.flip = self->priv->flip;

    auto result = nvmm::NvmmTransform::transform(src_buf, dst_buf, params);

    /* Release ownership — these surfaces belong to the pipeline allocator */
    src_buf.release();
    dst_buf.release();

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
    gobject_class->finalize = gst_nvmm_convert_finalize;

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

    GstCaps* caps = gst_caps_from_string(kCapsStr);
    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps));
    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
    gst_caps_unref(caps);

    transform_class->transform_caps = gst_nvmm_convert_transform_caps;
    transform_class->fixate_caps = gst_nvmm_convert_fixate_caps;
    transform_class->get_unit_size = gst_nvmm_convert_get_unit_size;
    transform_class->set_caps = gst_nvmm_convert_set_caps;
    transform_class->stop = gst_nvmm_convert_stop;
    transform_class->prepare_output_buffer = gst_nvmm_convert_prepare_output_buffer;
    transform_class->transform = gst_nvmm_convert_transform;

    transform_class->passthrough_on_same_caps = TRUE;
}

static gboolean
gst_nvmm_convert_stop(GstBaseTransform* trans)
{
    auto* self = GST_NVMM_CONVERT(trans);
    if (self->priv->pool) {
        gst_buffer_pool_set_active(self->priv->pool, FALSE);
        gst_object_unref(self->priv->pool);
        self->priv->pool = NULL;
    }
    return TRUE;
}

static void
gst_nvmm_convert_finalize(GObject* object)
{
    auto* self = GST_NVMM_CONVERT(object);
    if (self->priv->pool) {
        gst_buffer_pool_set_active(self->priv->pool, FALSE);
        gst_object_unref(self->priv->pool);
        self->priv->pool = NULL;
    }
    G_OBJECT_CLASS(gst_nvmm_convert_parent_class)->finalize(object);
}

static void gst_nvmm_convert_init(GstNvmmConvert* self) {
    self->priv = static_cast<GstNvmmConvertPrivate*>(
        gst_nvmm_convert_get_instance_private(self));
    self->priv->crop = {};
    self->priv->flip = nvmm::FlipMethod::kNone;
    self->priv->pool = NULL;
}
