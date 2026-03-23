#include "gstnvmmconvert.h"

#include "gstnvmmallocator.h"
#include "gstnvmmbufferpool.h"
#include "nvmm_buffer.hpp"
#include "nvmm_transform.hpp"
#include "nvmm_types.hpp"

#include <atomic>

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_convert_debug);
#define GST_CAT_DEFAULT gst_nvmm_convert_debug

namespace {

const char* kCapsStr =
    "video/x-raw(memory:NVMM), "
    "format=(string){NV12, RGBA, I420, BGRA}, "
    "width=(int)[1, 8192], height=(int)[1, 8192], "
    "framerate=(fraction)[0/1, 240/1]";

}  // namespace

#define GST_TYPE_NVMM_FLIP_METHOD (gst_nvmm_flip_method_get_type())
static GType
gst_nvmm_flip_method_get_type(void)
{
    static GType type = 0;
    if (g_once_init_enter(&type)) {
        static const GEnumValue values[] = {
            {0, "No flip", "none"},
            {1, "Rotate 90 CW", "rotate-90"},
            {2, "Rotate 180", "rotate-180"},
            {3, "Rotate 90 CCW", "rotate-270"},
            {4, "Flip horizontal", "horizontal-flip"},
            {5, "Transpose", "upper-right-diagonal"},
            {6, "Flip vertical", "vertical-flip"},
            {7, "Inverse transpose", "upper-left-diagonal"},
            {0, NULL, NULL}
        };
        GType tmp = g_enum_register_static("GstNvmmFlipMethod", values);
        g_once_init_leave(&type, tmp);
    }
    return type;
}

enum {
    PROP_0,
    PROP_CROP_X,
    PROP_CROP_Y,
    PROP_CROP_W,
    PROP_CROP_H,
    PROP_FLIP_METHOD,
};

struct _GstNvmmConvertPrivate {
    std::atomic<uint32_t> crop_x;
    std::atomic<uint32_t> crop_y;
    std::atomic<uint32_t> crop_w;
    std::atomic<uint32_t> crop_h;
    std::atomic<int> flip;
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
        case PROP_CROP_X: self->priv->crop_x.store(g_value_get_uint(value)); break;
        case PROP_CROP_Y: self->priv->crop_y.store(g_value_get_uint(value)); break;
        case PROP_CROP_W: self->priv->crop_w.store(g_value_get_uint(value)); break;
        case PROP_CROP_H: self->priv->crop_h.store(g_value_get_uint(value)); break;
        case PROP_FLIP_METHOD:
            self->priv->flip.store(g_value_get_enum(value));
            break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec); break;
    }
}

static void gst_nvmm_convert_get_property(GObject* object, guint prop_id,
                                            GValue* value, GParamSpec* pspec) {
    auto* self = GST_NVMM_CONVERT(object);
    switch (prop_id) {
        case PROP_CROP_X: g_value_set_uint(value, self->priv->crop_x.load()); break;
        case PROP_CROP_Y: g_value_set_uint(value, self->priv->crop_y.load()); break;
        case PROP_CROP_W: g_value_set_uint(value, self->priv->crop_w.load()); break;
        case PROP_CROP_H: g_value_set_uint(value, self->priv->crop_h.load()); break;
        case PROP_FLIP_METHOD:
            g_value_set_enum(value, self->priv->flip.load());
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
        uint32_t cw = self->priv->crop_w.load();
        uint32_t ch = self->priv->crop_h.load();
        if (cw > 0 && ch > 0) {
            gst_structure_fixate_field_nearest_int(outs, "width",
                static_cast<int>(cw));
            gst_structure_fixate_field_nearest_int(outs, "height",
                static_cast<int>(ch));
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
    gboolean no_transform = (self->priv->crop_w.load() == 0 ||
                             self->priv->crop_h.load() == 0) &&
                            self->priv->flip.load() == 0;
    gst_base_transform_set_passthrough(trans, same && no_transform);

    /* Pool setup is handled by decide_allocation */

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
    params.src_crop.x = self->priv->crop_x.load();
    params.src_crop.y = self->priv->crop_y.load();
    params.src_crop.width = self->priv->crop_w.load();
    params.src_crop.height = self->priv->crop_h.load();
    params.flip = static_cast<nvmm::FlipMethod>(self->priv->flip.load());

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

static gboolean
gst_nvmm_convert_propose_allocation(GstBaseTransform* trans,
                                     GstQuery* decide_query,
                                     GstQuery* query)
{
    (void)trans;
    (void)decide_query;

    /* Tell upstream we support video meta (non-standard strides) */
    gst_query_add_allocation_meta(query, GST_VIDEO_META_API_TYPE, NULL);

    /* Propose an NVMM buffer pool for upstream to use */
    GstCaps* caps;
    gst_query_parse_allocation(query, &caps, NULL);
    if (caps) {
        GstBufferPool* pool = gst_nvmm_buffer_pool_new();
        GstStructure* config = gst_buffer_pool_get_config(pool);

        GstVideoInfo info;
        if (gst_video_info_from_caps(&info, caps)) {
            gst_buffer_pool_config_set_params(config, caps,
                GST_VIDEO_INFO_SIZE(&info), 2, 8);
            if (gst_buffer_pool_set_config(pool, config)) {
                gst_query_add_allocation_pool(query, pool,
                    GST_VIDEO_INFO_SIZE(&info), 2, 8);
            }
        }
        gst_object_unref(pool);
    }

    return TRUE;
}

static gboolean
gst_nvmm_convert_decide_allocation(GstBaseTransform* trans,
                                    GstQuery* query)
{
    auto* self = GST_NVMM_CONVERT(trans);

    /* If passthrough, no output pool needed */
    if (gst_base_transform_is_passthrough(trans)) {
        return GST_BASE_TRANSFORM_CLASS(gst_nvmm_convert_parent_class)
            ->decide_allocation(trans, query);
    }

    /* Check if downstream provided a pool we can use */
    guint n_pools = gst_query_get_n_allocation_pools(query);
    GstBufferPool* pool = NULL;
    guint size = 0, min = 2, max = 8;

    if (n_pools > 0) {
        gst_query_parse_nth_allocation_pool(query, 0, &pool, &size, &min, &max);
    }

    /* If no pool from downstream, create our NVMM pool */
    if (!pool) {
        pool = gst_nvmm_buffer_pool_new();

        GstCaps* outcaps;
        gst_query_parse_allocation(query, &outcaps, NULL);
        if (outcaps) {
            GstVideoInfo info;
            if (gst_video_info_from_caps(&info, outcaps)) {
                size = GST_VIDEO_INFO_SIZE(&info);
            }
        }
    }

    /* Configure and activate */
    GstStructure* config = gst_buffer_pool_get_config(pool);
    GstCaps* outcaps;
    gst_query_parse_allocation(query, &outcaps, NULL);
    gst_buffer_pool_config_set_params(config, outcaps, size,
                                       min < 2 ? 2 : min, max < 4 ? 8 : max);
    gst_buffer_pool_set_config(pool, config);

    /* Replace the pool in our private data */
    if (self->priv->pool) {
        gst_buffer_pool_set_active(self->priv->pool, FALSE);
        gst_object_unref(self->priv->pool);
    }
    self->priv->pool = pool;
    gst_buffer_pool_set_active(self->priv->pool, TRUE);

    /* Update the query */
    if (n_pools > 0) {
        gst_query_set_nth_allocation_pool(query, 0, pool, size, min, max);
    } else {
        gst_query_add_allocation_pool(query, pool, size, min, max);
    }

    return TRUE;
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
        g_param_spec_enum("flip-method", "Flip Method",
                          "Video flip/rotation method",
                          GST_TYPE_NVMM_FLIP_METHOD, 0,
                          static_cast<GParamFlags>(G_PARAM_READWRITE |
                              G_PARAM_STATIC_STRINGS)));

    GST_DEBUG_CATEGORY_INIT(gst_nvmm_convert_debug, "nvmmconvert", 0,
                            "NVMM video converter");

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
    transform_class->propose_allocation = gst_nvmm_convert_propose_allocation;
    transform_class->decide_allocation = gst_nvmm_convert_decide_allocation;
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
    self->priv->crop_x = 0;
    self->priv->crop_y = 0;
    self->priv->crop_w = 0;
    self->priv->crop_h = 0;
    self->priv->flip = 0;
    self->priv->pool = NULL;
}
