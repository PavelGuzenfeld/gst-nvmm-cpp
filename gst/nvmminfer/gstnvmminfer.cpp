#include "config.h"  // PACKAGE_VERSION, HAVE_TENSORRT

#include "gstnvmminfer.h"
#include "trt_engine.hpp"
#include "nvmm_det_meta.h"
#include "gstnvmmallocator.h"

#include <nvbufsurface.h>
#include <cuda_runtime.h>

#include <string>

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_infer_debug);
#define GST_CAT_DEFAULT gst_nvmm_infer_debug

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

/* ---- color-order enum ---- */

enum { NVMM_INFER_RGB = 0, NVMM_INFER_BGR = 1 };

#define GST_TYPE_NVMM_INFER_COLOR_ORDER (gst_nvmm_infer_color_order_get_type())
static GType
gst_nvmm_infer_color_order_get_type(void)
{
    static GType t = 0;
    static const GEnumValue v[] = {
        { NVMM_INFER_RGB, "RGB (most ONNX detectors)", "rgb" },
        { NVMM_INFER_BGR, "BGR (Caffe-derived models)", "bgr" },
        { 0, nullptr, nullptr },
    };
    if (g_once_init_enter(&t)) {
        GType tmp = g_enum_register_static("GstNvmmInferColorOrder", v);
        g_once_init_leave(&t, tmp);
    }
    return t;
}

/* ---- element ---- */

struct _GstNvmmInfer {
    GstBaseTransform parent;

    /* properties */
    gchar   *engine_file;
    gdouble  net_scale;     /* multiply pixels by this (YOLO: 1/255) */
    gint     color_order;   /* NVMM_INFER_{RGB,BGR} */
    gint     dla_core;      /* -1 = GPU / as-built; 0/1 = DLA core (onnx-build path) */

    /* runtime state */
    nvmm::TrtEngine *engine;
    cudaStream_t     stream;
    gint             width, height;
};

G_DEFINE_TYPE(GstNvmmInfer, gst_nvmm_infer, GST_TYPE_BASE_TRANSFORM)

enum { PROP_0, PROP_ENGINE_FILE, PROP_NET_SCALE_FACTOR, PROP_COLOR_ORDER, PROP_DLA_CORE };

/* Detector input is fed from decoded NV12 NVMM video. Frame travels through
   unchanged; only detection meta is attached. */
static GstStaticPadTemplate sink_tmpl = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string)NV12, "
                    "width=(int)[32,8192], height=(int)[32,8192], "
                    "framerate=(fraction)[0/1, 240/1]"));
static GstStaticPadTemplate src_tmpl = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string)NV12, "
                    "width=(int)[32,8192], height=(int)[32,8192], "
                    "framerate=(fraction)[0/1, 240/1]"));

static NvBufSurface *
surface_of(GstBuffer *buf)
{
    GstMemory *m = gst_buffer_peek_memory(buf, 0);
    if (m && gst_is_nvmm_memory(m))
        return static_cast<NvBufSurface *>(gst_nvmm_memory_get_surface(m));
    GstMapInfo map;
    if (m && gst_buffer_map(buf, &map, GST_MAP_READ)) {
        auto *s = reinterpret_cast<NvBufSurface *>(map.data);
        gst_buffer_unmap(buf, &map);
        return s;
    }
    return nullptr;
}

static gboolean
gst_nvmm_infer_set_caps(GstBaseTransform *bt, GstCaps *incaps, GstCaps *)
{
    auto *self = GST_NVMM_INFER(bt);
    GstStructure *s = gst_caps_get_structure(incaps, 0);
    if (!gst_structure_get_int(s, "width", &self->width) ||
        !gst_structure_get_int(s, "height", &self->height)) {
        GST_ERROR_OBJECT(self, "caps missing width/height");
        return FALSE;
    }
    return TRUE;
}

static gboolean
gst_nvmm_infer_start(GstBaseTransform *bt)
{
    auto *self = GST_NVMM_INFER(bt);

    if (!self->engine_file || !self->engine_file[0]) {
        GST_ELEMENT_ERROR(self, RESOURCE, NOT_FOUND,
                          ("the \"engine-file\" property is required"),
                          ("build one on the target with trtexec --saveEngine"));
        return FALSE;
    }

    cudaError_t ce = cudaStreamCreate(&self->stream);
    if (ce != cudaSuccess) {
        GST_ELEMENT_ERROR(self, LIBRARY, INIT, ("cudaStreamCreate failed"),
                          ("%s", cudaGetErrorString(ce)));
        return FALSE;
    }

    std::string err;
    auto eng = nvmm::TrtEngine::load_file(self->engine_file, err);
    if (!eng) {
        GST_ELEMENT_ERROR(self, RESOURCE, OPEN_READ,
                          ("failed to load TensorRT engine \"%s\"", self->engine_file),
                          ("%s", err.c_str()));
        cudaStreamDestroy(self->stream);
        self->stream = nullptr;
        return FALSE;
    }
    self->engine = eng.release();

    /* Introspect + log the engine's I/O so a misbuilt engine is obvious. */
    GST_INFO_OBJECT(self, "loaded engine \"%s\": %zu I/O tensors",
                    self->engine_file, self->engine->tensors().size());
    for (const auto &t : self->engine->tensors()) {
        GST_INFO_OBJECT(self, "  %-6s %-20s %s [%s] %zu bytes",
                        t.is_input ? "INPUT" : "OUTPUT", t.name.c_str(),
                        nvmm::dims_str(t.dims).c_str(), nvmm::dtype_str(t.dtype), t.bytes);
    }

    const nvmm::TensorInfo *in = self->engine->input0();
    const nvmm::TensorInfo *out = self->engine->output0();
    if (!in || !out) {
        GST_ELEMENT_ERROR(self, RESOURCE, SETTINGS,
                          ("engine has no %s tensor", in ? "output" : "input"), (nullptr));
        return FALSE;
    }
    /* A detector input is expected to be 1x3xHxW (NCHW). Warn — don't fail —
       so other shapes can still be inspected during bring-up. */
    if (in->dims.nbDims != 4 || in->dims.d[1] != 3) {
        GST_WARNING_OBJECT(self, "input \"%s\" is %s, expected 1x3xHxW (NCHW)",
                           in->name.c_str(), nvmm::dims_str(in->dims).c_str());
    }
    return TRUE;
}

static GstFlowReturn
gst_nvmm_infer_transform_ip(GstBaseTransform *bt, GstBuffer *buf)
{
    auto *self = GST_NVMM_INFER(bt);

    NvBufSurface *surf = surface_of(buf);
    if (!surf) {
        GST_WARNING_OBJECT(self, "no NvBufSurface in buffer");
        return GST_FLOW_OK;
    }

    /* Phase-1 slice 1: engine is loaded + introspected at start(); the VIC+NPP
       preprocess, enqueueV3, YOLO parse and det_meta attach land in the next
       slices. For now the frame passes through unchanged. */
    GST_LOG_OBJECT(self, "frame %ux%u — inference path pending (preprocess+parse)",
                   self->width, self->height);
    return GST_FLOW_OK;
}

static gboolean
gst_nvmm_infer_stop(GstBaseTransform *bt)
{
    auto *self = GST_NVMM_INFER(bt);
    delete self->engine;
    self->engine = nullptr;
    if (self->stream) { cudaStreamDestroy(self->stream); self->stream = nullptr; }
    return TRUE;
}

static void
gst_nvmm_infer_finalize(GObject *o)
{
    auto *self = GST_NVMM_INFER(o);
    g_free(self->engine_file);
    G_OBJECT_CLASS(gst_nvmm_infer_parent_class)->finalize(o);
}

static void
gst_nvmm_infer_set_property(GObject *o, guint id, const GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_INFER(o);
    switch (id) {
        case PROP_ENGINE_FILE:      g_free(self->engine_file);
                                    self->engine_file = g_value_dup_string(v); break;
        case PROP_NET_SCALE_FACTOR: self->net_scale = g_value_get_double(v); break;
        case PROP_COLOR_ORDER:      self->color_order = g_value_get_enum(v); break;
        case PROP_DLA_CORE:         self->dla_core = g_value_get_int(v); break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p); break;
    }
}

static void
gst_nvmm_infer_get_property(GObject *o, guint id, GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_INFER(o);
    switch (id) {
        case PROP_ENGINE_FILE:      g_value_set_string(v, self->engine_file); break;
        case PROP_NET_SCALE_FACTOR: g_value_set_double(v, self->net_scale); break;
        case PROP_COLOR_ORDER:      g_value_set_enum(v, self->color_order); break;
        case PROP_DLA_CORE:         g_value_set_int(v, self->dla_core); break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p); break;
    }
}

static void
gst_nvmm_infer_class_init(GstNvmmInferClass *klass)
{
    auto *go = G_OBJECT_CLASS(klass);
    auto *el = GST_ELEMENT_CLASS(klass);
    auto *bt = GST_BASE_TRANSFORM_CLASS(klass);

    go->set_property = gst_nvmm_infer_set_property;
    go->get_property = gst_nvmm_infer_get_property;
    go->finalize     = gst_nvmm_infer_finalize;

    auto flags = (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS);
    g_object_class_install_property(go, PROP_ENGINE_FILE,
        g_param_spec_string("engine-file", "Engine file",
            "Path to a serialized TensorRT .engine (build on the target with trtexec)",
            nullptr, flags));
    g_object_class_install_property(go, PROP_NET_SCALE_FACTOR,
        g_param_spec_double("net-scale-factor", "Net scale factor",
            "Pixel scale applied during preprocess (e.g. 1/255 for YOLO)",
            0.0, G_MAXDOUBLE, 1.0 / 255.0, flags));
    g_object_class_install_property(go, PROP_COLOR_ORDER,
        g_param_spec_enum("color-order", "Color order",
            "Channel order the engine expects",
            GST_TYPE_NVMM_INFER_COLOR_ORDER, NVMM_INFER_RGB, flags));
    g_object_class_install_property(go, PROP_DLA_CORE,
        g_param_spec_int("dla-core", "DLA core",
            "DLA core for the onnx-build path (-1 = GPU/as-built); for a prebuilt "
            "engine-file the accelerator is already baked in",
            -1, 1, -1, flags));

    gst_element_class_add_static_pad_template(el, &sink_tmpl);
    gst_element_class_add_static_pad_template(el, &src_tmpl);
    gst_element_class_set_static_metadata(el,
        "NVMM Inference (TensorRT)", "Filter/Analyzer/Video",
        "DeepStream-free TensorRT inference on NVMM frames; passes the frame "
        "through and attaches detections as NvmmDetMeta",
        "Pavel Guzenfeld");

    bt->transform_ip = gst_nvmm_infer_transform_ip;
    bt->set_caps     = gst_nvmm_infer_set_caps;
    bt->start        = gst_nvmm_infer_start;
    bt->stop         = gst_nvmm_infer_stop;
    bt->passthrough_on_same_caps = FALSE;

    GST_DEBUG_CATEGORY_INIT(gst_nvmm_infer_debug, "nvmminfer", 0, "NVMM TensorRT inference");
}

static void
gst_nvmm_infer_init(GstNvmmInfer *self)
{
    self->engine_file = nullptr;
    self->net_scale   = 1.0 / 255.0;
    self->color_order = NVMM_INFER_RGB;
    self->dla_core    = -1;
    self->engine      = nullptr;
    self->stream      = nullptr;
    self->width = self->height = 0;
    /* In-place: same caps in/out, the frame's pixels are never copied —
       transform_ip gets the writable buffer and only attaches detection meta. */
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
}

/* ---- plugin ---- */

static gboolean
plugin_init(GstPlugin *plugin)
{
    return gst_element_register(plugin, "nvmminfer", GST_RANK_NONE, GST_TYPE_NVMM_INFER);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    nvmminfer, "DeepStream-free TensorRT inference on NVMM frames",
    plugin_init, PACKAGE_VERSION, "LGPL", "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp")
