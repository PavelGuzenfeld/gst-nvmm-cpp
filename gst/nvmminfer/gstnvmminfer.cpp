#include "config.h"  // PACKAGE_VERSION, HAVE_TENSORRT

#include "gstnvmminfer.h"
#include "trt_engine.hpp"
#include "preprocess.hpp"
#include "yolo_parser.hpp"
#include "nvmm_det_meta.h"
#include "gstnvmmallocator.h"

#include <nvbufsurface.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

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
    gdouble  conf_threshold;
    gdouble  iou_threshold;

    /* runtime state */
    nvmm::TrtEngine    *engine;
    nvmm::Preprocessor *pre;
    cudaStream_t        stream;
    gint                width, height;

    /* device I/O + introspected geometry */
    float  *d_input;
    float  *d_output;
    std::vector<float> *host_out;   /* output copied to host for parsing */
    std::string *in_name, *out_name;
    int     net_w, net_h;
    int     num_classes, num_proposals;
    guint64 frame_no;
};

G_DEFINE_TYPE(GstNvmmInfer, gst_nvmm_infer, GST_TYPE_BASE_TRANSFORM)

enum { PROP_0, PROP_ENGINE_FILE, PROP_NET_SCALE_FACTOR, PROP_COLOR_ORDER, PROP_DLA_CORE,
       PROP_CONF_THRESHOLD, PROP_IOU_THRESHOLD };

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
    /* Detector input must be 1x3xHxW (NCHW). */
    if (in->dims.nbDims != 4 || in->dims.d[1] != 3) {
        GST_ELEMENT_ERROR(self, RESOURCE, SETTINGS,
                          ("input \"%s\" is %s, expected 1x3xHxW (NCHW)",
                           in->name.c_str(), nvmm::dims_str(in->dims).c_str()), (nullptr));
        return FALSE;
    }
    self->net_h = (int)in->dims.d[2];
    self->net_w = (int)in->dims.d[3];

    /* YOLO head: [1, 4+num_classes, num_proposals] (channels-first). */
    if (out->dims.nbDims != 3 || out->dims.d[1] <= 4) {
        GST_ELEMENT_ERROR(self, RESOURCE, SETTINGS,
                          ("output \"%s\" is %s, expected [1, 4+classes, proposals]",
                           out->name.c_str(), nvmm::dims_str(out->dims).c_str()), (nullptr));
        return FALSE;
    }
    self->num_classes   = (int)out->dims.d[1] - 4;
    self->num_proposals = (int)out->dims.d[2];

    cudaError_t e1 = cudaMalloc((void **)&self->d_input, in->bytes);
    cudaError_t e2 = cudaMalloc((void **)&self->d_output, out->bytes);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        GST_ELEMENT_ERROR(self, RESOURCE, NO_SPACE_LEFT, ("cudaMalloc for engine I/O failed"),
                          ("%s / %s", cudaGetErrorString(e1), cudaGetErrorString(e2)));
        return FALSE;
    }
    self->host_out->resize((size_t)out->volume);

    *self->in_name = in->name;
    *self->out_name = out->name;
    if (!self->engine->bind(*self->in_name, self->d_input) ||
        !self->engine->bind(*self->out_name, self->d_output)) {
        GST_ELEMENT_ERROR(self, LIBRARY, INIT, ("failed to bind engine I/O addresses"), (nullptr));
        return FALSE;
    }

    GST_INFO_OBJECT(self, "ready: net %dx%d, %d classes, %d proposals",
                    self->net_w, self->net_h, self->num_classes, self->num_proposals);
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

    std::string err;

    /* Lazy preprocess config: needs both the network size (start) and the frame
       size (set_caps). Letterbox geometry is fixed once the frame size is known. */
    if (!self->pre->configured()) {
        if (self->width <= 0 || self->height <= 0) {
            GST_WARNING_OBJECT(self, "frame size unknown; skipping");
            return GST_FLOW_OK;
        }
        const bool rgb = self->color_order == NVMM_INFER_RGB;
        if (!self->pre->configure(self->net_w, self->net_h, self->width, self->height,
                                  rgb, (float)self->net_scale, self->stream, err)) {
            GST_ELEMENT_ERROR(self, LIBRARY, INIT, ("preprocess configure failed"),
                              ("%s", err.c_str()));
            return GST_FLOW_ERROR;
        }
    }

    nvmm::LetterboxInfo lb;
    if (!self->pre->run(surf, self->d_input, lb, err)) {
        GST_WARNING_OBJECT(self, "preprocess failed: %s", err.c_str());
        return GST_FLOW_OK;
    }

    if (!self->engine->infer(self->stream)) {
        GST_WARNING_OBJECT(self, "TensorRT enqueueV3 failed");
        return GST_FLOW_OK;
    }

    /* Copy the output back to host and wait for the whole stream (preprocess +
       inference + copy) to complete. */
    const size_t out_bytes = self->host_out->size() * sizeof(float);
    cudaError_t ce = cudaMemcpyAsync(self->host_out->data(), self->d_output, out_bytes,
                                     cudaMemcpyDeviceToHost, self->stream);
    if (ce == cudaSuccess) ce = cudaStreamSynchronize(self->stream);
    if (ce != cudaSuccess) {
        GST_WARNING_OBJECT(self, "CUDA output copy/sync failed: %s", cudaGetErrorString(ce));
        return GST_FLOW_OK;
    }

    /* Decode + NMS -> det_meta (boxes in original-frame pixel space). */
    NvmmFrameMeta fm{};
    fm.frame_number = self->frame_no++;
    fm.infer_width  = (guint32)lb.frame_w;
    fm.infer_height = (guint32)lb.frame_h;
    nvmm::YoloParams yp;
    yp.num_classes   = self->num_classes;
    yp.num_proposals = self->num_proposals;
    yp.conf_threshold = (float)self->conf_threshold;
    yp.iou_threshold  = (float)self->iou_threshold;
    bool truncated = false;
    fm.num_objects = nvmm::yolo_parse(self->host_out->data(), yp, lb, fm.objects, &truncated);
    fm.flags = truncated ? NVMM_FRAME_META_FLAG_TRUNCATED : 0u;

    gst_buffer_add_nvmm_det_meta(buf, &fm);
    GST_LOG_OBJECT(self, "frame %" G_GUINT64_FORMAT ": %u detections",
                   fm.frame_number, fm.num_objects);
    return GST_FLOW_OK;
}

static gboolean
gst_nvmm_infer_stop(GstBaseTransform *bt)
{
    auto *self = GST_NVMM_INFER(bt);
    delete self->engine;
    self->engine = nullptr;
    delete self->pre;                       /* releases its RGBA surface + planes */
    self->pre = new nvmm::Preprocessor();   /* fresh, so a re-start reconfigures */
    if (self->d_input)  { cudaFree(self->d_input);  self->d_input = nullptr; }
    if (self->d_output) { cudaFree(self->d_output); self->d_output = nullptr; }
    if (self->stream) { cudaStreamDestroy(self->stream); self->stream = nullptr; }
    return TRUE;
}

static void
gst_nvmm_infer_finalize(GObject *o)
{
    auto *self = GST_NVMM_INFER(o);
    g_free(self->engine_file);
    delete self->pre;
    delete self->host_out;
    delete self->in_name;
    delete self->out_name;
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
        case PROP_CONF_THRESHOLD:   self->conf_threshold = g_value_get_double(v); break;
        case PROP_IOU_THRESHOLD:    self->iou_threshold = g_value_get_double(v); break;
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
        case PROP_CONF_THRESHOLD:   g_value_set_double(v, self->conf_threshold); break;
        case PROP_IOU_THRESHOLD:    g_value_set_double(v, self->iou_threshold); break;
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
    g_object_class_install_property(go, PROP_CONF_THRESHOLD,
        g_param_spec_double("conf-threshold", "Confidence threshold",
            "Minimum class confidence to keep a detection",
            0.0, 1.0, 0.25, flags));
    g_object_class_install_property(go, PROP_IOU_THRESHOLD,
        g_param_spec_double("nms-iou-threshold", "NMS IoU threshold",
            "IoU above which same-class boxes are suppressed",
            0.0, 1.0, 0.45, flags));

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
    self->engine_file    = nullptr;
    self->net_scale      = 1.0 / 255.0;
    self->color_order    = NVMM_INFER_RGB;
    self->dla_core       = -1;
    self->conf_threshold = 0.25;
    self->iou_threshold  = 0.45;
    self->engine         = nullptr;
    self->pre            = new nvmm::Preprocessor();
    self->stream         = nullptr;
    self->width = self->height = 0;
    self->d_input = self->d_output = nullptr;
    self->host_out = new std::vector<float>();
    self->in_name  = new std::string();
    self->out_name = new std::string();
    self->net_w = self->net_h = 0;
    self->num_classes = self->num_proposals = 0;
    self->frame_no = 0;
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
