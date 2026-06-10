#include "config.h"  // PACKAGE_VERSION, HAVE_TENSORRT

#include "gstnvmmsecondaryinfer.h"
#include "roi_preprocess.hpp"
#include "secondary_cache.hpp"
#include "trt_engine.hpp"
#include "nvmm_class_meta.h"
#include "nvmm_det_meta.h"
#include "gstnvmmallocator.h"

#include <nvbufsurface.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_secondary_infer_debug);
#define GST_CAT_DEFAULT gst_nvmm_secondary_infer_debug

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

/* ---- enums ---- */

enum { NVMM_SEC_RGB = 0, NVMM_SEC_BGR = 1 };

#define GST_TYPE_NVMM_SECONDARY_INFER_COLOR_ORDER \
    (gst_nvmm_secondary_infer_color_order_get_type())
static GType
gst_nvmm_secondary_infer_color_order_get_type(void)
{
    static GType t = 0;
    static const GEnumValue v[] = {
        { NVMM_SEC_RGB, "RGB (most ONNX classifiers)", "rgb" },
        { NVMM_SEC_BGR, "BGR (Caffe-derived models)", "bgr" },
        { 0, nullptr, nullptr },
    };
    if (g_once_init_enter(&t)) {
        GType tmp = g_enum_register_static("GstNvmmSecondaryInferColorOrder", v);
        g_once_init_leave(&t, tmp);
    }
    return t;
}

enum { NVMM_SEC_ACT_SOFTMAX = 0, NVMM_SEC_ACT_NONE = 1 };

#define GST_TYPE_NVMM_SECONDARY_INFER_ACTIVATION \
    (gst_nvmm_secondary_infer_activation_get_type())
static GType
gst_nvmm_secondary_infer_activation_get_type(void)
{
    static GType t = 0;
    static const GEnumValue v[] = {
        { NVMM_SEC_ACT_SOFTMAX, "Softmax over the logits (raw classifier head)", "softmax" },
        { NVMM_SEC_ACT_NONE, "None (engine output is already probabilities)", "none" },
        { 0, nullptr, nullptr },
    };
    if (g_once_init_enter(&t)) {
        GType tmp = g_enum_register_static("GstNvmmSecondaryInferActivation", v);
        g_once_init_leave(&t, tmp);
    }
    return t;
}

/* ---- element ---- */

struct _GstNvmmSecondaryInfer {
    GstBaseTransform parent;

    /* properties */
    gchar   *engine_file;
    gchar   *labels_file;
    guint    infer_interval;  /* re-infer a track every N frames */
    guint    max_track_age;   /* drop a cached track unseen this many frames */
    guint    min_roi;         /* skip boxes smaller than this (surface px) */
    gdouble  net_scale;       /* multiply pixels by this (e.g. 1/255) */
    gint     color_order;     /* NVMM_SEC_{RGB,BGR} */
    gint     activation;      /* NVMM_SEC_ACT_* */
    gdouble  conf_threshold;  /* min top-1 score to attach a result */

    /* runtime state */
    nvmm::TrtEngine        *engine;
    nvmm::RoiPreprocessor  *pre;
    nvmm::SecondaryCache   *cache;
    cudaStream_t            stream;

    float  *d_input;
    float  *d_output;
    std::vector<float>       *host_out;  /* output copied to host for argmax */
    std::vector<std::string> *labels;    /* from labels-file (may be empty) */
    int     net_w, net_h;
    int     num_classes;
    guint64 frame_no;
};

G_DEFINE_TYPE(GstNvmmSecondaryInfer, gst_nvmm_secondary_infer, GST_TYPE_BASE_TRANSFORM)

enum { PROP_0, PROP_ENGINE_FILE, PROP_LABELS_FILE, PROP_INFER_INTERVAL,
       PROP_MAX_TRACK_AGE, PROP_MIN_ROI_SIZE, PROP_NET_SCALE_FACTOR,
       PROP_COLOR_ORDER, PROP_OUTPUT_ACTIVATION, PROP_CONF_THRESHOLD };

/* Cascade node: fed from the decoded NV12 NVMM stream that already carries
   det meta (nvmminfer/nvmmtracker/nvmmfusion upstream). Frame travels through
   unchanged; only classification meta is attached. */
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

static gboolean gst_nvmm_secondary_infer_stop(GstBaseTransform *bt);  /* unwind failed start */

static gboolean
load_labels(GstNvmmSecondaryInfer *self)
{
    self->labels->clear();
    if (!self->labels_file || !self->labels_file[0])
        return TRUE;
    gchar *contents = nullptr;
    GError *gerr = nullptr;
    if (!g_file_get_contents(self->labels_file, &contents, nullptr, &gerr)) {
        GST_ELEMENT_ERROR(self, RESOURCE, OPEN_READ,
                          ("failed to read labels-file \"%s\"", self->labels_file),
                          ("%s", gerr ? gerr->message : "unknown error"));
        g_clear_error(&gerr);
        return FALSE;
    }
    gchar **lines = g_strsplit(contents, "\n", -1);
    g_free(contents);
    for (gchar **l = lines; *l; l++) {
        gchar *s = g_strstrip(*l);
        if (s[0])
            self->labels->emplace_back(s);
    }
    g_strfreev(lines);
    return TRUE;
}

static gboolean
gst_nvmm_secondary_infer_start(GstBaseTransform *bt)
{
    auto *self = GST_NVMM_SECONDARY_INFER(bt);

    if (!self->engine_file || !self->engine_file[0]) {
        GST_ELEMENT_ERROR(self, RESOURCE, NOT_FOUND,
                          ("the \"engine-file\" property is required"),
                          ("build one on the target with trtexec --saveEngine"));
        return FALSE;
    }
    if (!load_labels(self))
        return FALSE;

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

    GST_INFO_OBJECT(self, "loaded engine \"%s\": %zu I/O tensors",
                    self->engine_file, self->engine->tensors().size());
    for (const auto &t : self->engine->tensors()) {
        GST_INFO_OBJECT(self, "  %-6s %-20s %s [%s] %zu bytes",
                        t.is_input ? "INPUT" : "OUTPUT", t.name.c_str(),
                        nvmm::dims_str(t.dims).c_str(), nvmm::dtype_str(t.dtype), t.bytes);
    }

    /* Exactly one input + one output (same contract as nvmminfer). */
    size_t n_in = 0, n_out = 0;
    for (const auto &t : self->engine->tensors()) (t.is_input ? n_in : n_out)++;
    if (n_in != 1 || n_out != 1) {
        GST_ELEMENT_ERROR(self, RESOURCE, SETTINGS,
                          ("engine has %zu input(s)/%zu output(s), expected exactly 1/1",
                           n_in, n_out), (nullptr));
        gst_nvmm_secondary_infer_stop(bt);
        return FALSE;
    }
    const nvmm::TensorInfo *in = self->engine->input0();
    const nvmm::TensorInfo *out = self->engine->output0();
    if (in->dtype != nvinfer1::DataType::kFLOAT ||
        out->dtype != nvinfer1::DataType::kFLOAT) {
        GST_ELEMENT_ERROR(self, RESOURCE, SETTINGS,
                          ("engine I/O dtype is %s/%s, only FP32 bindings are supported",
                           nvmm::dtype_str(in->dtype), nvmm::dtype_str(out->dtype)), (nullptr));
        gst_nvmm_secondary_infer_stop(bt);
        return FALSE;
    }
    /* Classifier input must be 1x3xHxW (NCHW). */
    if (in->dims.nbDims != 4 || in->dims.d[0] != 1 || in->dims.d[1] != 3) {
        GST_ELEMENT_ERROR(self, RESOURCE, SETTINGS,
                          ("input \"%s\" is %s, expected 1x3xHxW (NCHW)",
                           in->name.c_str(), nvmm::dims_str(in->dims).c_str()), (nullptr));
        gst_nvmm_secondary_infer_stop(bt);
        return FALSE;
    }
    self->net_h = (int)in->dims.d[2];
    self->net_w = (int)in->dims.d[3];

    /* Classification head: a flat per-class score vector — [1,C], [1,C,1,1] or
       plain [C]. Anything else (a detection head, multi-batch) is rejected. */
    bool head_ok = false;
    switch (out->dims.nbDims) {
        case 1: head_ok = true; break;
        case 2: head_ok = out->dims.d[0] == 1; break;
        case 4: head_ok = out->dims.d[0] == 1 && out->dims.d[2] == 1 &&
                          out->dims.d[3] == 1; break;
        default: break;
    }
    if (!head_ok || out->volume < 2) {
        GST_ELEMENT_ERROR(self, RESOURCE, SETTINGS,
                          ("output \"%s\" is %s, expected a per-class score vector "
                           "([1,C], [1,C,1,1] or [C])",
                           out->name.c_str(), nvmm::dims_str(out->dims).c_str()), (nullptr));
        gst_nvmm_secondary_infer_stop(bt);
        return FALSE;
    }
    self->num_classes = (int)out->volume;
    if (!self->labels->empty() && (int)self->labels->size() != self->num_classes)
        GST_WARNING_OBJECT(self, "labels-file has %zu labels but the engine has %d "
                           "classes — out-of-range ids fall back to \"class<N>\"",
                           self->labels->size(), self->num_classes);

    cudaError_t e1 = cudaMalloc((void **)&self->d_input, in->bytes);
    cudaError_t e2 = cudaMalloc((void **)&self->d_output, out->bytes);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        GST_ELEMENT_ERROR(self, RESOURCE, NO_SPACE_LEFT, ("cudaMalloc for engine I/O failed"),
                          ("%s / %s", cudaGetErrorString(e1), cudaGetErrorString(e2)));
        gst_nvmm_secondary_infer_stop(bt);
        return FALSE;
    }
    self->host_out->resize((size_t)out->volume);

    if (!self->engine->bind(in->name, self->d_input) ||
        !self->engine->bind(out->name, self->d_output)) {
        GST_ELEMENT_ERROR(self, LIBRARY, INIT, ("failed to bind engine I/O addresses"), (nullptr));
        gst_nvmm_secondary_infer_stop(bt);
        return FALSE;
    }

    if (!self->pre->configure(self->net_w, self->net_h,
                              self->color_order == NVMM_SEC_RGB,
                              (float)self->net_scale, self->stream, err)) {
        GST_ELEMENT_ERROR(self, LIBRARY, INIT, ("ROI preprocess configure failed"),
                          ("%s", err.c_str()));
        gst_nvmm_secondary_infer_stop(bt);
        return FALSE;
    }

    delete self->cache;
    self->cache = new nvmm::SecondaryCache(
        {self->infer_interval, self->max_track_age});
    self->frame_no = 0;

    GST_INFO_OBJECT(self, "ready: net %dx%d, %d classes, interval %u, %zu labels",
                    self->net_w, self->net_h, self->num_classes,
                    self->infer_interval, self->labels->size());
    return TRUE;
}

/* Top-1 over host_out (optionally softmaxed). Returns the class id, stores the
   score. With activation=none the raw value is reported as-is. */
static int
top1(GstNvmmSecondaryInfer *self, float *score)
{
    const std::vector<float> &v = *self->host_out;
    int best = 0;
    for (int i = 1; i < self->num_classes; i++)
        if (v[i] > v[best]) best = i;
    if (self->activation == NVMM_SEC_ACT_SOFTMAX) {
        /* exp-normalize against the max for stability; only the top-1
           probability is needed, so sum once. */
        double sum = 0.0;
        for (int i = 0; i < self->num_classes; i++)
            sum += std::exp((double)v[i] - (double)v[best]);
        *score = (float)(1.0 / sum);
    } else {
        *score = v[best];
    }
    return best;
}

static void
label_of(GstNvmmSecondaryInfer *self, int class_id, char *out, size_t len)
{
    if (class_id >= 0 && class_id < (int)self->labels->size())
        g_strlcpy(out, (*self->labels)[class_id].c_str(), len);
    else
        g_snprintf(out, len, "class%d", class_id);
}

static GstFlowReturn
gst_nvmm_secondary_infer_transform_ip(GstBaseTransform *bt, GstBuffer *buf)
{
    auto *self = GST_NVMM_SECONDARY_INFER(bt);

    const guint64 fno = self->frame_no++;
    self->cache->expire(fno);

    GstNvmmDetMeta *m = gst_buffer_get_nvmm_det_meta(buf);
    if (!m || !m->num_objects)
        return GST_FLOW_OK;

    NvBufSurface *surf = surface_of(buf);
    if (!surf) {
        GST_WARNING_OBJECT(self, "no NvBufSurface in buffer");
        return GST_FLOW_OK;
    }
    const int sw = (int)surf->surfaceList[0].width;
    const int sh = (int)surf->surfaceList[0].height;
    /* det-meta boxes live in infer_width/height space; map to surface px. */
    const float sx = m->infer_width  ? (float)sw / m->infer_width  : 1.f;
    const float sy = m->infer_height ? (float)sh / m->infer_height : 1.f;

    std::vector<NvmmClassEntry> entries(m->num_objects);
    for (auto &e : entries) { e.class_id = -1; e.confidence = 0.f; e.fresh = 0; e.label[0] = '\0'; }

    guint fresh_count = 0;
    for (guint32 i = 0; i < m->num_objects; i++) {
        const NvmmDetObject &o = m->objects[i];
        NvmmClassEntry &e = entries[i];

        const float L = o.left * sx, T = o.top * sy;
        const float Wd = o.width * sx, Hd = o.height * sy;
        if (Wd < (float)self->min_roi || Hd < (float)self->min_roi)
            continue;  /* too small to classify usefully; class_id stays -1 */

        /* Cached result still valid for this track? Untracked objects
           (tracker_id == 0) cannot be cached and re-infer every frame. */
        if (o.tracker_id && !self->cache->due(o.tracker_id, fno)) {
            const nvmm::ClassResult *r = self->cache->lookup(o.tracker_id, fno);
            if (r) {
                e.class_id = r->class_id;
                e.confidence = r->confidence;
                e.fresh = 0;
                g_strlcpy(e.label, r->label, sizeof e.label);
                continue;
            }
        }

        std::string err;
        if (!self->pre->run(surf, L, T, Wd, Hd, self->d_input, err)) {
            GST_WARNING_OBJECT(self, "ROI preprocess failed: %s", err.c_str());
            continue;
        }
        if (!self->engine->infer(self->stream)) {
            GST_WARNING_OBJECT(self, "TensorRT enqueueV3 failed");
            continue;
        }
        const size_t out_bytes = self->host_out->size() * sizeof(float);
        cudaError_t ce = cudaMemcpyAsync(self->host_out->data(), self->d_output,
                                         out_bytes, cudaMemcpyDeviceToHost, self->stream);
        if (ce == cudaSuccess) ce = cudaStreamSynchronize(self->stream);
        if (ce != cudaSuccess) {
            GST_WARNING_OBJECT(self, "CUDA output copy/sync failed: %s",
                               cudaGetErrorString(ce));
            continue;
        }

        float conf = 0.f;
        const int cls = top1(self, &conf);
        if (conf < (float)self->conf_threshold)
            continue;  /* below threshold: no result, no cache — retried next frame */

        e.class_id = cls;
        e.confidence = conf;
        e.fresh = 1;
        label_of(self, cls, e.label, sizeof e.label);
        fresh_count++;

        if (o.tracker_id) {
            nvmm::ClassResult r;
            r.class_id = cls;
            r.confidence = conf;
            memcpy(r.label, e.label, sizeof r.label);
            self->cache->store(o.tracker_id, r, fno);
        }
    }

    gst_buffer_add_nvmm_class_meta(buf, entries.data(), m->num_objects);
    GST_LOG_OBJECT(self, "frame %" G_GUINT64_FORMAT ": %u objects, %u inferred, "
                   "%zu tracks cached", fno, m->num_objects, fresh_count,
                   self->cache->size());
    return GST_FLOW_OK;
}

static gboolean
gst_nvmm_secondary_infer_stop(GstBaseTransform *bt)
{
    auto *self = GST_NVMM_SECONDARY_INFER(bt);
    delete self->engine;
    self->engine = nullptr;
    delete self->pre;                          /* releases its RGBA surface + planes */
    self->pre = new nvmm::RoiPreprocessor();   /* fresh, so a re-start reconfigures */
    delete self->cache;
    self->cache = nullptr;
    if (self->d_input)  { cudaFree(self->d_input);  self->d_input = nullptr; }
    if (self->d_output) { cudaFree(self->d_output); self->d_output = nullptr; }
    if (self->stream) { cudaStreamDestroy(self->stream); self->stream = nullptr; }
    return TRUE;
}

static void
gst_nvmm_secondary_infer_finalize(GObject *o)
{
    auto *self = GST_NVMM_SECONDARY_INFER(o);
    g_free(self->engine_file);
    g_free(self->labels_file);
    delete self->pre;
    delete self->cache;
    delete self->host_out;
    delete self->labels;
    G_OBJECT_CLASS(gst_nvmm_secondary_infer_parent_class)->finalize(o);
}

static void
gst_nvmm_secondary_infer_set_property(GObject *o, guint id, const GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_SECONDARY_INFER(o);
    switch (id) {
        case PROP_ENGINE_FILE:       g_free(self->engine_file);
                                     self->engine_file = g_value_dup_string(v); break;
        case PROP_LABELS_FILE:       g_free(self->labels_file);
                                     self->labels_file = g_value_dup_string(v); break;
        case PROP_INFER_INTERVAL:    self->infer_interval = g_value_get_uint(v); break;
        case PROP_MAX_TRACK_AGE:     self->max_track_age = g_value_get_uint(v); break;
        case PROP_MIN_ROI_SIZE:      self->min_roi = g_value_get_uint(v); break;
        case PROP_NET_SCALE_FACTOR:  self->net_scale = g_value_get_double(v); break;
        case PROP_COLOR_ORDER:       self->color_order = g_value_get_enum(v); break;
        case PROP_OUTPUT_ACTIVATION: self->activation = g_value_get_enum(v); break;
        case PROP_CONF_THRESHOLD:    self->conf_threshold = g_value_get_double(v); break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p); break;
    }
}

static void
gst_nvmm_secondary_infer_get_property(GObject *o, guint id, GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_SECONDARY_INFER(o);
    switch (id) {
        case PROP_ENGINE_FILE:       g_value_set_string(v, self->engine_file); break;
        case PROP_LABELS_FILE:       g_value_set_string(v, self->labels_file); break;
        case PROP_INFER_INTERVAL:    g_value_set_uint(v, self->infer_interval); break;
        case PROP_MAX_TRACK_AGE:     g_value_set_uint(v, self->max_track_age); break;
        case PROP_MIN_ROI_SIZE:      g_value_set_uint(v, self->min_roi); break;
        case PROP_NET_SCALE_FACTOR:  g_value_set_double(v, self->net_scale); break;
        case PROP_COLOR_ORDER:       g_value_set_enum(v, self->color_order); break;
        case PROP_OUTPUT_ACTIVATION: g_value_set_enum(v, self->activation); break;
        case PROP_CONF_THRESHOLD:    g_value_set_double(v, self->conf_threshold); break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p); break;
    }
}

static void
gst_nvmm_secondary_infer_class_init(GstNvmmSecondaryInferClass *klass)
{
    auto *go = G_OBJECT_CLASS(klass);
    auto *el = GST_ELEMENT_CLASS(klass);
    auto *bt = GST_BASE_TRANSFORM_CLASS(klass);

    go->set_property = gst_nvmm_secondary_infer_set_property;
    go->get_property = gst_nvmm_secondary_infer_get_property;
    go->finalize     = gst_nvmm_secondary_infer_finalize;

    auto flags = (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS);
    g_object_class_install_property(go, PROP_ENGINE_FILE,
        g_param_spec_string("engine-file", "Engine file",
            "Path to a serialized TensorRT classifier .engine (build on the "
            "target with trtexec)", nullptr, flags));
    g_object_class_install_property(go, PROP_LABELS_FILE,
        g_param_spec_string("labels-file", "Labels file",
            "Text file with one class label per line (else \"class<N>\")",
            nullptr, flags));
    g_object_class_install_property(go, PROP_INFER_INTERVAL,
        g_param_spec_uint("infer-interval", "Infer interval",
            "Re-classify a tracked object every N frames (1 = every frame); "
            "between runs the cached per-tracker_id result is attached",
            1, 600, 10, flags));
    g_object_class_install_property(go, PROP_MAX_TRACK_AGE,
        g_param_spec_uint("max-track-age", "Max track age",
            "Drop a cached track after this many frames without a detection",
            1, 10000, 60, flags));
    g_object_class_install_property(go, PROP_MIN_ROI_SIZE,
        g_param_spec_uint("min-roi-size", "Min ROI size",
            "Skip detections narrower or shorter than this (surface pixels)",
            2, 8192, 16, flags));
    g_object_class_install_property(go, PROP_NET_SCALE_FACTOR,
        g_param_spec_double("net-scale-factor", "Net scale factor",
            "Pixel scale applied during preprocess (e.g. 1/255)",
            0.0, G_MAXDOUBLE, 1.0 / 255.0, flags));
    g_object_class_install_property(go, PROP_COLOR_ORDER,
        g_param_spec_enum("color-order", "Color order",
            "Channel order the engine expects",
            GST_TYPE_NVMM_SECONDARY_INFER_COLOR_ORDER, NVMM_SEC_RGB, flags));
    g_object_class_install_property(go, PROP_OUTPUT_ACTIVATION,
        g_param_spec_enum("output-activation", "Output activation",
            "Applied to the engine output before taking top-1",
            GST_TYPE_NVMM_SECONDARY_INFER_ACTIVATION, NVMM_SEC_ACT_SOFTMAX, flags));
    g_object_class_install_property(go, PROP_CONF_THRESHOLD,
        g_param_spec_double("conf-threshold", "Confidence threshold",
            "Minimum top-1 score to attach (and cache) a classification",
            0.0, 1.0, 0.0, flags));

    gst_element_class_add_static_pad_template(el, &sink_tmpl);
    gst_element_class_add_static_pad_template(el, &src_tmpl);
    gst_element_class_set_static_metadata(el,
        "NVMM Secondary Inference (TensorRT cascade)", "Filter/Analyzer/Video",
        "DeepStream-free TensorRT cascade: classifies each upstream detection's "
        "ROI on an interval with a per-track cache and attaches the results as "
        "NvmmClassMeta",
        "Pavel Guzenfeld");

    bt->transform_ip = gst_nvmm_secondary_infer_transform_ip;
    bt->start        = gst_nvmm_secondary_infer_start;
    bt->stop         = gst_nvmm_secondary_infer_stop;
    bt->passthrough_on_same_caps = FALSE;

    GST_DEBUG_CATEGORY_INIT(gst_nvmm_secondary_infer_debug, "nvmmsecondaryinfer", 0,
                            "NVMM TensorRT secondary inference");
}

static void
gst_nvmm_secondary_infer_init(GstNvmmSecondaryInfer *self)
{
    self->engine_file    = nullptr;
    self->labels_file    = nullptr;
    self->infer_interval = 10;
    self->max_track_age  = 60;
    self->min_roi        = 16;
    self->net_scale      = 1.0 / 255.0;
    self->color_order    = NVMM_SEC_RGB;
    self->activation     = NVMM_SEC_ACT_SOFTMAX;
    self->conf_threshold = 0.0;
    self->engine         = nullptr;
    self->pre            = new nvmm::RoiPreprocessor();
    self->cache          = nullptr;
    self->stream         = nullptr;
    self->d_input = self->d_output = nullptr;
    self->host_out = new std::vector<float>();
    self->labels   = new std::vector<std::string>();
    self->net_w = self->net_h = 0;
    self->num_classes = 0;
    self->frame_no = 0;
    /* In-place: same caps in/out, the frame's pixels are never copied —
       transform_ip gets the writable buffer and only attaches class meta. */
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
}

/* ---- plugin ---- */

static gboolean
plugin_init(GstPlugin *plugin)
{
    return gst_element_register(plugin, "nvmmsecondaryinfer", GST_RANK_NONE,
                                GST_TYPE_NVMM_SECONDARY_INFER);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    nvmmsecondaryinfer, "DeepStream-free TensorRT secondary (cascade) inference on NVMM frames",
    plugin_init, PACKAGE_VERSION, "LGPL", "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp")
