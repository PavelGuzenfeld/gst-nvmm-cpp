/// GstNvmmSamurai — see gstnvmmsamurai.h. Phase B2: element shell — loads the
/// SamuraiTracker (5 engines), seeds from the first confident YOLO detection,
/// runs the (B2-stub) tracker per frame, and attaches GstNvmmTrackMeta. The real
/// per-frame dataflow lands in SamuraiTracker (Phase B3).

#include "config.h"  // PACKAGE_VERSION

#include "gstnvmmsamurai.h"

#include <gst/video/video.h>

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

#include <cstdio>
#include <string>

#include "samurai_tracker.hpp"
#include "nvmm_det_meta.h"
#include "nvmm_track_meta.h"
#include "gstnvmmallocator.h"    // gst_is_nvmm_memory / gst_nvmm_memory_get_surface

#include <nvbufsurface.h>

GST_DEBUG_CATEGORY(gst_nvmm_samurai_debug);
#define GST_CAT_DEFAULT gst_nvmm_samurai_debug

struct _GstNvmmSamurai {
    GstBaseTransform parent;

    /* properties */
    gchar  *engine_dir;
    gchar  *consts_file;
    gint    crop_size;
    gint    max_kf;
    gdouble kf_score_weight;
    gint    stable_frames_threshold;
    gdouble iou_threshold;
    gdouble kf_min_area;        /* min KF box area (px^2) to accept a KF update */
    gint    target_class;
    gdouble seed_conf;          /* min YOLO conf to autonomously seed */
    gboolean seed_prefer_center; /* seed the most-central det, not the most confident */
    guint    seed_delay;        /* don't auto-seed before this frame (camera settle) */
    gchar  *seed_roi;           /* "x,y,w,h" frame coords: force initial seed here,
                                   bypassing YOLO (for targets the detector misses) */
    gboolean gmc;               /* camera-motion compensation (handheld clips) */

    /* runtime */
    nvmm::SamuraiTracker *tracker;
    guint64 frame_no;
    guint   kf_count;           /* consecutive KF-only frames since last full inference */

    /* re-seed authority: nvmmfusekf (downstream) sends an upstream "nvmm-reseed"
       event with a box; we force a (re)seed on the next frame. */
    nvmm::TrackBox reseed_box;
    gboolean       reseed_pending;
    gboolean       roi_armed;    /* seed-roi pending; applied at frame >= seed_delay */
};

enum {
    PROP_0, PROP_ENGINE_DIR, PROP_CONSTS_FILE, PROP_CROP_SIZE, PROP_MAX_KF,
    PROP_KF_SCORE_WEIGHT, PROP_STABLE_FRAMES_THRESHOLD, PROP_IOU_THRESHOLD,
    PROP_KF_MIN_AREA, PROP_TARGET_CLASS, PROP_SEED_CONF, PROP_SEED_PREFER_CENTER,
    PROP_SEED_ROI, PROP_SEED_DELAY, PROP_GMC,
};

#define NVMM_NV12_CAPS \
    "video/x-raw(memory:NVMM), format=(string)NV12, " \
    "width=(int)[1,8192], height=(int)[1,8192], framerate=(fraction)[0/1,240/1]"

static GstStaticPadTemplate sink_tmpl = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS(NVMM_NV12_CAPS));
static GstStaticPadTemplate src_tmpl = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS(NVMM_NV12_CAPS));

G_DEFINE_TYPE(GstNvmmSamurai, gst_nvmm_samurai, GST_TYPE_BASE_TRANSFORM)

/* NVMM memory -> NvBufSurface (no copy); fall back to mapping for plain mem. */
static NvBufSurface *surface_of(GstBuffer *buf)
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

static gboolean gst_nvmm_samurai_start(GstBaseTransform *bt)
{
    auto *self = GST_NVMM_SAMURAI(bt);
    if (!self->engine_dir || !*self->engine_dir) {
        GST_ELEMENT_ERROR(self, RESOURCE, NOT_FOUND,
                          ("engine-dir property is required"), (nullptr));
        return FALSE;
    }
    nvmm::SamuraiConfig cfg;
    cfg.engine_dir = self->engine_dir;
    cfg.consts_file = self->consts_file ? self->consts_file : "";
    cfg.crop_size = self->crop_size;
    cfg.max_kf = self->max_kf;
    cfg.kf_score_weight = (float)self->kf_score_weight;
    cfg.stable_frames_threshold = self->stable_frames_threshold;
    cfg.iou_threshold = (float)self->iou_threshold;
    cfg.kf_min_area = (float)self->kf_min_area;
    cfg.target_class = self->target_class;
    cfg.gmc = self->gmc;

    self->tracker = new nvmm::SamuraiTracker();
    std::string err;
    if (!self->tracker->init(cfg, err)) {
        GST_ELEMENT_ERROR(self, LIBRARY, INIT, ("SAMURAI init failed: %s", err.c_str()), (nullptr));
        delete self->tracker; self->tracker = nullptr;
        return FALSE;
    }
    self->frame_no = 0;
    self->kf_count = 0;
    self->reseed_pending = FALSE;
    self->roi_armed = FALSE;
    /* seed-roi: force the initial seed at a fixed box (bypass YOLO auto-seed). */
    if (self->seed_roi && *self->seed_roi) {
        float x = 0, y = 0, w = 0, h = 0;
        if (sscanf(self->seed_roi, "%f,%f,%f,%f", &x, &y, &w, &h) == 4 && w > 0 && h > 0) {
            self->reseed_box = {x, y, w, h, 0.f, true};
            self->roi_armed = TRUE;   /* applied at frame >= seed-delay (settle) */
            GST_INFO_OBJECT(self, "seed-roi armed at (%.0f,%.0f %.0fx%.0f), delay=%u",
                            x, y, w, h, self->seed_delay);
        } else {
            GST_WARNING_OBJECT(self, "bad seed-roi '%s' (want x,y,w,h)", self->seed_roi);
        }
    }
    return TRUE;
}

static gboolean gst_nvmm_samurai_stop(GstBaseTransform *bt)
{
    auto *self = GST_NVMM_SAMURAI(bt);
    delete self->tracker; self->tracker = nullptr;
    return TRUE;
}

/* Pick a target-class detection to seed from, mapped from infer space to surface
   pixels. Among dets with conf >= seed-conf: by default the highest confidence;
   with seed-prefer-center=true, the one closest to the frame center (the target
   of interest is what the camera is pointed at — avoids locking onto a higher-
   confidence background false positive). Returns false if none qualifies. */
static gboolean best_seed_box(GstNvmmSamurai *self, GstBuffer *buf, NvBufSurface *surf,
                              nvmm::TrackBox *out)
{
    GstNvmmDetMeta *det = gst_buffer_get_nvmm_det_meta(buf);
    if (!det || det->num_objects == 0)
        return FALSE;
    const float sx = det->infer_width  ? (float)surf->surfaceList[0].width  / det->infer_width  : 1.f;
    const float sy = det->infer_height ? (float)surf->surfaceList[0].height / det->infer_height : 1.f;
    const float ccx = det->infer_width * 0.5f, ccy = det->infer_height * 0.5f;
    const NvmmDetObject *best = nullptr;
    float best_d = 0.f;
    for (guint32 i = 0; i < det->num_objects; i++) {
        const NvmmDetObject &o = det->objects[i];
        if (o.class_id != self->target_class || o.confidence < self->seed_conf)
            continue;
        if (self->seed_prefer_center) {
            const float dx = o.left + o.width * 0.5f - ccx, dy = o.top + o.height * 0.5f - ccy;
            const float d = dx * dx + dy * dy;
            if (!best || d < best_d) { best = &o; best_d = d; }
        } else if (!best || o.confidence > best->confidence) {
            best = &o;
        }
    }
    if (!best)
        return FALSE;
    out->left = best->left * sx; out->top = best->top * sy;
    out->width = best->width * sx; out->height = best->height * sy;
    out->score = best->confidence; out->valid = true;
    return TRUE;
}

/* Upstream re-seed authority: nvmmfusekf sends a CUSTOM_UPSTREAM "nvmm-reseed"
   event carrying a box (frame/surface coords) when the track is lost. We stash it
   and force a (re)seed on the next frame (we have the surface there, not here). */
static gboolean gst_nvmm_samurai_src_event(GstBaseTransform *bt, GstEvent *ev)
{
    auto *self = GST_NVMM_SAMURAI(bt);
    if (GST_EVENT_TYPE(ev) == GST_EVENT_CUSTOM_UPSTREAM) {
        const GstStructure *s = gst_event_get_structure(ev);
        if (s && gst_structure_has_name(s, "nvmm-reseed")) {
            gdouble x = 0, y = 0, w = 0, h = 0;
            gst_structure_get_double(s, "x", &x); gst_structure_get_double(s, "y", &y);
            gst_structure_get_double(s, "w", &w); gst_structure_get_double(s, "h", &h);
            self->reseed_box = {(float)x, (float)y, (float)w, (float)h, 0.f, true};
            self->reseed_pending = TRUE;
            GST_DEBUG_OBJECT(self, "reseed requested at (%.0f,%.0f %.0fx%.0f)", x, y, w, h);
            gst_event_unref(ev);
            return TRUE;  // consume
        }
    }
    return GST_BASE_TRANSFORM_CLASS(gst_nvmm_samurai_parent_class)->src_event(bt, ev);
}

static GstFlowReturn gst_nvmm_samurai_transform_ip(GstBaseTransform *bt, GstBuffer *buf)
{
    auto *self = GST_NVMM_SAMURAI(bt);
    const guint64 fno = self->frame_no++;
    NvBufSurface *surf = surface_of(buf);
    if (!surf) {
        GST_WARNING_OBJECT(self, "no NvBufSurface in buffer");
        return GST_FLOW_OK;
    }

    std::string err;
    nvmm::TrackResult res;
    /* Forced re-seed from the downstream fusekf authority takes precedence. */
    if (self->reseed_pending) {
        self->reseed_pending = FALSE;
        self->kf_count = 0;
        if (!self->tracker->seed(surf, self->reseed_box, err))
            GST_WARNING_OBJECT(self, "reseed failed: %s", err.c_str());
    }
    /* seed-roi: force the initial seed at the configured box once the image has
       settled (frame >= seed-delay), bypassing YOLO auto-seed. */
    if (self->roi_armed && !self->tracker->seeded() && fno >= self->seed_delay) {
        self->roi_armed = FALSE;
        if (!self->tracker->seed(surf, self->reseed_box, err))
            GST_WARNING_OBJECT(self, "seed-roi failed: %s", err.c_str());
    }
    if (!self->tracker->seeded() && fno >= self->seed_delay) {
        nvmm::TrackBox seed;
        if (best_seed_box(self, buf, surf, &seed)) {
            if (!self->tracker->seed(surf, seed, err))
                GST_WARNING_OBJECT(self, "seed failed: %s", err.c_str());
        }
    }
    if (self->tracker->seeded()) {
        /* max-kf fast frames: run full SAM inference, then up to max-kf KF-only
           frames (kf.predict only, no engines) — the box is extrapolated between
           full inferences. max-kf=0 => every frame full. */
        gboolean kf_only = FALSE;
        if ((gint)self->kf_count < self->max_kf) {
            kf_only = TRUE; self->kf_count++;
        } else {
            self->kf_count = 0;
        }
        if (!self->tracker->track(surf, kf_only, res, err)) {
            GST_WARNING_OBJECT(self, "track failed: %s", err.c_str());
            return GST_FLOW_OK;
        }
    }

    /* Attach the track meta (valid=FALSE until seeded). */
    GstNvmmTrackMeta *tm = gst_buffer_add_nvmm_track_meta(buf);
    if (tm) {
        tm->frame_number = fno;
        tm->frame_width = surf->surfaceList[0].width;
        tm->frame_height = surf->surfaceList[0].height;
        tm->valid = res.box.valid;
        tm->target_id = res.target_id;
        tm->left = res.box.left; tm->top = res.box.top;
        tm->width = res.box.width; tm->height = res.box.height;
        tm->object_score = res.box.score;
        tm->kf_left = res.kf_box.left; tm->kf_top = res.kf_box.top;
        tm->kf_width = res.kf_box.width; tm->kf_height = res.kf_box.height;
        tm->kf_score = res.kf_box.score;
        tm->is_kf_only = res.is_kf_only;
        tm->stable_frames = res.stable_frames;
    }
    return GST_FLOW_OK;
}

static void gst_nvmm_samurai_set_property(GObject *o, guint id, const GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_SAMURAI(o);
    switch (id) {
    case PROP_ENGINE_DIR:  g_free(self->engine_dir); self->engine_dir = g_value_dup_string(v); break;
    case PROP_CONSTS_FILE: g_free(self->consts_file); self->consts_file = g_value_dup_string(v); break;
    case PROP_CROP_SIZE:   self->crop_size = g_value_get_int(v); break;
    case PROP_MAX_KF:      self->max_kf = g_value_get_int(v); break;
    case PROP_KF_SCORE_WEIGHT: self->kf_score_weight = g_value_get_double(v); break;
    case PROP_STABLE_FRAMES_THRESHOLD: self->stable_frames_threshold = g_value_get_int(v); break;
    case PROP_IOU_THRESHOLD: self->iou_threshold = g_value_get_double(v); break;
    case PROP_KF_MIN_AREA: self->kf_min_area = g_value_get_double(v); break;
    case PROP_TARGET_CLASS: self->target_class = g_value_get_int(v); break;
    case PROP_SEED_CONF:   self->seed_conf = g_value_get_double(v); break;
    case PROP_SEED_PREFER_CENTER: self->seed_prefer_center = g_value_get_boolean(v); break;
    case PROP_SEED_ROI: g_free(self->seed_roi); self->seed_roi = g_value_dup_string(v); break;
    case PROP_SEED_DELAY: self->seed_delay = g_value_get_uint(v); break;
    case PROP_GMC: self->gmc = g_value_get_boolean(v); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p); break;
    }
}

static void gst_nvmm_samurai_get_property(GObject *o, guint id, GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_SAMURAI(o);
    switch (id) {
    case PROP_ENGINE_DIR:  g_value_set_string(v, self->engine_dir); break;
    case PROP_CONSTS_FILE: g_value_set_string(v, self->consts_file); break;
    case PROP_CROP_SIZE:   g_value_set_int(v, self->crop_size); break;
    case PROP_MAX_KF:      g_value_set_int(v, self->max_kf); break;
    case PROP_KF_SCORE_WEIGHT: g_value_set_double(v, self->kf_score_weight); break;
    case PROP_STABLE_FRAMES_THRESHOLD: g_value_set_int(v, self->stable_frames_threshold); break;
    case PROP_IOU_THRESHOLD: g_value_set_double(v, self->iou_threshold); break;
    case PROP_KF_MIN_AREA: g_value_set_double(v, self->kf_min_area); break;
    case PROP_TARGET_CLASS: g_value_set_int(v, self->target_class); break;
    case PROP_SEED_CONF:   g_value_set_double(v, self->seed_conf); break;
    case PROP_SEED_PREFER_CENTER: g_value_set_boolean(v, self->seed_prefer_center); break;
    case PROP_SEED_ROI: g_value_set_string(v, self->seed_roi); break;
    case PROP_SEED_DELAY: g_value_set_uint(v, self->seed_delay); break;
    case PROP_GMC: g_value_set_boolean(v, self->gmc); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p); break;
    }
}

static void gst_nvmm_samurai_finalize(GObject *o)
{
    auto *self = GST_NVMM_SAMURAI(o);
    g_free(self->engine_dir); g_free(self->consts_file); g_free(self->seed_roi);
    G_OBJECT_CLASS(gst_nvmm_samurai_parent_class)->finalize(o);
}

static void gst_nvmm_samurai_class_init(GstNvmmSamuraiClass *klass)
{
    auto *go = G_OBJECT_CLASS(klass);
    auto *el = GST_ELEMENT_CLASS(klass);
    auto *bt = GST_BASE_TRANSFORM_CLASS(klass);

    go->set_property = gst_nvmm_samurai_set_property;
    go->get_property = gst_nvmm_samurai_get_property;
    go->finalize = gst_nvmm_samurai_finalize;

    GParamFlags f = (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS);
    g_object_class_install_property(go, PROP_ENGINE_DIR,
        g_param_spec_string("engine-dir", "Engine dir",
            "Directory with the 5 SAMURAI .engine files", nullptr, f));
    g_object_class_install_property(go, PROP_CONSTS_FILE,
        g_param_spec_string("consts-file", "Constants file",
            "samurai_consts file (learned out-of-engine constants)", nullptr, f));
    g_object_class_install_property(go, PROP_CROP_SIZE,
        g_param_spec_int("crop-size", "Crop size", "Encoder input size (square)",
            64, 2048, 512, f));
    g_object_class_install_property(go, PROP_MAX_KF,
        g_param_spec_int("max-kf", "Max KF-only frames",
            "Max consecutive Kalman-only frames between full inferences", 0, 30, 2, f));
    g_object_class_install_property(go, PROP_KF_SCORE_WEIGHT,
        g_param_spec_double("kf-score-weight", "KF score weight",
            "SAMURAI weighted = w*kf_iou + (1-w)*iou", 0.0, 1.0, 0.25, f));
    g_object_class_install_property(go, PROP_STABLE_FRAMES_THRESHOLD,
        g_param_spec_int("stable-frames-threshold", "Stable frames threshold",
            "Frames of stable IoU before the 'stable' regime", 0, 1000, 10, f));
    g_object_class_install_property(go, PROP_IOU_THRESHOLD,
        g_param_spec_double("iou-threshold", "IoU threshold",
            "Min selected-candidate IoU to accept a Kalman update (stability gate)",
            0.0, 1.0, 0.5, f));
    g_object_class_install_property(go, PROP_KF_MIN_AREA,
        g_param_spec_double("kf-min-area", "KF min area (px^2)",
            "Min Kalman box area (pixels^2) to accept a Kalman update",
            0.0, 1e8, 25.0, f));
    g_object_class_install_property(go, PROP_TARGET_CLASS,
        g_param_spec_int("target-class", "Target class",
            "YOLO class id to seed/track", 0, 1000, 0, f));
    g_object_class_install_property(go, PROP_SEED_CONF,
        g_param_spec_double("seed-conf", "Seed confidence",
            "Min YOLO confidence to autonomously seed the tracker", 0.0, 1.0, 0.25, f));
    g_object_class_install_property(go, PROP_SEED_PREFER_CENTER,
        g_param_spec_boolean("seed-prefer-center", "Seed prefer center",
            "Seed the detection closest to the frame center (vs the most confident)",
            FALSE, f));
    g_object_class_install_property(go, PROP_SEED_ROI,
        g_param_spec_string("seed-roi", "Seed ROI",
            "Force the initial seed at \"x,y,w,h\" (frame pixels), bypassing YOLO "
            "auto-seed — for a target the detector misses", nullptr, f));
    g_object_class_install_property(go, PROP_SEED_DELAY,
        g_param_spec_uint("seed-delay", "Seed delay (frames)",
            "Don't auto-seed before this frame (skip unstable lead-in until the "
            "camera settles)", 0, 100000, 0, f));
    g_object_class_install_property(go, PROP_GMC,
        g_param_spec_boolean("gmc", "Camera-motion compensation",
            "Estimate per-frame camera translation and shift the KF/crop to cancel "
            "it (for handheld / moving-camera clips)", FALSE, f));

    gst_element_class_add_static_pad_template(el, &sink_tmpl);
    gst_element_class_add_static_pad_template(el, &src_tmpl);
    gst_element_class_set_static_metadata(el,
        "NVMM SAMURAI tracker", "Filter/Effect/Video",
        "SAM2.1 SAMURAI visual tracker (TensorRT) producing GstNvmmTrackMeta",
        "gst-nvmm-cpp");

    bt->transform_ip = gst_nvmm_samurai_transform_ip;
    bt->src_event = gst_nvmm_samurai_src_event;
    bt->start = gst_nvmm_samurai_start;
    bt->stop = gst_nvmm_samurai_stop;
}

static void gst_nvmm_samurai_init(GstNvmmSamurai *self)
{
    self->crop_size = 512;
    self->max_kf = 2;
    self->kf_score_weight = 0.25;
    self->stable_frames_threshold = 10;
    self->iou_threshold = 0.5;
    self->kf_min_area = 25.0;
    self->target_class = 0;
    self->seed_conf = 0.25;
    self->seed_prefer_center = FALSE;
    self->seed_roi = nullptr;
    self->seed_delay = 0;
    self->gmc = FALSE;
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), FALSE);
}

static gboolean plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_nvmm_samurai_debug, "nvmmsamurai", 0, "NVMM SAMURAI tracker");
    return gst_element_register(plugin, "nvmmsamurai", GST_RANK_NONE, GST_TYPE_NVMM_SAMURAI);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, nvmmsamurai,
                  "SAM2.1 SAMURAI visual tracker (NVMM/TensorRT)",
                  plugin_init, PACKAGE_VERSION, "LGPL", "gst-nvmm-cpp", "https://github.com/PavelGuzenfeld/gst-nvmm-cpp")
