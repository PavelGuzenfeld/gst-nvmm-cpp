/// GstNvmmFuseKf — see gstnvmmfusekf.h. Master constant-velocity KF (frame
/// coords) fusing SAMURAI (GstNvmmTrackMeta) + best YOLO det (GstNvmmDetMeta).

#include "gstnvmmfusekf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <gst/gst.h>

#include "kalman_box.hpp"     // gst/common
#include "nvmm_track_meta.h"  // gst/common
#include "nvmm_det_meta.h"    // gst/common

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_fusekf_debug);
#define GST_CAT_DEFAULT gst_nvmm_fusekf_debug

struct _GstNvmmFuseKf {
    GstBaseTransform parent;

    /* properties */
    gint    target_class;
    gdouble det_conf;        /* min YOLO confidence to fuse */
    gdouble gate_thresh;     /* Mahalanobis gating distance ceiling */
    gint    max_lost;        /* consecutive no-measurement frames before "lost" */
    gint    reseed_cooldown_frames; /* frames between upstream reseed emissions */

    /* runtime (master KF lives in C++; held via a raw pointer) */
    nvmm::KalmanBox *kf;
    gboolean have_pts;
    guint64  prev_pts;
    gint     lost;
    guint64  target_id;
    gint     reseed_cooldown;  /* frames to wait before re-emitting a reseed event */
    gboolean ever_valid;       /* had a track at least once (gates reseed-on-loss) */
    FILE    *csv;              /* $NVMMFUSEKF_CSV per-frame box dump (eval/overlay) */
    guint64  frame_n;
};

enum { PROP_0, PROP_TARGET_CLASS, PROP_DET_CONF, PROP_GATE_THRESH, PROP_MAX_LOST,
       PROP_RESEED_COOLDOWN };

#define NVMM_NV12_CAPS \
    "video/x-raw(memory:NVMM), format=(string)NV12, " \
    "width=(int)[1,8192], height=(int)[1,8192], framerate=(fraction)[0/1,240/1]"

static GstStaticPadTemplate sink_tmpl = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS(NVMM_NV12_CAPS));
static GstStaticPadTemplate src_tmpl = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS(NVMM_NV12_CAPS));

G_DEFINE_TYPE(GstNvmmFuseKf, gst_nvmm_fusekf, GST_TYPE_BASE_TRANSFORM)

/* Best target-class detection (frame coords) from the YOLO det meta. */
static gboolean best_det(GstNvmmFuseKf *self, GstNvmmDetMeta *det,
                         guint fw, guint fh, double *cx, double *cy, double *w, double *h)
{
    if (!det || det->num_objects == 0)
        return FALSE;
    const double sx = det->infer_width  ? (double)fw / det->infer_width  : 1.0;
    const double sy = det->infer_height ? (double)fh / det->infer_height : 1.0;
    const NvmmDetObject *best = nullptr;
    for (guint32 i = 0; i < det->num_objects; i++) {
        const NvmmDetObject &o = det->objects[i];
        if ((gint)o.class_id != self->target_class || o.confidence < self->det_conf)
            continue;
        if (!best || o.confidence > best->confidence)
            best = &o;
    }
    if (!best)
        return FALSE;
    *w = best->width * sx; *h = best->height * sy;
    *cx = best->left * sx + *w / 2.0;
    *cy = best->top * sy + *h / 2.0;
    return TRUE;
}

static GstFlowReturn gst_nvmm_fusekf_transform_ip(GstBaseTransform *bt, GstBuffer *buf)
{
    auto *self = GST_NVMM_FUSEKF(bt);
    GstNvmmTrackMeta *tm = gst_buffer_get_nvmm_track_meta(buf);
    if (!tm)
        return GST_FLOW_OK;  // nothing to fuse into

    /* dt in frames from PTS / buffer duration (fallback 1.0). */
    double dt = 1.0;
    const guint64 pts = GST_BUFFER_PTS(buf);
    const guint64 dur = GST_BUFFER_DURATION(buf);
    if (self->have_pts && GST_CLOCK_TIME_IS_VALID(pts) && GST_CLOCK_TIME_IS_VALID(dur) && dur > 0)
        dt = (double)(pts - self->prev_pts) / (double)dur;
    if (dt < 0.1) dt = 0.1;
    if (dt > 10.0) dt = 10.0;
    if (GST_CLOCK_TIME_IS_VALID(pts)) { self->prev_pts = pts; self->have_pts = TRUE; }

    const guint fw = tm->frame_width, fh = tm->frame_height;

    /* Measurement 1: SAMURAI track (already frame coords). */
    gboolean has_sam = tm->valid && tm->width > 0 && tm->height > 0;
    double sam_cx = tm->left + tm->width / 2.0, sam_cy = tm->top + tm->height / 2.0;
    double sam_w = tm->width, sam_h = tm->height;

    /* Measurement 2: best YOLO det (scaled to frame coords). */
    double yc = 0, yy = 0, yw = 0, yh = 0;
    gboolean has_yolo = best_det(self, gst_buffer_get_nvmm_det_meta(buf), fw, fh,
                                 &yc, &yy, &yw, &yh);

    /* SAMURAI is the trusted primary tracker (it already gates internally), so it
       updates ungated; YOLO is the gated secondary measurement (reject false
       positives / off-target dets). KalmanBox measurement noise scales with box
       size, so for tiny targets gating only the secondary keeps the master KF
       from starving. */
    int updated = 0, fused_yolo = 0;
    if (!self->kf->initiated()) {
        /* Init ONLY from the SAMURAI primary (it owns seeding incl. seed-delay /
           seed-roi). Initing from a lone YOLO det would jump the gun and lock onto
           a distractor before SAMURAI has settled on the target. */
        if (has_sam) { self->kf->initiate(sam_cx, sam_cy, sam_w, sam_h); updated = 1; }
    } else {
        self->kf->predict(dt);
        /* Gate YOLO by Euclidean center distance to the prediction (pixels), not
           Mahalanobis: KalmanBox measurement noise scales with box size, so for a
           ~5px drone the Mahalanobis gate is useless (reject-all / accept-all). A
           pixel radius cleanly fuses on-target YOLO dets and rejects far ones. */
        double pcx, pcy, pw, ph; self->kf->box(pcx, pcy, pw, ph);
        const double yd = std::hypot(yc - pcx, yy - pcy);
        const gboolean yolo_ok = has_yolo && yd < self->gate_thresh;
        if (has_sam) { self->kf->update(sam_cx, sam_cy, sam_w, sam_h); updated++; }
        if (yolo_ok) { self->kf->update(yc, yy, yw, yh); updated++; fused_yolo = 1; }
    }
    /* "lost" tracks the primary (SAMURAI); a lone gated YOLO doesn't keep it alive. */
    self->lost = has_sam ? 0 : self->lost + 1;

    /* Write the fused box back into the track meta. */
    if (self->kf->initiated() && self->lost <= self->max_lost) {
        double cx, cy, w, h; self->kf->box(cx, cy, w, h);
        tm->left = (float)(cx - w / 2.0); tm->top = (float)(cy - h / 2.0);
        tm->width = (float)w; tm->height = (float)h;
        tm->valid = TRUE;
        self->ever_valid = TRUE;
        if (!tm->target_id) tm->target_id = ++self->target_id;
    } else {
        tm->valid = FALSE;  /* lost */
    }

    /* Re-seed authority: when the track is lost and a fresh YOLO det is available,
       send an upstream "nvmm-reseed" event (box in frame coords) to nvmmsamurai.
       Cooldown prevents spamming while it recovers. */
    if (self->reseed_cooldown > 0) self->reseed_cooldown--;
    /* Only re-seed once a track has existed and been LOST — never during initial
       acquisition (SAMURAI owns that, incl. seed-delay/seed-roi). */
    if (self->ever_valid && self->lost > self->max_lost && has_yolo && self->reseed_cooldown == 0) {
        GstStructure *s = gst_structure_new("nvmm-reseed",
            "x", G_TYPE_DOUBLE, yc - yw / 2.0, "y", G_TYPE_DOUBLE, yy - yh / 2.0,
            "w", G_TYPE_DOUBLE, yw, "h", G_TYPE_DOUBLE, yh, nullptr);
        gst_pad_push_event(GST_BASE_TRANSFORM_SINK_PAD(bt),
                           gst_event_new_custom(GST_EVENT_CUSTOM_UPSTREAM, s));
        self->reseed_cooldown = self->reseed_cooldown_frames;
        GST_INFO_OBJECT(self, "lost (%d) — sent reseed at (%.0f,%.0f %.0fx%.0f)",
                        self->lost, yc - yw / 2.0, yy - yh / 2.0, yw, yh);
    }
    GST_LOG_OBJECT(self, "fuse dt=%.2f sam=%d yolo=%d(fused=%d) upd=%d lost=%d box(%.0f,%.0f %.0fx%.0f)",
                   dt, has_sam, has_yolo, fused_yolo, updated, self->lost,
                   tm->left, tm->top, tm->width, tm->height);
    if (self->csv)
        std::fprintf(self->csv, "%llu,%d,%.1f,%.1f,%.1f,%.1f,%.3f,%d\n",
                     (unsigned long long)self->frame_n, tm->valid ? 1 : 0,
                     tm->left, tm->top, tm->width, tm->height, tm->object_score, fused_yolo);
    self->frame_n++;
    return GST_FLOW_OK;
}

static gboolean gst_nvmm_fusekf_stop(GstBaseTransform *bt)
{
    auto *self = GST_NVMM_FUSEKF(bt);
    delete self->kf; self->kf = nullptr;
    if (self->csv) { std::fclose(self->csv); self->csv = nullptr; }
    return TRUE;
}

static gboolean gst_nvmm_fusekf_start(GstBaseTransform *bt)
{
    auto *self = GST_NVMM_FUSEKF(bt);
    delete self->kf;
    self->kf = new nvmm::KalmanBox();
    self->have_pts = FALSE;
    self->prev_pts = 0;
    self->lost = 0;
    self->target_id = 0;
    self->reseed_cooldown = 0;
    self->ever_valid = FALSE;
    self->frame_n = 0;
    self->csv = nullptr;
    if (const char *p = std::getenv("NVMMFUSEKF_CSV")) {
        self->csv = std::fopen(p, "w");
        if (self->csv) std::fprintf(self->csv, "frame,valid,left,top,width,height,score,yolo_fused\n");
    }
    return TRUE;
}

static void gst_nvmm_fusekf_set_property(GObject *o, guint id, const GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_FUSEKF(o);
    switch (id) {
    case PROP_TARGET_CLASS: self->target_class = g_value_get_int(v); break;
    case PROP_DET_CONF:     self->det_conf = g_value_get_double(v); break;
    case PROP_GATE_THRESH:  self->gate_thresh = g_value_get_double(v); break;
    case PROP_MAX_LOST:     self->max_lost = g_value_get_int(v); break;
    case PROP_RESEED_COOLDOWN: self->reseed_cooldown_frames = g_value_get_int(v); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p); break;
    }
}

static void gst_nvmm_fusekf_get_property(GObject *o, guint id, GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_FUSEKF(o);
    switch (id) {
    case PROP_TARGET_CLASS: g_value_set_int(v, self->target_class); break;
    case PROP_DET_CONF:     g_value_set_double(v, self->det_conf); break;
    case PROP_GATE_THRESH:  g_value_set_double(v, self->gate_thresh); break;
    case PROP_MAX_LOST:     g_value_set_int(v, self->max_lost); break;
    case PROP_RESEED_COOLDOWN: g_value_set_int(v, self->reseed_cooldown_frames); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p); break;
    }
}

static void gst_nvmm_fusekf_class_init(GstNvmmFuseKfClass *klass)
{
    auto *go = G_OBJECT_CLASS(klass);
    auto *el = GST_ELEMENT_CLASS(klass);
    auto *bt = GST_BASE_TRANSFORM_CLASS(klass);

    go->set_property = gst_nvmm_fusekf_set_property;
    go->get_property = gst_nvmm_fusekf_get_property;

    GParamFlags f = (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS);
    g_object_class_install_property(go, PROP_TARGET_CLASS,
        g_param_spec_int("target-class", "Target class", "YOLO class id to fuse", 0, 1000, 0, f));
    g_object_class_install_property(go, PROP_DET_CONF,
        g_param_spec_double("det-conf", "Det confidence", "Min YOLO confidence to fuse", 0.0, 1.0, 0.25, f));
    g_object_class_install_property(go, PROP_GATE_THRESH,
        g_param_spec_double("gate-threshold", "Gate threshold (px)",
            "Max YOLO-to-prediction center distance (pixels) to fuse the detection",
            0.0, 1e6, 100.0, f));
    g_object_class_install_property(go, PROP_MAX_LOST,
        g_param_spec_int("max-lost", "Max lost frames",
            "Consecutive no-measurement frames before the track is marked lost", 0, 10000, 30, f));
    g_object_class_install_property(go, PROP_RESEED_COOLDOWN,
        g_param_spec_int("reseed-cooldown", "Reseed cooldown (frames)",
            "Frames to wait after emitting an upstream reseed before emitting another",
            0, 10000, 15, f));

    gst_element_class_add_static_pad_template(el, &sink_tmpl);
    gst_element_class_add_static_pad_template(el, &src_tmpl);
    gst_element_class_set_static_metadata(el,
        "NVMM master-KF fusion", "Filter/Effect/Video",
        "Fuses SAMURAI track + YOLO detection with a master constant-velocity Kalman filter",
        "gst-nvmm-cpp");

    bt->transform_ip = gst_nvmm_fusekf_transform_ip;
    bt->start = gst_nvmm_fusekf_start;
    bt->stop = gst_nvmm_fusekf_stop;
}

static void gst_nvmm_fusekf_init(GstNvmmFuseKf *self)
{
    self->target_class = 0;
    self->det_conf = 0.25;
    self->gate_thresh = 100.0;   /* pixels */
    self->max_lost = 30;
    self->reseed_cooldown_frames = 15;
    self->kf = nullptr;
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), FALSE);
}

static gboolean plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_nvmm_fusekf_debug, "nvmmfusekf", 0, "NVMM master-KF fusion");
    return gst_element_register(plugin, "nvmmfusekf", GST_RANK_NONE, GST_TYPE_NVMM_FUSEKF);
}

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, nvmmfusekf,
                  "Master-KF YOLO+SAMURAI fusion",
                  plugin_init, "1.0", "LGPL", "gst-nvmm-cpp",
                  "https://github.com/PavelGuzenfeld/gst-nvmm-cpp")
