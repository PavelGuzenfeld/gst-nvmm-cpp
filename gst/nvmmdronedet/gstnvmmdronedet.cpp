/// nvmmdronedet — autonomous drone-seed gate for the YOLO→SAMURAI pipeline.
///
/// Sits between nvmminfer and nvmmsamurai. The drone-trained YOLO fires on the drone
/// AND on static terrain / sky haze, so SAMURAI's auto-seed cannot tell them apart on
/// hard clips. This element FILTERS the GstNvmmDetMeta to the single detection that is
/// independently MOVING — confirmed by dual-homography residual (terrain backgrounds)
/// or high-confidence sky-diff motion (sky backgrounds), held for KSUP frames (see
/// DroneDetGate / scripts/yolo_dualh_seed.py). Until a drone is confirmed it emits ZERO
/// detections, so SAMURAI never seeds on terrain; once confirmed it passes exactly the
/// drone det, so SAMURAI's existing auto-seed (and fusekf reseed) lock the real target.
/// SAMURAI and nvmmfusekf are unchanged.
///
/// The expensive ORB+homography runs only while SEARCHING; once locked the gate just
/// associates YOLO dets to the lock (cheap). It maps the NVMM Y plane via a VIC
/// downscale + CPU map (same pattern as nvmmsamurai's GMC).
#include "config.h"
#include "nvmm_det_meta.h"
#include "gstnvmmallocator.h"
#include "dronedet_gate.hpp"
#include "xfeat_matcher.hpp"   // gst/common: nvmm::XfeatMatcher / XfeatFrame
#include "xfeat_motion.hpp"    // gst/common: ransac_affine / combined_residuals_2ref

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <nvbufsurface.h>
#include <deque>
#include <string>
#include <vector>

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_dronedet_debug);
#define GST_CAT_DEFAULT gst_nvmm_dronedet_debug
#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

#define GST_TYPE_NVMM_DRONEDET (gst_nvmm_dronedet_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmDroneDet, gst_nvmm_dronedet, GST, NVMM_DRONEDET, GstBaseTransform)

struct _GstNvmmDroneDet {
    GstBaseTransform parent;
    /* props */
    gchar  *engine_dir;   /* dir holding xfeat.engine + lightglue.engine */
    gint   target_class;
    gdouble min_conf;
    gint   ds_factor;     /* (v1 unused: matcher works in fixed registration space) */
    gint   dlt;
    gdouble rmin, rminsky, confsky, dist;
    gint   amin, ksup, maxlost;
    gdouble borderfrac;   /* border reject (0 = off, reversible) */
    gboolean enabled;     /* false = pure passthrough (no filtering) */
    gboolean seed_on_motion; /* seed SAMURAI from a motion blob when YOLO is silent */
    gint    motion_silent;   /* YOLO-silent frames before motion-seeding activates */
    gdouble motion_rmin;     /* residual threshold for a motion blob */
    /* state */
    nvmm::DroneDetGate *gate;
    nvmm::XfeatMatcher *matcher;          /* XFeat+LightGlue (owns engines + stream) */
    gboolean matcher_ready;
    std::deque<nvmm::XfeatFrame> *hist;   /* rolling per-frame features (two past refs) */
    guint64 frame_no;
    gboolean announced;
};

G_DEFINE_TYPE(GstNvmmDroneDet, gst_nvmm_dronedet, GST_TYPE_BASE_TRANSFORM)

enum { PROP_0, PROP_ENGINE_DIR, PROP_TARGET_CLASS, PROP_MIN_CONF, PROP_DS, PROP_DLT, PROP_RMIN,
       PROP_RMINSKY, PROP_CONFSKY, PROP_DIST, PROP_AMIN, PROP_KSUP, PROP_MAXLOST,
       PROP_BORDERFRAC, PROP_ENABLED,
       PROP_SEED_ON_MOTION, PROP_MOTION_SILENT, PROP_MOTION_RMIN };

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

static NvBufSurface *surface_of(GstBuffer *buf)
{
    GstMemory *m = gst_buffer_peek_memory(buf, 0);
    if (m && gst_is_nvmm_memory(m))
        return static_cast<NvBufSurface *>(gst_nvmm_memory_get_surface(m));
    /* Fallback (matches nvmmsamurai): DeepStream-style NVMM buffers map to an
       NvBufSurface struct rather than carrying our allocator's nvmm tag. */
    GstMapInfo map;
    if (m && gst_buffer_map(buf, &map, GST_MAP_READ)) {
        auto *s = reinterpret_cast<NvBufSurface *>(map.data);
        gst_buffer_unmap(buf, &map);
        return s;
    }
    return nullptr;
}

static GstFlowReturn
gst_nvmm_dronedet_transform_ip(GstBaseTransform *bt, GstBuffer *buf)
{
    auto *self = GST_NVMM_DRONEDET(bt);
    self->frame_no++;
    if (!self->enabled) return GST_FLOW_OK;

    GstNvmmDetMeta *det = gst_buffer_get_nvmm_det_meta(buf);
    NvBufSurface *surf = surface_of(buf);
    if (!surf) { GST_WARNING_OBJECT(self, "no NvBufSurface; passthrough"); return GST_FLOW_OK; }

    const int surfW = (int)surf->surfaceList[0].width;
    const int surfH = (int)surf->surfaceList[0].height;

    /* lazy-init the XFeat matcher + gate on the first frame */
    if (!self->matcher_ready) {
        if (!self->engine_dir || !*self->engine_dir) {
            GST_ERROR_OBJECT(self, "engine-dir not set; passthrough");
            return GST_FLOW_OK;
        }
        self->matcher = new nvmm::XfeatMatcher();
        std::string e;
        if (!self->matcher->init(self->engine_dir, e)) {
            GST_ERROR_OBJECT(self, "xfeat matcher init failed: %s; passthrough", e.c_str());
            delete self->matcher; self->matcher = nullptr;
            return GST_FLOW_OK;
        }
        self->matcher_ready = TRUE;
        nvmm::GateCfg cfg;
        cfg.dlt = self->dlt; cfg.rmin = (float)self->rmin;
        cfg.rminsky = (float)self->rminsky; cfg.confsky = (float)self->confsky;
        cfg.dist = (float)self->dist; cfg.amin = self->amin; cfg.ksup = self->ksup;
        cfg.maxlost = self->maxlost; cfg.borderfrac = (float)self->borderfrac;
        cfg.seed_on_motion = self->seed_on_motion; cfg.motion_silent = self->motion_silent;
        cfg.motion_rmin = (float)self->motion_rmin;
        self->gate = new nvmm::DroneDetGate(cfg);
        GST_INFO_OBJECT(self, "init: surf %dx%d, XFeat matcher ready", surfW, surfH);
    }

    /* extract this frame's features + keep a rolling history (two past references) */
    self->hist->emplace_back();
    nvmm::XfeatFrame &cur = self->hist->back();
    { std::string e; if (!self->matcher->extract(surf, cur, e))
        GST_WARNING_OBJECT(self, "xfeat extract failed: %s", e.c_str()); }
    const size_t need = (size_t)2 * self->dlt + 1;
    while (self->hist->size() > need) self->hist->pop_front();

    /* collect target-class detections in SURFACE coords (DetMeta is inference coords) */
    std::vector<nvmm::GateDet> dets;
    if (det && det->num_objects) {
        const float sx = det->infer_width  ? (float)surfW / det->infer_width  : 1.f;
        const float sy = det->infer_height ? (float)surfH / det->infer_height : 1.f;
        for (guint32 i = 0; i < det->num_objects; i++) {
            const NvmmDetObject &o = det->objects[i];
            if (o.class_id != self->target_class || o.confidence < self->min_conf) continue;
            dets.push_back({ (float)(o.left + o.width / 2.0) * sx,
                             (float)(o.top + o.height / 2.0) * sy,
                             (float)o.width * sx, (float)o.height * sy,
                             o.confidence, (int)i });
        }
    }

    /* independent-motion residuals: match cur against the two past references, fit a
       background affine per reference (RANSAC rejects movers as outliers), take the
       per-anchor MIN residual, and convert to surface coords -> MotionSample list. */
    int keep = -1;
    if ((int)self->hist->size() >= (int)need && self->matcher_ready && !cur.empty()) {
        const auto &h = *self->hist;
        const nvmm::XfeatFrame &refA = h[h.size() - 1 - self->dlt];
        const nvmm::XfeatFrame &refB = h[h.size() - 1 - 2 * self->dlt];
        std::vector<nvmm::motion::MatchPair> mA, mB; std::string me;
        std::vector<nvmm::MotionSample> motion;
        if (self->matcher->match(cur, refA, mA, me) && self->matcher->match(cur, refB, mB, me)) {
            auto fA = nvmm::motion::ransac_affine(mA);   // global background (no exclusion)
            auto fB = nvmm::motion::ransac_affine(mB);
            if (fA.ok && fB.ok) {
                auto rp = nvmm::motion::combined_residuals_2ref(fA.M, mA, fB.M, mB);
                const double isx = (double)surfW / nvmm::XfeatMatcher::kRW;  // reg -> surface
                const double isy = (double)surfH / nvmm::XfeatMatcher::kRH;
                motion.reserve(rp.size());
                for (const auto &r : rp)
                    motion.push_back({ (float)(r.pt.x * isx), (float)(r.pt.y * isy), (float)r.resid });
            }
        }
        keep = self->gate->update(motion, dets, surfW, surfH);
    }

    /* FILTER DetMeta: until a drone is confirmed, emit ZERO dets; then exactly one. */
    if (det) {
        float scx, scy, sw, sh;
        if (keep >= 0 && keep < (int)det->num_objects) {
            if (keep != 0) det->objects[0] = det->objects[keep];
            det->num_objects = 1;
            if (!self->announced) {
                GST_INFO_OBJECT(self, "drone CONFIRMED at frame %" G_GUINT64_FORMAT
                                " -> seeding SAMURAI", self->frame_no);
                self->announced = TRUE;
            }
        } else if (keep == -2 && self->gate && self->gate->synth_seed(scx, scy, sw, sh)) {
            /* SEED-ON-MOTION: YOLO was silent; synthesize one det from the confirmed
               motion blob (surface coords; DetMeta infer_w/h == frame size, so scale 1)
               so SAMURAI auto-seeds the big mover the detector can't name. */
            NvmmDetObject &o = det->objects[0];
            o.left = scx - sw / 2.f; o.top = scy - sh / 2.f; o.width = sw; o.height = sh;
            o.class_id = self->target_class; o.confidence = 0.90f; o.tracker_id = 0;
            g_strlcpy(o.label, "drone", NVMM_META_LABEL_LEN);
            det->num_objects = 1;
            GST_INFO_OBJECT(self, "MOTION-SEED at frame %" G_GUINT64_FORMAT
                            " (%.0f,%.0f %.0fx%.0f) -> seeding SAMURAI",
                            self->frame_no, o.left, o.top, sw, sh);
        } else {
            det->num_objects = 0;
        }
    }
    return GST_FLOW_OK;
}

static void set_prop(GObject *o, guint id, const GValue *v, GParamSpec *p) {
    auto *s = GST_NVMM_DRONEDET(o);
    switch (id) {
    case PROP_ENGINE_DIR:   g_free(s->engine_dir); s->engine_dir = g_value_dup_string(v); break;
    case PROP_TARGET_CLASS: s->target_class = g_value_get_int(v); break;
    case PROP_MIN_CONF:     s->min_conf = g_value_get_double(v); break;
    case PROP_DS:           s->ds_factor = g_value_get_int(v); break;
    case PROP_DLT:          s->dlt = g_value_get_int(v); break;
    case PROP_RMIN:         s->rmin = g_value_get_double(v); break;
    case PROP_RMINSKY:      s->rminsky = g_value_get_double(v); break;
    case PROP_CONFSKY:      s->confsky = g_value_get_double(v); break;
    case PROP_DIST:         s->dist = g_value_get_double(v); break;
    case PROP_AMIN:         s->amin = g_value_get_int(v); break;
    case PROP_KSUP:         s->ksup = g_value_get_int(v); break;
    case PROP_MAXLOST:      s->maxlost = g_value_get_int(v); break;
    case PROP_BORDERFRAC:   s->borderfrac = g_value_get_double(v); break;
    case PROP_ENABLED:      s->enabled = g_value_get_boolean(v); break;
    case PROP_SEED_ON_MOTION: s->seed_on_motion = g_value_get_boolean(v); break;
    case PROP_MOTION_SILENT:  s->motion_silent = g_value_get_int(v); break;
    case PROP_MOTION_RMIN:    s->motion_rmin = g_value_get_double(v); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p);
    }
}
static void get_prop(GObject *o, guint id, GValue *v, GParamSpec *p) {
    auto *s = GST_NVMM_DRONEDET(o);
    switch (id) {
    case PROP_ENGINE_DIR:   g_value_set_string(v, s->engine_dir); break;
    case PROP_TARGET_CLASS: g_value_set_int(v, s->target_class); break;
    case PROP_MIN_CONF:     g_value_set_double(v, s->min_conf); break;
    case PROP_DS:           g_value_set_int(v, s->ds_factor); break;
    case PROP_DLT:          g_value_set_int(v, s->dlt); break;
    case PROP_RMIN:         g_value_set_double(v, s->rmin); break;
    case PROP_RMINSKY:      g_value_set_double(v, s->rminsky); break;
    case PROP_CONFSKY:      g_value_set_double(v, s->confsky); break;
    case PROP_DIST:         g_value_set_double(v, s->dist); break;
    case PROP_AMIN:         g_value_set_int(v, s->amin); break;
    case PROP_KSUP:         g_value_set_int(v, s->ksup); break;
    case PROP_MAXLOST:      g_value_set_int(v, s->maxlost); break;
    case PROP_BORDERFRAC:   g_value_set_double(v, s->borderfrac); break;
    case PROP_ENABLED:      g_value_set_boolean(v, s->enabled); break;
    case PROP_SEED_ON_MOTION: g_value_set_boolean(v, s->seed_on_motion); break;
    case PROP_MOTION_SILENT:  g_value_set_int(v, s->motion_silent); break;
    case PROP_MOTION_RMIN:    g_value_set_double(v, s->motion_rmin); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p);
    }
}

static gboolean gst_nvmm_dronedet_stop(GstBaseTransform *bt) {
    auto *s = GST_NVMM_DRONEDET(bt);
    delete s->gate; s->gate = nullptr;
    delete s->matcher; s->matcher = nullptr; s->matcher_ready = FALSE;
    if (s->hist) s->hist->clear();
    return TRUE;
}

static void gst_nvmm_dronedet_finalize(GObject *o) {
    auto *s = GST_NVMM_DRONEDET(o);
    delete s->gate; delete s->matcher; delete s->hist;
    g_free(s->engine_dir);
    G_OBJECT_CLASS(gst_nvmm_dronedet_parent_class)->finalize(o);
}

/* Teardown: nvmmfusekf sends an upstream "nvmm-reset" when a track is torn down
   (parked on edge clutter / target gone). Un-latch the gate so it re-acquires.
   Do NOT consume — other elements upstream may also want it. */
static gboolean
gst_nvmm_dronedet_src_event(GstBaseTransform *bt, GstEvent *ev)
{
    auto *self = GST_NVMM_DRONEDET(bt);
    if (GST_EVENT_TYPE(ev) == GST_EVENT_CUSTOM_UPSTREAM) {
        const GstStructure *s = gst_event_get_structure(ev);
        if (s && gst_structure_has_name(s, "nvmm-reset") && self->gate) {
            self->gate->reset();
            GST_INFO_OBJECT(self, "reset requested -> gate un-latched");
        }
    }
    return GST_BASE_TRANSFORM_CLASS(gst_nvmm_dronedet_parent_class)->src_event(bt, ev);
}

static void
gst_nvmm_dronedet_class_init(GstNvmmDroneDetClass *klass)
{
    auto *go = G_OBJECT_CLASS(klass);
    auto *el = GST_ELEMENT_CLASS(klass);
    auto *bt = GST_BASE_TRANSFORM_CLASS(klass);
    go->set_property = set_prop; go->get_property = get_prop;
    go->finalize = gst_nvmm_dronedet_finalize;

    auto F = (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS);
    g_object_class_install_property(go, PROP_ENGINE_DIR, g_param_spec_string(
        "engine-dir", "Engine dir", "Directory holding xfeat.engine + lightglue.engine "
        "(the XFeat matcher for the independent-motion gate)", nullptr, F));
    g_object_class_install_property(go, PROP_TARGET_CLASS, g_param_spec_int(
        "target-class", "Target class", "YOLO class id of the drone", 0, 9999, 0, F));
    g_object_class_install_property(go, PROP_MIN_CONF, g_param_spec_double(
        "min-conf", "Min conf", "Min YOLO conf to consider a detection", 0, 1, 0.25, F));
    g_object_class_install_property(go, PROP_DS, g_param_spec_int(
        "ds", "Downscale", "Y-plane downscale factor for the gate", 1, 8, 2, F));
    g_object_class_install_property(go, PROP_DLT, g_param_spec_int(
        "dlt", "Frame delta", "Past-frame delta for motion references", 1, 30, 5, F));
    g_object_class_install_property(go, PROP_RMIN, g_param_spec_double(
        "rmin", "Dual-H residual", "Min dual-homography residual at a det", 0, 255, 12, F));
    g_object_class_install_property(go, PROP_RMINSKY, g_param_spec_double(
        "rminsky", "Sky-diff motion", "Min sky-diff motion at a det", 0, 255, 8, F));
    g_object_class_install_property(go, PROP_CONFSKY, g_param_spec_double(
        "confsky", "Sky-diff conf gate", "Min YOLO conf for the sky-diff path", 0, 1, 0.55, F));
    g_object_class_install_property(go, PROP_DIST, g_param_spec_double(
        "dist", "Assoc dist", "Track association gate (surface px)", 1, 1000, 45, F));
    g_object_class_install_property(go, PROP_AMIN, g_param_spec_int(
        "amin", "Min age", "Min track age before confirm", 1, 1000, 6, F));
    g_object_class_install_property(go, PROP_KSUP, g_param_spec_int(
        "ksup", "Support frames", "Consecutive supported frames to confirm", 1, 1000, 4, F));
    g_object_class_install_property(go, PROP_MAXLOST, g_param_spec_int(
        "maxlost", "Max lost", "Frames a track may miss before death", 0, 1000, 2, F));
    g_object_class_install_property(go, PROP_BORDERFRAC, g_param_spec_double(
        "border-frac", "Border reject", "Reject seeds/locks in letterbox/pillarbox or within "
        "this fraction of the active edge (0 = off)", 0, 0.4, 0.02, F));
    g_object_class_install_property(go, PROP_ENABLED, g_param_spec_boolean(
        "enabled", "Enabled", "Filter dets (false = passthrough)", TRUE, F));
    g_object_class_install_property(go, PROP_SEED_ON_MOTION, g_param_spec_boolean(
        "seed-on-motion", "Seed on motion",
        "When YOLO is silent, seed SAMURAI from the strongest persistent independent-motion "
        "blob (detector-independent seeding for big movers YOLO can't name). Off by default.",
        FALSE, F));
    g_object_class_install_property(go, PROP_MOTION_SILENT, g_param_spec_int(
        "motion-silent", "Motion-seed silence",
        "YOLO-silent frames required before motion-seeding activates", 1, 1000, 12, F));
    g_object_class_install_property(go, PROP_MOTION_RMIN, g_param_spec_double(
        "motion-rmin", "Motion blob residual",
        "Independent-motion residual (downscaled) to threshold a motion blob", 0, 255, 16, F));

    gst_element_class_add_static_pad_template(el, &sink_tmpl);
    gst_element_class_add_static_pad_template(el, &src_tmpl);
    gst_element_class_set_static_metadata(el,
        "NVMM Drone Seed Gate", "Filter/Analyzer/Video",
        "Filters YOLO detections to the single independently-moving drone "
        "(dual-homography ∩ YOLO) so SAMURAI auto-seeds the real target",
        "Pavel Guzenfeld");

    bt->transform_ip = gst_nvmm_dronedet_transform_ip;
    bt->src_event = gst_nvmm_dronedet_src_event;
    bt->stop = gst_nvmm_dronedet_stop;
    bt->passthrough_on_same_caps = FALSE;
    GST_DEBUG_CATEGORY_INIT(gst_nvmm_dronedet_debug, "nvmmdronedet", 0, "NVMM drone seed gate");
}

static void
gst_nvmm_dronedet_init(GstNvmmDroneDet *self)
{
    self->target_class = 0; self->min_conf = 0.25; self->ds_factor = 2; self->dlt = 5;
    self->rmin = 12; self->rminsky = 8; self->confsky = 0.55; self->dist = 45;
    self->amin = 6; self->ksup = 4; self->maxlost = 2; self->borderfrac = 0.02; self->enabled = TRUE;
    self->seed_on_motion = FALSE; self->motion_silent = 12; self->motion_rmin = 16;
    self->engine_dir = nullptr;
    self->gate = nullptr; self->matcher = nullptr; self->matcher_ready = FALSE;
    self->hist = new std::deque<nvmm::XfeatFrame>(); self->frame_no = 0; self->announced = FALSE;
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
}

static gboolean plugin_init(GstPlugin *plugin) {
    return gst_element_register(plugin, "nvmmdronedet", GST_RANK_NONE, GST_TYPE_NVMM_DRONEDET);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    nvmmdronedet, "Autonomous drone-seed gate (YOLO ∩ dual-homography)",
    plugin_init, PACKAGE_VERSION, "LGPL", "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp")
