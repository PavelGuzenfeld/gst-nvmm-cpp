#include "config.h"

#include "gstnvmmdrawdet.h"
#include "nvmm_det_meta.h"
#include "nvmm_motion_meta.h"
#include "nvmm_class_meta.h"
#include "nvmm_track_meta.h"
#include "gstnvmmallocator.h"
#include "font6x8.h"

#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <cuda_runtime.h>
#include <cuda_egl_interop.h>

#include <cmath>
#include <cstring>
#include <string>

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_drawdet_debug);
#define GST_CAT_DEFAULT gst_nvmm_drawdet_debug

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

struct _GstNvmmDrawDet {
    GstBaseTransform parent;
    gint           width, height;
    gint           thickness;          /* property: box line thickness (px) */
    gboolean       draw_labels;        /* property: draw "label conf%" text */
    gboolean       draw_det;           /* property: draw raw YOLO det boxes */
    NvBufSurface  *rgba;               /* VIC-native RGBA, full-frame */
    cudaGraphicsResource_t egl_res;    /* CUDA view of rgba via EGL */
    /* SAMURAI/fusekf track-meta HUD: live FPS (EMA) + tracking coverage. */
    gboolean       draw_track;         /* property: draw GstNvmmTrackMeta + HUD */
    gdouble        fps_smoothing;      /* property: HUD FPS EMA weight on history (0..1) */
    gint           font_scale_div;    /* property: font px = max(1, height/this) */
    gint64         last_us;            /* monotonic time of previous frame */
    double         ema_fps;
    guint64        n_frames, n_valid;
};

G_DEFINE_TYPE(GstNvmmDrawDet, gst_nvmm_drawdet, GST_TYPE_BASE_TRANSFORM)

enum { PROP_0, PROP_THICKNESS, PROP_DRAW_LABELS, PROP_DRAW_TRACK, PROP_DRAW_DET,
       PROP_FPS_SMOOTHING, PROP_FONT_SCALE_DIV };

static GstStaticPadTemplate sink_tmpl = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string)NV12, "
                    "width=(int)[32,8192], height=(int)[32,8192], "
                    "framerate=(fraction)[0/1, 240/1]"));
static GstStaticPadTemplate src_tmpl = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw, format=(string)RGBA, "
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

/* NVMM NV12 <-> system RGBA, same dimensions/framerate. */
static GstCaps *
gst_nvmm_drawdet_transform_caps(GstBaseTransform *, GstPadDirection direction,
                                GstCaps *caps, GstCaps *filter)
{
    GstCaps *res = gst_caps_new_empty();
    for (guint i = 0; i < gst_caps_get_size(caps); i++) {
        GstStructure *st = gst_structure_copy(gst_caps_get_structure(caps, i));
        if (direction == GST_PAD_SINK) {  // -> src: system RGBA
            gst_structure_set(st, "format", G_TYPE_STRING, "RGBA", nullptr);
            gst_caps_append_structure_full(res, st, gst_caps_features_new_empty());
        } else {                          // -> sink: NVMM NV12
            gst_structure_set(st, "format", G_TYPE_STRING, "NV12", nullptr);
            gst_caps_append_structure_full(res, st,
                gst_caps_features_new("memory:NVMM", nullptr));
        }
    }
    if (filter) {
        GstCaps *tmp = gst_caps_intersect_full(filter, res, GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(res);
        res = tmp;
    }
    return res;
}

static gboolean
gst_nvmm_drawdet_set_caps(GstBaseTransform *bt, GstCaps *incaps, GstCaps *)
{
    auto *self = GST_NVMM_DRAWDET(bt);
    GstStructure *s = gst_caps_get_structure(incaps, 0);
    return gst_structure_get_int(s, "width", &self->width) &&
           gst_structure_get_int(s, "height", &self->height);
}

/* Output allocation size. The NVMM input's gst-buffer size is a surface
   HANDLE, not pixels, so the default unit-size scaling must never be used to
   size the packed-RGBA output: with a caps-any downstream (no pool from the
   ALLOCATION query, e.g. fakesink) that mis-sized the buffer and the host
   copy in transform() corrupted the heap. Compute the size from the raw-RGBA
   caps instead, whichever direction holds them. */
static gboolean
gst_nvmm_drawdet_transform_size(GstBaseTransform *bt, GstPadDirection direction,
                                GstCaps * /*caps*/, gsize /*size*/,
                                GstCaps *othercaps, gsize *othersize)
{
    GstCaps *rgba_caps = othercaps;  /* SINK direction: othercaps = src (RGBA) */
    if (direction == GST_PAD_SRC) {
        /* othercaps = sink (NVMM NV12): the NVMM buffer is a handle; report
           the only meaningful size we have, the surface-struct pointer slot. */
        *othersize = sizeof(NvBufSurface *);
        return TRUE;
    }
    GstStructure *s = gst_caps_get_structure(rgba_caps, 0);
    gint w = 0, h = 0;
    if (!gst_structure_get_int(s, "width", &w) ||
        !gst_structure_get_int(s, "height", &h) || w <= 0 || h <= 0) {
        GST_ERROR_OBJECT(bt, "RGBA caps without width/height: %" GST_PTR_FORMAT,
                         rgba_caps);
        return FALSE;
    }
    *othersize = (gsize)w * h * 4;
    return TRUE;
}

static gboolean
ensure_rgba(GstNvmmDrawDet *self)
{
    if (self->rgba) return TRUE;
    NvBufSurfaceCreateParams p{};
    p.width = (guint)self->width; p.height = (guint)self->height;
    p.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    p.layout = NVBUF_LAYOUT_PITCH;
    p.memType = NVBUF_MEM_DEFAULT;
    if (NvBufSurfaceCreate(&self->rgba, 1, &p) != 0 || !self->rgba) {
        GST_ERROR_OBJECT(self, "NvBufSurfaceCreate(RGBA) failed");
        return FALSE;
    }
    self->rgba->numFilled = 1;
    if (NvBufSurfaceMapEglImage(self->rgba, 0) != 0) {
        GST_ERROR_OBJECT(self, "NvBufSurfaceMapEglImage failed");
        return FALSE;
    }
    if (cudaGraphicsEGLRegisterImage(&self->egl_res,
            self->rgba->surfaceList[0].mappedAddr.eglImage,
            cudaGraphicsRegisterFlagsNone) != cudaSuccess) {
        GST_ERROR_OBJECT(self, "cudaGraphicsEGLRegisterImage failed");
        return FALSE;
    }
    return TRUE;
}

/* Vivid, high-contrast color per class id (RGBA), cycling a fixed palette. */
static void
class_color(int id, guint8 *r, guint8 *g, guint8 *b)
{
    static const guint8 pal[][3] = {
        {255,  64,  64}, { 64, 255,  64}, { 64, 160, 255}, {255, 255,  64},
        {255,  64, 255}, { 64, 255, 255}, {255, 160,  64}, {160,  64, 255},
    };
    const guint8 *c = pal[((guint)id) % (sizeof(pal) / sizeof(pal[0]))];
    *r = c[0]; *g = c[1]; *b = c[2];
}

static void
draw_rect(guint8 *rgba, int W, int H, int x, int y, int w, int h, int t,
          guint8 r, guint8 g, guint8 b)
{
    auto px = [&](int xx, int yy) {
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) return;
        guint8 *p = rgba + ((size_t)yy * W + xx) * 4;
        p[0] = r; p[1] = g; p[2] = b; p[3] = 255;
    };
    for (int k = 0; k < t; k++) {
        for (int xx = x; xx < x + w; xx++) { px(xx, y + k); px(xx, y + h - 1 - k); }
        for (int yy = y; yy < y + h; yy++) { px(x + k, yy); px(x + w - 1 - k, yy); }
    }
}

/* Fill the rect [x,x+w) x [y,y+h), clipped once to the frame. */
static void
fill_rect(guint8 *rgba, int W, int H, int x, int y, int w, int h,
          guint8 r, guint8 g, guint8 b)
{
    const int x0 = MAX(0, x), y0 = MAX(0, y), x1 = MIN(W, x + w), y1 = MIN(H, y + h);
    for (int yy = y0; yy < y1; yy++) {
        guint8 *p = rgba + ((size_t)yy * W + x0) * 4;
        for (int xx = x0; xx < x1; xx++, p += 4) { p[0] = r; p[1] = g; p[2] = b; p[3] = 255; }
    }
}

/* Draw `str` with the built-in 6x8 font scaled by `s`: a filled bar in the box
 * color (r,g,b) with black glyphs on top, for legibility over any background.
 * (x,y) is the glyph origin; the bar is padded one scaled pixel around it. */
static void
draw_text(guint8 *rgba, int W, int H, int x, int y, const char *str, int s,
          guint8 r, guint8 g, guint8 b)
{
    const int n = (int)strlen(str);
    fill_rect(rgba, W, H, x - s, y - s, n * FONT_W * s + 2 * s, FONT_H * s + 2 * s, r, g, b);
    for (int i = 0; i < n; i++) {
        unsigned char c = (unsigned char)str[i];
        if (c < FONT_FIRST || c > FONT_LAST) c = '?';
        const unsigned char *gph = kFont[c - FONT_FIRST];
        for (int row = 0; row < FONT_H; row++)
            for (int col = 0; col < FONT_W; col++)
                if (gph[row] & (1u << col))
                    fill_rect(rgba, W, H, x + (i * FONT_W + col) * s, y + row * s, s, s, 0, 0, 0);
    }
}

static GstFlowReturn
gst_nvmm_drawdet_transform(GstBaseTransform *bt, GstBuffer *inbuf, GstBuffer *outbuf)
{
    auto *self = GST_NVMM_DRAWDET(bt);
    if (self->width <= 0 || !ensure_rgba(self)) return GST_FLOW_ERROR;

    NvBufSurface *src = surface_of(inbuf);
    if (!src) { GST_WARNING_OBJECT(self, "no NvBufSurface"); return GST_FLOW_ERROR; }
    if (src->numFilled == 0) src->numFilled = src->batchSize ? src->batchSize : 1;

    /* NV12 -> RGBA (full frame) on the VIC. */
    NvBufSurfTransformParams xform{};
    if (NvBufSurfTransform(src, self->rgba, &xform) != NvBufSurfTransformError_Success) {
        GST_WARNING_OBJECT(self, "NvBufSurfTransform failed");
        return GST_FLOW_ERROR;
    }

    GstMapInfo omap;
    if (!gst_buffer_map(outbuf, &omap, GST_MAP_WRITE)) return GST_FLOW_ERROR;
    const int W = self->width, H = self->height;

    /* Refuse to scribble: a mis-negotiated output buffer fails loudly here
       instead of corrupting the heap (see transform_size above). */
    if (omap.size < (gsize)W * H * 4) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
                          ("output buffer is %" G_GSIZE_FORMAT " bytes, need %dx%dx4",
                           omap.size, W, H), (nullptr));
        gst_buffer_unmap(outbuf, &omap);
        return GST_FLOW_ERROR;
    }

    /* EGL/CUDA view of the RGBA surface -> copy to the (host) output buffer. */
    cudaEglFrame ef;
    cudaError_t r = cudaGraphicsResourceGetMappedEglFrame(&ef, self->egl_res, 0, 0);
    if (r == cudaSuccess) {
        if (ef.frameType == cudaEglFrameTypePitch)
            r = cudaMemcpy2D(omap.data, (size_t)W * 4, ef.frame.pPitch[0].ptr,
                             ef.frame.pPitch[0].pitch, (size_t)W * 4, H,
                             cudaMemcpyDeviceToHost);
        else
            r = cudaMemcpy2DFromArray(omap.data, (size_t)W * 4, ef.frame.pArray[0],
                                      0, 0, (size_t)W * 4, H, cudaMemcpyDeviceToHost);
    }
    if (r != cudaSuccess) {
        GST_WARNING_OBJECT(self, "RGBA copy to host failed: %s", cudaGetErrorString(r));
        gst_buffer_unmap(outbuf, &omap);
        return GST_FLOW_ERROR;
    }

    /* Draw raw detection boxes (scaled from the meta's inference-frame space).
       Off when draw-det=false — for the tracker result video we only want the
       single fused track box, not the noisy per-frame YOLO detections. */
    GstNvmmDetMeta *m = gst_buffer_get_nvmm_det_meta(inbuf);
    if (self->draw_det && m && m->num_objects) {
        /* Motion annotation (from nvmmfusion): entries align by index. */
        GstNvmmMotionMeta *mm = gst_buffer_get_nvmm_motion_meta(inbuf);
        /* Classifier annotation (from nvmmsecondaryinfer): same alignment. */
        GstNvmmClassMeta *cm = gst_buffer_get_nvmm_class_meta(inbuf);
        const float sx = m->infer_width  ? (float)W / m->infer_width  : 1.f;
        const float sy = m->infer_height ? (float)H / m->infer_height : 1.f;
        const int ts = MAX(1, H / 360);  /* font scale: ~3 at 1080p, ~2 at 720p */
        for (guint32 i = 0; i < m->num_objects; i++) {
            const NvmmDetObject &o = m->objects[i];
            const gboolean moving =
                mm && i < mm->num_objects && mm->objects[i].moving;
            guint8 r8, g8, b8;
            class_color(o.class_id, &r8, &g8, &b8);
            const int bx = (int)(o.left * sx), by = (int)(o.top * sy);
            /* Movers get a heavier box so motion reads at a glance. */
            draw_rect((guint8 *)omap.data, W, H, bx, by,
                      (int)(o.width * sx), (int)(o.height * sy),
                      moving ? self->thickness * 2 : self->thickness, r8, g8, b8);

            if (!self->draw_labels) continue;
            /* "car #4 82% >> [taxi 91%]" — tracker id when assigned, ">>"
             * when moving, secondary-classifier result when attached. */
            char idbuf[24] = "";
            if (o.tracker_id)
                g_snprintf(idbuf, sizeof idbuf, " #%" G_GUINT64_FORMAT, o.tracker_id);
            char clsbuf[NVMM_META_LABEL_LEN + 16] = "";
            if (cm && i < cm->num_objects && cm->objects[i].class_id >= 0)
                g_snprintf(clsbuf, sizeof clsbuf, " [%s %.0f%%]",
                           cm->objects[i].label,
                           (double)cm->objects[i].confidence * 100.0);
            char text[2 * NVMM_META_LABEL_LEN + 64];
            g_snprintf(text, sizeof text, "%s%s %.0f%%%s%s",
                       o.label[0] ? o.label : "obj", idbuf,
                       (double)o.confidence * 100.0, moving ? " >>" : "", clsbuf);
            /* Label bar just above the box; tuck it inside the top if no room. */
            int ty = by - FONT_H * ts - ts;
            if (ty < ts) ty = by + ts;
            draw_text((guint8 *)omap.data, W, H, bx + ts, ty, text, ts, r8, g8, b8);
        }
    }
    /* SAMURAI/fusekf track-meta box (frame coords == W x H) + live HUD. */
    if (self->draw_track) {
        const gint64 now = g_get_monotonic_time();
        if (self->last_us) {
            const double inst = 1e6 / (double)(now - self->last_us);
            const double a = self->fps_smoothing;   /* prop fps-smoothing (def 0.9) */
            self->ema_fps = self->ema_fps > 0 ? a * self->ema_fps + (1.0 - a) * inst : inst;
        }
        self->last_us = now;
        self->n_frames++;
        GstNvmmTrackMeta *tm = gst_buffer_get_nvmm_track_meta(inbuf);
        const int ts = MAX(1, H / self->font_scale_div);  /* prop font-scale-divisor (def 540) */
        const int line = MAX(1, self->thickness);   /* thin track box (thickness prop) */
        if (tm && tm->valid && tm->width > 0) {
            self->n_valid++;
            draw_rect((guint8 *)omap.data, W, H, (int)tm->left, (int)tm->top,
                      (int)tm->width, (int)tm->height, line, 0, 255, 255);
            char tl[32];
            const double conf = 1.0 / (1.0 + exp(-(double)tm->object_score));
            g_snprintf(tl, sizeof tl, "target %.0f%%", conf * 100.0);
            int ty = (int)tm->top - FONT_H * ts - ts; if (ty < ts) ty = (int)tm->top + ts;
            draw_text((guint8 *)omap.data, W, H, (int)tm->left + ts, ty, tl, ts, 0, 255, 255);
        }
        const double cov = 100.0 * self->n_valid / self->n_frames;  /* n_frames++ above guarantees >= 1 */
        char hud[64];
        g_snprintf(hud, sizeof hud, "FPS %.1f  TRACK %.0f%%", self->ema_fps, cov);
        draw_text((guint8 *)omap.data, W, H, 8, 8, hud, ts, 0, 255, 0);
    }
    gst_buffer_unmap(outbuf, &omap);
    return GST_FLOW_OK;
}

static void
gst_nvmm_drawdet_set_property(GObject *o, guint id, const GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_DRAWDET(o);
    if (id == PROP_THICKNESS) self->thickness = g_value_get_int(v);
    else if (id == PROP_DRAW_LABELS) self->draw_labels = g_value_get_boolean(v);
    else if (id == PROP_DRAW_TRACK) self->draw_track = g_value_get_boolean(v);
    else if (id == PROP_DRAW_DET) self->draw_det = g_value_get_boolean(v);
    else if (id == PROP_FPS_SMOOTHING) self->fps_smoothing = g_value_get_double(v);
    else if (id == PROP_FONT_SCALE_DIV) self->font_scale_div = g_value_get_int(v);
    else G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p);
}
static void
gst_nvmm_drawdet_get_property(GObject *o, guint id, GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_DRAWDET(o);
    if (id == PROP_THICKNESS) g_value_set_int(v, self->thickness);
    else if (id == PROP_DRAW_LABELS) g_value_set_boolean(v, self->draw_labels);
    else if (id == PROP_DRAW_TRACK) g_value_set_boolean(v, self->draw_track);
    else if (id == PROP_DRAW_DET) g_value_set_boolean(v, self->draw_det);
    else if (id == PROP_FPS_SMOOTHING) g_value_set_double(v, self->fps_smoothing);
    else if (id == PROP_FONT_SCALE_DIV) g_value_set_int(v, self->font_scale_div);
    else G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p);
}

static gboolean
gst_nvmm_drawdet_stop(GstBaseTransform *bt)
{
    auto *self = GST_NVMM_DRAWDET(bt);
    if (self->egl_res) { cudaGraphicsUnregisterResource(self->egl_res); self->egl_res = nullptr; }
    if (self->rgba) {
        NvBufSurfaceUnMapEglImage(self->rgba, 0);
        NvBufSurfaceDestroy(self->rgba);
        self->rgba = nullptr;
    }
    return TRUE;
}

static void
gst_nvmm_drawdet_class_init(GstNvmmDrawDetClass *klass)
{
    auto *go = G_OBJECT_CLASS(klass);
    auto *el = GST_ELEMENT_CLASS(klass);
    auto *bt = GST_BASE_TRANSFORM_CLASS(klass);

    go->set_property = gst_nvmm_drawdet_set_property;
    go->get_property = gst_nvmm_drawdet_get_property;
    g_object_class_install_property(go, PROP_THICKNESS,
        g_param_spec_int("thickness", "Box thickness", "Detection box line width (px)",
            1, 32, 3, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(go, PROP_DRAW_TRACK,
        g_param_spec_boolean("draw-track", "Draw track + HUD",
            "Draw the GstNvmmTrackMeta box + live FPS/coverage HUD", TRUE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(go, PROP_DRAW_DET,
        g_param_spec_boolean("draw-det", "Draw raw detections",
            "Draw the raw YOLO GstNvmmDetMeta boxes/labels", TRUE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(go, PROP_DRAW_LABELS,
        g_param_spec_boolean("draw-labels", "Draw labels",
            "Draw the class label and confidence above each box", TRUE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(go, PROP_FPS_SMOOTHING,
        g_param_spec_double("fps-smoothing", "HUD FPS smoothing",
            "EMA weight on FPS history for the HUD (0 = instantaneous, ->1 = smoother)",
            0.0, 1.0, 0.9, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(go, PROP_FONT_SCALE_DIV,
        g_param_spec_int("font-scale-divisor", "Font scale divisor",
            "Overlay font pixel size = max(1, frame_height / this)",
            1, 10000, 540, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_add_static_pad_template(el, &sink_tmpl);
    gst_element_class_add_static_pad_template(el, &src_tmpl);
    gst_element_class_set_static_metadata(el,
        "NVMM Detection Overlay", "Filter/Editor/Video",
        "Draw NvmmDetMeta boxes onto the frame (NVMM NV12 -> system RGBA) for "
        "viewing and end-to-end tests",
        "Pavel Guzenfeld");

    bt->transform_caps = gst_nvmm_drawdet_transform_caps;
    bt->transform_size = gst_nvmm_drawdet_transform_size;
    bt->set_caps       = gst_nvmm_drawdet_set_caps;
    bt->transform      = gst_nvmm_drawdet_transform;
    bt->stop           = gst_nvmm_drawdet_stop;

    GST_DEBUG_CATEGORY_INIT(gst_nvmm_drawdet_debug, "nvmmdrawdet", 0, "NVMM detection overlay");
}

static void
gst_nvmm_drawdet_init(GstNvmmDrawDet *self)
{
    self->width = self->height = 0;
    self->thickness = 3;
    self->draw_labels = TRUE;
    self->draw_track = TRUE;
    self->draw_det = TRUE;
    self->fps_smoothing = 0.9;
    self->font_scale_div = 540;
    self->last_us = 0; self->ema_fps = 0; self->n_frames = 0; self->n_valid = 0;
    self->rgba = nullptr;
    self->egl_res = nullptr;
}

static gboolean
plugin_init(GstPlugin *plugin)
{
    return gst_element_register(plugin, "nvmmdrawdet", GST_RANK_NONE, GST_TYPE_NVMM_DRAWDET);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    nvmmdrawdet, "Draw NvmmDetMeta detections onto the frame for viewing",
    plugin_init, PACKAGE_VERSION, "LGPL", "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp")
