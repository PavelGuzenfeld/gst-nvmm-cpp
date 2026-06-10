#include "config.h"

#include "gstnvmmdrawdet.h"
#include "nvmm_det_meta.h"
#include "gstnvmmallocator.h"

#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <cuda_runtime.h>
#include <cuda_egl_interop.h>

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
    NvBufSurface  *rgba;               /* VIC-native RGBA, full-frame */
    cudaGraphicsResource_t egl_res;    /* CUDA view of rgba via EGL */
};

G_DEFINE_TYPE(GstNvmmDrawDet, gst_nvmm_drawdet, GST_TYPE_BASE_TRANSFORM)

enum { PROP_0, PROP_THICKNESS };

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

    /* Draw detection boxes (scaled from the meta's inference-frame space). */
    GstNvmmDetMeta *m = gst_buffer_get_nvmm_det_meta(inbuf);
    if (m && m->num_objects) {
        const float sx = m->infer_width  ? (float)W / m->infer_width  : 1.f;
        const float sy = m->infer_height ? (float)H / m->infer_height : 1.f;
        for (guint32 i = 0; i < m->num_objects; i++) {
            const NvmmDetObject &o = m->objects[i];
            guint8 r8, g8, b8;
            class_color(o.class_id, &r8, &g8, &b8);
            draw_rect((guint8 *)omap.data, W, H,
                      (int)(o.left * sx), (int)(o.top * sy),
                      (int)(o.width * sx), (int)(o.height * sy),
                      self->thickness, r8, g8, b8);
        }
    }
    gst_buffer_unmap(outbuf, &omap);
    return GST_FLOW_OK;
}

static void
gst_nvmm_drawdet_set_property(GObject *o, guint id, const GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_DRAWDET(o);
    if (id == PROP_THICKNESS) self->thickness = g_value_get_int(v);
    else G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p);
}
static void
gst_nvmm_drawdet_get_property(GObject *o, guint id, GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_DRAWDET(o);
    if (id == PROP_THICKNESS) g_value_set_int(v, self->thickness);
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

    gst_element_class_add_static_pad_template(el, &sink_tmpl);
    gst_element_class_add_static_pad_template(el, &src_tmpl);
    gst_element_class_set_static_metadata(el,
        "NVMM Detection Overlay", "Filter/Editor/Video",
        "Draw NvmmDetMeta boxes onto the frame (NVMM NV12 -> system RGBA) for "
        "viewing and end-to-end tests",
        "Pavel Guzenfeld");

    bt->transform_caps = gst_nvmm_drawdet_transform_caps;
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
