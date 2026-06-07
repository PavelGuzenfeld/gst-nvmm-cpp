#include "config.h"  // PACKAGE_VERSION

#include "gstnvmmofa.h"
#include "gstnvmmflowstats.h"
#include "nvmm_optical_flow_meta.h"
#include "gstnvmmallocator.h"

#include <nvbufsurface.h>

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/Status.h>
#include <vpi/algo/OpticalFlowDense.h>

#include <cstring>

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_ofa_debug);
#define GST_CAT_DEFAULT gst_nvmm_ofa_debug

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

/* ---- grid-size and quality enums ---- */

#define GST_TYPE_NVMM_OFA_GRID (gst_nvmm_ofa_grid_get_type())
static GType
gst_nvmm_ofa_grid_get_type(void)
{
    static GType t = 0;
    static const GEnumValue v[] = {
        { 1, "1x1 (dense)", "1" },
        { 2, "2x2",         "2" },
        { 4, "4x4",         "4" },
        { 8, "8x8",         "8" },
        { 0, nullptr, nullptr },
    };
    if (g_once_init_enter(&t)) {
        GType tmp = g_enum_register_static("GstNvmmOfaGrid", v);
        g_once_init_leave(&t, tmp);
    }
    return t;
}

#define GST_TYPE_NVMM_OFA_QUALITY (gst_nvmm_ofa_quality_get_type())
static GType
gst_nvmm_ofa_quality_get_type(void)
{
    static GType t = 0;
    static const GEnumValue v[] = {
        { VPI_OPTICAL_FLOW_QUALITY_LOW,    "Low",    "low" },
        { VPI_OPTICAL_FLOW_QUALITY_MEDIUM, "Medium", "medium" },
        { VPI_OPTICAL_FLOW_QUALITY_HIGH,   "High",   "high" },
        { 0, nullptr, nullptr },
    };
    if (g_once_init_enter(&t)) {
        GType tmp = g_enum_register_static("GstNvmmOfaQuality", v);
        g_once_init_leave(&t, tmp);
    }
    return t;
}

/* ---- element ---- */

struct _GstNvmmOfa {
    GstBaseTransform parent;

    gint grid_size;                 /* property: 1/2/4/8 */
    VPIOpticalFlowQuality quality;  /* property */

    /* runtime state */
    gint width, height;
    VPIStream stream;
    VPIPayload payload;
    VPIImage mv;                    /* OFA|CPU 2S16_BL output, reused per frame */
    VPIImage prev;                  /* wraps the previous frame's NVMM surface */
    GstBuffer *prev_buf;            /* ref keeps prev's surface alive/unmodified */
    gboolean configured;
};

G_DEFINE_TYPE(GstNvmmOfa, gst_nvmm_ofa, GST_TYPE_BASE_TRANSFORM)

enum { PROP_0, PROP_GRID_SIZE, PROP_QUALITY };

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

static VPIImage
wrap_ofa(NvBufSurface *surf)
{
    VPIImageData d;
    memset(&d, 0, sizeof(d));
    d.bufferType = VPI_IMAGE_BUFFER_NVBUFFER;
    d.buffer.fd  = (int)surf->surfaceList[0].bufferDesc;
    VPIImage img = nullptr;
    if (vpiImageCreateWrapper(&d, nullptr, VPI_BACKEND_OFA, &img) != VPI_SUCCESS)
        return nullptr;
    return img;
}

static void
release_state(GstNvmmOfa *self)
{
    if (self->prev)     { vpiImageDestroy(self->prev);     self->prev = nullptr; }
    if (self->prev_buf) { gst_buffer_unref(self->prev_buf); self->prev_buf = nullptr; }
    if (self->mv)       { vpiImageDestroy(self->mv);       self->mv = nullptr; }
    if (self->payload)  { vpiPayloadDestroy(self->payload); self->payload = nullptr; }
    if (self->stream)   { vpiStreamDestroy(self->stream);  self->stream = nullptr; }
    self->configured = FALSE;
}

static gboolean
gst_nvmm_ofa_set_caps(GstBaseTransform *bt, GstCaps *incaps, GstCaps *)
{
    auto *self = GST_NVMM_OFA(bt);
    GstStructure *s = gst_caps_get_structure(incaps, 0);
    if (!gst_structure_get_int(s, "width", &self->width) ||
        !gst_structure_get_int(s, "height", &self->height)) {
        GST_ERROR_OBJECT(self, "caps missing width/height");
        return FALSE;
    }
    /* dims changed — drop any previously-built OFA state */
    release_state(self);
    return TRUE;
}

/* Build the OFA payload, stream and reusable MV image once we know the input
   format (queried from the first wrapped frame) and dimensions. */
static gboolean
configure(GstNvmmOfa *self, VPIImage sample)
{
    VPIImageFormat inFmt;
    vpiImageGetFormat(sample, &inFmt);

    const int32_t grid = self->grid_size;
    const int32_t mvW = (self->width  + grid - 1) / grid;
    const int32_t mvH = (self->height + grid - 1) / grid;

    if (vpiStreamCreate(VPI_BACKEND_OFA, &self->stream) != VPI_SUCCESS) {
        GST_ERROR_OBJECT(self, "vpiStreamCreate(OFA) failed");
        return FALSE;
    }
    if (vpiCreateOpticalFlowDense(VPI_BACKEND_OFA, self->width, self->height, inFmt,
                                  &grid, 1, self->quality, &self->payload) != VPI_SUCCESS) {
        GST_ERROR_OBJECT(self, "vpiCreateOpticalFlowDense failed (grid=%d, %dx%d)",
                         grid, self->width, self->height);
        return FALSE;
    }
    /* OFA writes the flow; CPU backend lets us lock it to host for the meta. */
    if (vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16_BL,
                       VPI_BACKEND_OFA | VPI_BACKEND_CPU, &self->mv) != VPI_SUCCESS) {
        GST_ERROR_OBJECT(self, "vpiImageCreate(mv) failed");
        return FALSE;
    }
    self->configured = TRUE;
    GST_INFO_OBJECT(self, "OFA configured: %dx%d grid=%d -> mv %dx%d",
                    self->width, self->height, grid, mvW, mvH);
    return TRUE;
}

/* Lock the OFA output to host and copy it tightly-packed into a host GstMemory. */
static GstMemory *
flow_to_host_memory(GstNvmmOfa *self, gint *out_w, gint *out_h)
{
    VPIImageData d;
    if (vpiImageLockData(self->mv, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &d) != VPI_SUCCESS)
        return nullptr;
    const VPIImagePlanePitchLinear &pl = d.buffer.pitch.planes[0];
    const gint w = pl.width, h = pl.height;
    const gsize row = (gsize)w * 4;  /* 2 x int16 per cell */

    GstMemory *mem = gst_allocator_alloc(nullptr, row * h, nullptr);
    GstMapInfo map;
    if (!mem || !gst_memory_map(mem, &map, GST_MAP_WRITE)) {
        if (mem) gst_memory_unref(mem);
        vpiImageUnlock(self->mv);
        return nullptr;
    }
    const guint8 *src = static_cast<const guint8 *>(pl.data);
    for (gint y = 0; y < h; y++)
        memcpy(map.data + (gsize)y * row, src + (gsize)y * pl.pitchBytes, row);
    gst_memory_unmap(mem, &map);
    vpiImageUnlock(self->mv);

    *out_w = w; *out_h = h;
    return mem;
}

static GstFlowReturn
gst_nvmm_ofa_transform_ip(GstBaseTransform *bt, GstBuffer *buf)
{
    auto *self = GST_NVMM_OFA(bt);

    NvBufSurface *surf = surface_of(buf);
    if (!surf) {
        GST_WARNING_OBJECT(self, "no NvBufSurface in buffer");
        return GST_FLOW_OK;
    }
    surf->numFilled = surf->batchSize ? surf->batchSize : 1;

    VPIImage cur = wrap_ofa(surf);
    if (!cur) {
        GST_WARNING_OBJECT(self, "failed to wrap frame for OFA");
        return GST_FLOW_OK;
    }

    /* First frame: no predecessor — just remember it. */
    if (!self->prev) {
        self->prev = cur;
        self->prev_buf = gst_buffer_ref(buf);
        return GST_FLOW_OK;
    }

    if (!self->configured && !configure(self, cur)) {
        vpiImageDestroy(cur);
        return GST_FLOW_ERROR;
    }

    GstFlowReturn ret = GST_FLOW_OK;
    if (vpiSubmitOpticalFlowDense(self->stream, VPI_BACKEND_OFA, self->payload,
                                  self->prev, cur, self->mv) != VPI_SUCCESS ||
        vpiStreamSync(self->stream) != VPI_SUCCESS) {
        GST_WARNING_OBJECT(self, "OFA submit/sync failed");
    } else {
        gint mvW = 0, mvH = 0;
        GstMemory *flow = flow_to_host_memory(self, &mvW, &mvH);
        if (flow) {
            gst_buffer_add_nvmm_optical_flow_meta(buf, flow, mvW, mvH,
                                                  self->grid_size, self->width, self->height);
            gst_memory_unref(flow);  /* meta took its own ref */
        }
    }

    /* current frame becomes the predecessor for the next pair */
    vpiImageDestroy(self->prev);
    gst_buffer_unref(self->prev_buf);
    self->prev = cur;
    self->prev_buf = gst_buffer_ref(buf);
    return ret;
}

static gboolean
gst_nvmm_ofa_stop(GstBaseTransform *bt)
{
    release_state(GST_NVMM_OFA(bt));
    return TRUE;
}

static void
gst_nvmm_ofa_set_property(GObject *o, guint id, const GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_OFA(o);
    switch (id) {
        case PROP_GRID_SIZE: self->grid_size = g_value_get_enum(v); break;
        case PROP_QUALITY:   self->quality = (VPIOpticalFlowQuality)g_value_get_enum(v); break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p); break;
    }
}

static void
gst_nvmm_ofa_get_property(GObject *o, guint id, GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_OFA(o);
    switch (id) {
        case PROP_GRID_SIZE: g_value_set_enum(v, self->grid_size); break;
        case PROP_QUALITY:   g_value_set_enum(v, self->quality); break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p); break;
    }
}

static void
gst_nvmm_ofa_class_init(GstNvmmOfaClass *klass)
{
    auto *go = G_OBJECT_CLASS(klass);
    auto *el = GST_ELEMENT_CLASS(klass);
    auto *bt = GST_BASE_TRANSFORM_CLASS(klass);

    go->set_property = gst_nvmm_ofa_set_property;
    go->get_property = gst_nvmm_ofa_get_property;

    auto flags = (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS);
    g_object_class_install_property(go, PROP_GRID_SIZE,
        g_param_spec_enum("grid-size", "Grid size",
            "OFA output grid: pixels per motion vector (smaller = denser)",
            GST_TYPE_NVMM_OFA_GRID, 4, flags));
    g_object_class_install_property(go, PROP_QUALITY,
        g_param_spec_enum("quality", "Quality",
            "OFA optical-flow quality/speed trade-off",
            GST_TYPE_NVMM_OFA_QUALITY, VPI_OPTICAL_FLOW_QUALITY_MEDIUM, flags));

    gst_element_class_add_static_pad_template(el, &sink_tmpl);
    gst_element_class_add_static_pad_template(el, &src_tmpl);
    gst_element_class_set_static_metadata(el,
        "NVMM Optical Flow (OFA)", "Filter/Analyzer/Video",
        "Dense optical flow on the Orin OFA engine; passes the frame through and "
        "attaches the motion-vector field as NvmmOpticalFlowMeta",
        "Pavel Guzenfeld, Stereolabs");

    bt->transform_ip = gst_nvmm_ofa_transform_ip;
    bt->set_caps = gst_nvmm_ofa_set_caps;
    bt->stop = gst_nvmm_ofa_stop;
    bt->passthrough_on_same_caps = FALSE;

    GST_DEBUG_CATEGORY_INIT(gst_nvmm_ofa_debug, "nvmmofa", 0, "NVMM OFA optical flow");
}

static void
gst_nvmm_ofa_init(GstNvmmOfa *self)
{
    self->grid_size = 4;
    self->quality = VPI_OPTICAL_FLOW_QUALITY_MEDIUM;
    self->stream = nullptr;
    self->payload = nullptr;
    self->mv = nullptr;
    self->prev = nullptr;
    self->prev_buf = nullptr;
    self->configured = FALSE;
    /* In-place: same caps in/out, the frame's data is never copied — transform_ip
       gets the (writable) buffer and only attaches the flow meta. Not full
       passthrough, so the buffer is guaranteed writable for gst_buffer_add_meta. */
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
}

/* ---- plugin ---- */

static gboolean
plugin_init(GstPlugin *plugin)
{
    if (!gst_element_register(plugin, "nvmmofa", GST_RANK_NONE, GST_TYPE_NVMM_OFA))
        return FALSE;
    return gst_element_register(plugin, "nvmmflowstats", GST_RANK_NONE,
                                gst_nvmm_flowstats_get_type());
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    nvmmofa, "NVMM OFA optical flow + flow-meta consumer",
    plugin_init, PACKAGE_VERSION, "LGPL", "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp")
