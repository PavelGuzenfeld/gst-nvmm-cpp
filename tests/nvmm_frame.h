#ifndef NVMM_TEST_FRAME_H
#define NVMM_TEST_FRAME_H

#include <gst/gst.h>
#include <gst/video/video.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

#include "gstnvmmallocator.h"

struct Yuv { uint8_t y, u, v; };

constexpr Yuv kBlack{16, 128, 128};
constexpr Yuv kWhite{235, 128, 128};
constexpr Yuv kRed{81, 90, 240};
constexpr Yuv kBlue{41, 240, 110};

using Rgba = std::array<uint8_t, 4>;

using Nv12Painter = std::function<Yuv(int x, int y)>;

inline NvBufSurface *create_surface(int w, int h, NvBufSurfaceColorFormat fmt,
                                    NvBufSurfaceLayout layout)
{
    NvBufSurfaceCreateParams p{};
    p.width = (uint32_t)w;
    p.height = (uint32_t)h;
    p.colorFormat = fmt;
    p.layout = layout;
    p.memType = NVBUF_MEM_DEFAULT;
    NvBufSurface *s = nullptr;
    if (NvBufSurfaceCreate(&s, 1, &p) != 0 || !s)
        throw std::runtime_error("NvBufSurfaceCreate failed");
    s->numFilled = 1;
    return s;
}

/// Chroma of each 2x2 block comes from its top-left pixel.
inline void paint_nv12(NvBufSurface *s, const Nv12Painter &paint)
{
    if (NvBufSurfaceMap(s, 0, -1, NVBUF_MAP_READ_WRITE) != 0)
        throw std::runtime_error("NvBufSurfaceMap failed");
    NvBufSurfaceSyncForCpu(s, 0, -1);
    const NvBufSurfaceParams &p = s->surfaceList[0];
    auto *luma = static_cast<uint8_t *>(p.mappedAddr.addr[0]);
    auto *chroma = static_cast<uint8_t *>(p.mappedAddr.addr[1]);
    for (int y = 0; y < (int)p.height; y++)
        for (int x = 0; x < (int)p.width; x++) {
            const Yuv c = paint(x, y);
            luma[(size_t)y * p.planeParams.pitch[0] + x] = c.y;
            if (x % 2 == 0 && y % 2 == 0) {
                uint8_t *uv = chroma + (size_t)(y / 2) * p.planeParams.pitch[1] + x;
                uv[0] = c.u;
                uv[1] = c.v;
            }
        }
    NvBufSurfaceSyncForDevice(s, 0, -1);
    NvBufSurfaceUnMap(s, 0, -1);
}

inline NvBufSurface *pitch_nv12(int w, int h, const Nv12Painter &paint)
{
    NvBufSurface *s = create_surface(w, h, NVBUF_COLOR_FORMAT_NV12, NVBUF_LAYOUT_PITCH);
    paint_nv12(s, paint);
    return s;
}

inline GstBuffer *nvmm_nv12_buffer(int w, int h, const Nv12Painter &paint)
{
    static GstAllocator *alloc = gst_nvmm_allocator_new(0);
    GstMemory *mem = gst_nvmm_allocator_alloc_video(alloc, GST_VIDEO_FORMAT_NV12,
                                                    (guint)w, (guint)h);
    if (!mem) throw std::runtime_error("gst_nvmm_allocator_alloc_video failed");
    paint_nv12(static_cast<NvBufSurface *>(gst_nvmm_memory_get_surface(mem)), paint);
    GstBuffer *buf = gst_buffer_new();
    gst_buffer_append_memory(buf, mem);
    return buf;
}

/// DeepStream-style: the buffer bytes are the NvBufSurface struct, as nvvidconv emits.
inline GstBuffer *wrapped_surface_buffer(NvBufSurface *s)
{
    return gst_buffer_new_wrapped_full(
        (GstMemoryFlags)0, s, sizeof(NvBufSurface), 0, sizeof(NvBufSurface), s,
        [](gpointer p) { NvBufSurfaceDestroy(static_cast<NvBufSurface *>(p)); });
}

inline GstBuffer *block_linear_nv12_buffer(int w, int h, const Nv12Painter &paint)
{
    NvBufSurface *pitch = pitch_nv12(w, h, paint);
    NvBufSurface *bl = create_surface(w, h, NVBUF_COLOR_FORMAT_NV12,
                                      NVBUF_LAYOUT_BLOCK_LINEAR);
    NvBufSurfTransformParams xp{};
    const NvBufSurfTransform_Error e = NvBufSurfTransform(pitch, bl, &xp);
    NvBufSurfaceDestroy(pitch);
    if (e != NvBufSurfTransformError_Success)
        throw std::runtime_error("NvBufSurfTransform to block-linear failed");
    return wrapped_surface_buffer(bl);
}

/// The VIC's own NV12 to RGBA conversion of one pixel: the reference every
/// colour-converting element must reproduce.
inline Rgba vic_rgba_at(const Nv12Painter &paint, int w, int h, int x, int y)
{
    NvBufSurface *src = pitch_nv12(w, h, paint);
    NvBufSurface *dst = create_surface(w, h, NVBUF_COLOR_FORMAT_RGBA, NVBUF_LAYOUT_PITCH);
    NvBufSurfTransformParams xp{};
    if (NvBufSurfTransform(src, dst, &xp) != NvBufSurfTransformError_Success)
        throw std::runtime_error("NvBufSurfTransform to RGBA failed");
    NvBufSurfaceMap(dst, 0, 0, NVBUF_MAP_READ);
    NvBufSurfaceSyncForCpu(dst, 0, 0);
    const NvBufSurfaceParams &p = dst->surfaceList[0];
    const uint8_t *px = static_cast<const uint8_t *>(p.mappedAddr.addr[0]) +
                        (size_t)y * p.planeParams.pitch[0] + (size_t)x * 4;
    const Rgba out{px[0], px[1], px[2], px[3]};
    NvBufSurfaceUnMap(dst, 0, 0);
    NvBufSurfaceDestroy(dst);
    NvBufSurfaceDestroy(src);
    return out;
}

inline Nv12Painter solid(Yuv c)
{
    return [c](int, int) { return c; };
}

/// Empty when the element reached PAUSED; otherwise the bus error text.
inline std::string start_failure(GstElement *e)
{
    GstBus *bus = gst_bus_new();
    gst_element_set_bus(e, bus);
    const GstStateChangeReturn r = gst_element_set_state(e, GST_STATE_PAUSED);
    std::string text;
    GstMessage *m = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR);
    if (m) {
        GError *err = nullptr;
        gst_message_parse_error(m, &err, nullptr);
        text = err->message;
        g_error_free(err);
        gst_message_unref(m);
    } else if (r == GST_STATE_CHANGE_FAILURE) {
        text = "<failed without an error message>";
    }
    gst_element_set_state(e, GST_STATE_NULL);
    gst_element_set_bus(e, nullptr);
    gst_object_unref(bus);
    return text;
}

#endif
