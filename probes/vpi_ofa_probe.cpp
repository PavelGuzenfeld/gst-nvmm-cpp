// Phase-0 go/no-go probe for B3 (nvmmofa — Optical Flow Accelerator via VPI).
//
// OFA is Orin-only hardware. Unlike B4's PVA morphology (which needed
// pitch-linear), OFA dense optical flow REQUIRES block-linear input
// (VPI_IMAGE_FORMAT_NV12_BL / Y8_BL / ...), which is exactly what nvvidconv
// emits for NVMM — so the natural pipeline feeds OFA zero-copy. This probe
// proves: wrap two block-linear NV12 NvBufSurfaces zero-copy, run
// vpiSubmitOpticalFlowDense on VPI_BACKEND_OFA, get motion vectors out.
//
// Build (Orin / JP6, VPI 3):
//   g++ -std=c++17 -O2 vpi_ofa_probe.cpp -o vpi_ofa_probe \
//     -I/usr/src/jetson_multimedia_api/include -I/opt/nvidia/vpi3/include \
//     -L/usr/lib/aarch64-linux-gnu/tegra -L/opt/nvidia/vpi3/lib/aarch64-linux-gnu \
//     -lnvbufsurface -lnvvpi

#include <cstdio>
#include <cstring>
#include <cstdint>

#include <nvbufsurface.h>

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/Status.h>
#include <vpi/algo/OpticalFlowDense.h>

static bool vpi_ok(const char* what, VPIStatus s) {
    if (s == VPI_SUCCESS) { printf("  [ OK ] %s\n", what); return true; }
    char msg[VPI_MAX_STATUS_MESSAGE_LENGTH];
    vpiGetLastStatusMessage(msg, sizeof(msg));
    printf("  [FAIL] %s -> %s: %s\n", what, vpiStatusGetName(s), msg);
    return false;
}

static NvBufSurface* make_nv12_bl(uint32_t w, uint32_t h) {
    NvBufSurfaceCreateParams p;
    memset(&p, 0, sizeof(p));
    p.width = w; p.height = h;
    p.colorFormat = NVBUF_COLOR_FORMAT_NV12;
    p.layout = NVBUF_LAYOUT_BLOCK_LINEAR;   // OFA requires block-linear
    p.memType = NVBUF_MEM_SURFACE_ARRAY;    // Jetson NVMM
    NvBufSurface* surf = nullptr;
    if (NvBufSurfaceCreate(&surf, 1, &p) != 0) { printf("  [FAIL] NvBufSurfaceCreate NV12/BL\n"); return nullptr; }
    surf->numFilled = surf->batchSize ? surf->batchSize : 1;
    NvBufSurfaceSyncForDevice(surf, 0, 0);
    return surf;
}

static VPIImage wrap(NvBufSurface* surf, uint64_t backends, const char* label) {
    VPIImageData d;
    memset(&d, 0, sizeof(d));
    d.bufferType = VPI_IMAGE_BUFFER_NVBUFFER;
    d.buffer.fd  = (int)surf->surfaceList[0].bufferDesc;
    VPIImage img = nullptr;
    char what[64]; snprintf(what, sizeof(what), "wrap %s (fd=%d)", label, d.buffer.fd);
    vpi_ok(what, vpiImageCreateWrapper(&d, nullptr, backends, &img));
    return img;
}

int main() {
    const uint32_t W = 640, H = 480;     // >= 32x32; /gridSize stays >= 4x4
    const int32_t  gridSize = 4;         // OFA allows 1,2,4,8
    const int32_t  outW = (W + gridSize - 1) / gridSize;
    const int32_t  outH = (H + gridSize - 1) / gridSize;
    const uint64_t be = VPI_BACKEND_OFA;
    printf("== B3 OFA probe: NV12 block-linear %ux%u, grid=%d -> mv %dx%d (2S16_BL) ==\n",
           W, H, gridSize, outW, outH);

    NvBufSurface* prev_s = make_nv12_bl(W, H);
    NvBufSurface* cur_s  = make_nv12_bl(W, H);
    if (!prev_s || !cur_s) return 2;

    VPIImage prev = wrap(prev_s, be, "prev");
    VPIImage cur  = wrap(cur_s,  be, "cur");
    if (!prev || !cur) return 3;

    /* The payload's inputFmt must match what VPI actually assigned the wrapped
       NvBufSurface (NvBuffer NV12 maps to an ER/BL variant, not plain NV12_BL),
       so query it rather than hardcoding. */
    VPIImageFormat inFmt;
    vpiImageGetFormat(prev, &inFmt);
    printf("  wrapped input format = 0x%llx\n", (unsigned long long)inFmt);

    /* OFA only accepts a VPI-native 2S16_BL output (a wrapped SIGNED_R16G16
       NvBufSurface is rejected — verified), so allocate the MV in VPI and read
       it back to host with a lock (de-tiled to pitch-linear) — the path the
       element uses to copy the small flow field into per-frame metadata. */
    VPIImage mv = nullptr;
    if (!vpi_ok("vpiImageCreate(mv, 2S16_BL, OFA|CPU)",
                vpiImageCreate(outW, outH, VPI_IMAGE_FORMAT_2S16_BL,
                               VPI_BACKEND_OFA | VPI_BACKEND_CPU, &mv))) return 4;

    VPIPayload payload = nullptr;
    if (!vpi_ok("vpiCreateOpticalFlowDense(OFA, <wrapped fmt>)",
                vpiCreateOpticalFlowDense(be, W, H, inFmt,
                                          &gridSize, 1, VPI_OPTICAL_FLOW_QUALITY_MEDIUM,
                                          &payload))) return 5;

    VPIStream stream = nullptr;
    if (!vpi_ok("vpiStreamCreate(OFA)", vpiStreamCreate(be, &stream))) return 6;

    bool ofa_ok = vpi_ok("vpiSubmitOpticalFlowDense(OFA) on zero-copy NVMM",
        vpiSubmitOpticalFlowDense(stream, be, payload, prev, cur, mv))
        && vpi_ok("vpiStreamSync after OFA", vpiStreamSync(stream));

    /* Read the flow field back to host (what the element copies into meta). */
    bool read_ok = false;
    if (ofa_ok) {
        VPIImageData out;
        if (vpi_ok("vpiImageLockData(mv, HOST_PITCH_LINEAR)",
                   vpiImageLockData(mv, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &out))) {
            const VPIImagePlanePitchLinear& pl = out.buffer.pitch.planes[0];
            printf("  mv host plane: %dx%d pitch=%d bytes (2S16 = 4 B/cell)\n",
                   pl.width, pl.height, pl.pitchBytes);
            read_ok = (pl.data != nullptr && pl.width == outW && pl.height == outH);
            vpiImageUnlock(mv);
        }
    }

    printf("VERDICT: zero-copy-wrap=%s OFA=%s host-readback=%s\n",
           (prev && cur) ? "OK" : "FAIL", ofa_ok ? "OK" : "FAIL", read_ok ? "OK" : "FAIL");
    ofa_ok = ofa_ok && read_ok;

    vpiStreamDestroy(stream);
    vpiPayloadDestroy(payload);
    vpiImageDestroy(prev); vpiImageDestroy(cur); vpiImageDestroy(mv);
    NvBufSurfaceDestroy(prev_s); NvBufSurfaceDestroy(cur_s);
    return ofa_ok ? 0 : 1;
}
