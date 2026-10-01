/// Kept as the reproducible evidence for the parked nvmmcv NO-GO (docs/HW_ACCEL_EXPLORATION.md): does
/// PVA erode a zero-copy-wrapped NvBufSurface? PVA morphology takes U8/S8/U16/S16 (not NV12), > 128x128.
/// Build: g++ -std=c++17 -O2 vpi_pva_probe.cpp -o vpi_pva_probe -I/usr/src/jetson_multimedia_api/include -I/opt/nvidia/vpi3/include -L/usr/lib/aarch64-linux-gnu/tegra -L/opt/nvidia/vpi3/lib/aarch64-linux-gnu -lnvbufsurface -lnvvpi; run: ./vpi_pva_probe [block|pitch]
#include <cstdio>
#include <cstring>
#include <cstdint>

#include <nvbufsurface.h>

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/Status.h>
#include <vpi/algo/MorphologicalFilter.h>

static bool vpi_ok(const char* what, VPIStatus s) {
    if (s == VPI_SUCCESS) { printf("  [ OK ] %s\n", what); return true; }
    char msg[VPI_MAX_STATUS_MESSAGE_LENGTH];
    vpiGetLastStatusMessage(msg, sizeof(msg));
    printf("  [FAIL] %s -> %s: %s\n", what, vpiStatusGetName(s), msg);
    return false;
}

static VPIImage wrap_nvmm(NvBufSurface* surf, uint64_t backends, const char* label) {
    int fd = (int)surf->surfaceList[0].bufferDesc;
    VPIImageData d;
    memset(&d, 0, sizeof(d));
    d.bufferType = VPI_IMAGE_BUFFER_NVBUFFER;
    d.buffer.fd = fd;
    VPIImage img = nullptr;
    char what[64]; snprintf(what, sizeof(what), "vpiImageCreateWrapper(%s, fd=%d)", label, fd);
    vpi_ok(what, vpiImageCreateWrapper(&d, nullptr, backends, &img));
    return img;
}

static NvBufSurface* make_gray8(uint32_t w, uint32_t h, NvBufSurfaceLayout layout) {
    NvBufSurfaceCreateParams p;
    memset(&p, 0, sizeof(p));
    p.width = w; p.height = h;
    p.colorFormat = NVBUF_COLOR_FORMAT_GRAY8;
    p.layout = layout;
    p.memType = NVBUF_MEM_SURFACE_ARRAY;
    NvBufSurface* surf = nullptr;
    if (NvBufSurfaceCreate(&surf, 1, &p) != 0) { printf("  [FAIL] NvBufSurfaceCreate GRAY8\n"); return nullptr; }
    surf->numFilled = surf->batchSize ? surf->batchSize : 1;
    NvBufSurfaceSyncForDevice(surf, 0, 0);
    return surf;
}

int main(int argc, char** argv) {
    NvBufSurfaceLayout layout = NVBUF_LAYOUT_BLOCK_LINEAR;
    const char* layout_name = "BLOCK_LINEAR";
    if (argc > 1 && strcmp(argv[1], "pitch") == 0) { layout = NVBUF_LAYOUT_PITCH; layout_name = "PITCH_LINEAR"; }

    const uint32_t W = 256, H = 256;
    printf("== B4 PVA probe: GRAY8 %ux%u, layout=%s ==\n", W, H, layout_name);

    NvBufSurface* in_surf  = make_gray8(W, H, layout);
    NvBufSurface* out_surf = make_gray8(W, H, layout);
    if (!in_surf || !out_surf) return 2;

    const uint64_t backends = VPI_BACKEND_PVA | VPI_BACKEND_CUDA;
    VPIImage in  = wrap_nvmm(in_surf,  backends, "input");
    VPIImage out = wrap_nvmm(out_surf, backends, "output");
    if (!in || !out) return 3;

    VPIStream stream = nullptr;
    if (!vpi_ok("vpiStreamCreate(PVA|CUDA)", vpiStreamCreate(backends, &stream))) return 4;

    bool cuda_ok = vpi_ok("vpiSubmitErode(CUDA) on wrapped NVMM",
        vpiSubmitErode(stream, VPI_BACKEND_CUDA, in, out, nullptr, 3, 3, VPI_BORDER_ZERO))
        && vpi_ok("vpiStreamSync after CUDA", vpiStreamSync(stream));

    bool pva_ok = vpi_ok("vpiSubmitErode(PVA) on wrapped NVMM",
        vpiSubmitErode(stream, VPI_BACKEND_PVA, in, out, nullptr, 3, 3, VPI_BORDER_ZERO))
        && vpi_ok("vpiStreamSync after PVA", vpiStreamSync(stream));

    printf("VERDICT layout=%s: zero-copy-wrap=%s CUDA=%s PVA=%s\n",
           layout_name, (in && out) ? "OK" : "FAIL",
           cuda_ok ? "OK" : "FAIL", pva_ok ? "OK" : "FAIL");

    vpiStreamDestroy(stream);
    vpiImageDestroy(in); vpiImageDestroy(out);
    NvBufSurfaceDestroy(in_surf); NvBufSurfaceDestroy(out_surf);
    return pva_ok ? 0 : 1;
}
