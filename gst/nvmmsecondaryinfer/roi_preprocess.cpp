#include "roi_preprocess.hpp"

#include <algorithm>
#include <cmath>

#include <npp.h>
#include <cuda_egl_interop.h>  // runtime EGL interop (cudaGraphicsEGLRegisterImage, cudaEglFrame)

namespace nvmm {

namespace {

// VIC-native dst: surface-array RGBA (same rationale as nvmminfer's
// preprocess: NvBufSurfTransform reliably writes these; CUDA-memory dsts are
// rejected for some source memtypes). Pixels reached from CUDA via EGL.
NvBufSurface *create_rgba(int w, int h, std::string &err) {
    NvBufSurfaceCreateParams p{};
    p.width       = (uint32_t)w;
    p.height      = (uint32_t)h;
    p.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    p.layout      = NVBUF_LAYOUT_PITCH;
    p.memType     = NVBUF_MEM_DEFAULT;
    p.gpuId       = 0;
    NvBufSurface *s = nullptr;
    if (NvBufSurfaceCreate(&s, 1, &p) != 0 || !s) {
        err = "NvBufSurfaceCreate(RGBA) failed";
        return nullptr;
    }
    s->numFilled = 1;
    return s;
}

}  // namespace

bool RoiPreprocessor::configure(int net_w, int net_h, bool color_rgb, float scale,
                                cudaStream_t stream, std::string &err) {
    net_w_ = net_w; net_h_ = net_h;
    color_rgb_ = color_rgb; scale_ = scale; stream_ = stream;

    rgba_ = create_rgba(net_w, net_h, err);
    if (!rgba_) return false;

    // Map the VIC-native surface into CUDA via EGL (zero-copy view).
    if (NvBufSurfaceMapEglImage(rgba_, 0) != 0) {
        err = "NvBufSurfaceMapEglImage failed";
        return false;
    }
    cudaError_t r = cudaGraphicsEGLRegisterImage(
        &egl_res_, rgba_->surfaceList[0].mappedAddr.eglImage,
        cudaGraphicsRegisterFlagsNone);
    if (r != cudaSuccess) {
        err = std::string("cudaGraphicsEGLRegisterImage: ") + cudaGetErrorString(r);
        return false;
    }

    // Linear RGBA (NPP-usable) + 4 split planes (RGBA C4->P4; A ignored).
    if (cudaMalloc((void **)&rgba_lin_, (size_t)4 * net_w * net_h) != cudaSuccess ||
        cudaMalloc((void **)&planes_,   (size_t)4 * net_w * net_h) != cudaSuccess) {
        err = "cudaMalloc(rgba_lin/planes) failed";
        return false;
    }

    // Explicit per-call stream context (the _Ctx API) instead of nppSetStream:
    // the global NPP stream is process-wide, and nvmminfer in the same pipeline
    // sets it to ITS stream — sharing it would break per-element ordering.
    NppStatus st = nppGetStreamContext(&npp_ctx_);
    if (st != NPP_SUCCESS) {
        err = "nppGetStreamContext failed: " + std::to_string((int)st);
        return false;
    }
    npp_ctx_.hStream = stream;
    return true;
}

bool RoiPreprocessor::run(NvBufSurface *src, float left, float top,
                          float width, float height, float *d_input,
                          std::string &err) {
    if (!src) { err = "null src surface"; return false; }
    if (src->numFilled == 0)
        src->numFilled = src->batchSize ? src->batchSize : 1;

    const int sw = (int)src->surfaceList[0].width;
    const int sh = (int)src->surfaceList[0].height;

    // Clamp the box to the surface and even-align (NV12 chroma is 2x2
    // subsampled; VIC wants even crop coordinates/sizes).
    int x0 = std::max(0, (int)std::floor(left)) & ~1;
    int y0 = std::max(0, (int)std::floor(top)) & ~1;
    int x1 = std::min(sw, (int)std::ceil(left + width));
    int y1 = std::min(sh, (int)std::ceil(top + height));
    int w = (x1 - x0) & ~1;
    int h = (y1 - y0) & ~1;
    if (w < 2 || h < 2) { err = "ROI degenerate after clamping"; return false; }

    NvBufSurfTransformRect src_rect{};
    src_rect.left = (uint32_t)x0; src_rect.top = (uint32_t)y0;
    src_rect.width = (uint32_t)w; src_rect.height = (uint32_t)h;

    // NV12 ROI -> RGBA, stretch-resized to the full net-size dst (VIC).
    NvBufSurfTransformParams xform{};
    xform.transform_flag = NVBUFSURF_TRANSFORM_CROP_SRC;
    xform.src_rect = &src_rect;
    const NvBufSurfTransform_Error e = NvBufSurfTransform(src, rgba_, &xform);
    if (e != NvBufSurfTransformError_Success) {
        err = "NvBufSurfTransform failed: " + std::to_string((int)e);
        return false;
    }

    const int W = net_w_, H = net_h_;

    // Pull the freshly-written surface from the EGL/CUDA view into the linear
    // buffer — device-to-device.
    cudaEglFrame ef;
    cudaError_t r = cudaGraphicsResourceGetMappedEglFrame(&ef, egl_res_, 0, 0);
    if (r != cudaSuccess) {
        err = std::string("GetMappedEglFrame: ") + cudaGetErrorString(r);
        return false;
    }
    if (ef.frameType == cudaEglFrameTypePitch) {
        const cudaPitchedPtr &pp = ef.frame.pPitch[0];
        r = cudaMemcpy2DAsync(rgba_lin_, (size_t)W * 4, pp.ptr, pp.pitch,
                              (size_t)W * 4, H, cudaMemcpyDeviceToDevice, stream_);
    } else {  // cudaEglFrameTypeArray
        r = cudaMemcpy2DFromArrayAsync(rgba_lin_, (size_t)W * 4, ef.frame.pArray[0],
                                       0, 0, (size_t)W * 4, H,
                                       cudaMemcpyDeviceToDevice, stream_);
    }
    if (r != cudaSuccess) {
        err = std::string("copy to linear RGBA: ") + cudaGetErrorString(r);
        return false;
    }

    const NppiSize roi = {W, H};
    // RGBA (interleaved) -> 4 packed uint8 planes [R,G,B,A]; we use R,G,B.
    Npp8u *planes4[4] = {planes_, planes_ + (size_t)W * H,
                         planes_ + 2 * (size_t)W * H, planes_ + 3 * (size_t)W * H};
    if (nppiCopy_8u_C4P4R_Ctx(rgba_lin_, W * 4, planes4, W, roi, npp_ctx_) != NPP_SUCCESS) {
        err = "nppiCopy_8u_C4P4R failed";
        return false;
    }

    // uint8 planes -> f32 NCHW (channel order RGB or BGR; alpha plane unused).
    const int map[3] = {color_rgb_ ? 0 : 2, 1, color_rgb_ ? 2 : 0};
    for (int c = 0; c < 3; c++) {
        if (nppiConvert_8u32f_C1R_Ctx(planes4[map[c]], W,
                                      d_input + (size_t)c * W * H,
                                      W * (int)sizeof(float), roi, npp_ctx_) != NPP_SUCCESS) {
            err = "nppiConvert_8u32f_C1R failed";
            return false;
        }
    }

    // Scale all 3 contiguous planes by `scale` in one pass.
    const NppiSize all = {W, 3 * H};
    if (nppiMulC_32f_C1IR_Ctx((Npp32f)scale_, d_input, W * (int)sizeof(float), all,
                              npp_ctx_) != NPP_SUCCESS) {
        err = "nppiMulC_32f_C1IR failed";
        return false;
    }
    return true;
}

RoiPreprocessor::~RoiPreprocessor() {
    if (egl_res_) cudaGraphicsUnregisterResource(egl_res_);
    if (rgba_) {
        NvBufSurfaceUnMapEglImage(rgba_, 0);
        NvBufSurfaceDestroy(rgba_);
    }
    if (rgba_lin_) cudaFree(rgba_lin_);
    if (planes_) cudaFree(planes_);
}

}  // namespace nvmm
