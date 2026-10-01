#include "preprocess.hpp"

#include <algorithm>
#include <cmath>

#include <npp.h>
#include <cuda_egl_interop.h>

namespace nvmm {

namespace {

/// NvBufSurfTransform rejects a CUDA-memory dst for some source memtypes here, so
/// write a VIC-native surface-array RGBA and reach it from CUDA via EGL.
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

}

/// Letterbox gray is filled once: each frame copies only the image rect, so the pad survives.
bool Preprocessor::configure(int net_w, int net_h, int frame_w, int frame_h,
                             bool color_rgb, float scale, cudaStream_t stream,
                             std::string &err) {
    net_w_ = net_w; net_h_ = net_h;
    color_rgb_ = color_rgb; scale_ = scale; stream_ = stream;

    rgba_ = create_rgba(net_w, net_h, err);
    if (!rgba_) return false;

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

    if (cudaMalloc((void **)&rgba_lin_, (size_t)4 * net_w * net_h) != cudaSuccess ||
        cudaMalloc((void **)&planes_,   (size_t)4 * net_w * net_h) != cudaSuccess) {
        err = "cudaMalloc(rgba_lin/planes) failed";
        return false;
    }

    const float s = std::min((float)net_w / frame_w, (float)net_h / frame_h);
    const int new_w = (int)std::lround(frame_w * s);
    const int new_h = (int)std::lround(frame_h * s);
    const int pad_x = (net_w - new_w) / 2;
    const int pad_y = (net_h - new_h) / 2;
    lb_ = LetterboxInfo{s, (float)pad_x, (float)pad_y, frame_w, frame_h};
    dst_rect_.top = (uint32_t)pad_y; dst_rect_.left = (uint32_t)pad_x;
    dst_rect_.width = (uint32_t)new_w; dst_rect_.height = (uint32_t)new_h;

    nppSetStream(stream);

    const Npp8u pad[4] = {114, 114, 114, 255};
    const NppiSize full = {net_w, net_h};
    if (nppiSet_8u_C4R(pad, rgba_lin_, net_w * 4, full) != NPP_SUCCESS) {
        err = "nppiSet (pad fill) failed";
        return false;
    }
    return true;
}

bool Preprocessor::run(NvBufSurface *src, float *d_input, LetterboxInfo &lb,
                       std::string &err) {
    if (!src) { err = "null src surface"; return false; }
    if (src->numFilled == 0)
        src->numFilled = src->batchSize ? src->batchSize : 1;

    NvBufSurfTransformParams xform{};
    xform.transform_flag = NVBUFSURF_TRANSFORM_CROP_DST;
    xform.dst_rect = &dst_rect_;
    const NvBufSurfTransform_Error e = NvBufSurfTransform(src, rgba_, &xform);
    if (e != NvBufSurfTransformError_Success) {
        err = "NvBufSurfTransform failed: " + std::to_string((int)e);
        return false;
    }

    const int W = net_w_;
    const int px = (int)dst_rect_.left, py = (int)dst_rect_.top;
    const int rw = (int)dst_rect_.width, rh = (int)dst_rect_.height;

    cudaEglFrame ef;
    cudaError_t r = cudaGraphicsResourceGetMappedEglFrame(&ef, egl_res_, 0, 0);
    if (r != cudaSuccess) {
        err = std::string("GetMappedEglFrame: ") + cudaGetErrorString(r);
        return false;
    }
    uint8_t *dst = rgba_lin_ + ((size_t)py * W + px) * 4;
    if (ef.frameType == cudaEglFrameTypePitch) {
        const cudaPitchedPtr &pp = ef.frame.pPitch[0];
        r = cudaMemcpy2DAsync(dst, W * 4,
                              (const uint8_t *)pp.ptr + (size_t)py * pp.pitch + (size_t)px * 4,
                              pp.pitch, (size_t)rw * 4, rh, cudaMemcpyDeviceToDevice, stream_);
    } else {
        r = cudaMemcpy2DFromArrayAsync(dst, W * 4, ef.frame.pArray[0],
                                       (size_t)px * 4, py, (size_t)rw * 4, rh,
                                       cudaMemcpyDeviceToDevice, stream_);
    }
    if (r != cudaSuccess) {
        err = std::string("rect copy to linear RGBA: ") + cudaGetErrorString(r);
        return false;
    }

    const NppiSize roi = {W, net_h_};
    Npp8u *planes4[4] = {planes_, planes_ + (size_t)W * net_h_,
                         planes_ + 2 * (size_t)W * net_h_, planes_ + 3 * (size_t)W * net_h_};
    if (nppiCopy_8u_C4P4R(rgba_lin_, W * 4, planes4, W, roi) != NPP_SUCCESS) {
        err = "nppiCopy_8u_C4P4R failed";
        return false;
    }

    const int map[3] = {color_rgb_ ? 0 : 2, 1, color_rgb_ ? 2 : 0};
    for (int c = 0; c < 3; c++) {
        if (nppiConvert_8u32f_C1R(planes4[map[c]], W,
                                  d_input + (size_t)c * W * net_h_,
                                  W * (int)sizeof(float), roi) != NPP_SUCCESS) {
            err = "nppiConvert_8u32f_C1R failed";
            return false;
        }
    }

    const NppiSize all = {W, 3 * net_h_};
    if (nppiMulC_32f_C1IR((Npp32f)scale_, d_input, W * (int)sizeof(float), all) != NPP_SUCCESS) {
        err = "nppiMulC_32f_C1IR failed";
        return false;
    }

    lb = lb_;
    return true;
}

Preprocessor::~Preprocessor() {
    if (egl_res_) cudaGraphicsUnregisterResource(egl_res_);
    if (rgba_) {
        NvBufSurfaceUnMapEglImage(rgba_, 0);
        NvBufSurfaceDestroy(rgba_);
    }
    if (rgba_lin_) cudaFree(rgba_lin_);
    if (planes_) cudaFree(planes_);
}

}
