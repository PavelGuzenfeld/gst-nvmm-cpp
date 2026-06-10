#include "preprocess.hpp"

#include <algorithm>
#include <cmath>

#include <npp.h>

namespace nvmm {

namespace {

NvBufSurface *create_rgba(int w, int h, std::string &err) {
    NvBufSurfaceCreateParams p{};
    p.width       = (uint32_t)w;
    p.height      = (uint32_t)h;
    p.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    p.layout      = NVBUF_LAYOUT_PITCH;
    p.memType     = NVBUF_MEM_CUDA_DEVICE;   // CUDA-addressable (Phase-0 verified)
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

bool Preprocessor::configure(int net_w, int net_h, int frame_w, int frame_h,
                             bool color_rgb, float scale, cudaStream_t stream,
                             std::string &err) {
    net_w_ = net_w; net_h_ = net_h;
    color_rgb_ = color_rgb; scale_ = scale; stream_ = stream;

    rgba_ = create_rgba(net_w, net_h, err);
    if (!rgba_) return false;

    if (cudaMalloc((void **)&planes_, (size_t)3 * net_w * net_h) != cudaSuccess) {
        err = "cudaMalloc(planes) failed";
        return false;
    }

    // Letterbox geometry (frame size is fixed for the stream).
    const float s = std::min((float)net_w / frame_w, (float)net_h / frame_h);
    const int new_w = (int)std::lround(frame_w * s);
    const int new_h = (int)std::lround(frame_h * s);
    const int pad_x = (net_w - new_w) / 2;
    const int pad_y = (net_h - new_h) / 2;
    lb_ = LetterboxInfo{s, (float)pad_x, (float)pad_y, frame_w, frame_h};
    dst_rect_.top = (uint32_t)pad_y; dst_rect_.left = (uint32_t)pad_x;
    dst_rect_.width = (uint32_t)new_w; dst_rect_.height = (uint32_t)new_h;

    // Transform on GPU, ordered on our inference stream (no host bounce, no VIC
    // memtype constraint on the CUDA-memory dst).
    NvBufSurfTransformConfigParams cfg{};
    cfg.compute_mode = NvBufSurfTransformCompute_GPU;
    cfg.gpu_id = 0;
    cfg.cuda_stream = stream;
    if (NvBufSurfTransformSetSessionParams(&cfg) != NvBufSurfTransformError_Success) {
        err = "NvBufSurfTransformSetSessionParams failed";
        return false;
    }

    nppSetStream(stream);

    // Pre-fill the whole RGBA with letterbox gray (114). Per-frame transforms
    // only write dst_rect, so the pad survives — fill once.
    const Npp8u pad[4] = {114, 114, 114, 255};
    const NppiSize full = {net_w, net_h};
    if (nppiSet_8u_C4R(pad, (Npp8u *)rgba_->surfaceList[0].dataPtr,
                       (int)rgba_->surfaceList[0].pitch, full) != NPP_SUCCESS) {
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

    // NV12 -> RGBA, resized + letterboxed into dst_rect (pad preserved).
    NvBufSurfTransformParams xform{};
    xform.transform_flag = NVBUFSURF_TRANSFORM_CROP_DST;
    xform.dst_rect = &dst_rect_;
    const NvBufSurfTransform_Error e = NvBufSurfTransform(src, rgba_, &xform);
    if (e != NvBufSurfTransformError_Success) {
        err = "NvBufSurfTransform failed: " + std::to_string((int)e);
        return false;
    }

    const int W = net_w_, H = net_h_;
    const NppiSize roi = {W, H};
    Npp8u *rgba = (Npp8u *)rgba_->surfaceList[0].dataPtr;
    const int pitch = (int)rgba_->surfaceList[0].pitch;

    // RGBA (interleaved) -> 3 packed uint8 planes [R,G,B] (alpha dropped).
    Npp8u *planes3[3] = {planes_, planes_ + (size_t)W * H, planes_ + 2 * (size_t)W * H};
    if (nppiCopy_8u_C4P3R(rgba, pitch, planes3, W, roi) != NPP_SUCCESS) {
        err = "nppiCopy_8u_C4P3R failed";
        return false;
    }

    // uint8 planes -> f32 NCHW (channel order RGB or BGR).
    const int map[3] = {color_rgb_ ? 0 : 2, 1, color_rgb_ ? 2 : 0};
    for (int c = 0; c < 3; c++) {
        if (nppiConvert_8u32f_C1R(planes3[map[c]], W,
                                  d_input + (size_t)c * W * H,
                                  W * (int)sizeof(float), roi) != NPP_SUCCESS) {
            err = "nppiConvert_8u32f_C1R failed";
            return false;
        }
    }

    // Scale all 3 contiguous planes by `scale` in one pass.
    const NppiSize all = {W, 3 * H};
    if (nppiMulC_32f_C1IR((Npp32f)scale_, d_input, W * (int)sizeof(float), all) != NPP_SUCCESS) {
        err = "nppiMulC_32f_C1IR failed";
        return false;
    }

    lb = lb_;
    return true;
}

Preprocessor::~Preprocessor() {
    if (planes_) cudaFree(planes_);
    if (rgba_) NvBufSurfaceDestroy(rgba_);
}

}  // namespace nvmm
