/// RoiPreprocessor — one NV12 NvBufSurface ROI -> normalized NCHW float tensor.
///
/// The per-object twin of nvmminfer's Preprocessor: instead of letterboxing the
/// whole frame, each call VIC-crops a source rectangle (the detection box) and
/// stretch-resizes it to the classifier's input size; NPP then planarizes,
/// converts and scales it into the f32 NCHW tensor TensorRT binds. All
/// device-side, ordered on one stream — same "no host round-trip" contract.
#pragma once

#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <npp.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

namespace nvmm {

class RoiPreprocessor {
public:
    /// net_w/net_h: classifier input size. color_rgb: RGB plane order (else
    /// BGR). scale: pixel multiplier (e.g. 1/255). offsets/std_values: optional
    /// (nullptr = skip) per-channel normalization in the ENGINE's channel
    /// order: y = (x * scale - offset[c]) / std[c]. Allocates the intermediate
    /// RGBA surface and sets the transform to GPU/stream. Frame-size free:
    /// the crop rect is clamped per call against the source surface.
    bool configure(int net_w, int net_h, bool color_rgb, float scale,
                   const float *offsets, const float *std_values,
                   cudaStream_t stream, std::string &err);

    /// Preprocess the `src` rectangle (surface pixel coords; clamped to the
    /// surface and even-aligned for NV12 chroma) into `d_input` (caller-owned,
    /// 3*net_w*net_h floats, NCHW) on the configured stream. Async — caller
    /// syncs after inference.
    bool run(NvBufSurface *src, float left, float top, float width, float height,
             float *d_input, std::string &err);

    bool configured() const { return rgba_ != nullptr; }
    ~RoiPreprocessor();

private:
    int net_w_ = 0, net_h_ = 0;
    bool color_rgb_ = true;
    float scale_ = 1.f / 255.f;
    bool has_offsets_ = false, has_std_ = false;
    float offsets_[3] = {0.f, 0.f, 0.f};  // engine channel order
    float std_[3] = {1.f, 1.f, 1.f};
    cudaStream_t stream_ = nullptr;
    NppStreamContext npp_ctx_{};  // per-call stream binding (see configure)

    NvBufSurface *rgba_ = nullptr;       // net RGBA, VIC-native surface-array dst
    cudaGraphicsResource_t egl_res_ = nullptr;  // CUDA view of rgba_ via EGL
    uint8_t *rgba_lin_ = nullptr;        // linear RGBA device buffer (NPP-usable)
    uint8_t *planes_ = nullptr;          // 4 * net_w * net_h packed uint8
};

}  // namespace nvmm
