/// analytics_kernels.hpp — host-callable wrappers for the CUDA implementation of
/// low_texture_motion (analytics_kernels.cu). The one analytics component where
/// the benchmark evidence justified a GPU path: every stage is embarrassingly
/// parallel and the fused CPU version still trails OpenCV's SIMD kernels ~1.7x
/// at 1080p. Parity-checked against the (golden-validated) host implementation
/// by tests/analytics_kernel_probe.cpp.
///
/// Two entry points:
///  - run_device(): ZERO-COPY path for the plugin zoo — takes pitched DEVICE
///    pointers (e.g. the CUDA mapping of an NvBufSurface luma plane, the same
///    pointer the nvmm elements already pass to their kernels) and writes a
///    pitched device float map, all on the caller's stream. No host round-trip.
///  - run(): host-buffer convenience wrapper (uploads, runs, downloads) for
///    tests, benchmarks and host-resident callers.
///
/// Gated behind -Danalytics_cuda=enabled — needs only the CUDA toolkit, not
/// TensorRT/NvBufSurface (unlike the SAMURAI kernels), so it builds on any
/// CUDA-capable host, not just Jetson.
#pragma once

#include <cuda_runtime.h>

#include "image.hpp"
#include "low_texture_motion.hpp"

namespace nvmm {
namespace motion {

/// Pitched single-channel device plane — the device-pointer counterpart of
/// img::View<T>, with the same field names (`stride` in ELEMENTS of T; a
/// mapped NvBufSurface plane's pitch is in bytes — divide by sizeof(T)). Kept
/// as its own type rather than reusing View<T> directly so host code can't
/// accidentally dereference a device pointer through View::at()/row().
template <typename T>
struct DevicePlane {
    T *data = nullptr;
    int width = 0, height = 0;
    std::ptrdiff_t stride = 0;
};

/// Device-resident fused low_texture_motion. Owns its device SCRATCH buffers
/// (never the input/output planes) so per-frame use does not reallocate;
/// scratch resizes on dimension change. Not thread-safe; one instance per
/// stream. Returns false on any CUDA error — callers fall back to the host
/// path; last_error() has the cudaGetErrorString for logging.
class LowTextureMotionCuda {
public:
    LowTextureMotionCuda();
    ~LowTextureMotionCuda();
    LowTextureMotionCuda(const LowTextureMotionCuda &) = delete;
    LowTextureMotionCuda &operator=(const LowTextureMotionCuda &) = delete;

    /// Zero-copy: inputs and output are device planes; all work is enqueued on
    /// `stream` and NOT synchronised — the caller owns the sync point (or
    /// chains further kernels on the same stream).
    bool run_device(DevicePlane<const uint8_t> cur, DevicePlane<const uint8_t> ref_a,
                    DevicePlane<const uint8_t> ref_b, const LowTextureMotionParams &p,
                    DevicePlane<float> out, cudaStream_t stream);

    /// Host convenience: upload the three views, run_device on `stream` (default
    /// stream when null), download into `out`. Synchronises before returning.
    bool run(img::View<const uint8_t> cur, img::View<const uint8_t> ref_a,
             img::View<const uint8_t> ref_b, const LowTextureMotionParams &p,
             img::Image<float> &out, cudaStream_t stream = nullptr);

    const char *last_error() const;

private:
    struct Impl;
    Impl *impl_;
};

}  // namespace motion
}  // namespace nvmm
