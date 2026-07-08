/// gmc_vpi_fft.hpp — GMC fft-cuda backend: FFT phase correlation on the GPU via
/// VPI (VPI_BACKEND_CUDA). Same algorithm as PhaseCorrelator (gst/common), but the
/// two forward FFTs, the cross-power normalize, and the inverse FFT run on-device;
/// only the tiny n x n real correlation surface is read back for the shared
/// sub-pixel refine (refine_correlation_peak). Windowing + cross-power are custom
/// CUDA kernels (VPI has neither); the FFTs are VPI/cuFFT.
///
/// Compiled only when NVMM_HAVE_VPI is defined (VPI present at build); otherwise
/// available() is false and the fusekf/tracker falls back to fft-cpu. VPI FFT also
/// needs cuFFT at run time, so available() actually creates a payload to confirm.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "phase_correlation.hpp"  // PhaseShift + refine_correlation_peak (shared)

#ifdef NVMM_HAVE_VPI
#include <cstring>
#include <cuda_runtime.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/FFT.h>

#include "samurai_kernels.hpp"  // k_gmc_window / k_gmc_cross_power

namespace nvmm {

class GmcVpiFft {
public:
    GmcVpiFft() = default;
    GmcVpiFft(const GmcVpiFft &) = delete;             // owns VPI handles + device mem
    GmcVpiFft &operator=(const GmcVpiFft &) = delete;

    // Cheap availability probe: create+destroy a CUDA FFT payload. Confirms VPI +
    // cuFFT are usable on this box (the FFT payload dlopens cuFFT). Static so the
    // tracker can decide the backend before allocating anything.
    static bool available() {
        VPIStream s = nullptr;
        if (vpiStreamCreate(VPI_BACKEND_CUDA, &s) != VPI_SUCCESS) return false;
        VPIPayload p = nullptr;
        const VPIStatus st = vpiCreateFFT(VPI_BACKEND_CUDA, 128, 128,
                                          VPI_IMAGE_FORMAT_2F32, VPI_IMAGE_FORMAT_2F32, &p);
        if (p) vpiPayloadDestroy(p);
        vpiStreamDestroy(s);
        return st == VPI_SUCCESS;
    }

    ~GmcVpiFft() { destroy(); }

    bool init(int n, std::string &err) {
        n_ = n;
        const size_t n2 = (size_t)n * n;
        // Dedicated CUDA stream for the window/cross-power kernels + copies, so GMC
        // never issues a device-wide sync that would stall SAM2.1's concurrent stream.
        if (cudaStreamCreate(&cs_) != cudaSuccess) { err = "cudaStreamCreate(gmc fft)"; return false; }
        if (vpiStreamCreate(VPI_BACKEND_CUDA, &stream_) != VPI_SUCCESS) { err = "vpiStreamCreate(CUDA)"; return false; }
        if (vpiCreateFFT(VPI_BACKEND_CUDA, n, n, VPI_IMAGE_FORMAT_2F32, VPI_IMAGE_FORMAT_2F32, &fft_) != VPI_SUCCESS) { err = "vpiCreateFFT"; return false; }
        if (vpiCreateIFFT(VPI_BACKEND_CUDA, n, n, VPI_IMAGE_FORMAT_2F32, VPI_IMAGE_FORMAT_2F32, &ifft_) != VPI_SUCCESS) { err = "vpiCreateIFFT"; return false; }
        if (cudaMalloc(&d_yp_, n2) != cudaSuccess || cudaMalloc(&d_yc_, n2) != cudaSuccess ||
            cudaMalloc(&d_hann_, n * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_prevwin_, n2 * sizeof(float2)) != cudaSuccess ||
            cudaMalloc(&d_currwin_, n2 * sizeof(float2)) != cudaSuccess ||
            cudaMalloc(&d_aspec_, n2 * sizeof(float2)) != cudaSuccess ||
            cudaMalloc(&d_bspec_, n2 * sizeof(float2)) != cudaSuccess ||
            cudaMalloc(&d_corr_, n2 * sizeof(float2)) != cudaSuccess) { err = "cudaMalloc(gmc fft)"; return false; }
        // separable Hann, identical to PhaseCorrelator's window.
        std::vector<float> hann((size_t)n);
        for (int i = 0; i < n; i++)
            hann[(size_t)i] = 0.5f * (1.f - std::cos(2.f * 3.14159265358979323846f * i / (n - 1)));
        if (cudaMemcpy(d_hann_, hann.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) { err = "cudaMemcpy(hann)"; return false; }
        if (!wrap(d_prevwin_, img_prevwin_) || !wrap(d_currwin_, img_currwin_) ||
            !wrap(d_aspec_, img_aspec_) || !wrap(d_bspec_, img_bspec_) ||
            !wrap(d_corr_, img_corr_)) { err = "vpiImageCreateWrapper(gmc fft)"; return false; }
        h_corr_.resize(2 * n2);
        real_.resize(n2);
        return true;
    }

    // Same sign convention as PhaseCorrelator::correlate(prev, curr): the content's
    // motion prev -> curr. `prev`/`curr` are host uint8, n*n (row stride n).
    PhaseShift estimate(const uint8_t *prev, const uint8_t *curr) {
        const int n = n_;
        const size_t n2 = (size_t)n * n;
        // All kernels + copies on cs_; sync is stream-scoped (cudaStreamSynchronize),
        // never device-wide, so a concurrent TRT stream (SAM2.1) is not stalled. Any
        // CUDA/VPI failure returns {} — gated out by the caller (no garbage shift).
        if (cudaMemcpyAsync(d_yp_, prev, n2, cudaMemcpyHostToDevice, cs_) != cudaSuccess ||
            cudaMemcpyAsync(d_yc_, curr, n2, cudaMemcpyHostToDevice, cs_) != cudaSuccess) return {};
        k_gmc_window(d_yp_, n, n, d_hann_, d_prevwin_, cs_);
        k_gmc_window(d_yc_, n, n, d_hann_, d_currwin_, cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) return {};   // windows done before VPI reads them
        if (vpiSubmitFFT(stream_, VPI_BACKEND_CUDA, fft_, img_prevwin_, img_aspec_, 0) != VPI_SUCCESS ||
            vpiSubmitFFT(stream_, VPI_BACKEND_CUDA, fft_, img_currwin_, img_bspec_, 0) != VPI_SUCCESS ||
            vpiStreamSync(stream_) != VPI_SUCCESS) return {};
        k_gmc_cross_power(d_aspec_, d_bspec_, (int)n2, cs_);  // A = FFT(prev)*conj(FFT(curr))/|.|
        if (cudaStreamSynchronize(cs_) != cudaSuccess) return {};
        if (vpiSubmitIFFT(stream_, VPI_BACKEND_CUDA, ifft_, img_aspec_, img_corr_, 0) != VPI_SUCCESS ||
            vpiStreamSync(stream_) != VPI_SUCCESS) return {};
        if (cudaMemcpyAsync(h_corr_.data(), d_corr_, n2 * sizeof(float2), cudaMemcpyDeviceToHost, cs_) != cudaSuccess ||
            cudaStreamSynchronize(cs_) != cudaSuccess) return {};
        // VPI's inverse FFT is 1/N-scaled; PhaseCorrelator uses an UNscaled inverse.
        // Rescale by N so `response` lands on the same scale as fft-cpu (the shift is
        // scale-invariant) and a single confidence gate works for both FFT backends.
        // (This ties the conf scale to VPI's 1/N convention — see the confidence-model
        // follow-up in the plan; the shift itself is unaffected by any scaling.)
        const double nrm = (double)n2;
        for (size_t i = 0; i < n2; i++) real_[i] = (double)h_corr_[2 * i] * nrm;  // real part
        return refine_correlation_peak(real_, n, n);
    }

private:
    bool wrap(void *dev, VPIImage &img) {
        VPIImageData d;
        std::memset(&d, 0, sizeof(d));
        d.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
        d.buffer.pitch.format = VPI_IMAGE_FORMAT_2F32;
        d.buffer.pitch.numPlanes = 1;
        d.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_2F32;
        d.buffer.pitch.planes[0].width = n_;
        d.buffer.pitch.planes[0].height = n_;
        d.buffer.pitch.planes[0].pitchBytes = n_ * (int)sizeof(float2);
        d.buffer.pitch.planes[0].data = dev;
        return vpiImageCreateWrapper(&d, nullptr, VPI_BACKEND_CUDA, &img) == VPI_SUCCESS;
    }
    void destroy() {
        for (VPIImage *p : {&img_prevwin_, &img_currwin_, &img_aspec_, &img_bspec_, &img_corr_})
            if (*p) { vpiImageDestroy(*p); *p = nullptr; }
        if (fft_)    { vpiPayloadDestroy(fft_);   fft_ = nullptr; }
        if (ifft_)   { vpiPayloadDestroy(ifft_);  ifft_ = nullptr; }
        if (stream_) { vpiStreamDestroy(stream_); stream_ = nullptr; }
        if (cs_)     { cudaStreamDestroy(cs_);    cs_ = nullptr; }
        for (void **p : {(void **)&d_yp_, (void **)&d_yc_, (void **)&d_hann_,
                         (void **)&d_prevwin_, (void **)&d_currwin_, (void **)&d_aspec_,
                         (void **)&d_bspec_, (void **)&d_corr_})
            if (*p) { cudaFree(*p); *p = nullptr; }
    }

    int n_ = 0;
    cudaStream_t cs_ = nullptr;    // dedicated stream for kernels+copies (no device sync)
    VPIStream stream_ = nullptr;
    VPIPayload fft_ = nullptr, ifft_ = nullptr;
    VPIImage img_prevwin_ = nullptr, img_currwin_ = nullptr, img_aspec_ = nullptr,
             img_bspec_ = nullptr, img_corr_ = nullptr;
    unsigned char *d_yp_ = nullptr, *d_yc_ = nullptr;
    float *d_hann_ = nullptr;
    float2 *d_prevwin_ = nullptr, *d_currwin_ = nullptr, *d_aspec_ = nullptr,
           *d_bspec_ = nullptr, *d_corr_ = nullptr;
    std::vector<float> h_corr_;   // host readback of d_corr_ (2*n2 floats: re,im pairs)
    std::vector<double> real_;    // real part for refine_correlation_peak
};

}  // namespace nvmm

#else  // !NVMM_HAVE_VPI — stub so callers compile on non-VPI (mock/CI) builds.
namespace nvmm {
class GmcVpiFft {
public:
    static bool available() { return false; }
    bool init(int, std::string &err) { err = "VPI not built"; return false; }
    PhaseShift estimate(const uint8_t *, const uint8_t *) { return {}; }
};
}  // namespace nvmm
#endif  // NVMM_HAVE_VPI
