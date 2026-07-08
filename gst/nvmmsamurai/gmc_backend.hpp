/// gmc_backend.hpp — GMC (camera-motion compensation) backend selection policy
/// (host, dependency-free, unit-testable). The estimator implementations live in
/// samurai_tracker.cpp (they need NvBufSurface / VPI / CUDA); this header only
/// owns the enum, its string mapping, the `auto` resolution order, and the patch
/// size each backend needs — the pure decisions that gate surface allocation.
///
/// Backends (see docs plan):
///   ncc      — CPU zero-mean NCC brute-force (samurai_gmc.hpp), the baseline.
///   fft-cpu  — CPU FFT phase correlation (gst/common/phase_correlation.hpp).
///   fft-cuda — VPI FFT (VPI_BACKEND_CUDA) phase correlation.
///   pva      — VPI HarrisCorners + OpticalFlowPyrLK (VPI_BACKEND_PVA) -> median.
/// `auto` resolves at tracker init to the best backend actually available.
#pragma once
#include <cstring>

namespace nvmm {

enum class GmcBackend { Auto, Ncc, FftCpu, FftCuda, Pva };

/// Canonical GObject-property string for a backend (also used in logs).
inline const char *gmc_backend_name(GmcBackend b) {
    switch (b) {
        case GmcBackend::Auto:    return "auto";
        case GmcBackend::Ncc:     return "ncc";
        case GmcBackend::FftCpu:  return "fft-cpu";
        case GmcBackend::FftCuda: return "fft-cuda";
        case GmcBackend::Pva:     return "pva";
    }
    return "auto";
}

/// Parse a property string; unknown strings fall back to Auto.
inline GmcBackend gmc_backend_from_string(const char *s) {
    if (!s) return GmcBackend::Auto;
    if (!std::strcmp(s, "ncc"))      return GmcBackend::Ncc;
    if (!std::strcmp(s, "fft-cpu"))  return GmcBackend::FftCpu;
    if (!std::strcmp(s, "fft-cuda")) return GmcBackend::FftCuda;
    if (!std::strcmp(s, "pva"))      return GmcBackend::Pva;
    return GmcBackend::Auto;
}

/// Resolve a requested backend to a concrete one given what the platform offers.
/// `auto` order: fft-cuda -> pva -> fft-cpu -> ncc. An explicitly requested backend
/// that is unavailable degrades down the same order (never silently to a *higher*
/// tier), so an off-Orin/CI build always lands on a CPU backend that builds & runs.
inline GmcBackend resolve_gmc_backend(GmcBackend requested, bool have_cuda_fft,
                                      bool have_pva) {
    switch (requested) {
        case GmcBackend::FftCuda: return have_cuda_fft ? GmcBackend::FftCuda
                                : have_pva            ? GmcBackend::Pva
                                                      : GmcBackend::FftCpu;
        case GmcBackend::Pva:     return have_pva ? GmcBackend::Pva : GmcBackend::FftCpu;
        case GmcBackend::FftCpu:  return GmcBackend::FftCpu;
        case GmcBackend::Ncc:     return GmcBackend::Ncc;
        case GmcBackend::Auto:
        default:
            if (have_cuda_fft) return GmcBackend::FftCuda;
            if (have_pva)      return GmcBackend::Pva;
            return GmcBackend::FftCpu;
    }
}

/// Square patch side the backend's estimator consumes. The FFT paths need a
/// power-of-two (radix-2 FFT); NCC matches them at 128. The PVA path needs
/// >= 160x120 for VPI HarrisCorners, so it uses 256 (also pow2-friendly for VIC).
inline int gmc_patch_size(GmcBackend resolved) {
    return resolved == GmcBackend::Pva ? 256 : 128;
}

}  // namespace nvmm
