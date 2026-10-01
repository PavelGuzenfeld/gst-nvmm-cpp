#pragma once
#include <cstring>

namespace nvmm {

enum class GmcBackend { Auto, Ncc, FftCpu, FftCuda, Pva };

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

/// auto walks fft-cuda -> pva -> fft-cpu -> ncc. An unavailable explicit request
/// degrades to the CPU form of the same algorithm, never to a higher tier or to a
/// different algorithm, so a CI build always lands on a CPU backend.
inline GmcBackend resolve_gmc_backend(GmcBackend requested, bool have_cuda_fft,
                                      bool have_pva) {
    switch (requested) {
        case GmcBackend::FftCuda: return have_cuda_fft ? GmcBackend::FftCuda : GmcBackend::FftCpu;
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

/// FFT paths need a power of two (radix-2); NCC matches them at 128. VPI
/// HarrisCorners on PVA needs at least 160x120, so pva uses 256.
inline int gmc_patch_size(GmcBackend resolved) {
    return resolved == GmcBackend::Pva ? 256 : 128;
}

}
