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

inline GmcBackend gmc_backend_from_string(const char *s) {
    if (!s) return GmcBackend::Auto;
    if (!std::strcmp(s, "ncc"))      return GmcBackend::Ncc;
    if (!std::strcmp(s, "fft-cpu"))  return GmcBackend::FftCpu;
    if (!std::strcmp(s, "fft-cuda")) return GmcBackend::FftCuda;
    if (!std::strcmp(s, "pva"))      return GmcBackend::Pva;
    return GmcBackend::Auto;
}

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

inline int gmc_patch_size(GmcBackend resolved) {
    return resolved == GmcBackend::Pva ? 256 : 128;
}

}
