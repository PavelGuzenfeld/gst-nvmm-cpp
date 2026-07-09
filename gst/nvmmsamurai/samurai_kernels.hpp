/// samurai_kernels.hpp — host-callable wrappers for the SAMURAI per-frame CUDA
/// kernels (samurai_kernels.cu, compiled by nvcc/CUDA 12.6, -std=c++20, sm_87).
/// These replace the correctness-first host round-trips in samurai_tracker.cpp so
/// the per-frame dataflow stays on-device (zero-copy). Each kernel is parity-
/// checked against its host reference (which is itself golden-validated).
///
/// `compute-sanitizer --tool memcheck|initcheck|racecheck|synccheck` against
/// benchmarks/bench_gmc (exercises k_gmc_window/k_gmc_cross_power): all 4 tools
/// pass clean (0 errors/hazards) on the Orin JP6 test host. If you hit "GPU
/// debugging features are disabled" running this yourself on Jetson/Tegra, it is
/// NOT a kernel boot-flag issue (a `nvgpu.support_gpu_tools=1` cmdline flag does
/// nothing on L4T R36 — that module parameter doesn't exist) — the launching user
/// must be in the `debug` group: `sudo usermod -aG debug $USER`, then start a new
/// login session (group membership applies on next login, no reboot needed).
#pragma once

#include <cuda_runtime.h>

namespace nvmm {

/// Transpose a row-major (rows x cols) tensor to (cols x rows). Used for the
/// (C,HW)<->(HW,C) reshapes between encoder/decoder (C,HW) and memory_attention
/// (HW,C). out[c*rows + r] = in[r*cols + c].
void k_transpose(const float *in, float *out, int rows, int cols, cudaStream_t s);

/// out[c*HW + i] = in[c*HW + i] + bias[c]   (per-channel add; no_mem_embed).
void k_add_per_channel(const float *in, const float *bias, float *out,
                       int C, int HW, cudaStream_t s);

/// out[i] = sigmoid(in[i]) * scale + bias   (tracking memory mask: 20/-10).
void k_sigmoid_scale(const float *in, float *out, int n, float scale, float bias,
                     cudaStream_t s);

/// out[i] = (in[i] > 0) ? hi : lo   (seed memory mask binarize: +10/-10).
void k_threshold_scale(const float *in, float *out, int n, float hi, float lo,
                       cudaStream_t s);

/// Bilinear resize src(hi x wi) -> dst(ho x wo), align_corners=False (PyTorch
/// F.interpolate clone). Single channel.
void k_bilinear(const float *src, float *dst, int hi, int wi, int ho, int wo,
                cudaStream_t s);

/// Tight bbox of {mask > 0} over an h x w grid. Writes 4 ints to d_box (device):
/// [xmin, ymin, xmax, ymax]; xmax < 0 means empty. Caller pre-fills d_box with
/// {w, h, -1, -1}. Uses atomics.
void k_mask_bbox(const float *mask, int h, int w, int *d_box, cudaStream_t s);

/// Assemble the static (7*tok+64)x1x64 memory + memory_pos on-device (device-ring
/// path; mirrors host assemble_memory in samurai_memory.hpp). tok = encoder grid
/// token count ((crop/16)^2 — 1024 @512 crop, 576 @384, 256 @256).
///   maskmem : device array of 7 device pointers (each 64*tok), slot 0=cond.
///   objptr  : packed device (16*256), p=0 = cond.
///   pos_list: device (16) signed temporal positions.
///   maskmem_pos (64*tok), tpos (7*64), tposproj_w (64*256), tposproj_b (64): device consts.
///   memory, memory_pos: device outputs ((7*tok+64)*64).
void k_assemble_memory(const float *const *maskmem, const float *objptr,
                       const float *pos_list, const float *maskmem_pos,
                       const float *tpos, const float *tposproj_w, const float *tposproj_b,
                       float *memory, float *memory_pos, int tok, cudaStream_t s);

/// GMC fft-cuda backend helpers (gmc_vpi_fft.hpp). Window an n x n uint8 grayscale
/// patch (row stride `pitch`) by a separable Hann window (`hann` length n) into a
/// complex 2F32 image (out[i] = {y*win, 0}), matching PhaseCorrelator's windowing
/// so the CUDA and CPU FFT paths are directly comparable.
void k_gmc_window(const unsigned char *y, int pitch, int n, const float *hann,
                  float2 *out, cudaStream_t s);

/// Normalized cross-power spectrum in place: a[i] = R/|R|, R = a[i] * conj(b[i])
/// (or {0,0} if |R| ~ 0). Same convention as PhaseCorrelator's cross-power step.
void k_gmc_cross_power(float2 *a, const float2 *b, int n2, cudaStream_t s);

}  // namespace nvmm
