/// samurai_kernels.hpp — host-callable wrappers for the SAMURAI per-frame CUDA
/// kernels (samurai_kernels.cu, compiled by nvcc/CUDA 12.6, -std=c++20, sm_87).
/// These replace the correctness-first host round-trips in samurai_tracker.cpp so
/// the per-frame dataflow stays on-device (zero-copy). Each kernel is parity-
/// checked against its host reference (which is itself golden-validated).
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

/// Assemble the static 7232x1x64 memory + memory_pos on-device (device-ring path;
/// mirrors host assemble_memory in samurai_memory.hpp).
///   maskmem : device array of 7 device pointers (each 64*1024), slot 0=cond.
///   objptr  : packed device (16*256), p=0 = cond.
///   pos_list: device (16) signed temporal positions.
///   maskmem_pos (64*1024), tpos (7*64), tposproj_w (64*256), tposproj_b (64): device consts.
///   memory, memory_pos: device outputs (7232*64).
void k_assemble_memory(const float *const *maskmem, const float *objptr,
                       const float *pos_list, const float *maskmem_pos,
                       const float *tpos, const float *tposproj_w, const float *tposproj_b,
                       float *memory, float *memory_pos, cudaStream_t s);

}  // namespace nvmm
