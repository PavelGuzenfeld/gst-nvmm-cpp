#pragma once

#include <cuda_runtime.h>

namespace nvmm {

void k_transpose(const float *in, float *out, int rows, int cols, cudaStream_t s);

void k_add_per_channel(const float *in, const float *bias, float *out,
                       int C, int HW, cudaStream_t s);

void k_sigmoid_scale(const float *in, float *out, int n, float scale, float bias,
                     cudaStream_t s);

void k_threshold_scale(const float *in, float *out, int n, float hi, float lo,
                       cudaStream_t s);

void k_bilinear(const float *src, float *dst, int hi, int wi, int ho, int wo,
                cudaStream_t s);

/// Caller pre-fills d_box with {w, h, -1, -1}. Result is [xmin, ymin, xmax, ymax];
/// xmax < 0 means empty.
void k_mask_bbox(const float *mask, int h, int w, int *d_box, cudaStream_t s);

/// Device mirror of assemble_memory, tok = (crop/16)^2. maskmem: 7 device ptrs of
/// 64*tok, slot 0 = cond; objptr (16*256), p=0 = cond; outputs (7*tok+64)*64.
void k_assemble_memory(const float *const *maskmem, const float *objptr,
                       const float *pos_list, const float *maskmem_pos,
                       const float *tpos, const float *tposproj_w, const float *tposproj_b,
                       float *memory, float *memory_pos, int tok, cudaStream_t s);

/// Same separable Hann as PhaseCorrelator, so the CUDA and CPU FFT paths compare.
void k_gmc_window(const unsigned char *y, int pitch, int n, const float *hann,
                  float2 *out, cudaStream_t s);

/// a[i] = R/|R| with R = a[i]*conj(b[i]), {0,0} when |R| ~ 0, as PhaseCorrelator.
void k_gmc_cross_power(float2 *a, const float2 *b, int n2, cudaStream_t s);

}
