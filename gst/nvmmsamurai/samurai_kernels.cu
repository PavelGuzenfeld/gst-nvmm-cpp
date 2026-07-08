/// samurai_kernels.cu — SAMURAI per-frame CUDA kernels (CUDA 12.6, -std=c++20,
/// sm_87). Host wrappers declared in samurai_kernels.hpp. See that file for the
/// contract; each kernel mirrors a host reference in samurai_seed_math.hpp /
/// samurai_tracker.cpp and is parity-checked (tests/samurai_kernel_probe).
#include "samurai_kernels.hpp"

#include <cuda/std/cmath>   // cuda::std::fmaxf (libcu++ / CCCL; CUDA 12.6 has no <algorithm>)

namespace nvmm {
namespace {
constexpr int kBlk = 256;
inline int grid(int n) { return (n + kBlk - 1) / kBlk; }

__global__ void transpose_k(const float *in, float *out, int rows, int cols)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    const int r = idx / cols, c = idx % cols;
    out[c * rows + r] = in[idx];
}

__global__ void add_per_channel_k(const float *in, const float *bias, float *out,
                                  int C, int HW)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * HW) return;
    out[idx] = in[idx] + bias[idx / HW];
}

__global__ void sigmoid_scale_k(const float *in, float *out, int n, float scale, float bias)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (1.f / (1.f + __expf(-in[idx]))) * scale + bias;
}

__global__ void threshold_scale_k(const float *in, float *out, int n, float hi, float lo)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = in[idx] > 0.f ? hi : lo;
}

// Bilinear, align_corners=False: src = (dst+0.5)*scale - 0.5, edge-clamped.
__global__ void bilinear_k(const float *src, float *dst, int hi, int wi, int ho, int wo)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ho * wo) return;
    const int oy = idx / wo, ox = idx % wo;
    const float sy = (float)hi / ho, sx = (float)wi / wo;
    const float fy = cuda::std::fmaxf((oy + 0.5f) * sy - 0.5f, 0.f);
    const float fx = cuda::std::fmaxf((ox + 0.5f) * sx - 0.5f, 0.f);
    const int y0 = (int)fy, x0 = (int)fx;
    const int y1 = y0 + 1 < hi ? y0 + 1 : hi - 1;
    const int x1 = x0 + 1 < wi ? x0 + 1 : wi - 1;
    const float wy = fy - y0, wx = fx - x0;
    const float a = src[y0 * wi + x0], b = src[y0 * wi + x1];
    const float c = src[y1 * wi + x0], d = src[y1 * wi + x1];
    const float top = a + (b - a) * wx, bot = c + (d - c) * wx;
    dst[idx] = top + (bot - top) * wy;
}

// d_box = [xmin, ymin, xmax, ymax]; caller pre-sets {w, h, -1, -1}.
__global__ void mask_bbox_k(const float *mask, int h, int w, int *box)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= h * w) return;
    if (mask[idx] > 0.f) {
        const int y = idx / w, x = idx % w;
        atomicMin(&box[0], x); atomicMin(&box[1], y);
        atomicMax(&box[2], x); atomicMax(&box[3], y);
    }
}

// One element of get_1d_sine_pe(x, dim=256): in<128 -> sin, else cos.
__device__ inline float sine_pe_elem(float x, int in)
{
    const int pe_dim = 128;
    const int j = in < pe_dim ? in : in - pe_dim;
    const float dim_t = powf(10000.f, (float)(2 * (j / 2)) / pe_dim);
    const float v = x / dim_t;
    return in < pe_dim ? __sinf(v) : __cosf(v);
}

// Mirrors samurai_memory.hpp assemble_memory. (7*tok+64)*64 outputs; tok = the
// encoder grid token count ((crop/16)^2 — 1024 @512, 576 @384, 256 @256).
__global__ void assemble_k(const float *const *maskmem, const float *objptr,
                           const float *pos_list, const float *maskmem_pos,
                           const float *tpos, const float *tposproj_w, const float *tposproj_b,
                           float *memory, float *memory_pos, int tok)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int kMaskRows = 7 * tok, kTotal = kMaskRows + 64;
    if (idx >= kTotal * 64) return;
    const int row = idx / 64, ch = idx % 64;
    if (row < kMaskRows) {                       // maskmem block
        const int s = row / tok, i = row % tok;
        memory[idx] = maskmem[s][ch * tok + i];
        memory_pos[idx] = maskmem_pos[ch * tok + i] + tpos[(6 - s) * 64 + ch];
    } else {                                     // obj_ptr block
        const int o = row - kMaskRows, p = o / 4, k = o % 4;
        memory[idx] = objptr[p * 256 + k * 64 + ch];
        const float x = pos_list[p] / 15.f;      // t_diff_max = 15
        float acc = tposproj_b[ch];
        const float *wr = tposproj_w + (size_t)ch * 256;
        for (int in = 0; in < 256; in++) acc += wr[in] * sine_pe_elem(x, in);
        memory_pos[idx] = acc;
    }
}
// GMC fft-cuda: window a uint8 patch by a separable Hann into complex 2F32.
__global__ void gmc_window_k(const unsigned char *y, int pitch, int n,
                             const float *hann, float2 *out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    const int r = idx / n, c = idx % n;
    const float win = hann[r] * hann[c];
    out[idx] = make_float2((float)y[(size_t)r * pitch + c] * win, 0.f);
}

// GMC fft-cuda: normalized cross-power a = R/|R|, R = a * conj(b), in place.
__global__ void gmc_cross_power_k(float2 *a, const float2 *b, int n2)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    const float2 av = a[i], bv = b[i];
    const float rx = av.x * bv.x + av.y * bv.y;   // Re(a * conj(b))
    const float ry = av.y * bv.x - av.x * bv.y;   // Im(a * conj(b))
    const float m = sqrtf(rx * rx + ry * ry);
    a[i] = m > 1e-12f ? make_float2(rx / m, ry / m) : make_float2(0.f, 0.f);
}
}  // namespace

void k_gmc_window(const unsigned char *y, int pitch, int n, const float *hann,
                  float2 *out, cudaStream_t s)
{ gmc_window_k<<<grid(n * n), kBlk, 0, s>>>(y, pitch, n, hann, out); }

void k_gmc_cross_power(float2 *a, const float2 *b, int n2, cudaStream_t s)
{ gmc_cross_power_k<<<grid(n2), kBlk, 0, s>>>(a, b, n2); }

void k_transpose(const float *in, float *out, int rows, int cols, cudaStream_t s)
{ transpose_k<<<grid(rows * cols), kBlk, 0, s>>>(in, out, rows, cols); }

void k_add_per_channel(const float *in, const float *bias, float *out, int C, int HW, cudaStream_t s)
{ add_per_channel_k<<<grid(C * HW), kBlk, 0, s>>>(in, bias, out, C, HW); }

void k_sigmoid_scale(const float *in, float *out, int n, float scale, float bias, cudaStream_t s)
{ sigmoid_scale_k<<<grid(n), kBlk, 0, s>>>(in, out, n, scale, bias); }

void k_threshold_scale(const float *in, float *out, int n, float hi, float lo, cudaStream_t s)
{ threshold_scale_k<<<grid(n), kBlk, 0, s>>>(in, out, n, hi, lo); }

void k_bilinear(const float *src, float *dst, int hi, int wi, int ho, int wo, cudaStream_t s)
{ bilinear_k<<<grid(ho * wo), kBlk, 0, s>>>(src, dst, hi, wi, ho, wo); }

void k_mask_bbox(const float *mask, int h, int w, int *d_box, cudaStream_t s)
{ mask_bbox_k<<<grid(h * w), kBlk, 0, s>>>(mask, h, w, d_box); }

void k_assemble_memory(const float *const *maskmem, const float *objptr,
                       const float *pos_list, const float *maskmem_pos,
                       const float *tpos, const float *tposproj_w, const float *tposproj_b,
                       float *memory, float *memory_pos, int tok, cudaStream_t s)
{ assemble_k<<<grid((7 * tok + 64) * 64), kBlk, 0, s>>>(maskmem, objptr, pos_list, maskmem_pos,
                                              tpos, tposproj_w, tposproj_b, memory, memory_pos, tok); }

}  // namespace nvmm
