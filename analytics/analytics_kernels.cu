/// analytics_kernels.cu — fused CUDA implementation of low_texture_motion.
/// Stage-for-stage mirror of the fused host pipeline (low_texture_motion.hpp):
///   1. Sobel-x/y + gradient magnitude, one kernel (REFLECT_101 border);
///   2. separable Gaussian blur, threshold fused into the column pass -> 0/1 mask;
///   3. morphological close as separable window-OR (dilate) then window-AND
///      (erode) — exact on a binary mask with a square SE, and the clipped
///      window matches OpenCV's +-inf morphology border semantics;
///   4. min(|cur-ref_a|, |cur-ref_b|) * mask -> float, one kernel;
///   5. optional separable blur of the output; border zeroing fused into the
///      last kernel that touches the frame.
/// Weights match img::gaussian_kernel (OpenCV's tables/formula), so parity with
/// the host path is tight; the probe test asserts it. All kernels take pitched
/// inputs/outputs so mapped NvBufSurface planes work in place (zero-copy);
/// internal scratch is packed.

#include "analytics_kernels.hpp"
#include "image_ops.hpp"

#include <cuda_runtime.h>
#ifdef ANALYTICS_KERNELS_DEBUG
#include <cstdio>
#endif

namespace nvmm {
namespace motion {

constexpr int kMaxKernel = 31;
// named-namespace scope on purpose: anonymous-namespace __constant__ symbols
// fail cudaMemcpyToSymbol registration ("invalid device symbol") on CUDA 12
__constant__ float c_blur[kMaxKernel];

namespace {

__device__ __forceinline__ int reflect101(int i, int n)
{
    if (i < 0) return -i;
    if (i >= n) return 2 * n - 2 - i;
    return i;
}

__global__ void k_sobel_mag(const uint8_t *src, long spitch, float *mag, int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const uint8_t *ym = src + (size_t)reflect101(y - 1, h) * spitch;
    const uint8_t *y0 = src + (size_t)y * spitch;
    const uint8_t *yp = src + (size_t)reflect101(y + 1, h) * spitch;
    const int xm = reflect101(x - 1, w), xp = reflect101(x + 1, w);
    const int gx = ((int)ym[xp] + 2 * y0[xp] + yp[xp]) - ((int)ym[xm] + 2 * y0[xm] + yp[xm]);
    const int gy = ((int)yp[xm] + 2 * yp[x] + yp[xp]) - ((int)ym[xm] + 2 * ym[x] + ym[xp]);
    mag[(size_t)y * w + x] = sqrtf((float)(gx * gx + gy * gy));
}

__global__ void k_blur_rows(const float *src, float *dst, int w, int h, int k)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int r = k / 2;
    const float *s = src + (size_t)y * w;
    float acc = 0.f;
    for (int i = 0; i < k; i++) acc += c_blur[i] * s[reflect101(x - r + i, w)];
    dst[(size_t)y * w + x] = acc;
}

/// Column blur into a 0/1 mask (threshold fused).
__global__ void k_blur_cols_mask(const float *src, uint8_t *dst, int w, int h, int k,
                                 float thresh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int r = k / 2;
    float acc = 0.f;
    for (int i = 0; i < k; i++)
        acc += c_blur[i] * src[(size_t)reflect101(y - r + i, h) * w + x];
    dst[(size_t)y * w + x] = acc < thresh ? 1 : 0;
}

/// Column blur into the pitched float output, border zeroed (final pass).
__global__ void k_blur_cols_out(const float *src, float *dst, long dpitch, int w, int h,
                                int k, int border)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const bool edge = x < border || y < border || x >= w - border || y >= h - border;
    float acc = 0.f;
    if (!edge) {
        const int r = k / 2;
        for (int i = 0; i < k; i++)
            acc += c_blur[i] * src[(size_t)reflect101(y - r + i, h) * w + x];
    }
    dst[(size_t)y * dpitch + x] = acc;
}

/// Separable binary morphology: OR (dilate) / AND (erode) over a 2r+1 window
/// along one axis. Window clipped to the frame: outside pixels never dilate
/// and never block erosion (OpenCV's +-inf border).
__global__ void k_morph_axis(const uint8_t *src, uint8_t *dst, int w, int h,
                             int r, bool horizontal, bool dilate)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int c = horizontal ? x : y, n = horizontal ? w : h;
    const int lo = max(0, c - r), hi = min(n - 1, c + r);
    uint8_t v = dilate ? 0 : 1;
    for (int i = lo; i <= hi; i++) {
        const uint8_t s = horizontal ? src[(size_t)y * w + i] : src[(size_t)i * w + x];
        if (dilate) v |= s;
        else v &= s;
        if (dilate ? v : !v) break;
    }
    dst[(size_t)y * w + x] = v;
}

/// out = mask ? min(|cur-ref_a|, |cur-ref_b|) : 0. Writes packed scratch when a
/// blur pass follows (dpitch == w, border == 0) or the pitched output with the
/// border zeroed when it is the final pass.
__global__ void k_masked_min_diff(const uint8_t *cur, long cpitch, const uint8_t *ra,
                                  long apitch, const uint8_t *rb, long bpitch,
                                  const uint8_t *mask, float *out, long dpitch,
                                  int w, int h, int border)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const bool edge = x < border || y < border || x >= w - border || y >= h - border;
    float v = 0.f;
    if (!edge && mask[(size_t)y * w + x]) {
        const int c = cur[(size_t)y * cpitch + x];
        const int da = abs(c - (int)ra[(size_t)y * apitch + x]);
        const int db = abs(c - (int)rb[(size_t)y * bpitch + x]);
        v = (float)min(da, db);
    }
    out[(size_t)y * dpitch + x] = v;
}

}  // namespace

struct LowTextureMotionCuda::Impl {
    int w = 0, h = 0;
    uint8_t *d_mask = nullptr, *d_morph = nullptr;
    float *d_a = nullptr, *d_b = nullptr;
    // host-wrapper staging (allocated only by run())
    uint8_t *d_cur = nullptr, *d_ra = nullptr, *d_rb = nullptr;
    float *d_out = nullptr;
    cudaError_t err = cudaSuccess;

    bool ok(cudaError_t e) {
#ifdef ANALYTICS_KERNELS_DEBUG
        if (e != cudaSuccess) fprintf(stderr, "cuda err: %s\n", cudaGetErrorString(e));
#endif
        if (e != cudaSuccess && err == cudaSuccess) err = e;
        return e == cudaSuccess;
    }

    void release() {
        cudaFree(d_mask); cudaFree(d_morph); cudaFree(d_a); cudaFree(d_b);
        cudaFree(d_cur); cudaFree(d_ra); cudaFree(d_rb); cudaFree(d_out);
        d_mask = d_morph = d_cur = d_ra = d_rb = nullptr;
        d_a = d_b = d_out = nullptr;
        w = h = 0;
    }

    bool scratch(int nw, int nh, bool staging) {
        if ((nw != w || nh != h) || (staging && !d_cur)) {
            release();
            const size_t n = (size_t)nw * nh;
            if (!ok(cudaMalloc(&d_mask, n)) || !ok(cudaMalloc(&d_morph, n)) ||
                !ok(cudaMalloc(&d_a, n * sizeof(float))) ||
                !ok(cudaMalloc(&d_b, n * sizeof(float)))) {
                release();
                return false;
            }
            if (staging &&
                (!ok(cudaMalloc(&d_cur, n)) || !ok(cudaMalloc(&d_ra, n)) ||
                 !ok(cudaMalloc(&d_rb, n)) || !ok(cudaMalloc(&d_out, n * sizeof(float))))) {
                release();
                return false;
            }
            w = nw; h = nh;
        }
        return true;
    }
};

LowTextureMotionCuda::LowTextureMotionCuda() : impl_(new Impl) {}
LowTextureMotionCuda::~LowTextureMotionCuda()
{
    impl_->release();
    delete impl_;
}

const char *LowTextureMotionCuda::last_error() const
{
    return cudaGetErrorString(impl_->err);
}

bool LowTextureMotionCuda::run_device(DevicePlane<const uint8_t> cur,
                                      DevicePlane<const uint8_t> ref_a,
                                      DevicePlane<const uint8_t> ref_b,
                                      const LowTextureMotionParams &p,
                                      DevicePlane<float> out, cudaStream_t stream)
{
    const int w = cur.width, h = cur.height;
    if (w <= 1 || h <= 1 || p.grad_blur > kMaxKernel || p.diff_blur > kMaxKernel ||
        !cur.data || !ref_a.data || !ref_b.data || !out.data)
        return false;
    if (!impl_->scratch(w, h, impl_->d_cur != nullptr)) return false;

    const dim3 blk(32, 8);
    const dim3 grd((unsigned)(w + 31) / 32, (unsigned)(h + 7) / 8);

    // 1. gradient magnitude
    k_sobel_mag<<<grd, blk, 0, stream>>>(cur.data, (long)cur.pitch, impl_->d_a, w, h);

    // 2. blur + fused threshold -> binary mask
    const std::vector<float> kg = img::gaussian_kernel(p.grad_blur);
    if (!impl_->ok(cudaMemcpyToSymbolAsync(c_blur, kg.data(), kg.size() * sizeof(float), 0,
                                           cudaMemcpyHostToDevice, stream)))
        return false;
    k_blur_rows<<<grd, blk, 0, stream>>>(impl_->d_a, impl_->d_b, w, h, p.grad_blur);
    k_blur_cols_mask<<<grd, blk, 0, stream>>>(impl_->d_b, impl_->d_mask, w, h, p.grad_blur,
                                              p.grad_thresh);

    // 3. close = separable dilate then separable erode
    const int r = p.close_k / 2;
    k_morph_axis<<<grd, blk, 0, stream>>>(impl_->d_mask, impl_->d_morph, w, h, r, true, true);
    k_morph_axis<<<grd, blk, 0, stream>>>(impl_->d_morph, impl_->d_mask, w, h, r, false, true);
    k_morph_axis<<<grd, blk, 0, stream>>>(impl_->d_mask, impl_->d_morph, w, h, r, true, false);
    k_morph_axis<<<grd, blk, 0, stream>>>(impl_->d_morph, impl_->d_mask, w, h, r, false, false);

    // 4. masked min-diff (final pass when no output blur follows)
    const bool blur_out = p.diff_blur > 0;
    k_masked_min_diff<<<grd, blk, 0, stream>>>(
        cur.data, (long)cur.pitch, ref_a.data, (long)ref_a.pitch, ref_b.data,
        (long)ref_b.pitch, impl_->d_mask, blur_out ? impl_->d_a : out.data,
        blur_out ? (long)w : (long)out.pitch, w, h, blur_out ? 0 : p.border);

    // 5. optional output blur, border zero fused into its column pass
    if (blur_out) {
        const std::vector<float> kd = img::gaussian_kernel(p.diff_blur);
        if (!impl_->ok(cudaMemcpyToSymbolAsync(c_blur, kd.data(), kd.size() * sizeof(float),
                                               0, cudaMemcpyHostToDevice, stream)))
            return false;
        k_blur_rows<<<grd, blk, 0, stream>>>(impl_->d_a, impl_->d_b, w, h, p.diff_blur);
        k_blur_cols_out<<<grd, blk, 0, stream>>>(impl_->d_b, out.data, (long)out.pitch, w, h,
                                                 p.diff_blur, p.border);
    }
    return impl_->ok(cudaGetLastError());
}

bool LowTextureMotionCuda::run(img::View<const uint8_t> cur, img::View<const uint8_t> ref_a,
                               img::View<const uint8_t> ref_b,
                               const LowTextureMotionParams &p, img::Image<float> &out,
                               cudaStream_t stream)
{
    const int w = cur.width, h = cur.height;
    if (w <= 1 || h <= 1) return false;
    if (!impl_->scratch(w, h, true)) return false;

    auto upload = [&](uint8_t *dst, img::View<const uint8_t> v) {
        return impl_->ok(cudaMemcpy2DAsync(dst, (size_t)w, v.data, (size_t)v.stride,
                                           (size_t)w, (size_t)h, cudaMemcpyHostToDevice,
                                           stream));
    };
    if (!upload(impl_->d_cur, cur) || !upload(impl_->d_ra, ref_a) ||
        !upload(impl_->d_rb, ref_b))
        return false;

    DevicePlane<const uint8_t> dc{impl_->d_cur, w, h, w}, da{impl_->d_ra, w, h, w},
        db{impl_->d_rb, w, h, w};
    DevicePlane<float> dout{impl_->d_out, w, h, w};
    if (!run_device(dc, da, db, p, dout, stream)) return false;

    if (out.width() != w || out.height() != h) out = img::Image<float>(w, h);
    if (!impl_->ok(cudaMemcpyAsync(out.data(), impl_->d_out, (size_t)w * h * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream)))
        return false;
    return impl_->ok(cudaStreamSynchronize(stream));
}

}  // namespace motion
}  // namespace nvmm
