/// samurai_seed_math.hpp — host-side math for the seed/track selection, kept
/// dependency-free so it can be unit-tested off-target. Correctness-first (host);
/// the per-frame mask ops are CUDA-optimization candidates later.
///   - bilinear_upsample: PyTorch F.interpolate(align_corners=False) clone
///   - mask_to_box: argwhere(mask>0) -> [x,y,w,h] (w=xmax-xmin), matches sam2_base
///   - mlp3_relu: obj_ptr_proj (3 Linear layers, ReLU between, none after last)
#pragma once

#include <cstddef>
#include <vector>

namespace nvmm {

/// Box in pixel coords (within the upsampled mask frame). valid=false if empty.
struct MaskBox {
    float x = 0.f, y = 0.f, w = 0.f, h = 0.f;
    bool valid = false;
};

/// Bilinear resize src[hi*wi] -> dst[ho*wo], matching torch interpolate with
/// align_corners=False (src = (dst+0.5)*scale - 0.5, edge-clamped).
inline std::vector<float> bilinear_upsample(const float *src, int hi, int wi,
                                            int ho, int wo)
{
    std::vector<float> dst((size_t)ho * wo);
    const float sy = (float)hi / ho, sx = (float)wi / wo;
    for (int oy = 0; oy < ho; oy++) {
        float fy = (oy + 0.5f) * sy - 0.5f;
        if (fy < 0.f) fy = 0.f;
        int y0 = (int)fy; int y1 = y0 + 1 < hi ? y0 + 1 : hi - 1;
        float wy = fy - y0;
        for (int ox = 0; ox < wo; ox++) {
            float fx = (ox + 0.5f) * sx - 0.5f;
            if (fx < 0.f) fx = 0.f;
            int x0 = (int)fx; int x1 = x0 + 1 < wi ? x0 + 1 : wi - 1;
            float wx = fx - x0;
            const float a = src[(size_t)y0 * wi + x0], b = src[(size_t)y0 * wi + x1];
            const float c = src[(size_t)y1 * wi + x0], d = src[(size_t)y1 * wi + x1];
            const float top = a + (b - a) * wx, bot = c + (d - c) * wx;
            dst[(size_t)oy * wo + ox] = top + (bot - top) * wy;
        }
    }
    return dst;
}

/// Tight box around mask>0 pixels (w = xmax-xmin, h = ymax-ymin), matching
/// sam2_base's BoundingBox(x=x_min,y=y_min,x2=x_max,y2=y_max).
inline MaskBox mask_to_box(const float *mask, int h, int w, float thresh = 0.f)
{
    int xmin = w, ymin = h, xmax = -1, ymax = -1;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (mask[(size_t)y * w + x] > thresh) {
                if (x < xmin) xmin = x;
                if (x > xmax) xmax = x;
                if (y < ymin) ymin = y;
                if (y > ymax) ymax = y;
            }
    MaskBox b;
    if (xmax < 0) return b;  // empty
    b.x = (float)xmin; b.y = (float)ymin;
    b.w = (float)(xmax - xmin); b.h = (float)(ymax - ymin);
    b.valid = true;
    return b;
}

/// 3-layer MLP with ReLU between layers, no activation after the last (SAM2 MLP
/// default). weights row-major [out,in]; in==out==dim for obj_ptr_proj.
inline std::vector<float> mlp3_relu(const float *x, int dim,
                                    const float *w0, const float *b0,
                                    const float *w1, const float *b1,
                                    const float *w2, const float *b2)
{
    auto layer = [dim](const std::vector<float> &in, const float *W, const float *B,
                       bool relu) {
        std::vector<float> out((size_t)dim);
        for (int o = 0; o < dim; o++) {
            float acc = B[o];
            const float *wr = W + (size_t)o * dim;
            for (int i = 0; i < dim; i++) acc += wr[i] * in[i];
            out[o] = (relu && acc < 0.f) ? 0.f : acc;
        }
        return out;
    };
    std::vector<float> h0(x, x + dim);
    h0 = layer(h0, w0, b0, true);
    h0 = layer(h0, w1, b1, true);
    h0 = layer(h0, w2, b2, false);
    return h0;
}

}  // namespace nvmm
