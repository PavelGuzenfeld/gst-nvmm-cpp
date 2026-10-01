#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace nvmm {

namespace memdims {
constexpr int kTok = 1024, kMem = 64, kHid = 256, kMask = 7, kPtr = 16;
constexpr int kPtrTok = kHid / kMem;
constexpr int kObjTok = kPtr * kPtrTok;
constexpr int kTotal = kMask * kTok + kObjTok;
constexpr float kTdiffMax = kPtr - 1;
}

inline void get_1d_sine_pe(float pos, int dim, float *out, float temp = 10000.f)
{
    const int pe_dim = dim / 2;
    for (int i = 0; i < pe_dim; i++) {
        const float dim_t = std::pow(temp, (float)(2 * (i / 2)) / pe_dim);
        const float v = pos / dim_t;
        out[i] = std::sin(v);
        out[pe_dim + i] = std::cos(v);
    }
}

struct MemConsts {
    const float *cond_maskmem_pos;
    const float *maskmem_tpos_enc;
    const float *obj_ptr_tpos_proj_w;
    const float *obj_ptr_tpos_proj_b;
};

inline void assemble_memory(const float *const *maskmem, const float *const *objptr,
                            const float *pos_list, const MemConsts &c,
                            float *memory, float *memory_pos)
{
    using namespace memdims;
    for (int s = 0; s < kMask; s++) {
        const float *feat = maskmem[s];
        const float *tpos = c.maskmem_tpos_enc + (size_t)(kMask - s - 1) * kMem;
        for (int i = 0; i < kTok; i++) {
            float *mrow = memory     + (size_t)(s * kTok + i) * kMem;
            float *prow = memory_pos + (size_t)(s * kTok + i) * kMem;
            for (int ch = 0; ch < kMem; ch++) {
                mrow[ch] = feat[(size_t)ch * kTok + i];
                prow[ch] = c.cond_maskmem_pos[(size_t)ch * kTok + i] + tpos[ch];
            }
        }
    }
    float sine[kHid];
    for (int p = 0; p < kPtr; p++) {
        get_1d_sine_pe(pos_list[p] / kTdiffMax, kHid, sine);
        float opos[kMem];
        for (int o = 0; o < kMem; o++) {
            float acc = c.obj_ptr_tpos_proj_b[o];
            const float *wr = c.obj_ptr_tpos_proj_w + (size_t)o * kHid;
            for (int in = 0; in < kHid; in++) acc += wr[in] * sine[in];
            opos[o] = acc;
        }
        for (int k = 0; k < kPtrTok; k++) {
            const int row = kMask * kTok + p * kPtrTok + k;
            float *mrow = memory     + (size_t)row * kMem;
            float *prow = memory_pos + (size_t)row * kMem;
            for (int ch = 0; ch < kMem; ch++) {
                mrow[ch] = objptr[p][(size_t)k * kMem + ch];
                prow[ch] = opos[ch];
            }
        }
    }
}

}
