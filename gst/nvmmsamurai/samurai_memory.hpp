/// samurai_memory.hpp — MemoryBank assembly for the static 7232x1x64 memory +
/// memory_pos that the memory_attention engine consumes. Header-only, host-side,
/// dependency-free (testable off-target). Every transform here is golden-anchored
/// against memattn_real.npz (precheck_mem.py checks A/C/D/E):
///   memory[0:7168]   = 7 maskmem frames (cond, tpos1..6), flatten(2).permute(2,0,1)
///   memory[7168:7232]= 16 obj_ptr split into 4x64 tokens (ptr p token k -> row 4p+k)
///   memory_pos[maskmem] = maskmem_pos_enc(frame-invariant) + maskmem_tpos_enc[6-slot]
///   memory_pos[objptr]  = obj_ptr_tpos_proj( get_1d_sine_pe(pos/15, 256) ), repeat x4
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace nvmm {

// Dims: top-level feature 32x32 (=1024 tokens), mem_dim 64, hidden 256,
// num_maskmem 7, max_obj_ptr 16, obj_ptr split into C/mem_dim = 4 tokens.
namespace memdims {
constexpr int kTok = 1024, kMem = 64, kHid = 256, kMask = 7, kPtr = 16;
constexpr int kPtrTok = kHid / kMem;             // 4 tokens per obj_ptr
constexpr int kObjTok = kPtr * kPtrTok;          // 64
constexpr int kTotal = kMask * kTok + kObjTok;   // 7232
constexpr float kTdiffMax = kPtr - 1;            // 15
}

/// 1D sine PE, exact clone of sam2_utils.get_1d_sine_pe(pos, dim) for one scalar.
/// out has `dim` elements: [sin(pos/dim_t)... , cos(pos/dim_t)...].
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
    const float *cond_maskmem_pos;     // (64*1024) frame-invariant pos, layout c*1024+i
    const float *maskmem_tpos_enc;     // (7*64)
    const float *obj_ptr_tpos_proj_w;  // (64*256) row-major [out,in]
    const float *obj_ptr_tpos_proj_b;  // (64)
};

/// Assemble the static memory + memory_pos.
///   maskmem[s]: 7 pointers to (64*1024) maskmem content (layout c*1024+i), slot
///               0=cond, 1..6 = tpos1..6 (oldest..newest).
///   objptr[p]:  16 pointers to (256) obj_ptr vectors, p=0 is the cond ptr.
///   pos_list[p]: 16 signed temporal positions (cond = frame_idx, others = t_diff).
///   memory, memory_pos: caller-owned (kTotal*64) outputs.
inline void assemble_memory(const float *const *maskmem, const float *const *objptr,
                            const float *pos_list, const MemConsts &c,
                            float *memory, float *memory_pos)
{
    using namespace memdims;
    // ---- maskmem blocks (slots 0..6) ----
    for (int s = 0; s < kMask; s++) {
        const float *feat = maskmem[s];
        const float *tpos = c.maskmem_tpos_enc + (size_t)(kMask - s - 1) * kMem;  // [6-s]
        for (int i = 0; i < kTok; i++) {
            float *mrow = memory     + (size_t)(s * kTok + i) * kMem;
            float *prow = memory_pos + (size_t)(s * kTok + i) * kMem;
            for (int ch = 0; ch < kMem; ch++) {
                mrow[ch] = feat[(size_t)ch * kTok + i];
                prow[ch] = c.cond_maskmem_pos[(size_t)ch * kTok + i] + tpos[ch];
            }
        }
    }
    // ---- obj_ptr blocks (16 ptrs, 4 tokens each) ----
    float sine[kHid];
    for (int p = 0; p < kPtr; p++) {
        get_1d_sine_pe(pos_list[p] / kTdiffMax, kHid, sine);   // (256,)
        float opos[kMem];                                       // proj 256->64
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
                mrow[ch] = objptr[p][(size_t)k * kMem + ch];   // ptr[64k : 64k+64]
                prow[ch] = opos[ch];                            // shared across k
            }
        }
    }
}

}  // namespace nvmm
