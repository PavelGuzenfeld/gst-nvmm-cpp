#include "samurai_tracker.hpp"

#include "samurai_view.hpp"
#include "samurai_consts.hpp"
#include "samurai_seed_math.hpp"
#include "samurai_memory.hpp"
#include "samurai_kernels.hpp"
#include "samurai_gmc.hpp"
#include "phase_correlation.hpp"
#include "gmc_vpi_fft.hpp"
#include "gmc_vpi_pva.hpp"
#include "gmc_mask.hpp"
#include "kalman_box.hpp"
#include "vit_grid.hpp"

#include <nvbufsurftransform.h>
#include "trt_engine.hpp"
#include "roi_preprocess.hpp"

#include <gst/gst.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

GST_DEBUG_CATEGORY_EXTERN(gst_nvmm_samurai_debug);
#define GST_CAT_DEFAULT gst_nvmm_samurai_debug

namespace nvmm {

/// SAM2 preprocess_image normalization, RGB order: y = (x/255 - mean) / std.
static const float kSamMean[3] = {0.485f, 0.456f, 0.406f};
static const float kSamStd[3]  = {0.229f, 0.224f, 0.225f};

namespace {
float *find_out(const std::vector<std::pair<std::string, float *>> &v, const char *n)
{
    for (auto &p : v)
        if (p.first == n) return p.second;
    return nullptr;
}

/// Matches SAM2 BoundingBox.iou (w = x2 - x1 convention).
float box_iou(const MaskBox &a, const MaskBox &b)
{
    if (!a.valid || !b.valid) return 0.f;
    const float ax2 = a.x + a.w, ay2 = a.y + a.h, bx2 = b.x + b.w, by2 = b.y + b.h;
    const float ix = std::fmax(0.f, std::fmin(ax2, bx2) - std::fmax(a.x, b.x));
    const float iy = std::fmax(0.f, std::fmin(ay2, by2) - std::fmax(a.y, b.y));
    const float inter = ix * iy;
    const float uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

void dump_bin(const std::string &dir, const char *name, const float *d_ptr,
              size_t bytes, cudaStream_t stream)
{
    std::vector<float> host(bytes / sizeof(float));
    if (cudaMemcpyAsync(host.data(), d_ptr, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return;
    cudaStreamSynchronize(stream);
    std::string path = dir + "/" + name + ".bin";
    if (FILE *f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(host.data(), 1, bytes, f);
        std::fclose(f);
        GST_DEBUG("dumped %s (%zu bytes)", path.c_str(), bytes);
    }
}
}

struct SamuraiTracker::Impl {
    cudaStream_t stream = nullptr;
    std::unique_ptr<TrtEngine> encoder, prompt, decoder, mem_encoder, mem_attn;

    RoiPreprocessor pre;
    float *d_crop = nullptr;
    std::vector<std::pair<std::string, float *>> enc_out;
    /// Views into enc_out at a 512 crop: out6 image_embed 1x256x32x32, out4 feat_s0
    /// 1x256x128x128, out5 feat_s1 1x256x64x64, out3 curr_pos 1x256x32x32.
    float *d_image_embed = nullptr;
    float *d_feat_s0 = nullptr;
    float *d_feat_s1 = nullptr;
    float *d_curr_pos = nullptr;

    SamuraiConsts consts;

    float *d_coords = nullptr;
    float *d_psparse = nullptr;
    float *d_pdense = nullptr;
    float *d_image_pe = nullptr;
    float *d_dense_const = nullptr;
    float *d_empty_sparse = nullptr;
    float *d_dmasks = nullptr;
    float *d_dious = nullptr;
    float *d_dtokens = nullptr;
    float *d_dobj = nullptr;
    float *d_mem_mask = nullptr;
    float *d_maskmem_feat = nullptr;
    float *d_maskmem_pos = nullptr;
    /// Decoder image_embed in (C,HW): seed = out6 + no_mem_embed,
    /// tracking = memory_attention output.
    float *d_dec_embed = nullptr;
    float *d_no_mem = nullptr;
    float *d_high = nullptr;
    int   *d_box = nullptr;
    std::vector<float> no_mem_embed;

    /// memory_attention I/O in (HW,1,C): curr/curr_pos are out6/out3 transposed;
    /// memory/memory_pos are (7*tok+64,1,64).
    float *d_ma_curr = nullptr;
    float *d_ma_curr_pos = nullptr;
    float *d_ma_memory = nullptr;
    float *d_ma_memory_pos = nullptr;
    float *d_ma_attn = nullptr;

    std::vector<float> cond_maskmem_feat, cond_maskmem_pos, cond_obj_ptr;
    float cond_obj_score = 0.f, cond_best_iou = 0.f;
    bool has_cond = false;

    /// The maskmem ring lives on device (D2D, no host round-trip); the obj_ptr ring
    /// is tiny and stays host, packed and uploaded each full frame.
    std::vector<std::vector<float>> ring_objptr;
    float *d_cond_maskmem = nullptr;
    float *d_ring_bufs[6] = {nullptr};
    std::vector<float *> ring_dev;
    int ring_write = 0;
    float *d_mm_ptrs = nullptr;
    float *d_objptr_packed = nullptr;
    float *d_pos_list = nullptr;
    float *d_const_maskmem_pos = nullptr;
    float *d_tpos = nullptr;
    float *d_tposproj_w = nullptr;
    float *d_tposproj_b = nullptr;

    KalmanBox kf;
    bool gmc_enabled = false;
    GmcBackend gmc_backend = GmcBackend::Ncc;
    NvBufSurface *gmc_surf = nullptr;
    std::vector<uint8_t> gmc_prev, gmc_curr;
    std::vector<uint8_t> gmc_pm, gmc_cm;
    std::vector<float>   gmc_pf, gmc_cf;
    std::unique_ptr<PhaseCorrelator> gmc_pc;
    std::unique_ptr<GmcVpiFft> gmc_fft;
    std::unique_ptr<GmcVpiPva> gmc_pva;
    bool gmc_have_prev = false;
    float gmc_scale = 1.f;
    /// Patch side, set from the backend at init: 128 for ncc/fft, 256 for the PVA Harris minimum.
    int   gmc_n_ = 128;
    /// NCC search radius tuned for the 128-px patch; rescale if gmc_n_ changes.
    static constexpr int kGmcSearch = 24;
    /// Dead-band in frame px: residual sub-px FFT jitter on a static camera would
    /// otherwise accumulate as drift. Real camera motion is far larger.
    static constexpr float kGmcMinShift = 0.5f;

    int stable_frames = 0;
    int stable_frames_threshold = 10;
    float kf_score_weight = 0.25f;
    float iou_threshold   = 0.5f;
    float kf_min_area     = 25.f;
    int64_t frame_idx = 0;

    int crop_size = 512;
    /// The engine set must be exported for crop_size: grid = crop/16, tokens = grid^2,
    /// memory bank rows = 7*tok+64, low-res mask = crop/4.
    int grid() const { return vit_grid_side(crop_size, 16); }
    int tok()  const { return vit_grid_tokens(crop_size, 16); }
    int mlow() const { return crop_size / 4; }
    SamuraiView view{};
    TrackBox last;

    std::string dump_dir;
    bool dump_done = false;

    bool run_encoder(NvBufSurface *frame, const TrackBox &box, std::string &err);
    bool seed_cond_frame(const TrackBox &box, std::string &err);
    bool track_frame(TrackResult &out, std::string &err);
    void apply_gmc(NvBufSurface *frame);
    GmcShift gmc_estimate(const uint8_t *prev, const uint8_t *curr);
};

/// Every backend returns content motion prev->curr: curr[y,x] ~= prev[y-dy, x-dx].
/// A flipped sign doubles camera motion through kf.shift instead of cancelling it.
GmcShift SamuraiTracker::Impl::gmc_estimate(const uint8_t *prev, const uint8_t *curr)
{
    switch (gmc_backend) {
        case GmcBackend::Pva:
            return gmc_pva->estimate(prev, curr);
        case GmcBackend::FftCuda: {
            const PhaseShift s = gmc_fft->estimate(prev, curr);
            GmcShift out;
            out.dx = (float)s.x; out.dy = (float)s.y; out.conf = (float)s.response;
            return out;
        }
        case GmcBackend::FftCpu: {
            const size_t n2 = (size_t)gmc_n_ * gmc_n_;
            gmc_pf.resize(n2); gmc_cf.resize(n2);
            for (size_t i = 0; i < n2; i++) {
                gmc_pf[i] = (float)prev[i];
                gmc_cf[i] = (float)curr[i];
            }
            const PhaseCorrelator::Shift s = gmc_pc->correlate(gmc_pf.data(), gmc_cf.data());
            GmcShift out;
            out.dx = (float)s.x; out.dy = (float)s.y; out.conf = (float)s.response;
            return out;
        }
        case GmcBackend::Ncc:
        default:
            return estimate_shift(prev, curr, gmc_n_, kGmcSearch);
    }
}

void SamuraiTracker::Impl::apply_gmc(NvBufSurface *frame)
{
    if (!gmc_surf) return;
    const int W = (int)frame->surfaceList[0].width, H = (int)frame->surfaceList[0].height;
    const int sq = W < H ? W : H;
    NvBufSurfTransformRect src{(uint32_t)((H - sq) / 2), (uint32_t)((W - sq) / 2),
                               (uint32_t)sq, (uint32_t)sq};
    NvBufSurfTransformParams p{};
    p.transform_flag = NVBUFSURF_TRANSFORM_CROP_SRC | NVBUFSURF_TRANSFORM_FILTER;
    p.transform_filter = NvBufSurfTransformInter_Bilinear;
    p.src_rect = &src;
    if (NvBufSurfTransform(frame, gmc_surf, &p) != NvBufSurfTransformError_Success) return;
    if (NvBufSurfaceMap(gmc_surf, 0, 0, NVBUF_MAP_READ) != 0) return;
    NvBufSurfaceSyncForCpu(gmc_surf, 0, 0);
    const uint8_t *y = (const uint8_t *)gmc_surf->surfaceList[0].mappedAddr.addr[0];
    const int pitch = gmc_surf->surfaceList[0].planeParams.pitch[0];
    for (int r = 0; r < gmc_n_; r++)
        std::memcpy(gmc_curr.data() + (size_t)r * gmc_n_, y + (size_t)r * pitch, gmc_n_);
    NvBufSurfaceUnMap(gmc_surf, 0, 0);
    gmc_scale = (float)sq / gmc_n_;

    if (gmc_have_prev) {
        /// Mask the target out of both patches so its own motion is not read as camera
        /// motion; FFT backends otherwise report a spurious shift on a static camera.
        /// Raw prev/curr stay unmasked for the next-frame swap.
        const uint8_t *pe = gmc_prev.data(), *ce = gmc_curr.data();
        const GmcMaskBox mb = gmc_map_box_to_patch(last.left, last.top, last.width, last.height,
                                                    W, H, sq, gmc_n_);
        if (mb.overlaps) {
            gmc_pm.assign(gmc_prev.begin(), gmc_prev.end());
            gmc_cm.assign(gmc_curr.begin(), gmc_curr.end());
            gmc_mask_box_to_mean(gmc_pm.data(), gmc_n_, mb.x0, mb.y0, mb.x1, mb.y1);
            gmc_mask_box_to_mean(gmc_cm.data(), gmc_n_, mb.x0, mb.y0, mb.x1, mb.y1);
            pe = gmc_pm.data(); ce = gmc_cm.data();
        }
        const GmcShift s = gmc_estimate(pe, ce);
        /// Confidence is peak NCC in [-1,1] for ncc, the phase-correlation response for
        /// fft, and the fraction of corners tracked for pva.
        float min_conf = 0.05f;
        if (gmc_backend == GmcBackend::Ncc) min_conf = 0.3f;
        else if (gmc_backend == GmcBackend::Pva) min_conf = 0.3f;
        if (s.conf > min_conf) {
            const float dx = s.dx * gmc_scale, dy = s.dy * gmc_scale;
            if (std::fabs(dx) >= kGmcMinShift || std::fabs(dy) >= kGmcMinShift) {
                last.left += dx; last.top += dy;
                kf.shift(dx, dy);
                GST_LOG("gmc[%s] shift (%.1f,%.1f) conf=%.3f",
                        gmc_backend_name(gmc_backend), dx, dy, s.conf);
            }
        }
    }
    gmc_curr.swap(gmc_prev);
    gmc_have_prev = true;
}

bool SamuraiTracker::Impl::seed_cond_frame(const TrackBox &box, std::string &err)
{
    const int M = mlow(), HI = crop_size, TOK = tok(), MM = 64 * TOK;
    constexpr int T = 256;
    /// Crop coords with no scale: the crop equals the export's image_size.
    const float coords[4] = {box.left - view.x,             box.top - view.y,
                             box.left + box.width - view.x, box.top + box.height - view.y};
    cudaMemcpyAsync(d_coords, coords, sizeof(coords), cudaMemcpyHostToDevice, stream);
    if (!prompt->infer(stream)) { err = "prompt_encoder infer failed"; return false; }
    k_add_per_channel(d_image_embed, d_no_mem, d_dec_embed, 256, TOK, stream);
    decoder->bind("sparse", d_psparse);
    decoder->bind("dense", d_pdense);
    if (!decoder->set_input_shape("sparse", {1, 3, T})) { err = "set sparse shape"; return false; }
    if (!decoder->infer(stream)) { err = "mask_decoder infer failed"; return false; }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { err = "decoder sync"; return false; }
    float ious[4] = {0}, obj_score = 0.f;
    cudaMemcpy(ious, d_dious, sizeof(ious), cudaMemcpyDeviceToHost);
    cudaMemcpy(&obj_score, d_dobj, sizeof(float), cudaMemcpyDeviceToHost);
    /// A box prompt (2 pts > multimask_max_pt_num=1) runs multimask_output=False, so
    /// the seed takes mask token 0, not the 4-candidate argmax of the tracking path.
    const int best = 0;
    /// SAM2 min_obj_score_logits = -1.
    const bool appearing = obj_score > -1.0f;
    k_bilinear(d_dmasks + (size_t)best * M * M, d_high, M, M, HI, HI, stream);
    const int initbox[4] = {HI, HI, -1, -1};
    cudaMemcpyAsync(d_box, initbox, sizeof(initbox), cudaMemcpyHostToDevice, stream);
    k_mask_bbox(d_high, HI, HI, d_box, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) { err = "bbox sync"; return false; }
    int bx[4]; cudaMemcpy(bx, d_box, sizeof(bx), cudaMemcpyDeviceToHost);
    MaskBox mb{};
    if (appearing && bx[2] >= 0) {
        mb.x = (float)bx[0]; mb.y = (float)bx[1];
        mb.w = (float)(bx[2] - bx[0]); mb.h = (float)(bx[3] - bx[1]); mb.valid = true;
    }
    last = box;
    if (mb.valid) {
        last.left = mb.x + view.x; last.top = mb.y + view.y;
        last.width = mb.w; last.height = mb.h;
    }
    last.score = obj_score; last.valid = true;
    std::vector<float> token(T);
    cudaMemcpy(token.data(), d_dtokens + (size_t)best * T, T * sizeof(float),
               cudaMemcpyDeviceToHost);
    cond_obj_ptr = mlp3_relu(token.data(), T,
        consts.data("obj_ptr_proj.layers.0.weight"), consts.data("obj_ptr_proj.layers.0.bias"),
        consts.data("obj_ptr_proj.layers.1.weight"), consts.data("obj_ptr_proj.layers.1.bias"),
        consts.data("obj_ptr_proj.layers.2.weight"), consts.data("obj_ptr_proj.layers.2.bias"));
    const float lambda = appearing ? 1.f : 0.f;
    const float *no_ptr = consts.data("no_obj_ptr");
    for (int i = 0; i < T; i++)
        cond_obj_ptr[i] = lambda * cond_obj_ptr[i] + (1.f - lambda) * no_ptr[i];
    /// Seed memory mask: binarize_mask_from_pts * 20 - 10.
    k_threshold_scale(d_high, d_mem_mask, HI * HI, 10.f, -10.f, stream);
    if (!mem_encoder->infer(stream)) { err = "memory_encoder infer failed"; return false; }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { err = "memenc sync"; return false; }
    if (appearing) {
        cudaMemcpy(d_cond_maskmem, d_maskmem_feat, MM * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        cond_maskmem_feat.resize(MM);
        cudaMemcpy(cond_maskmem_feat.data(), d_maskmem_feat, MM * sizeof(float), cudaMemcpyDeviceToHost);
        const float *noe = consts.data("no_obj_embed_spatial");
        for (int c = 0; c < 64; c++)
            for (int s = 0; s < TOK; s++)
                cond_maskmem_feat[(size_t)c * TOK + s] += noe[c];
        cudaMemcpy(d_cond_maskmem, cond_maskmem_feat.data(), MM * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_const_maskmem_pos, d_maskmem_pos, MM * sizeof(float), cudaMemcpyDeviceToDevice);
    cond_obj_score = obj_score; cond_best_iou = ious[best]; has_cond = true;
    if (mb.valid) kf.initiate(last.left + last.width / 2.f, last.top + last.height / 2.f,
                              last.width, last.height);
    stable_frames = 1;
    frame_idx = 0;
    ring_dev.clear();
    ring_write = 0;
    ring_objptr.clear();
    if (!dump_dir.empty()) {
        auto dh = [&](const char *n, const std::vector<float> &v) {
            if (FILE *f = std::fopen((dump_dir + "/" + n + ".bin").c_str(), "wb")) {
                std::fwrite(v.data(), sizeof(float), v.size(), f); std::fclose(f);
            }
        };
        std::vector<float> high((size_t)HI * HI), cmm(MM);
        cudaMemcpy(high.data(), d_high, high.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cmm.data(), d_cond_maskmem, MM * sizeof(float), cudaMemcpyDeviceToHost);
        dh("seed_token", token); dh("seed_obj_ptr", cond_obj_ptr);
        dh("seed_high", high); dh("seed_maskmem_feat", cmm);
    }
    GST_DEBUG("seed cond frame: best=%d iou=%.3f obj=%.3f box(%.0f,%.0f %.0fx%.0f) maskpx=%s",
             best, ious[best], obj_score, last.left, last.top, last.width, last.height,
             mb.valid ? "yes" : "EMPTY");
    return true;
}

bool SamuraiTracker::Impl::track_frame(TrackResult &out, std::string &err)
{
    using namespace memdims;
    const int M = mlow(), HI = crop_size, HW = tok();
    constexpr int CH = kHid;
    frame_idx++;

#ifdef SAMURAI_HOST_OPS
    { std::vector<float> o6((size_t)CH*HW), o3((size_t)CH*HW), cu((size_t)HW*CH), cp((size_t)HW*CH);
      cudaMemcpy(o6.data(), d_image_embed, o6.size()*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(o3.data(), d_curr_pos, o3.size()*sizeof(float), cudaMemcpyDeviceToHost);
      for (int c=0;c<CH;c++) for (int i=0;i<HW;i++){ cu[(size_t)i*CH+c]=o6[(size_t)c*HW+i]; cp[(size_t)i*CH+c]=o3[(size_t)c*HW+i]; }
      cudaMemcpyAsync(d_ma_curr, cu.data(), cu.size()*sizeof(float), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(d_ma_curr_pos, cp.data(), cp.size()*sizeof(float), cudaMemcpyHostToDevice, stream); }
#else
    k_transpose(d_image_embed, d_ma_curr, CH, HW, stream);
    k_transpose(d_curr_pos, d_ma_curr_pos, CH, HW, stream);
#endif

    const float *mm[kMask];
    mm[0] = d_cond_maskmem;
    const int nm = (int)ring_dev.size();
    for (int s = 1; s < kMask; s++) {
        const int from_newest = kMask - 1 - s;
        mm[s] = (from_newest < nm) ? ring_dev[nm - 1 - from_newest] : d_cond_maskmem;
    }
    cudaMemcpyAsync(d_mm_ptrs, mm, sizeof(mm), cudaMemcpyHostToDevice, stream);
    std::vector<float> objpack((size_t)kPtr * kHid);
    float pos_list[kPtr];
    std::copy(cond_obj_ptr.begin(), cond_obj_ptr.end(), objpack.begin());
    pos_list[0] = (float)frame_idx;
    const int no = (int)ring_objptr.size();
    for (int p = 1; p < kPtr; p++) {
        pos_list[p] = (float)p;
        const int from_newest = p - 1;
        const std::vector<float> &src = (from_newest < no) ? ring_objptr[no - 1 - from_newest] : cond_obj_ptr;
        std::copy(src.begin(), src.end(), objpack.begin() + (size_t)p * kHid);
    }
    cudaMemcpyAsync(d_objptr_packed, objpack.data(), objpack.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pos_list, pos_list, sizeof(pos_list), cudaMemcpyHostToDevice, stream);
    k_assemble_memory(reinterpret_cast<const float *const *>(d_mm_ptrs), d_objptr_packed,
                      d_pos_list, d_const_maskmem_pos, d_tpos, d_tposproj_w, d_tposproj_b,
                      d_ma_memory, d_ma_memory_pos, HW, stream);

    if (!mem_attn->infer(stream)) { err = "memory_attention infer failed"; return false; }
#ifdef SAMURAI_HOST_OPS
    { if (cudaStreamSynchronize(stream) != cudaSuccess) { err = "memattn sync"; return false; }
      std::vector<float> at((size_t)HW*CH), de((size_t)CH*HW);
      cudaMemcpy(at.data(), d_ma_attn, at.size()*sizeof(float), cudaMemcpyDeviceToHost);
      for (int c=0;c<CH;c++) for (int i=0;i<HW;i++) de[(size_t)c*HW+i]=at[(size_t)i*CH+c];
      cudaMemcpyAsync(d_dec_embed, de.data(), de.size()*sizeof(float), cudaMemcpyHostToDevice, stream); }
#else
    k_transpose(d_ma_attn, d_dec_embed, HW, CH, stream);
#endif

    decoder->bind("sparse", d_empty_sparse);
    decoder->bind("dense", d_dense_const);
    if (!decoder->set_input_shape("sparse", {1, 2, kHid})) { err = "set sparse"; return false; }
    if (!decoder->infer(stream)) { err = "mask_decoder infer failed"; return false; }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { err = "decoder sync"; return false; }

    float ious[4] = {0}, obj_score = 0.f;
    cudaMemcpy(ious, d_dious, sizeof(ious), cudaMemcpyDeviceToHost);
    cudaMemcpy(&obj_score, d_dobj, sizeof(float), cudaMemcpyDeviceToHost);
    /// A degenerate memory-attention frame can emit non-finite objectness. Coast and
    /// skip the memory update: one NaN maskmem poisons every later assemble.
    const bool finite_obj = std::isfinite(obj_score);
    const bool appearing = finite_obj && obj_score > -1.0f;

    /// Candidates are un-viewed to frame coords: the KF runs in frame coords so a
    /// recentered crop does not move it.
    MaskBox cbox[3];
    float ciou[3] = {ious[1], ious[2], ious[3]};
#ifdef SAMURAI_HOST_OPS
    std::vector<float> cand_high[3];
    { std::vector<float> low((size_t)M * M);
      for (int j = 0; j < 3; j++) {
          cudaMemcpy(low.data(), d_dmasks + (size_t)(1 + j) * M * M, low.size() * sizeof(float), cudaMemcpyDeviceToHost);
          cand_high[j] = bilinear_upsample(low.data(), M, M, HI, HI);
          MaskBox b = appearing ? mask_to_box(cand_high[j].data(), HI, HI, 0.f) : MaskBox{};
          if (b.valid) cbox[j] = MaskBox{b.x + view.x, b.y + view.y, b.w, b.h, true};
      } }
#else
    const int initbox[4] = {HI, HI, -1, -1};
    for (int j = 0; j < 3; j++) {
        k_bilinear(d_dmasks + (size_t)(1 + j) * M * M, d_high, M, M, HI, HI, stream);
        cudaMemcpyAsync(d_box, initbox, sizeof(initbox), cudaMemcpyHostToDevice, stream);
        k_mask_bbox(d_high, HI, HI, d_box, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) { err = "cand bbox sync"; return false; }
        int bx[4]; cudaMemcpy(bx, d_box, sizeof(bx), cudaMemcpyDeviceToHost);
        if (appearing && bx[2] >= 0)
            cbox[j] = MaskBox{bx[0] + view.x, bx[1] + view.y, (float)(bx[2] - bx[0]),
                              (float)(bx[3] - bx[1]), true};
    }
#endif

    /// SamuraiSelector, sam2_base.py:430-511.
    int sel = 0;
    MaskBox kfbox;
    auto argmax3 = [](const float *v) { int b = 0; if (v[1] > v[b]) b = 1; if (v[2] > v[b]) b = 2; return b; };
    if (!kf.initiated() || stable_frames == 0) {
        sel = argmax3(ciou);
        if (cbox[sel].valid) {
            kf.initiate(cbox[sel].x + cbox[sel].w / 2.f, cbox[sel].y + cbox[sel].h / 2.f,
                        cbox[sel].w, cbox[sel].h);
            stable_frames = 1;
        }
    } else {
        kf.predict(1.0);
        double kcx, kcy, kw, kh; kf.box(kcx, kcy, kw, kh);
        kfbox = MaskBox{(float)(kcx - kw / 2), (float)(kcy - kh / 2), (float)kw, (float)kh, true};
        const float kf_area = (float)(kw * kh);
        if (stable_frames < stable_frames_threshold) {
            sel = argmax3(ciou);
        } else {
            const float kw = kf_score_weight;
            float w[3];
            for (int j = 0; j < 3; j++)
                w[j] = kw * box_iou(kfbox, cbox[j]) + (1.f - kw) * ciou[j];
            sel = argmax3(w);
        }
        if (ciou[sel] > iou_threshold && kf_area > kf_min_area && cbox[sel].valid) {
            kf.update(cbox[sel].x + cbox[sel].w / 2.f, cbox[sel].y + cbox[sel].h / 2.f,
                      cbox[sel].w, cbox[sel].h);
            stable_frames++;
        } else {
            stable_frames = 0;
        }
    }

    out = TrackResult{};
    if (cbox[sel].valid) {
        out.box.left = cbox[sel].x; out.box.top = cbox[sel].y;
        out.box.width = cbox[sel].w; out.box.height = cbox[sel].h;
        out.box.score = obj_score; out.box.valid = true;
        last = out.box;
    } else {
        out.box = last;
    }
    if (kfbox.valid) {
        out.kf_box.left = kfbox.x; out.kf_box.top = kfbox.y;
        out.kf_box.width = kfbox.w; out.kf_box.height = kfbox.h; out.kf_box.valid = true;
    }
    out.stable_frames = (uint32_t)stable_frames;
    out.target_id = 1;

    if (!finite_obj) {
        GST_WARNING("track f=%ld: non-finite objectness, coasting (skipped memory update)",
                    (long)frame_idx);
        return true;
    }

    std::vector<float> token(CH);
    cudaMemcpy(token.data(), d_dtokens + (size_t)(1 + sel) * CH, CH * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> optr = mlp3_relu(token.data(), CH,
        consts.data("obj_ptr_proj.layers.0.weight"), consts.data("obj_ptr_proj.layers.0.bias"),
        consts.data("obj_ptr_proj.layers.1.weight"), consts.data("obj_ptr_proj.layers.1.bias"),
        consts.data("obj_ptr_proj.layers.2.weight"), consts.data("obj_ptr_proj.layers.2.bias"));
    const float lambda = appearing ? 1.f : 0.f;
    const float *no_ptr = consts.data("no_obj_ptr");
    for (int i = 0; i < CH; i++) optr[i] = lambda * optr[i] + (1.f - lambda) * no_ptr[i];

#ifdef SAMURAI_HOST_OPS
    { std::vector<float> mem((size_t)HI * HI); const float *h = cand_high[sel].data();
      for (size_t i = 0; i < mem.size(); i++) mem[i] = (1.f / (1.f + std::exp(-h[i]))) * 20.f - 10.f;
      cudaMemcpyAsync(d_mem_mask, mem.data(), mem.size() * sizeof(float), cudaMemcpyHostToDevice, stream); }
#else
    k_bilinear(d_dmasks + (size_t)(1 + sel) * M * M, d_high, M, M, HI, HI, stream);
    /// Tracking memory mask is sigmoid(mask) * 20 - 10, not the seed's binarize.
    k_sigmoid_scale(d_high, d_mem_mask, HI * HI, 20.f, -10.f, stream);
#endif
    if (!mem_encoder->infer(stream)) { err = "memory_encoder infer failed"; return false; }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { err = "memenc sync"; return false; }
    float *slot = d_ring_bufs[ring_write];
    if (appearing) {
        cudaMemcpy(slot, d_maskmem_feat, (size_t)kMem * HW * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        std::vector<float> nf((size_t)kMem * HW);
        cudaMemcpy(nf.data(), d_maskmem_feat, nf.size() * sizeof(float), cudaMemcpyDeviceToHost);
        const float *noe = consts.data("no_obj_embed_spatial");
        for (int c = 0; c < kMem; c++)
            for (int s = 0; s < HW; s++) nf[(size_t)c * HW + s] += noe[c];
        cudaMemcpy(slot, nf.data(), nf.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    ring_write = (ring_write + 1) % (kMask - 1);
    ring_dev.push_back(slot);
    if ((int)ring_dev.size() > kMask - 1) ring_dev.erase(ring_dev.begin());
    ring_objptr.push_back(std::move(optr));
    if ((int)ring_objptr.size() > kPtr - 1) ring_objptr.erase(ring_objptr.begin());
    GST_LOG("track f=%ld sel=%d obj=%.2f stable=%d box(%.0f,%.0f %.0fx%.0f)%s",
              (long)frame_idx, sel, obj_score, stable_frames,
              out.box.left, out.box.top, out.box.width, out.box.height,
              appearing ? "" : " OCCLUDED");
    return true;
}

bool SamuraiTracker::Impl::run_encoder(NvBufSurface *frame, const TrackBox &box,
                                       std::string &err)
{
    const int fw = (int)frame->surfaceList[0].width;
    const int fh = (int)frame->surfaceList[0].height;
    view = get_view_around_bbox(box.left, box.top, box.width, box.height,
                                crop_size, fw, fh);

    if (!pre.run(frame, view.x, view.y, view.width, view.height, d_crop, err))
        return false;
    if (!encoder->infer(stream)) {
        err = "image_encoder infer failed";
        return false;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        err = "encoder stream sync failed";
        return false;
    }

    if (!dump_dir.empty() && !dump_done) {
        const size_t crop_bytes = (size_t)3 * crop_size * crop_size * sizeof(float);
        dump_bin(dump_dir, "crop_input", d_crop, crop_bytes, stream);
        for (auto &p : enc_out) {
            const TensorInfo *ti = nullptr;
            for (auto &t : encoder->tensors())
                if (t.name == p.first) { ti = &t; break; }
            if (ti) dump_bin(dump_dir, p.first.c_str(), p.second, ti->bytes, stream);
        }
        dump_done = true;
    }
    return true;
}

SamuraiTracker::SamuraiTracker() : impl_(std::make_unique<Impl>()) {}
SamuraiTracker::~SamuraiTracker()
{
    if (impl_) {
        if (impl_->d_crop) cudaFree(impl_->d_crop);
        for (auto &p : impl_->enc_out)
            if (p.second) cudaFree(p.second);
        for (float *p : {impl_->d_coords, impl_->d_psparse, impl_->d_pdense,
                         impl_->d_image_pe, impl_->d_dense_const, impl_->d_empty_sparse,
                         impl_->d_dmasks, impl_->d_dious, impl_->d_dtokens, impl_->d_dobj,
                         impl_->d_mem_mask, impl_->d_maskmem_feat, impl_->d_maskmem_pos,
                         impl_->d_dec_embed, impl_->d_ma_curr, impl_->d_ma_curr_pos,
                         impl_->d_ma_memory, impl_->d_ma_memory_pos, impl_->d_ma_attn,
                         impl_->d_no_mem, impl_->d_high, impl_->d_cond_maskmem,
                         impl_->d_mm_ptrs, impl_->d_objptr_packed, impl_->d_pos_list,
                         impl_->d_const_maskmem_pos, impl_->d_tpos, impl_->d_tposproj_w,
                         impl_->d_tposproj_b})
            if (p) cudaFree(p);
        for (float *p : impl_->d_ring_bufs) if (p) cudaFree(p);
        if (impl_->d_box) cudaFree(impl_->d_box);
        if (impl_->gmc_surf) NvBufSurfaceDestroy(impl_->gmc_surf);
        if (impl_->stream) cudaStreamDestroy(impl_->stream);
    }
}

static std::unique_ptr<TrtEngine>
load_one(const std::string &dir, const char *name, std::string &err)
{
    std::string path = dir;
    if (!path.empty() && path.back() != '/')
        path += '/';
    path += name;
    std::string e;
    auto eng = TrtEngine::load_file(path, e);
    if (!eng)
        err = std::string("load ") + path + ": " + e;
    return eng;
}

bool SamuraiTracker::init(const SamuraiConfig &cfg, std::string &err)
{
    cfg_ = cfg;
    impl_->crop_size = cfg.crop_size;
    impl_->stable_frames_threshold = cfg.stable_frames_threshold;
    impl_->kf_score_weight = cfg.kf_score_weight;
    impl_->iou_threshold   = cfg.iou_threshold;
    impl_->kf_min_area     = cfg.kf_min_area;
    if (cudaStreamCreate(&impl_->stream) != cudaSuccess) {
        err = "cudaStreamCreate failed";
        return false;
    }
    struct { const char *file; std::unique_ptr<TrtEngine> *slot; } engines[] = {
        {"image_encoder_bplus_512.engine", &impl_->encoder},
        {"prompt_encoder.engine",          &impl_->prompt},
        {"mask_decoder.engine",            &impl_->decoder},
        {"memory_encoder.engine",          &impl_->mem_encoder},
        {"memory_attention.engine",        &impl_->mem_attn},
    };
    for (auto &e : engines) {
        *e.slot = load_one(cfg.engine_dir, e.file, err);
        if (!*e.slot)
            return false;
        GST_INFO("loaded engine %s", e.file);
    }

    if (!impl_->pre.configure(impl_->crop_size, impl_->crop_size, true,
                              1.f / 255.f, kSamMean, kSamStd,
                              impl_->stream, err))
        return false;

    const TensorInfo *in = impl_->encoder->input0();
    if (!in) { err = "encoder has no input tensor"; return false; }
    if (cudaMalloc((void **)&impl_->d_crop, in->bytes) != cudaSuccess) {
        err = "cudaMalloc(encoder input) failed";
        return false;
    }
    if (!impl_->encoder->bind(in->name, impl_->d_crop)) {
        err = "bind encoder input failed";
        return false;
    }
    for (const TensorInfo &t : impl_->encoder->tensors()) {
        if (t.is_input) continue;
        float *d = nullptr;
        if (cudaMalloc((void **)&d, t.bytes) != cudaSuccess) {
            err = "cudaMalloc(encoder output) failed";
            return false;
        }
        if (!impl_->encoder->bind(t.name, d)) {
            err = "bind encoder output " + t.name + " failed";
            return false;
        }
        impl_->enc_out.emplace_back(t.name, d);
    }
    impl_->d_image_embed = find_out(impl_->enc_out, "out6");
    impl_->d_feat_s0     = find_out(impl_->enc_out, "out4");
    impl_->d_feat_s1     = find_out(impl_->enc_out, "out5");
    impl_->d_curr_pos    = find_out(impl_->enc_out, "out3");
    if (!impl_->d_image_embed || !impl_->d_feat_s0 || !impl_->d_feat_s1 || !impl_->d_curr_pos) {
        err = "encoder output names (out3..out6) not found";
        return false;
    }

    if (const char *dd = std::getenv("SAMURAI_DUMP_DIR"))
        impl_->dump_dir = dd;

    if (cfg.consts_file.empty()) {
        err = "consts-file is required (samurai_consts.bin)";
        return false;
    }
    if (!impl_->consts.load(cfg.consts_file, err))
        return false;
    for (const char *need : {"image_pe", "dense_no_mask", "empty_sparse",
                             "maskmem_tpos_enc", "no_mem_embed", "no_obj_ptr",
                             "no_obj_embed_spatial", "obj_ptr_proj.layers.0.weight"}) {
        if (!impl_->consts.get(need)) {
            err = std::string("consts missing tensor: ") + need;
            return false;
        }
    }

    auto bytes_of = [](TrtEngine *e, const char *name) -> size_t {
        for (const TensorInfo &t : e->tensors())
            if (t.name == name) return t.bytes;
        return 0;
    };
    auto alloc_bind = [&](TrtEngine *e, const char *name, float **slot) -> bool {
        size_t b = bytes_of(e, name);
        if (!b) { err = std::string("no tensor ") + name; return false; }
        if (cudaMalloc((void **)slot, b) != cudaSuccess) { err = "cudaMalloc failed"; return false; }
        if (!e->bind(name, *slot)) { err = std::string("bind ") + name + " failed"; return false; }
        return true;
    };
    auto upload_const = [&](const char *cname, float **slot) -> bool {
        const ConstTensor *c = impl_->consts.get(cname);
        if (!c) { err = std::string("const missing ") + cname; return false; }
        if (cudaMalloc((void **)slot, c->count() * sizeof(float)) != cudaSuccess) { err = "cudaMalloc const"; return false; }
        return cudaMemcpy(*slot, c->data.data(), c->count() * sizeof(float),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    };

    if (!alloc_bind(impl_->prompt.get(), "coords", &impl_->d_coords) ||
        !alloc_bind(impl_->prompt.get(), "sparse", &impl_->d_psparse) ||
        !alloc_bind(impl_->prompt.get(), "dense",  &impl_->d_pdense))
        return false;
    if (!upload_const("image_pe", &impl_->d_image_pe) ||
        !upload_const("dense_no_mask", &impl_->d_dense_const) ||
        !upload_const("empty_sparse", &impl_->d_empty_sparse))
        return false;
    if (cudaMalloc((void **)&impl_->d_dec_embed, bytes_of(impl_->decoder.get(), "image_embed")) != cudaSuccess) {
        err = "cudaMalloc(dec_embed) failed"; return false;
    }
    { const ConstTensor *nme = impl_->consts.get("no_mem_embed");
      impl_->no_mem_embed.assign(nme->data.begin(), nme->data.end()); }
    if (cudaMalloc((void **)&impl_->d_no_mem, 256 * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&impl_->d_high, (size_t)impl_->crop_size * impl_->crop_size * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&impl_->d_box, 4 * sizeof(int)) != cudaSuccess) {
        err = "cudaMalloc(kernel scratch) failed"; return false;
    }
    cudaMemcpy(impl_->d_no_mem, impl_->no_mem_embed.data(), 256 * sizeof(float), cudaMemcpyHostToDevice);
    const size_t mm_floats = (size_t)64 * impl_->tok();
    bool ok = cudaMalloc((void **)&impl_->d_cond_maskmem, mm_floats * sizeof(float)) == cudaSuccess
        && cudaMalloc((void **)&impl_->d_mm_ptrs, 7 * sizeof(float *)) == cudaSuccess
        && cudaMalloc((void **)&impl_->d_objptr_packed, (size_t)16 * 256 * sizeof(float)) == cudaSuccess
        && cudaMalloc((void **)&impl_->d_pos_list, 16 * sizeof(float)) == cudaSuccess
        && cudaMalloc((void **)&impl_->d_const_maskmem_pos, mm_floats * sizeof(float)) == cudaSuccess
        && cudaMalloc((void **)&impl_->d_tpos, (size_t)7 * 64 * sizeof(float)) == cudaSuccess
        && cudaMalloc((void **)&impl_->d_tposproj_w, (size_t)64 * 256 * sizeof(float)) == cudaSuccess
        && cudaMalloc((void **)&impl_->d_tposproj_b, 64 * sizeof(float)) == cudaSuccess;
    for (int r = 0; r < 6 && ok; r++)
        ok = cudaMalloc((void **)&impl_->d_ring_bufs[r], mm_floats * sizeof(float)) == cudaSuccess;
    if (!ok) { err = "cudaMalloc(device ring) failed"; return false; }
    cudaMemcpy(impl_->d_tpos, impl_->consts.data("maskmem_tpos_enc"), (size_t)7 * 64 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(impl_->d_tposproj_w, impl_->consts.data("obj_ptr_tpos_proj.weight"), (size_t)64 * 256 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(impl_->d_tposproj_b, impl_->consts.data("obj_ptr_tpos_proj.bias"), 64 * sizeof(float), cudaMemcpyHostToDevice);
    impl_->gmc_enabled = cfg.gmc;
    if (cfg.gmc) {
        /// Probe accelerators only when the request can use one: a PVA payload can start
        /// PVA firmware (seconds) and log VPI errors on boxes without PVA.
        bool have_cuda_fft = false, have_pva = false;
        if (cfg.gmc_backend != GmcBackend::Ncc && cfg.gmc_backend != GmcBackend::FftCpu) {
            have_cuda_fft = GmcVpiFft::available();
            have_pva      = GmcVpiPva::available();
        }
        impl_->gmc_backend = resolve_gmc_backend(cfg.gmc_backend, have_cuda_fft, have_pva);
        impl_->gmc_n_ = gmc_patch_size(impl_->gmc_backend);
        NvBufSurfaceCreateParams cp{};
        cp.width = (uint32_t)impl_->gmc_n_; cp.height = (uint32_t)impl_->gmc_n_;
        cp.colorFormat = NVBUF_COLOR_FORMAT_NV12;
        cp.layout = NVBUF_LAYOUT_PITCH; cp.memType = NVBUF_MEM_DEFAULT; cp.gpuId = 0;
        if (NvBufSurfaceCreate(&impl_->gmc_surf, 1, &cp) != 0) { err = "gmc surface create failed"; return false; }
        impl_->gmc_surf->numFilled = 1;
        impl_->gmc_prev.assign((size_t)impl_->gmc_n_ * impl_->gmc_n_, 0);
        impl_->gmc_curr.assign((size_t)impl_->gmc_n_ * impl_->gmc_n_, 0);
        if (impl_->gmc_backend == GmcBackend::FftCpu)
            impl_->gmc_pc = std::make_unique<PhaseCorrelator>(impl_->gmc_n_, impl_->gmc_n_);
        if (impl_->gmc_backend == GmcBackend::FftCuda) {
            impl_->gmc_fft = std::make_unique<GmcVpiFft>();
            std::string ferr;
            if (!impl_->gmc_fft->init(impl_->gmc_n_, ferr)) { err = "gmc fft-cuda init: " + ferr; return false; }
        }
        if (impl_->gmc_backend == GmcBackend::Pva) {
            impl_->gmc_pva = std::make_unique<GmcVpiPva>();
            std::string perr;
            if (!impl_->gmc_pva->init(impl_->gmc_n_, perr)) { err = "gmc pva init: " + perr; return false; }
        }
        GST_INFO("SamuraiTracker GMC backend=%s patch=%d",
                 gmc_backend_name(impl_->gmc_backend), impl_->gmc_n_);
    }
    if (!impl_->decoder->bind("image_embed", impl_->d_dec_embed) ||
        !impl_->decoder->bind("image_pe", impl_->d_image_pe) ||
        !impl_->decoder->bind("feat_s0", impl_->d_feat_s0) ||
        !impl_->decoder->bind("feat_s1", impl_->d_feat_s1)) {
        err = "bind decoder persistent inputs failed";
        return false;
    }
    if (!alloc_bind(impl_->decoder.get(), "masks", &impl_->d_dmasks) ||
        !alloc_bind(impl_->decoder.get(), "ious", &impl_->d_dious) ||
        !alloc_bind(impl_->decoder.get(), "tokens", &impl_->d_dtokens) ||
        !alloc_bind(impl_->decoder.get(), "obj_score", &impl_->d_dobj))
        return false;
    if (cudaMalloc((void **)&impl_->d_mem_mask, bytes_of(impl_->mem_encoder.get(), "mask")) != cudaSuccess) {
        err = "cudaMalloc(mem mask) failed"; return false;
    }
    if (!impl_->mem_encoder->bind("pix_feat", impl_->d_image_embed) ||
        !impl_->mem_encoder->bind("mask", impl_->d_mem_mask)) {
        err = "bind memory_encoder inputs failed"; return false;
    }
    if (!alloc_bind(impl_->mem_encoder.get(), "maskmem_feat", &impl_->d_maskmem_feat) ||
        !alloc_bind(impl_->mem_encoder.get(), "maskmem_pos", &impl_->d_maskmem_pos))
        return false;
    if (!alloc_bind(impl_->mem_attn.get(), "curr", &impl_->d_ma_curr) ||
        !alloc_bind(impl_->mem_attn.get(), "memory", &impl_->d_ma_memory) ||
        !alloc_bind(impl_->mem_attn.get(), "curr_pos", &impl_->d_ma_curr_pos) ||
        !alloc_bind(impl_->mem_attn.get(), "memory_pos", &impl_->d_ma_memory_pos) ||
        !alloc_bind(impl_->mem_attn.get(), "attn", &impl_->d_ma_attn))
        return false;

    GST_INFO("SamuraiTracker init OK (5 engines, %zu consts, crop=%d, target_class=%d%s)",
             impl_->consts.size(), cfg.crop_size, cfg.target_class,
             impl_->dump_dir.empty() ? "" : ", DUMP on");
    return true;
}

bool SamuraiTracker::seed(NvBufSurface *frame, const TrackBox &box, std::string &err)
{
    impl_->apply_gmc(frame);
    if (!impl_->run_encoder(frame, box, err)) {
        GST_WARNING("seed encoder failed: %s", err.c_str());
        return false;
    }
    if (!impl_->seed_cond_frame(box, err)) {
        GST_WARNING("seed cond-frame failed: %s", err.c_str());
        return false;
    }
    seeded_ = true;
    GST_INFO("SamuraiTracker seeded at (%.0f,%.0f %.0fx%.0f), view (%.0f,%.0f %.0fx%.0f)",
             box.left, box.top, box.width, box.height,
             impl_->view.x, impl_->view.y, impl_->view.width, impl_->view.height);
    return true;
}

bool SamuraiTracker::track(NvBufSurface *frame, bool kf_only, TrackResult &out,
                           std::string &err)
{
    impl_->apply_gmc(frame);
    if (kf_only) {
        out = TrackResult{};
        if (impl_->kf.initiated()) {
            impl_->kf.predict(1.0);
            double cx, cy, w, h; impl_->kf.box(cx, cy, w, h);
            out.box.left = (float)(cx - w / 2);
            out.box.top = (float)(cy - h / 2);
            out.box.width = (float)w; out.box.height = (float)h;
            out.box.score = 1.f; out.box.valid = true;
            out.kf_box = out.box;
            impl_->last = out.box;
        } else {
            out.box = impl_->last;
        }
        out.is_kf_only = true;
        out.stable_frames = (uint32_t)impl_->stable_frames;
        out.target_id = seeded_ ? 1u : 0u;
        return true;
    }
    if (!impl_->run_encoder(frame, impl_->last, err)) {
        GST_WARNING("track encoder failed: %s", err.c_str());
        return false;
    }
    if (!impl_->track_frame(out, err)) {
        GST_WARNING("track_frame failed: %s", err.c_str());
        return false;
    }
    out.is_kf_only = false;
    return true;
}

}
