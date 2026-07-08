/// gmc_vpi_pva.hpp — GMC pva backend: global translation from VPI HarrisCorners +
/// OpticalFlowPyrLK on the PVA (VPI_BACKEND_PVA), reduced to a median (dx,dy). The
/// classic sparse-optical-flow GMC, offloaded to the PVA engine (frees CPU + GPU).
///
/// Verified pipeline (see probe): Harris on PVA needs an S16 image (convert the U8
/// patch on CUDA); PyrLK needs U8 pyramids (built on CUDA). Harris on PVA also
/// needs input >= 160x120, so the caller uses a 256x256 patch, and minNMSDistance
/// must be 8; PyrLK windowDimension must be 7/9/11 on PVA. conf = fraction of
/// corners successfully tracked. Sign convention matches the other backends:
/// median(cur - prev) = content motion prev -> curr.
///
/// Compiled only when NVMM_HAVE_VPI is defined; else available() is false and the
/// tracker falls back.
#pragma once
#include <cstdint>
#include <string>

#include "samurai_gmc.hpp"  // GmcShift

#ifdef NVMM_HAVE_VPI
#include <algorithm>
#include <cstring>
#include <vector>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/OpticalFlowPyrLK.h>

namespace nvmm {

class GmcVpiPva {
public:
    // Cheap availability probe: create+destroy the PVA Harris + PyrLK payloads.
    static bool available() {
        VPIStream s = nullptr;
        if (vpiStreamCreate(VPI_BACKEND_PVA, &s) != VPI_SUCCESS) return false;
        VPIPayload h = nullptr, lk = nullptr;
        const VPIStatus a = vpiCreateHarrisCornerDetector(VPI_BACKEND_PVA, 256, 256, &h);
        const VPIStatus b = vpiCreateOpticalFlowPyrLK(VPI_BACKEND_PVA, 256, 256,
                                                      VPI_IMAGE_FORMAT_U8, kLevels, kScale, &lk);
        if (h) vpiPayloadDestroy(h);
        if (lk) vpiPayloadDestroy(lk);
        vpiStreamDestroy(s);
        return a == VPI_SUCCESS && b == VPI_SUCCESS;
    }

    ~GmcVpiPva() { destroy(); }

    bool init(int n, std::string &err) {
        n_ = n;
        const uint64_t B = VPI_BACKEND_PVA | VPI_BACKEND_CPU | VPI_BACKEND_CUDA;
        if (vpiStreamCreate(B, &stream_) != VPI_SUCCESS) { err = "vpiStreamCreate(PVA|CPU|CUDA)"; return false; }
        if (vpiImageCreate(n, n, VPI_IMAGE_FORMAT_U8, B, &prev_u8_) != VPI_SUCCESS ||
            vpiImageCreate(n, n, VPI_IMAGE_FORMAT_U8, B, &cur_u8_) != VPI_SUCCESS ||
            vpiImageCreate(n, n, VPI_IMAGE_FORMAT_S16, B, &prev_s16_) != VPI_SUCCESS) { err = "vpiImageCreate"; return false; }
        if (vpiCreateHarrisCornerDetector(VPI_BACKEND_PVA, n, n, &harris_) != VPI_SUCCESS) { err = "vpiCreateHarrisCornerDetector(PVA)"; return false; }
        if (vpiCreateOpticalFlowPyrLK(VPI_BACKEND_PVA, n, n, VPI_IMAGE_FORMAT_U8, kLevels, kScale, &lk_) != VPI_SUCCESS) { err = "vpiCreateOpticalFlowPyrLK(PVA)"; return false; }
        if (vpiArrayCreate(kCap, VPI_ARRAY_TYPE_KEYPOINT_F32, B, &kp_prev_) != VPI_SUCCESS ||
            vpiArrayCreate(kCap, VPI_ARRAY_TYPE_U32, B, &scores_) != VPI_SUCCESS ||
            vpiArrayCreate(kCap, VPI_ARRAY_TYPE_KEYPOINT_F32, B, &kp_cur_) != VPI_SUCCESS ||
            vpiArrayCreate(kCap, VPI_ARRAY_TYPE_U8, B, &status_) != VPI_SUCCESS) { err = "vpiArrayCreate"; return false; }
        if (vpiPyramidCreate(n, n, VPI_IMAGE_FORMAT_U8, kLevels, kScale, B, &pyr_p_) != VPI_SUCCESS ||
            vpiPyramidCreate(n, n, VPI_IMAGE_FORMAT_U8, kLevels, kScale, B, &pyr_c_) != VPI_SUCCESS) { err = "vpiPyramidCreate"; return false; }
        return true;
    }

    // Global translation (median of tracked corner displacements), same sign as
    // PhaseCorrelator: content motion prev -> curr. conf = tracked fraction [0,1].
    GmcShift estimate(const uint8_t *prev, const uint8_t *curr) {
        GmcShift out;  // {0,0,0} on any failure -> gated out by the caller
        fill_u8(prev_u8_, prev);
        fill_u8(cur_u8_, curr);
        VPIHarrisCornerDetectorParams hp;
        vpiInitHarrisCornerDetectorParams(&hp);
        hp.minNMSDistance = 8;  // required on PVA
        if (vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, prev_u8_, prev_s16_, nullptr) != VPI_SUCCESS ||
            vpiSubmitHarrisCornerDetector(stream_, VPI_BACKEND_PVA, harris_, prev_s16_, kp_prev_, scores_, &hp) != VPI_SUCCESS ||
            vpiStreamSync(stream_) != VPI_SUCCESS) return out;
        int nkp = 0;
        vpiArrayGetSize(kp_prev_, &nkp);
        if (nkp < 4) return out;  // too few corners (low texture) -> coast
        VPIOpticalFlowPyrLKParams lp;
        vpiInitOpticalFlowPyrLKParams(VPI_BACKEND_PVA, &lp);
        lp.windowDimension = 11;  // must be 7/9/11 on PVA
        if (vpiSubmitGaussianPyramidGenerator(stream_, VPI_BACKEND_CUDA, prev_u8_, pyr_p_, VPI_BORDER_CLAMP) != VPI_SUCCESS ||
            vpiSubmitGaussianPyramidGenerator(stream_, VPI_BACKEND_CUDA, cur_u8_, pyr_c_, VPI_BORDER_CLAMP) != VPI_SUCCESS ||
            vpiSubmitOpticalFlowPyrLK(stream_, VPI_BACKEND_PVA, lk_, pyr_p_, pyr_c_, kp_prev_, kp_cur_, status_, &lp) != VPI_SUCCESS ||
            vpiStreamSync(stream_) != VPI_SUCCESS) return out;
        VPIArrayData dp, dc, ds;
        if (vpiArrayLockData(kp_prev_, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &dp) != VPI_SUCCESS) return out;
        if (vpiArrayLockData(kp_cur_, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &dc) != VPI_SUCCESS) { vpiArrayUnlock(kp_prev_); return out; }
        if (vpiArrayLockData(status_, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &ds) != VPI_SUCCESS) { vpiArrayUnlock(kp_prev_); vpiArrayUnlock(kp_cur_); return out; }
        const VPIKeypointF32 *kp = (const VPIKeypointF32 *)dp.buffer.aos.data;
        const VPIKeypointF32 *kc = (const VPIKeypointF32 *)dc.buffer.aos.data;
        const uint8_t *st = (const uint8_t *)ds.buffer.aos.data;
        dxs_.clear(); dys_.clear();
        for (int i = 0; i < nkp; i++)
            if (st[i] == 0) { dxs_.push_back(kc[i].x - kp[i].x); dys_.push_back(kc[i].y - kp[i].y); }
        vpiArrayUnlock(kp_prev_); vpiArrayUnlock(kp_cur_); vpiArrayUnlock(status_);
        if (dxs_.empty()) return out;
        out.dx = median(dxs_);
        out.dy = median(dys_);
        out.conf = (float)dxs_.size() / (float)nkp;
        return out;
    }

private:
    static constexpr int kLevels = 4;
    static constexpr float kScale = 0.5f;
    static constexpr int kCap = 8192;

    static float median(std::vector<float> &v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }
    void fill_u8(VPIImage img, const uint8_t *src) {
        VPIImageData d;
        if (vpiImageLockData(img, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &d) != VPI_SUCCESS) return;
        const VPIImagePlanePitchLinear &pl = d.buffer.pitch.planes[0];
        uint8_t *p = (uint8_t *)pl.data;
        for (int y = 0; y < n_; y++)
            std::memcpy(p + (size_t)y * pl.pitchBytes, src + (size_t)y * n_, (size_t)n_);
        vpiImageUnlock(img);
    }
    void destroy() {
        for (VPIImage *p : {&prev_u8_, &cur_u8_, &prev_s16_}) if (*p) { vpiImageDestroy(*p); *p = nullptr; }
        for (VPIArray *p : {&kp_prev_, &scores_, &kp_cur_, &status_}) if (*p) { vpiArrayDestroy(*p); *p = nullptr; }
        for (VPIPyramid *p : {&pyr_p_, &pyr_c_}) if (*p) { vpiPyramidDestroy(*p); *p = nullptr; }
        if (harris_) { vpiPayloadDestroy(harris_); harris_ = nullptr; }
        if (lk_)     { vpiPayloadDestroy(lk_);     lk_ = nullptr; }
        if (stream_) { vpiStreamDestroy(stream_);  stream_ = nullptr; }
    }

    int n_ = 0;
    VPIStream stream_ = nullptr;
    VPIImage prev_u8_ = nullptr, cur_u8_ = nullptr, prev_s16_ = nullptr;
    VPIPayload harris_ = nullptr, lk_ = nullptr;
    VPIArray kp_prev_ = nullptr, scores_ = nullptr, kp_cur_ = nullptr, status_ = nullptr;
    VPIPyramid pyr_p_ = nullptr, pyr_c_ = nullptr;
    std::vector<float> dxs_, dys_;
};

}  // namespace nvmm

#else  // !NVMM_HAVE_VPI — stub so callers compile on non-VPI (mock/CI) builds.
namespace nvmm {
class GmcVpiPva {
public:
    static bool available() { return false; }
    bool init(int, std::string &err) { err = "VPI not built"; return false; }
    GmcShift estimate(const uint8_t *, const uint8_t *) { return {}; }
};
}  // namespace nvmm
#endif  // NVMM_HAVE_VPI
