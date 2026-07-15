/// xfeat_matcher.cpp — see xfeat_matcher.hpp.
/// Ported from ../gst-nvmm-ostrack/gst/gstnvmmostrack.cpp (init_xfeat/init_lightglue/
/// xfeat_extract/register_point), substituting nvmm::TrtEngine for raw TensorRT.
#include "xfeat_matcher.hpp"

#include "xfeat_sparse.hpp"   // get_kpts_heatmap / nms / score_and_sort / grid_sample_bicubic

#include <cmath>
#include <cstring>

namespace nvmm {

namespace {
// cudaMalloc-or-fail helper.
bool cu_malloc(void** p, size_t bytes, std::string& err, const char* what) {
    if (cudaMalloc(p, bytes) != cudaSuccess) { err = std::string("cudaMalloc failed: ") + what; return false; }
    return true;
}
}  // namespace

XfeatMatcher::~XfeatMatcher() { free_buffers(); }

void XfeatMatcher::free_buffers() {
    for (void** p : {&d_img_, &d_feats_, &d_kpts_, &d_heat_,
                     &d_d0_, &d_d1_, &d_k0_, &d_k1_, &d_sim_, &d_z0_, &d_z1_}) {
        if (*p) { cudaFree(*p); *p = nullptr; }
    }
    if (rgba_)  { NvBufSurfaceDestroy(rgba_); rgba_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

bool XfeatMatcher::init(const std::string& engine_dir, std::string& err) {
    xf_ = TrtEngine::load_file(engine_dir + "/xfeat.engine", err);
    if (!xf_) return false;
    lg_ = TrtEngine::load_file(engine_dir + "/lightglue.engine", err);
    if (!lg_) return false;

    if (cudaStreamCreate(&stream_) != cudaSuccess) { err = "cudaStreamCreate failed"; return false; }

    // XFeat IO (static shapes): size each device buffer from the engine's
    // TensorInfo::bytes so a half-precision engine still binds (NOTE: an fp16 XFeat
    // engine would also need the D2H reads + host post-proc in extract() to handle
    // the output dtype — those currently assume fp32).
    auto xbytes = [&](const char* nm) -> size_t {
        for (const auto& t : xf_->tensors()) if (t.name == nm) return t.bytes;
        return 0;
    };
    const size_t b_img = xbytes("image"), b_feats = xbytes("feats"),
                 b_kpts = xbytes("keypoints"), b_heat = xbytes("heatmap");
    if (!b_img || !b_feats || !b_kpts || !b_heat) {
        err = "xfeat engine missing an expected tensor (image/feats/keypoints/heatmap)"; return false;
    }
    const size_t f4 = sizeof(float);
    if (!cu_malloc(&d_img_,   b_img,   err, "xfeat image")) return false;
    if (!cu_malloc(&d_feats_, b_feats, err, "xfeat feats")) return false;
    if (!cu_malloc(&d_kpts_,  b_kpts,  err, "xfeat kpts"))  return false;
    if (!cu_malloc(&d_heat_,  b_heat,  err, "xfeat heat"))  return false;
    if (!xf_->bind("image", d_img_) || !xf_->bind("feats", d_feats_) ||
        !xf_->bind("keypoints", d_kpts_) || !xf_->bind("heatmap", d_heat_)) {
        err = "xfeat tensor bind failed (name mismatch vs engine)"; return false;
    }

    // LightGlue IO (allocated to kTopK; shapes set per match()).
    if (!cu_malloc(&d_d0_, (size_t)kTopK * 64 * f4, err, "lg desc0")) return false;
    if (!cu_malloc(&d_d1_, (size_t)kTopK * 64 * f4, err, "lg desc1")) return false;
    if (!cu_malloc(&d_k0_, (size_t)kTopK * 2  * f4, err, "lg nkpts0")) return false;
    if (!cu_malloc(&d_k1_, (size_t)kTopK * 2  * f4, err, "lg nkpts1")) return false;
    if (!cu_malloc(&d_sim_, (size_t)kTopK * kTopK * f4, err, "lg sim")) return false;
    if (!cu_malloc(&d_z0_, (size_t)kTopK * f4, err, "lg z0")) return false;
    if (!cu_malloc(&d_z1_, (size_t)kTopK * f4, err, "lg z1")) return false;
    if (!lg_->bind("desc0", d_d0_) || !lg_->bind("desc1", d_d1_) ||
        !lg_->bind("nkpts0", d_k0_) || !lg_->bind("nkpts1", d_k1_) ||
        !lg_->bind("sim", d_sim_) || !lg_->bind("z0", d_z0_) || !lg_->bind("z1", d_z1_)) {
        err = "lightglue tensor bind failed (name mismatch vs engine)"; return false;
    }

    // RGBA VIC stretch destination.
    NvBufSurfaceCreateParams cp{};
    cp.width = kXW; cp.height = kXH;
    cp.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    cp.layout = NVBUF_LAYOUT_PITCH;
    cp.memType = NVBUF_MEM_SURFACE_ARRAY;
    cp.gpuId = 0;
    if (NvBufSurfaceCreate(&rgba_, 1, &cp) != 0) { err = "xfeat RGBA surface create failed"; return false; }

    return true;
}

bool XfeatMatcher::extract(NvBufSurface* src, XfeatFrame& out, std::string& err) {
    out.kpts.clear(); out.descs.clear();

    const uint32_t W = src->surfaceList[0].width, H = src->surfaceList[0].height;
    // VIC stretch full-frame -> 480x256 RGBA (non-aspect-preserving, matches reference).
    NvBufSurfTransformRect sr{0, 0, W, H}, dr{0, 0, (uint32_t)kXW, (uint32_t)kXH};
    NvBufSurfTransformParams tp{};
    tp.src_rect = &sr; tp.dst_rect = &dr;
    tp.transform_flag = NVBUFSURF_TRANSFORM_CROP_SRC | NVBUFSURF_TRANSFORM_CROP_DST | NVBUFSURF_TRANSFORM_FILTER;
    tp.transform_filter = NvBufSurfTransformInter_Bilinear;
    if (NvBufSurfTransform(src, rgba_, &tp) != NvBufSurfTransformError_Success) { err = "xfeat VIC transform failed"; return false; }
    if (NvBufSurfaceMap(rgba_, 0, 0, NVBUF_MAP_READ) != 0) { err = "xfeat surface map failed"; return false; }
    NvBufSurfaceSyncForCpu(rgba_, 0, 0);
    const uint8_t* rgba = (const uint8_t*)rgba_->surfaceList[0].mappedAddr.addr[0];
    const int pitch = rgba_->surfaceList[0].pitch;

    // Deinterleave RGBA -> planar RGB /255 (VIC px[0]=R). CNN takes RGB, no mean/std.
    std::vector<float> in((size_t)3 * kXH * kXW);
    const int plane = kXH * kXW;
    for (int y = 0; y < kXH; ++y)
        for (int x = 0; x < kXW; ++x) {
            const uint8_t* px = rgba + (size_t)y * pitch + x * 4;
            const int p = y * kXW + x;
            in[p]             = px[0] / 255.f;
            in[plane + p]     = px[1] / 255.f;
            in[2 * plane + p] = px[2] / 255.f;
        }
    NvBufSurfaceUnMap(rgba_, 0, 0);

    if (cudaMemcpyAsync(d_img_, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
        err = "xfeat H2D copy failed"; return false;
    }
    if (!xf_->infer(stream_)) { err = "xfeat inference failed"; return false; }
    cudaStreamSynchronize(stream_);

    std::vector<float> K((size_t)65 * kXHC * kXWC), F((size_t)64 * kXHC * kXWC), Hh((size_t)kXHC * kXWC);
    cudaMemcpy(K.data(),  d_kpts_,  K.size()  * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(F.data(),  d_feats_, F.size()  * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Hh.data(), d_heat_,  Hh.size() * sizeof(float), cudaMemcpyDeviceToHost);

    auto K1h = nvmm::xfeat::get_kpts_heatmap(K.data(), kXHC, kXWC);
    auto kp = nvmm::xfeat::nms(K1h.data(), kXH, kXW, 0.05f, 5);
    auto sorted = nvmm::xfeat::score_and_sort(kp, K1h.data(), kXH, kXW, Hh.data(), kXHC, kXWC, kXH, kXW);
    if ((int)sorted.size() > kTopK) sorted.resize(kTopK);   // Top-K cap

    // L2-normalize feats over the 64 channels at each grid cell.
    std::vector<float> M1n(F.size());
    for (int h = 0; h < kXHC; ++h)
        for (int x = 0; x < kXWC; ++x) {
            double s = 0;
            for (int c = 0; c < 64; ++c) { double v = F[((size_t)c * kXHC + h) * kXWC + x]; s += v * v; }
            double nr = std::sqrt(s) + 1e-12;
            for (int c = 0; c < 64; ++c)
                M1n[((size_t)c * kXHC + h) * kXWC + x] = (float)(F[((size_t)c * kXHC + h) * kXWC + x] / nr);
        }

    const double rh = kRH / kXH;   // y rescale net(256)->reg(270); x scale = kRW/kXW = 1
    out.kpts.reserve(sorted.size());
    out.descs.reserve(sorted.size());
    std::array<float, 64> dn;
    float d[64];
    for (const auto& sc : sorted) {
        nvmm::xfeat::grid_sample_bicubic(M1n.data(), 64, kXHC, kXWC, sc.x, sc.y, kXH, kXW, d);
        double nr = 0; for (int c = 0; c < 64; ++c) nr += (double)d[c] * d[c];
        nr = std::sqrt(nr) + 1e-12;
        for (int c = 0; c < 64; ++c) dn[c] = (float)(d[c] / nr);
        out.descs.push_back(dn);
        out.kpts.push_back({ (double)sc.x, (double)sc.y * rh });
    }
    return true;
}

bool XfeatMatcher::match(const XfeatFrame& a, const XfeatFrame& b,
                         std::vector<nvmm::motion::MatchPair>& out, std::string& err) {
    out.clear();
    const int N0 = (int)a.kpts.size(), N1 = (int)b.kpts.size();
    if (N0 < 9 || N1 < 9) return true;                 // no verdict (not an error)
    if (N0 > kTopK || N1 > kTopK) { err = "xfeat match: N exceeds kTopK"; return false; }

    std::vector<nvmm::xfeat::Pt2> nk0, nk1;
    nvmm::xfeat::normalize_keypoints(a.kpts, kRW, kRH, nk0);
    nvmm::xfeat::normalize_keypoints(b.kpts, kRW, kRH, nk1);

    std::vector<float> d0((size_t)N0 * 64), d1((size_t)N1 * 64), k0((size_t)N0 * 2), k1((size_t)N1 * 2);
    for (int i = 0; i < N0; ++i) {
        std::memcpy(&d0[(size_t)i * 64], a.descs[i].data(), 64 * sizeof(float));
        k0[i * 2] = (float)nk0[i].x; k0[i * 2 + 1] = (float)nk0[i].y;
    }
    for (int i = 0; i < N1; ++i) {
        std::memcpy(&d1[(size_t)i * 64], b.descs[i].data(), 64 * sizeof(float));
        k1[i * 2] = (float)nk1[i].x; k1[i * 2 + 1] = (float)nk1[i].y;
    }

    if (!lg_->set_input_shape("desc0",  {1, N0, 64}) || !lg_->set_input_shape("desc1",  {1, N1, 64}) ||
        !lg_->set_input_shape("nkpts0", {1, N0, 2})  || !lg_->set_input_shape("nkpts1", {1, N1, 2})) {
        err = "lightglue set_input_shape failed"; return false;
    }
    cudaMemcpyAsync(d_d0_, d0.data(), d0.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_d1_, d1.data(), d1.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_k0_, k0.data(), k0.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_k1_, k1.data(), k1.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    if (!lg_->infer(stream_)) { err = "lightglue inference failed"; return false; }
    cudaStreamSynchronize(stream_);

    std::vector<float> sim((size_t)N0 * N1), z0(N0), z1(N1);
    cudaMemcpy(sim.data(), d_sim_, sim.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(z0.data(),  d_z0_,  z0.size()  * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(z1.data(),  d_z1_,  z1.size()  * sizeof(float), cudaMemcpyDeviceToHost);

    auto matches = nvmm::xfeat::filter_matches(sim.data(), z0.data(), z1.data(), N0, N1, 0.1);
    out.reserve(matches.size());
    for (const auto& m : matches)
        out.push_back({ m.i, a.kpts[m.i], b.kpts[m.j] });   // idx = anchor (A) keypoint index
    return true;
}

}  // namespace nvmm
