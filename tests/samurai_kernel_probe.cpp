/// samurai_kernel_probe — parity-check the SAMURAI CUDA kernels against their host
/// references (samurai_seed_math.hpp + inline). Deterministic inputs; pure
/// correctness. Pass = every kernel matches host within fp tolerance.
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "samurai_kernels.hpp"
#include "samurai_seed_math.hpp"
#include "samurai_memory.hpp"   // host assemble_memory + memdims (parity ref)

static float pat(int i) { return std::sin(0.1f * i) * 3.f - 1.f; }  // deterministic
static float *dev(const std::vector<float> &h)
{
    float *d = nullptr; cudaMalloc(&d, h.size() * sizeof(float));
    cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}
static std::vector<float> host(const float *d, size_t n)
{
    std::vector<float> h(n); cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}
static double maxabs(const std::vector<float> &a, const std::vector<float> &b)
{
    double m = 0; for (size_t i = 0; i < a.size(); i++) m = std::fmax(m, std::fabs((double)a[i] - b[i]));
    return m;
}

int main()
{
    cudaStream_t s; cudaStreamCreate(&s);
    bool ok = true;
    auto report = [&](const char *n, double e, double tol) {
        bool p = e <= tol; ok = ok && p;
        std::printf("  %-16s maxabs=%.4g %s\n", n, e, p ? "PASS" : "FAIL");
    };

    // transpose (C=5, HW=7) -> (7,5)
    { int C = 5, HW = 7; std::vector<float> in(C * HW); for (int i = 0; i < C * HW; i++) in[i] = pat(i);
      float *di = dev(in), *dout; cudaMalloc(&dout, in.size() * sizeof(float));
      nvmm::k_transpose(di, dout, C, HW, s); cudaStreamSynchronize(s);
      auto g = host(dout, in.size()); std::vector<float> ref(C * HW);
      for (int r = 0; r < C; r++) for (int c = 0; c < HW; c++) ref[c * C + r] = in[r * HW + c];
      report("transpose", maxabs(g, ref), 0); cudaFree(di); cudaFree(dout); }

    // add_per_channel (C=4, HW=6)
    { int C = 4, HW = 6; std::vector<float> in(C * HW), bias(C);
      for (int i = 0; i < C * HW; i++) in[i] = pat(i);
      for (int c = 0; c < C; c++) bias[c] = pat(100 + c);
      float *di = dev(in), *db = dev(bias), *dout; cudaMalloc(&dout, in.size() * sizeof(float));
      nvmm::k_add_per_channel(di, db, dout, C, HW, s); cudaStreamSynchronize(s);
      auto g = host(dout, in.size()); std::vector<float> ref(C * HW);
      for (int c = 0; c < C; c++) for (int i = 0; i < HW; i++) ref[c * HW + i] = in[c * HW + i] + bias[c];
      report("add_per_channel", maxabs(g, ref), 1e-6); cudaFree(di); cudaFree(db); cudaFree(dout); }

    // sigmoid_scale
    { int n = 50; std::vector<float> in(n); for (int i = 0; i < n; i++) in[i] = pat(i);
      float *di = dev(in), *dout; cudaMalloc(&dout, n * sizeof(float));
      nvmm::k_sigmoid_scale(di, dout, n, 20.f, -10.f, s); cudaStreamSynchronize(s);
      auto g = host(dout, n); std::vector<float> ref(n);
      for (int i = 0; i < n; i++) ref[i] = (1.f / (1.f + std::exp(-in[i]))) * 20.f - 10.f;
      report("sigmoid_scale", maxabs(g, ref), 1e-3); cudaFree(di); cudaFree(dout); }

    // threshold_scale
    { int n = 50; std::vector<float> in(n); for (int i = 0; i < n; i++) in[i] = pat(i);
      float *di = dev(in), *dout; cudaMalloc(&dout, n * sizeof(float));
      nvmm::k_threshold_scale(di, dout, n, 10.f, -10.f, s); cudaStreamSynchronize(s);
      auto g = host(dout, n); std::vector<float> ref(n);
      for (int i = 0; i < n; i++) ref[i] = in[i] > 0.f ? 10.f : -10.f;
      report("threshold_scale", maxabs(g, ref), 0); cudaFree(di); cudaFree(dout); }

    // bilinear 8x8 -> 16x16 vs host bilinear_upsample
    { int hi = 8, wi = 8, ho = 16, wo = 16; std::vector<float> in(hi * wi);
      for (int i = 0; i < hi * wi; i++) in[i] = pat(i);
      float *di = dev(in), *dout; cudaMalloc(&dout, (size_t)ho * wo * sizeof(float));
      nvmm::k_bilinear(di, dout, hi, wi, ho, wo, s); cudaStreamSynchronize(s);
      auto g = host(dout, (size_t)ho * wo);
      auto ref = nvmm::bilinear_upsample(in.data(), hi, wi, ho, wo);
      report("bilinear", maxabs(g, ref), 1e-5); cudaFree(di); cudaFree(dout); }

    // mask_bbox vs host mask_to_box
    { int h = 20, w = 24; std::vector<float> m((size_t)h * w, -1.f);
      for (int y = 5; y <= 12; y++) for (int x = 7; x <= 17; x++) m[y * w + x] = 1.f;  // box
      float *dm = dev(m); int *dbox; cudaMalloc(&dbox, 4 * sizeof(int));
      int init[4] = {w, h, -1, -1}; cudaMemcpy(dbox, init, sizeof(init), cudaMemcpyHostToDevice);
      nvmm::k_mask_bbox(dm, h, w, dbox, s); cudaStreamSynchronize(s);
      int box[4]; cudaMemcpy(box, dbox, sizeof(box), cudaMemcpyDeviceToHost);
      nvmm::MaskBox ref = nvmm::mask_to_box(m.data(), h, w, 0.f);
      double e = std::abs(box[0] - ref.x) + std::abs(box[1] - ref.y) +
                 std::abs((box[2] - box[0]) - ref.w) + std::abs((box[3] - box[1]) - ref.h);
      report("mask_bbox", e, 0); cudaFree(dm); cudaFree(dbox); }

    // assemble_memory: device k_assemble_memory vs host assemble_memory.
    {
        using namespace nvmm::memdims;
        std::vector<float> maskmem[kMask], objpack((size_t)kPtr * kHid), pos(kPtr);
        std::vector<float> cmpos((size_t)kMem * kTok), tpos((size_t)kMask * kMem),
            tpw((size_t)kMem * kHid), tpb(kMem);
        for (int s = 0; s < kMask; s++) { maskmem[s].resize((size_t)kMem * kTok);
            for (size_t i = 0; i < maskmem[s].size(); i++) maskmem[s][i] = pat((int)i + s * 7); }
        for (size_t i = 0; i < objpack.size(); i++) objpack[i] = pat((int)i + 11);
        for (int p = 0; p < kPtr; p++) pos[p] = (p == 0) ? 60.f : (float)p;
        for (size_t i = 0; i < cmpos.size(); i++) cmpos[i] = pat((int)i + 3);
        for (size_t i = 0; i < tpos.size(); i++) tpos[i] = pat((int)i + 5);
        for (size_t i = 0; i < tpw.size(); i++) tpw[i] = pat((int)i + 9) * 0.01f;
        for (int i = 0; i < kMem; i++) tpb[i] = pat(i + 13);
        // host reference
        const float *mm_h[kMask]; for (int s = 0; s < kMask; s++) mm_h[s] = maskmem[s].data();
        const float *op_h[kPtr];  for (int p = 0; p < kPtr; p++) op_h[p] = objpack.data() + (size_t)p * kHid;
        std::vector<float> hm((size_t)kTotal * kMem), hp((size_t)kTotal * kMem);
        nvmm::MemConsts mc{cmpos.data(), tpos.data(), tpw.data(), tpb.data()};
        nvmm::assemble_memory(mm_h, op_h, pos.data(), mc, hm.data(), hp.data());
        // device
        float *dmm[kMask]; for (int s = 0; s < kMask; s++) dmm[s] = dev(maskmem[s]);
        float *dptrs; cudaMalloc(&dptrs, sizeof(dmm)); cudaMemcpy(dptrs, dmm, sizeof(dmm), cudaMemcpyHostToDevice);
        float *dop = dev(objpack), *dpos = dev(pos), *dcm = dev(cmpos), *dtp = dev(tpos), *dtw = dev(tpw), *dtb = dev(tpb);
        float *dmem, *dmp; cudaMalloc(&dmem, hm.size() * sizeof(float)); cudaMalloc(&dmp, hp.size() * sizeof(float));
        nvmm::k_assemble_memory(reinterpret_cast<const float *const *>(dptrs), dop, dpos, dcm, dtp, dtw, dtb, dmem, dmp, kTok, s);
        cudaStreamSynchronize(s);
        auto gm = host(dmem, hm.size()), gp = host(dmp, hp.size());
        report("assemble.mem", maxabs(gm, hm), 1e-4);
        report("assemble.pos", maxabs(gp, hp), 1e-3);
        for (int sl = 0; sl < kMask; sl++) cudaFree(dmm[sl]);
        cudaFree(dptrs); cudaFree(dop); cudaFree(dpos); cudaFree(dcm); cudaFree(dtp); cudaFree(dtw); cudaFree(dtb); cudaFree(dmem); cudaFree(dmp);
    }

    std::printf("%s\n", ok ? "KERNEL_PARITY_PASS" : "KERNEL_PARITY_FAIL");
    return ok ? 0 : 1;
}
