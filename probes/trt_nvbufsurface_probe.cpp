// Phase-0 go/no-go probe for B5 (nvmminfer — DLA/GPU inference via TensorRT).
//
// THE GATE: does an NvBufSurface's device pointer flow into NPP and TensorRT
// with NO host (CPU) round-trip? "Zero-copy" for inference honestly means
// no host bounce — NOT binding the camera surface directly (TRT wants NCHW
// planar float at network res; the surface is RGBA/NV12 at camera res, so a
// device-side preprocess always produces a new input tensor). This probe
// proves the *honest* claim end to end:
//
//   NvBufSurface.dataPtr  --(NPP, device->device)-->  d_input (NCHW f32)
//                         --(TRT enqueueV3 on a CUDA stream)-->  d_output
//   ...with every buffer in device memory and not one cudaMemcpy to host.
//
// Steps (each prints [ OK ] / [FAIL]):
//   1. Alloc a PITCH-LINEAR RGBA surface (the VIC-preprocess *output* layout
//      that NPP/TRT consume). Confirm CUDA sees dataPtr as DEVICE memory
//      (cudaPointerGetAttributes) — the core zero-copy proof.
//   2. Run an NPP op on dataPtr (device->device) — proves NPP reads the surface
//      pointer in place, no copy.
//   3. Build a trivial TRT engine at runtime, bind device input/output addresses
//      (setInputTensorAddress on a pointer WE own), enqueueV3 + sync — proves
//      TRT binds raw device pointers we control (the model the element relies on).
//   4. (informational) Alloc a BLOCK-LINEAR surface (the nvv4l2decoder default)
//      and check whether its dataPtr is CUDA-addressable or needs an EGL bounce.
//      Documents the input-layout reality that drives the preprocess design.
//
// Build (Orin / JP6, TensorRT 10.3, CUDA 12.x):
//   g++ -std=c++17 -O2 trt_nvbufsurface_probe.cpp -o trt_nvbufsurface_probe \
//     -I/usr/src/jetson_multimedia_api/include -I/usr/local/cuda/include \
//     -L/usr/lib/aarch64-linux-gnu/tegra -L/usr/local/cuda/lib64 \
//     -L/usr/lib/aarch64-linux-gnu \
//     -lnvbufsurface -lnvinfer -lcudart -lnppc -lnppidei -lnppig
//
// Run on the HOST (not a container): the NVMM/NvBufSurface path needs /dev/nvmap.

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <memory>

#include <nvbufsurface.h>

#include <cuda_runtime.h>
#include <nppi_data_exchange_and_initialization.h>

#include <NvInfer.h>

// ---- tiny status helpers (match the VPI probes' [ OK ]/[FAIL] style) --------

static bool cuda_ok(const char* what, cudaError_t e) {
    if (e == cudaSuccess) { printf("  [ OK ] %s\n", what); return true; }
    printf("  [FAIL] %s -> %s\n", what, cudaGetErrorString(e));
    return false;
}

static bool npp_ok(const char* what, NppStatus s) {
    if (s == NPP_SUCCESS) { printf("  [ OK ] %s\n", what); return true; }
    printf("  [FAIL] %s -> NppStatus %d\n", what, (int)s);
    return false;
}

static bool check(const char* what, bool cond) {
    printf(cond ? "  [ OK ] %s\n" : "  [FAIL] %s\n", what);
    return cond;
}

struct Logger : nvinfer1::ILogger {
    void log(Severity sev, const char* msg) noexcept override {
        if (sev <= Severity::kWARNING) printf("  [TRT ] %s\n", msg);
    }
} gLogger;

// ---- NvBufSurface allocation (mirrors the VPI probe's make_* helper) --------

static NvBufSurface* make_surface(uint32_t w, uint32_t h,
                                  NvBufSurfaceColorFormat fmt,
                                  NvBufSurfaceLayout layout,
                                  NvBufSurfaceMemType mem) {
    NvBufSurfaceCreateParams p;
    memset(&p, 0, sizeof(p));
    p.width = w; p.height = h;
    p.colorFormat = fmt;
    p.layout = layout;
    p.memType = mem;
    p.gpuId = 0;
    NvBufSurface* surf = nullptr;
    if (NvBufSurfaceCreate(&surf, 1, &p) != 0 || !surf) return nullptr;
    return surf;
}

// Is `ptr` ordinary CUDA device memory usable without a host bounce?
static bool is_cuda_device_ptr(const char* what, void* ptr) {
    cudaPointerAttributes attr;
    cudaError_t e = cudaPointerGetAttributes(&attr, ptr);
    if (e != cudaSuccess) {
        printf("  [FAIL] %s: cudaPointerGetAttributes -> %s (likely needs EGL bounce)\n",
               what, cudaGetErrorString(e));
        cudaGetLastError();  // clear sticky error
        return false;
    }
    bool dev = (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged);
    printf(dev ? "  [ OK ] %s: CUDA memoryType=%d (device-addressable)\n"
               : "  [FAIL] %s: CUDA memoryType=%d (NOT device — host bounce required)\n",
           what, (int)attr.type);
    return dev;
}

int main() {
    const uint32_t W = 640, H = 640;   // typical detector network size
    bool gate = true;

    printf("== Step 1: pitch-linear RGBA surface -> CUDA device pointer ==\n");
    NvBufSurface* rgba = make_surface(W, H, NVBUF_COLOR_FORMAT_RGBA,
                                      NVBUF_LAYOUT_PITCH, NVBUF_MEM_CUDA_DEVICE);
    if (!check("NvBufSurfaceCreate(RGBA, pitch-linear, CUDA_DEVICE)", rgba != nullptr))
        return 2;
    NvBufSurfaceParams& sp = rgba->surfaceList[0];
    void* surf_ptr = sp.dataPtr;
    printf("  surface: %ux%u pitch=%u dataPtr=%p\n", sp.width, sp.height, sp.pitch, surf_ptr);
    gate &= is_cuda_device_ptr("surface.dataPtr", surf_ptr);

    printf("== Step 2: NPP op on the surface pointer (device->device) ==\n");
    // Write a constant into the surface via NPP — proves NPP consumes dataPtr
    // in place with no host copy.
    {
        const Npp8u val[4] = {16, 32, 48, 255};
        NppiSize roi = { (int)W, (int)H };
        gate &= npp_ok("nppiSet_8u_C4R on surface.dataPtr",
                       nppiSet_8u_C4R(val, (Npp8u*)surf_ptr, (int)sp.pitch, roi));
        gate &= cuda_ok("cudaDeviceSynchronize after NPP", cudaDeviceSynchronize());
    }

    printf("== Step 3: build trivial TRT engine + bind device pointers ==\n");
    bool trt_ok = false;
    {
        std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));
        std::unique_ptr<nvinfer1::INetworkDefinition> net(
            builder ? builder->createNetworkV2(0) : nullptr);
        std::unique_ptr<nvinfer1::IBuilderConfig> cfg(
            builder ? builder->createBuilderConfig() : nullptr);
        if (check("createInferBuilder / network / config", builder && net && cfg)) {
            // Input: 1x3xHxW float (NCHW), one ReLU, output same shape.
            auto* in = net->addInput("in", nvinfer1::DataType::kFLOAT,
                                     nvinfer1::Dims4{1, 3, (int)H, (int)W});
            auto* act = net->addActivation(*in, nvinfer1::ActivationType::kRELU);
            act->getOutput(0)->setName("out");
            net->markOutput(*act->getOutput(0));
            if (builder->platformHasFastFp16()) cfg->setFlag(nvinfer1::BuilderFlag::kFP16);

            std::unique_ptr<nvinfer1::IHostMemory> plan(
                builder->buildSerializedNetwork(*net, *cfg));
            std::unique_ptr<nvinfer1::IRuntime> rt(nvinfer1::createInferRuntime(gLogger));
            std::unique_ptr<nvinfer1::ICudaEngine> eng(
                (plan && rt) ? rt->deserializeCudaEngine(plan->data(), plan->size()) : nullptr);
            std::unique_ptr<nvinfer1::IExecutionContext> ctx(
                eng ? eng->createExecutionContext() : nullptr);

            if (check("build + deserialize engine + context", plan && rt && eng && ctx)) {
                const size_t elems = (size_t)3 * H * W;
                void* d_in = nullptr; void* d_out = nullptr;
                bool mem = cuda_ok("cudaMalloc d_input (NCHW f32)",
                                   cudaMalloc(&d_in, elems * sizeof(float)))
                         & cuda_ok("cudaMalloc d_output",
                                   cudaMalloc(&d_out, elems * sizeof(float)));
                cudaStream_t stream = nullptr;
                mem &= cuda_ok("cudaStreamCreate", cudaStreamCreate(&stream));

                // The honest preprocess hop: surface(device) -> d_input(device),
                // device-to-device, NO host. (A real element does NPP convert/
                // normalize/planarize here; D2D copy proves the addressing.)
                mem &= cuda_ok("cudaMemcpy surface->d_input (D2D, no host)",
                               cudaMemcpy(d_in, surf_ptr,
                                          std::min(elems * sizeof(float),
                                                   (size_t)sp.pitch * sp.height),
                                          cudaMemcpyDeviceToDevice));

                if (mem) {
                    // THE proof: TRT binds raw device pointers we own.
                    bool bound = check("setInputTensorAddress(in, d_input)",
                                       ctx->setInputTensorAddress("in", d_in))
                               & check("setOutputTensorAddress(out, d_output)",
                                       ctx->setTensorAddress("out", d_out));
                    bool ran = bound
                        && check("enqueueV3 on stream", ctx->enqueueV3(stream))
                        && cuda_ok("cudaStreamSynchronize", cudaStreamSynchronize(stream));
                    trt_ok = ran;
                }
                if (d_in) cudaFree(d_in);
                if (d_out) cudaFree(d_out);
                if (stream) cudaStreamDestroy(stream);
            }
        }
    }
    gate &= check("TensorRT bound + ran on device pointers", trt_ok);

    printf("== Step 4 (info): block-linear (decoder default) surface addressability ==\n");
    {
        NvBufSurface* bl = make_surface(W, H, NVBUF_COLOR_FORMAT_NV12,
                                        NVBUF_LAYOUT_BLOCK_LINEAR, NVBUF_MEM_CUDA_DEVICE);
        if (bl) {
            bool direct = is_cuda_device_ptr("block-linear NV12 dataPtr", bl->surfaceList[0].dataPtr);
            printf("  -> decoder-side block-linear is %s; preprocess must %s\n",
                   direct ? "CUDA-addressable" : "NOT directly CUDA-addressable",
                   direct ? "still de-tile to planar (VIC) before TRT"
                          : "go through VIC (NvBufSurfTransform) / EGL — expected");
            NvBufSurfaceDestroy(bl);
        } else {
            printf("  [info] block-linear CUDA_DEVICE alloc unsupported here (also informative)\n");
        }
    }

    printf("\nVERDICT: surface->NPP->TRT zero-copy (no host bounce) = %s\n",
           gate ? "GO" : "NO-GO");
    NvBufSurfaceDestroy(rgba);
    return gate ? 0 : 1;
}
