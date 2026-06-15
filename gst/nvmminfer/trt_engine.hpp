/// TrtEngine — thin owner of a deserialized TensorRT engine + execution context.
///
/// Phase-0 (probes/trt_nvbufsurface_probe.cpp) proved TRT binds raw CUDA device
/// pointers we own; this wraps that into the element. Inference binds device
/// pointers via bind() and runs enqueueV3 on a caller-supplied CUDA stream — the
/// caller owns the input/output device buffers (produced by the VIC+NPP
/// preprocess and consumed by the parser, both added in later Phase-1 slices).
///
/// Engine-file path only for now (engines are device + TRT-version specific, so
/// they are built on the target with trtexec); the onnx-build path is deferred.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>

namespace nvmm {

struct TensorInfo {
    std::string      name;
    bool             is_input = false;
    nvinfer1::DataType dtype  = nvinfer1::DataType::kFLOAT;
    nvinfer1::Dims   dims{};
    int64_t          volume = 0;   // element count (non-positive dims counted as 1)
    size_t           bytes  = 0;   // volume * sizeof(dtype)
};

class TrtEngine {
public:
    /// Load a serialized .engine file. Returns nullptr and fills `err` on failure.
    static std::unique_ptr<TrtEngine> load_file(const std::string &path, std::string &err);

    const std::vector<TensorInfo> &tensors() const { return tensors_; }
    const TensorInfo *input0() const;   // first input tensor  (nullptr if none)
    const TensorInfo *output0() const;  // first output tensor (nullptr if none)

    /// Bind a device pointer for a named tensor. False if the name is unknown.
    bool bind(const std::string &name, void *device_ptr);

    /// Set the runtime shape of a dynamic input (e.g. the mask_decoder's variable
    /// sparse-prompt axis). Must be called before infer() for any input with a -1
    /// dim. `dims` is the full shape. False on failure.
    bool set_input_shape(const std::string &name, const std::vector<int64_t> &dims);

    /// Enqueue inference on `stream` (async). All I/O addresses must be bound.
    bool infer(cudaStream_t stream);

    ~TrtEngine();

private:
    TrtEngine() = default;

    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    std::vector<TensorInfo>                      tensors_;
};

/// Human-readable "1x3x640x640" for logging.
std::string dims_str(const nvinfer1::Dims &d);
const char *dtype_str(nvinfer1::DataType t);

}  // namespace nvmm
