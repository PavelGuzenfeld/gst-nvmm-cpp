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
    /// Non-positive (dynamic) dims count as 1.
    int64_t          volume = 0;
    size_t           bytes  = 0;
};

class TrtEngine {
public:
    static std::unique_ptr<TrtEngine> load_file(const std::string &path, std::string &err);

    const std::vector<TensorInfo> &tensors() const { return tensors_; }
    const TensorInfo *input0() const;
    const TensorInfo *output0() const;

    bool bind(const std::string &name, void *device_ptr);

    /// Required before infer() for any input with a -1 dim; dims is the full shape.
    bool set_input_shape(const std::string &name, const std::vector<int64_t> &dims);

    bool infer(cudaStream_t stream);

    ~TrtEngine();

private:
    TrtEngine() = default;

    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    std::vector<TensorInfo>                      tensors_;
};

std::string dims_str(const nvinfer1::Dims &d);
const char *dtype_str(nvinfer1::DataType t);

}
