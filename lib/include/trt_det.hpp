#pragma once

#include "NvInfer.h"
#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

namespace detector
{
struct Deleter
{
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

template <typename T>
using unique_ptr = std::unique_ptr<T, Deleter>;

// 打印日志文件
class Logger : public nvinfer1::ILogger
{
public:
    Logger(bool verbose) : verbose_(verbose) {}
    void log(Severity severity, const char * msg) noexcept override {
        if (verbose_ || ((severity != Severity::kINFO) && (severity != Severity::kVERBOSE)))
            std::cout << msg << std::endl;
    }

private:
    bool verbose_{false};
};

struct Detection {
    float unknown;
    float score;
    float start_x;
    float start_y;
    float length;
    float lane_xs[72];
};

class Net
{
public:
    // create engine from engine path
    Net(const std::string & engine_path, bool verbose = false);

    // create engine from serialized onnx model
    Net(
      const std::string & onnx_file_path, const std::string & precision, const int max_batch_size,
      bool verbose = false, size_t workspace_size = (1U << 30));
    ~Net();

    // save model to path
    void save(const std::string & path);

    // Infer using pre-allocated GPU buffers {data}
    void infer(const cv::Mat &in_img, std::vector<void *> & buffers, const int batch_size);

    // Get (c, h, w) size of the fixed input
    std::vector<int> getInputSize() const;

    std::vector<int> getOutputSize() const;

    // Get max allowed batch size
    int getMaxBatchSize() const;

    // Get (c, h, w) size of the fixed input
    std::vector<int> getInputDims() const;

    float LaneIoU(const Detection & a, const Detection & b) const;

    void PostProcess(const cv::Mat & lane_image, float conf_thresh=0.4f, float nms_thresh=50.f, int nms_topk=4) const;

private:
    unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
     unique_ptr<nvinfer1::IHostMemory> plan_ = nullptr;
    unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
    unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::vector<float> input_d_;
    std::vector<Detection> detections_;

    void load(const std::string & path);
    bool prepare();
};

} // namespace segment



