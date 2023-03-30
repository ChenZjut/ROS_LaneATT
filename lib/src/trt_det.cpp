#include <fstream>
#include <stdexcept>

#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>

#include "trt_det.hpp"

#define N_OFFSETS 72
#define N_STRIPS (N_OFFSETS - 1)
#define MAX_COL_BLOCKS 1000

namespace detector
{
void Net::load(const std::string &path)
{
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) {
        std::cout << "read serialized file failed\n";
        return;
    }

    // 读出文件里面的内容
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    char * buffer = new char[size];
    file.read(buffer, size);
    file.close();
    std::cout << "modle size: " << size << std::endl;
    if (runtime_) {
        engine_ =
          unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(buffer, size, nullptr));
    }
    delete[] buffer;
}

bool Net::prepare()
{
    if (!engine_) {
        return false;
    }
    context_ = unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        return false;
    }
    input_d_.resize(getInputSize()[0] * getInputSize()[1] * getInputSize()[2]);
    cudaStreamCreate(&stream_);
    return true;
}

Net::Net(const std::string &engine_path, bool verbose)
{
    Logger logger(verbose);
    initLibNvInferPlugins(&logger, "");
    runtime_ = unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    load(engine_path);
    if (!prepare()) {
        std::cout << "Fail to prepare engine" << std::endl;
        return;
    }
}

Net::~Net()
{
    if (stream_) cudaStreamDestroy(stream_);
}

Net::Net(const std::string& onnx_file_path, const std::string& precision, const int max_batch_size,
         bool verbose, size_t workspace_size)
{
    Logger logger(verbose);
    runtime_ = unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime_) {
        std::cout << "Fail to create runtime" << std::endl;
        return;
    }
    bool fp16 = precision.compare("FP16") == 0;
    bool int8 = precision.compare("INT8") == 0;

    // Create builder
    auto builder = unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        std::cout << "Fail to create builder" << std::endl;
        return;
    }

    auto config = unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cout << "Fail to create config" << std::endl;
        return;
    }

    // Allow use of FP16 layers when running in INT8
    if (fp16 || int8) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    config->setMaxWorkspaceSize(workspace_size);
    // Use INT8 precision
    // if (int8) {
    //     // Create calibrator
    //     auto calibrator = unique_ptr<nvinfer1::IInt8Calibrator>(new Int8Calibrator());
    //     config->setInt8Calibrator(calibrator.get());
    //     config->setFlag(nvinfer1::BuilderFlag::kINT8);
    // }

    std::cout << "Building " << precision << " core model..." << std::endl;
    const auto flag =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    if (!network) {
        std::cout << "Fail to create network" << std::endl;
        return;
    }

    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser) {
        std::cout << "Fail to create parser" << std::endl;
        return;
    }

    parser->parseFromFile(onnx_file_path.c_str(),
                           static_cast<int>(nvinfer1::ILogger::Severity::kERROR));

    network->getInput(0)->setDimensions(nvinfer1::Dims4{1, 3, 360, 640});

    std::cout << "Applying optimizations and building TRT CUDA engine..." << std::endl;
    plan_ = unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan_) {
        std::cout << "Fail to create serialized network" << std::endl;
        return;
    }
    engine_ =
        unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan_->data(), plan_->size()));
    if (!prepare()) {
        std::cout << "Fail to prepare engine" << std::endl;
        return;
    }
}

void Net::save(const std::string & path)
{
    std::cout << "Writing to " << path << "..." << std::endl;
    std::ofstream file(path, std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<const char *>(plan_->data()), plan_->size());
}

void Net::infer(const cv::Mat &in_img, std::vector<void *> & buffers, const int batch_size)
{
    const int INPUT_H = getInputSize()[1];
    const int INPUT_W = getInputSize()[2];
    cv::Mat img_resize;
    cv::resize(in_img, img_resize, cv::Size(INPUT_W, INPUT_H), cv::INTER_LINEAR);
    uint8_t* data_hwc = reinterpret_cast<uint8_t*>(img_resize.data);
    float* data_chw = input_d_.data();
    for (int c = 0; c < 3; ++c) {
        for (unsigned j = 0, img_size = INPUT_H * INPUT_W; j < img_size; ++j) {
            data_chw[c * img_size + j] = data_hwc[j * 3 + c] / 255.f;
        }
    }
    const int in_size{static_cast<int>(input_d_.size())};
    const int out_size(MAX_COL_BLOCKS * (5 + N_OFFSETS));
    detections_.resize(MAX_COL_BLOCKS);
    if (!context_) {
        throw std::runtime_error("Fail to create context");
    }

    cudaError_t state;
    state = cudaMalloc(&buffers[0], in_size * sizeof(float));
    if (state) {
        std::cout << "Allocate memory failed" << std::endl;
        return;
    }

    state = cudaMalloc(&buffers[1], out_size * sizeof(int));
    if (state) {
        std::cout << "Allocate memory failed" << std::endl;
        return;
    }

    state = cudaMemcpyAsync(
            buffers[0], input_d_.data(), in_size * sizeof(float),
            cudaMemcpyHostToDevice, stream_);

    if (state) {
        std::cout << "Transmit to device failed" << std::endl;
        return;
    }
    context_->enqueueV2(&buffers[0], stream_, nullptr);
    state = cudaMemcpyAsync(
            detections_.data(), buffers[1], out_size * sizeof(float ),
            cudaMemcpyDeviceToHost, stream_);
    if (state) {
        std::cout << "Transmit to host failed" << std::endl;
        return;
    }

    cudaStreamSynchronize(stream_);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    PostProcess(img_resize);

}

std::vector<int> Net::getInputSize() const
{
    auto dims = engine_->getBindingDimensions(0);
    return {dims.d[1], dims.d[2], dims.d[3]};
}

std::vector<int> Net::getOutputSize() const
{
    auto dims = engine_->getBindingDimensions(1);
    return {dims.d[2], dims.d[3]};
}

int Net::getMaxBatchSize() const
{
    return engine_->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMAX).d[0];
}

std::vector<int> Net::getInputDims() const
{
    auto dims = engine_->getBindingDimensions(0);
    return {dims.d[1], dims.d[2], dims.d[3]};
}

float Net::LaneIoU(const Detection& a, const Detection& b) const
{
    int start_a = static_cast<int>(a.start_y * N_STRIPS + 0.5f);
    int start_b = static_cast<int>(b.start_y * N_STRIPS + 0.5f);
    int start = std::max(start_a, start_b);
    int end_a = start_a + static_cast<int>(a.length + 0.5f) - 1;
    int end_b = start_b + static_cast<int>(b.length + 0.5f) - 1;
    int end = std::min(std::min(end_a, end_b), N_STRIPS);
    float dist = 0.0f;
    for (int i = start; i <= end; ++i) {
        dist += fabs(a.lane_xs[i] - b.lane_xs[i]);
    }
    dist /= static_cast<float>(end - start + 1);
    return dist;
}

void Net::PostProcess(const cv::Mat& lane_image, float conf_thresh, float nms_thresh, int nms_topk) const
{
    const int INPUT_H = getInputSize()[1];
    // 1.Do NMS
    std::vector<Detection> candidates;
    std::vector<Detection> proposals;
    for (auto det : detections_) {
        if (det.score > conf_thresh) {
            candidates.push_back(det);
        }
    }
    std::cout << candidates.size() << std::endl;
    std::sort(candidates.begin(), candidates.end(), [=](const Detection& a, const Detection& b) { return a.score > b.score; });
    for (int i = 0; i < candidates.size(); ++i) {
        if (candidates[i].score < 0.0f) {
            continue;
        }
        proposals.push_back(candidates[i]);
        if (proposals.size() == nms_topk) {
            break;
        }
        for (int j = i + 1; j < candidates.size(); ++j) {
            if (candidates[j].score > 0.0f && LaneIoU(candidates[j], candidates[i]) < nms_thresh) {
                candidates[j].score = -1.0f;
            }
        }
    }

    // 2.Decoding
    std::vector<float> anchor_ys;
    for (int i = 0; i < N_OFFSETS; ++i) {
        anchor_ys.push_back(1.0f - i / float(N_STRIPS));
    }
    std::vector<std::vector<cv::Point2f>> lanes;
    for (const auto& lane: proposals) { 
        int start = static_cast<int>(lane.start_y * N_STRIPS + 0.5f);
        int end = start + static_cast<int>(lane.length + 0.5f) - 1;
        end = std::min(end, N_STRIPS);
        std::vector<cv::Point2f> points;
        for (int i = start; i <= end; ++i) {
            points.push_back(cv::Point2f(lane.lane_xs[i], anchor_ys[i] * INPUT_H));
        }
        lanes.push_back(points);
    }

    // 3.Visualize
    for (const auto& lane_points : lanes) {
        for (const auto& point : lane_points) {
            cv::circle(lane_image, point, 1, cv::Scalar(0, 255, 0), -1);
        }
    }
    cv::imshow("laneatt_trt", lane_image);
    cv::waitKey(10);
}
}