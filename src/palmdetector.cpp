#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"

#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/core/options.h"

#include "palmy/anchor.hpp"
#include "palmy/detector.hpp"
#include <iostream>
#include <random>

box decode(const anchor &anc, const float *predicted, float score) {

  float dx = predicted[0] / 192;
  float dy = predicted[1] / 192;
  float dw = predicted[2] / 192;
  float dh = predicted[3] / 192;

  dw = std::clamp(dw, -5.0f, 5.0f);
  dh = std::clamp(dh, -5.0f, 5.0f);

  float cx = anc.x + dx * anc.w;
  float cy = anc.y + dy * anc.h;

  float w = anc.w * std::exp(dw);
  float h = anc.h * std::exp(dh);

  box b;

  const float *key = (predicted + 4);
  for (int k = 0; k < 7; k++) {
    float x = anc.x + key[2 * k] / 192;
    float y = anc.y + key[2 * k + 1] / 192;
    b.circles.push_back({x, y});
  }

  b.x1 = std::clamp(cx - w / 2.0f, 0.0f, 1.0f);
  b.y1 = std::clamp(cy - h / 2.0f, 0.0f, 1.0f);
  b.x2 = std::clamp(cx + w / 2.0f, 0.0f, 1.0f);
  b.y2 = std::clamp(cy + h / 2.0f, 0.0f, 1.0f);
  b.score = score;

  return b;
}
// -----------------------------
// IoU
// -----------------------------
float IoU(const box &a, const box &b) {
  float xx1 = std::max(a.x1, b.x1);
  float yy1 = std::max(a.y1, b.y1);
  float xx2 = std::min(a.x2, b.x2);
  float yy2 = std::min(a.y2, b.y2);

  float w = std::max(0.0f, xx2 - xx1);
  float h = std::max(0.0f, yy2 - yy1);

  float inter = w * h;
  float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

  return inter / (areaA + areaB - inter + 1e-6f);
}

std::vector<box> NMS(std::vector<box> &boxes, float iou_thresh) {
  std::sort(boxes.begin(), boxes.end(),
            [](const box &a, const box &b) { return a.score > b.score; });

  std::vector<box> result;
  std::vector<bool> removed(boxes.size(), false);

  for (size_t i = 0; i < boxes.size(); ++i) {
    if (removed[i])
      continue;

    result.push_back(boxes[i]);

    for (size_t j = i + 1; j < boxes.size(); ++j) {
      if (removed[j])
        continue;

      if (IoU(boxes[i], boxes[j]) > iou_thresh) {
        removed[j] = true;
      }
    }
  }

  return result;
}

namespace palmy {

int detectorPalm::init(const char *modelName, litert::Environment &env) {

  LITERT_ASSIGN_OR_ABORT(
      palmModel, litert::CompiledModel::Create(env, modelName,
                                               litert::HwAccelerators::kGpu));

  LITERT_ASSIGN_OR_RETURN(inputBuffers, palmModel.CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(outputBuffers, palmModel.CreateOutputBuffers());

  auto type = palmModel.GetInputTensorType("input_1");

  LITERT_ABORT_IF_ERROR(type.HasValue());
  auto format = type->Layout().Dimensions();

  resizeDimension.width = format[1];
  resizeDimension.height = format[2];
  return 0;
};

detectorPalm::detectorPalm(const char *modelName, litert::Environment &env) {
  init(modelName, env);
}

palmy::container detectorPalm::operator >> (cv::Mat &frame) {

  boxes.clear();
  std::vector<float> data(palmy::numAnchor * palmy::numAttr);
  std::vector<float> score(palmy::numAnchor);

  inputBuffers[0].Write<float>(absl::MakeConstSpan(
      frame.ptr<float>(), frame.total() * frame.channels()));

  bool async = false;
  LITERT_ABORT_IF_ERROR(
      palmModel.RunAsync(0, inputBuffers, outputBuffers, async));

  outputBuffers[0].Read<float>(absl::MakeSpan(data));
  outputBuffers[1].Read<float>(absl::MakeSpan(score));

  for (ssize_t i = 0; i < score.size(); i++) {
    if (score[i] < 0.7)
      continue;
    const float *a = data.data() + (i * palmy::numAttr);
    boxes.push_back(decode(anc[i], a, score[i]));
  };

  outputBuffers[0].ClearEvent();
  outputBuffers[1].ClearEvent();

  return NMS(boxes, 0.3f);
};
}; // namespace palmy

int run() {

  LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));
  LITERT_ASSIGN_OR_ABORT(auto palmModel, litert::CompiledModel::Create(
                                             env, "palm_detection_full.tflite",
                                             litert::HwAccelerators::kGpu));

  LITERT_ASSIGN_OR_RETURN(auto input_buffers, palmModel.CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(auto output_buffers, palmModel.CreateOutputBuffers());

  auto type = palmModel.GetInputTensorType("input_1");
  cv::Size resizeDimension;

  LITERT_ABORT_IF_ERROR(type.HasValue());
  auto format = type->Layout().Dimensions();

  resizeDimension.width = format[1];
  resizeDimension.height = format[2];

  cv::VideoCapture cap(0);

  if (!cap.isOpened()) {
    std::cerr << "camera not workding \n" << std::endl;
    return -1;
  }

  constexpr int NUM_ANCHORS = 2016;
  constexpr int NUM_ATTR = 18;

  std::vector<float> data(NUM_ANCHORS * NUM_ATTR);
  std::vector<float> score(NUM_ANCHORS);

  while (1) {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty())
      break;
    cv::Mat clone = frame.clone();

    // Resize
    cv::Mat resized;
    cv::resize(frame, resized, resizeDimension);
    // cv::flip(resized, resized, 1);

    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    input_buffers[0].Write<float>(absl::MakeConstSpan(
        resized.ptr<float>(), resized.total() * resized.channels()));

    bool async = false;
    LITERT_ABORT_IF_ERROR(
        palmModel.RunAsync(0, input_buffers, output_buffers, async));

    output_buffers[0].Read<float>(absl::MakeSpan(data));
    output_buffers[1].Read<float>(absl::MakeSpan(score));

    std::vector<box> boxes;

    for (ssize_t i = 0; i < score.size(); i++) {
      if (score[i] < 0.7)
        continue;
      const float *a = data.data() + (i * NUM_ATTR);
      boxes.push_back(decode(anc[i], a, score[i]));
    };

    auto final_boxes = NMS(boxes, 0.3f);
    output_buffers[0].ClearEvent();
    output_buffers[1].ClearEvent();

    // -----------------------------
    // Draw
    // -----------------------------
    for (auto &b : final_boxes) {
      int x1 = b.x1 * clone.cols;
      int y1 = b.y1 * clone.rows;
      int x2 = b.x2 * clone.cols;
      int y2 = b.y2 * clone.rows;

      cv::rectangle(clone, cv::Point(x1, y1), cv::Point(x2, y2), {0, 255, 0},
                    2);
      for (auto &itr : b.circles) {
        int x = itr.x * clone.cols;
        int y = itr.y * clone.rows;
        cv::circle(clone, cv::Point(x, y), 3, cv::Scalar(255, 255, 0), -1);
      }
    }
    cv::flip(clone, clone, 1);
    cv::imshow("palm detection", clone);
    auto key = cv::waitKey(16);
    if (key == 27)
      break; // ESC to exit
  }

  return 0;
};
