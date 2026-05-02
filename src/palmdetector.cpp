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

palmy::palmOut1 processBoxes(palmy::palmIn &in, palmy::container &final_boxes) {

  palmy::palmOut1 points;

  if (final_boxes.size() > 2)
    final_boxes.resize(2);

  int rows = in.rows;
  int cols = in.cols;

  for (auto &b : final_boxes) {
    for (auto &itr : b.circles) {
      int x = itr.x * cols;
      int y = itr.y * rows;
      cv::circle(in, cv::Point(x, y), 3, cv::Scalar(255, 255, 0), -1);
    }
    auto vec = final_boxes[0].circles;
    struct {
      float x, y;
    } center;
    float boxw = b.x2 - b.x1;
    float boxy = b.y2 - b.y1;
    center.x = b.x2 - (boxw / 2);
    center.y = b.y2 - (boxy / 2);

    float angle =
        -std::atan2(vec[2].y - center.y, vec[2].x - center.x) * (180 / M_PI);
    angle += angle < 0 ? 360 : 0;

    float distancey = std::sqrt((std::pow((vec[2].x - vec[0].x), 2)) +
                                std::pow((vec[2].y - vec[0].y), 2));

    int x1 = std::ceil(b.x1 * (float)cols);
    int y1 = std::ceil(((b.y1 - (distancey * 1.5)) * (float)rows));
    int x2 = std::ceil(b.x2 * (float)cols);
    int y2 = std::ceil(b.y2 * (float)rows);

    int diff = ((y2 - y1) - (x2 - x1)) / 2;
    x1 -= diff;
    x2 += diff;

    points.push_back({std::clamp(x1, 1, (cols - (x2 - x1))),
                      std::clamp(y1, 1, rows), std::clamp(x2 - x1, 1, cols),
                      std::clamp(y2 - y1, 1, rows)});

    cv::rectangle(in, cv::Point(x1, y1), cv::Point(x2, y2), {0, 255, 0}, 2);
  }

  return points;
}
palmy::palmOut detectorPalm::operator<<(palmy::palmIn in) {

  cv::Mat frame = in.clone();

  cv::resize(frame, frame, resizeDimension);
  cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
  frame.convertTo(frame, CV_32F, 1.0 / 255.0);

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
  auto final_boxes = NMS(boxes, 0.3f);

  return {in, processBoxes(in, final_boxes)};
};
}; // namespace palmy
