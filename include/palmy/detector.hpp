#pragma once

#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_tensor_buffer.h"

#include "./anchor.hpp"
#include <memory>
#include <opencv2/opencv.hpp>

namespace palmy {
using container = std::vector<box>;
using tensorBuffer = std::vector<litert::TensorBuffer>;
using envExp = litert::Expected<litert::Environment>;
constexpr int numAnchor = 2016;
constexpr int numAttr = 18;
}; // namespace palmy

namespace palmy {
class detectorPalm {
public:
  detectorPalm() {};
  detectorPalm(const char *modelName, litert::Environment &env);

  int init(const char *modelName, litert::Environment &env);

  ~detectorPalm() = default;
  palmy::container operator>>(cv::Mat &frame);
  cv::Size resizeVal() const { return resizeDimension; }

private:
  litert::CompiledModel palmModel;
  palmy::tensorBuffer inputBuffers;
  palmy::tensorBuffer outputBuffers;
  std::vector<box> boxes;
  cv::Size resizeDimension;
};
} // namespace palmy

int run();
