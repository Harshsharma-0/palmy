#pragma once

#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "common.hpp"

namespace palmy {
class detectorPalm {
public:
  detectorPalm() {};
  detectorPalm(const char *modelName, litert::Environment &env);

  int init(const char *modelName, litert::Environment &env);

  ~detectorPalm() = default;
  palmy::palmOut operator<<(palmy::palmIn in);
  cv::Size resizeVal() const { return resizeDimension; }

private:
  litert::CompiledModel palmModel;
  palmy::tensorBuffer inputBuffers;
  palmy::tensorBuffer outputBuffers;
  std::vector<box> boxes;
  cv::Size resizeDimension;
};
} // namespace palmy
