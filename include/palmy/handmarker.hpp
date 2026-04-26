#pragma once
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_tensor_buffer.h"

#include <opencv2/opencv.hpp>
#include "common.hpp"

namespace palmy {
class handMarker {
public:
  handMarker(){};
  handMarker(const char *modelName,litert::Environment &env);
  int init(const char *modelName,litert::Environment &env);
  palmy::markerOut operator<<(palmy::palmOut in);

private:
  litert::CompiledModel handModel;
  palmy::tensorBuffer inputBuffers;
  palmy::tensorBuffer outputBuffers;
  palmy::pointContainer points;
  cv::Size resizeDimension;
};
}; // namespace palmy
