#pragma once
#include "litert/cc/litert_tensor_buffer.h"
#include <opencv2/opencv.hpp>
#include <utility>
#include "anchor.hpp"

namespace palmy {
using container = std::vector<box>;
using pointContainer = std::vector<palmy::point3D>;
using tensorBuffer = std::vector<litert::TensorBuffer>;
using envExp = litert::Expected<litert::Environment>;
using palmIn = cv::Mat;
using palmOut1 = std::vector<palmy::region>;
using palmOut = std::pair<cv::Mat ,palmOut1>;
using markerOut = std::pair<cv::Mat,pointContainer&>;
constexpr int numAnchor = 2016;
constexpr int numAttr = 18;
}; // namespace palmy

