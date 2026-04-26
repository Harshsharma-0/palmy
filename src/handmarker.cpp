#include "palmy/handmarker.hpp"

namespace palmy {
handMarker::handMarker(const char *modelName, litert::Environment &env) {
  init(modelName, env);
};

int handMarker::init(const char *modelName, litert::Environment &env) {
  LITERT_ASSIGN_OR_ABORT(
      handModel, litert::CompiledModel::Create(env, modelName,
                                               litert::HwAccelerators::kGpu));

  LITERT_ASSIGN_OR_RETURN(inputBuffers, handModel.CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(outputBuffers, handModel.CreateOutputBuffers());

  auto type = handModel.GetInputTensorType("input_1");
  auto modelLay = handModel.GetOutputTensorLayouts(0);

  LITERT_ABORT_IF_ERROR(type.HasValue());
  LITERT_ABORT_IF_ERROR(modelLay.HasValue());
  auto format = type->Layout().Dimensions();
  for (auto lay : modelLay.Value()) {
    auto nums = lay.Dimensions();
    for (auto ten : nums)
      std::cout << ten << " x ";
    std::cout << std::endl;
  };

  resizeDimension.width = format[1];
  resizeDimension.height = format[2];
  return 0;
};

palmy::markerOut handMarker::operator<<(palmy::palmOut in) {

  points.clear();
  auto [pic, boxes] = in;

  struct lpoint {
    float x;
    float y;
    float z;
  };
  std::vector<float> jojo(63);
  float score[1024];

  for (auto &entry : boxes) {

    cv::Mat frame = pic(cv::Rect(entry.x, entry.y, entry.w, entry.h));
    cv::resize(frame, frame, resizeDimension);
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    frame.convertTo(frame, CV_32F, 1.0 / 255.0);
    inputBuffers[0].Write<float>(absl::MakeConstSpan(
        frame.ptr<float>(), frame.total() * frame.channels()));

    bool async = false;
    LITERT_ABORT_IF_ERROR(
        handModel.RunAsync(0, inputBuffers, outputBuffers, async));

    outputBuffers[0].Read<float>(absl::MakeSpan(jojo));
    struct lpoint *looped = (struct lpoint *)jojo.data();

    for (size_t i = 0; i < 21; i++) {
      int x = std::ceil(((looped[i].x / 224) * entry.w) + entry.x);
      int y = std::ceil(((looped[i].y / 224) * entry.h) + entry.y);
      cv::circle(pic, cv::Point(x, y), 3, cv::Scalar(0,0,255), -1);
      points.push_back({static_cast<float>(x),static_cast<float>(y),looped[i].y});
    }
    for (auto &output : outputBuffers)
      output.ClearEvent();
  }

  return {pic, points};
}
}; // namespace palmy
