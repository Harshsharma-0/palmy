#include "palmy/detector.hpp"
#include "palmy/handmarker.hpp"

int main() {

  LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));
  palmy::detectorPalm palmer("./palm_detection_full.tflite", env);
  palmy::handMarker marker("./hand_landmark_full.tflite", env);

  cv::VideoCapture cap(0);

  if (!cap.isOpened()) {
    std::cerr << "camera not workding \n" << std::endl;
    return -1;
  }

  while (1) {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty())
      break;

    auto data  = palmer << frame;
    auto [clone,boxes] = marker << data;

    cv::flip(clone, clone, 1);
    cv::imshow("palm detection", clone);
    auto key = cv::waitKey(16);
    if (key == 27)
      break; // ESC to exit
  }

  return 0;
}
