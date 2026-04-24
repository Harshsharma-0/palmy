#include "palmy/detector.hpp"
int main() {

  LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));
  palmy::detectorPalm palmer("./palm_detection_full.tflite",env);
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
    cv::Mat clone = frame.clone();

    // Resize
    cv::Mat resized;
    cv::resize(frame, resized,palmer.resizeVal());
    // cv::flip(resized, resized, 1);

    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    auto final_boxes = palmer >> resized;
    
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
}
