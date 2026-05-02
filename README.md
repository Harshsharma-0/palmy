### PALMY 
 - A simple HandLand-Mark Detection pipeline using opencv and liteRT.
> - Detects 21 HandLand-Marks and give 3-D point's relative to the frame.
> - Use LiteRT successor of tensorflow-lite.
> - Usage google palm Detection Model and HandLandMarker model to process hand

## INSTALLATION 
 - Copy the repo
    ```bash
    git clone https://github.com/Harshsharma-0/palmy.git && cd palmy
   ```
 - Run Makefile
    ```bash
    make init
    ```
 - Run the example
    ```bash
    cd build && make && ./open
    ```
  - Using in your own Code
    - CMake: Add these lines
      ```cmake
       target_link_libraries(${path_to_palmy}/build/libpalmy.a)
       include_directories(${path_to_palmy}/include)
       ```
    - Makefile: Use Compile Flags
       ```make 
        -I${path_to_palmy}/include -l${path_to_palmy}/build/libpalmy.a
       ```
      


### USAGE EXAMPLE 
```cpp
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
    auto [clone,points] = marker << data;

    cv::flip(clone, clone, 1);
    cv::imshow("palm detection", clone);
    auto key = cv::waitKey(16);
    if (key == 27)
      break; // ESC to exit
  }

  return 0;
}
```


