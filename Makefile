TARGET_LIBS= -lopencv_core -lopencv_video -lopencv_videoio -lopencv_highgui -lopencv_imgproc
LITERT_CMDLINE = -DLITERT_LINK="${TARGET_LIBS} ${HOME}/opt/liteRT/libLiteRtWebGpuAccelerator.so"
LITERT_CMDLINE += -DLITERT_DIR=${LITERT_INSTALLED_DIR}
LITERT_CMDLINE += -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++

PALM_MODEL_URL=https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite 
PALM_MODEL_LITE_URL=https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite 
HAND_MODEL_URL=https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite 
HAND_MODEL_LITE_URL=https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite

BUILD_DIR=./build


gen: genanchor.js
	node ./genanchor.js > ./src/anchor.cpp

cmake:CMakeLists.txt
	 cmake ./ -B build ${LITERT_CMDLINE}

model:
	wget ${PALM_MODEL_URL} -O ${BUILD_DIR}/palm_detection_full.tflite
	wget ${PALM_MODEL_LITE_URL} -O ${BUILD_DIR}/palm_detection_lite.tflite
	wget ${HAND_MODEL_URL} -O ${BUILD_DIR}/hand_landmark_full.tflite
	wget ${HAND_MODEL_LITE_URL} -O ${BUILD_DIR}/hand_landmark_lite.tflite


init:gen cmake model

