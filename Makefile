TARGET_LIBS= -lopencv_core -lopencv_video -lopencv_videoio -lopencv_highgui -lopencv_imgproc
LITERT_CMDLINE = -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++

PALM_MODEL_URL=https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite 
PALM_MODEL_LITE_URL=https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite 
HAND_MODEL_URL=https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite 
HAND_MODEL_LITE_URL=https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite

LITERT_MAIN_SO_URL=https://storage.googleapis.com/litert/binaries/2.1.4/linux_x86_64/libLiteRt.so
LITERT_WEBGPU_SO_URL=https://storage.googleapis.com/litert/binaries/2.1.4/linux_x86_64/libLiteRtWebGpuAccelerator.so
LITERT_SDK_URL=https://github.com/google-ai-edge/LiteRT/releases/download/v2.1.4/litert_cc_sdk.zip

BUILD_DIR=./build

all:

litert:
	mkdir -p liteRT
	wget ${LITERT_SDK_URL} &&\
	unzip ./litert_cc_sdk.zip -d ./ &&\
	cd ./litert_cc_sdk &&\
	wget ${LITERT_MAIN_SO_URL} && \
	wget ${LITERT_WEBGPU_SO_URL} &&\
	cd ../ &&\
	rm ./litert_cc_sdk.zip


gen: genanchor.js
	node ./genanchor.js > ./src/anchor.cpp

cmake:CMakeLists.txt
	 cmake ./ -B build ${LITERT_CMDLINE} && ln -s $(shell pwd)/build/compile_commands.json $(shell pwd)/compile_commands.json

model:
	wget ${PALM_MODEL_URL} -O ${BUILD_DIR}/palm_detection_full.tflite
	wget ${PALM_MODEL_LITE_URL} -O ${BUILD_DIR}/palm_detection_lite.tflite
	wget ${HAND_MODEL_URL} -O ${BUILD_DIR}/hand_landmark_full.tflite
	wget ${HAND_MODEL_LITE_URL} -O ${BUILD_DIR}/hand_landmark_lite.tflite


init:gen cmake model

