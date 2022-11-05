cmake_minimum_required (VERSION 2.8)

#project (boe_sdk)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -std=c++14 -march=native -O3 -pthread -DELPP_NO_DEFAULT_LOG_FILE -DELPP_THREAD_SAFE")


#include
include_directories(${PROJECT_SOURCE_DIR}/3rd-party/opencv3.4.9/include)
include_directories(${PROJECT_SOURCE_DIR}/3rd-party/MNN/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

#lib
link_directories(${PROJECT_SOURCE_DIR}/3rd-party/opencv3.4.9/lib)
link_directories(${PROJECT_SOURCE_DIR}/3rd-party/MNN/lib)

set(FACE_LIST ${PROJECT_SOURCE_DIR}/src/main.cpp ${PROJECT_SOURCE_DIR}/src/boeFaceLandmark.cpp ${PROJECT_SOURCE_DIR}/src/boeFaceFeature.cpp ${PROJECT_SOURCE_DIR}/src/boeRetinaFace.cpp 
 ${PROJECT_SOURCE_DIR}/src/boeRec.cpp ${PROJECT_SOURCE_DIR}/src/boeFaceQuality.cpp ${PROJECT_SOURCE_DIR}/src/face_library.cpp ${PROJECT_SOURCE_DIR}/src/common_alignment.cpp)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(RecInfer ${FACE_LIST})

target_link_libraries(RecInfer)
target_link_libraries(RecInfer opencv_shape opencv_stitching opencv_objdetect opencv_superres opencv_videostab opencv_calib3d opencv_features2d
opencv_highgui opencv_videoio opencv_imgcodecs opencv_video opencv_photo opencv_ml opencv_imgproc opencv_flann opencv_core)
target_link_libraries(RecInfer -Wl,--whole-archive libMNN.so -Wl,--no-whole-archive)