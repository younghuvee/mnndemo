#ifndef __BOEFACELANDMARK_HPP__
#define __BOEFACELANDMARK_HPP__
#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#include "common_struct.hpp"
#include "opencv2/opencv.hpp"
#include <string>

using namespace MNN;
using namespace MNN::CV;

using namespace std;
using namespace cv;

class boeFaceLandmark {
public:
  boeFaceLandmark(const char *modelPath);
  ~boeFaceLandmark();

  ErrCode landmarkInfer(cv::Mat& cropImg, const boeRect rect, std::vector<boePointF> &points); // 输入裁剪之后的图片 人脸框 输出关键点


private:
  int numThread;
  // ncnn::Net net;
  const float meanValues[3] = {127.5, 127.5, 127.5};
  const float normValues[3] = {1.0 / 0.5 / 255.0, 1.0 / 0.5 / 255.0,
                               1.0 / 0.5 / 255.0};

  ErrCode m_errorCode;
  char m_modelPath[256];
  int m_modelW;
  int m_modelH;
  int m_modelC;

  MNN::Interpreter *m_mnnNet_landmark;
  MNN::Session *m_mnnSession_landmark;
  ImageProcess *m_pretreat_landmark;
};



#endif // __BOEFACELANDMARK_HPP__