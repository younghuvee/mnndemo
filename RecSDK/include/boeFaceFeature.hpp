#ifndef __BOEFACEFEATURE_HPP__
#define __BOEFACEFEATURE_HPP__
#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#include "common_struct.hpp"
#include "opencv2/opencv.hpp"
#include <string>

using namespace MNN;
using namespace MNN::CV;

using namespace std;
using namespace cv;

class boeFaceFeature {
public:
  boeFaceFeature(const char *modelPath);
  ~boeFaceFeature();

  ErrCode featureInfer(cv::Mat& originImg, float* feature);

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

  MNN::Interpreter *m_mnnNet_feature;
  MNN::Session *m_mnnSession_feature;
  ImageProcess *m_pretreat_feature;
};


#endif // __BOEFACEFEATURE_HPP__