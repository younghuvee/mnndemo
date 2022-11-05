#ifndef __BOERETINAFACE_HPP__
#define __BOERETINAFACE_HPP__
#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#include "common_struct.hpp"
#include "opencv2/opencv.hpp"
#include <string>

using namespace MNN;
using namespace MNN::CV;

using namespace std;
using namespace cv;

class boeRetinaFace {
public:
  boeRetinaFace(const char *modelPath);
  ~boeRetinaFace();


  ErrCode detectInfer(cv::Mat& originImg, std::vector<boeRect>& rects);


private:
  void create_anchors(std::vector<boeBox>& anchors, int w, int h) const;
  void clip_bboxes(boeBBox& bbox, int w, int h) const;
  void nms(std::vector<boeBBox>& input_bboxes, float nms_threshold) const;
  int numThread;
  // ncnn::Net net;

  const int _in_w = 640;
  const int _in_h = 480;

  float _nms_threshold = 0.2;
  float _score_threshold = 0.6;

  const float meanValues[3] = {104.f, 117.f, 123.f};
  // const float normValues[3] = {1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0,
  //                              1.0 / 0.225 / 255.0};

  ErrCode m_errorCode;
  char m_modelPath[256];
  int m_modelW;
  int m_modelH;
  int m_modelC;

  MNN::Interpreter *m_mnnNet_detect;
  MNN::Session *m_mnnSession_detect;
  ImageProcess *m_pretreat_detect;

  std::vector<boeBox> anchors;
};


#endif // __BOERETINAFACE_HPP__