#ifndef __BOEREC_HPP__
#define __BOEREC_HPP__
#include "common_struct.hpp"
#include "opencv2/opencv.hpp"
#include <string>
#include "boeRetinaFace.hpp"
#include "boeFaceLandmark.hpp"
#include "boeFaceQuality.hpp"
#include "boeFaceFeature.hpp"
#include "face_library.h"
#include "common_alignment.hpp"

using namespace std;
using namespace cv;

#define FEATURE_LENGTH 512
#define QUALITY_THRESHOLD 0.5
#define REC_THRESHOLD 0.5


class BoeRec {
public:
    BoeRec(const char* faceLibPath, const char *detModelPath, const char *lmModelPath, const char *quaModelPath, const char *recModelPath);
    ~BoeRec();

    ErrCode registerInfer(cv::Mat& inp_img, int Id);
    ErrCode recInfer(cv::Mat& inp_img, int& id, float& similarity);


private:

    ErrCode getFeatureInfer(cv::Mat& inpImg, float* feature);
    boeRetinaFace* detObj;
    boeFaceLandmark* landmarkObj;
    boeFaceQuality* qualityObj;
    boeFaceFeature* featureObj;
    boeFaceLibrary facelibrary;
};


#endif // __BOERETINAFACE_HPP__