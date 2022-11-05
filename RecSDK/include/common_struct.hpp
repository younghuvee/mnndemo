#ifndef COMMON_STRUCT_H
#define COMMON_STRUCT_H
#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;


#define WINDOWS_SYSTEM 0
#if WINDOWS_SYSTEM
#define DLL_EXPORT extern "C" __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

struct RegisterFace
{
    int id;
    char name[64];
};


enum ErrCode : unsigned int
{
    // EC_OK = 0,
    EC_MODEL_NOT_FOUND = 1001,
    EC_MODEL_PARSE_ERROR = 1002,
    EC_MODEL_WRONG = 1003,
    EC_INPUT_ERROR = 1004,

    EC_FACE_LIBRARY_SUCCESS = 2000,
    EC_FACE_LIBRARY_NOT_FOUND = 2001,
    EC_FACE_LIBRARY_FULL = 2002,
    EC_FACE_ID_EXIST = 2003,
    EC_FACE_NOT_FOUND = 2004,
    EC_FACE_TOO_SMALL = 2005,
    EC_IMAGE_TOO_SMALL = 2006,
    EC_FACECROP_SUCCESS = 2007,
    EC_LOW_QUALITY = 2007,

    EC_INTERNAL_ERROR = 3000,
    EC_NO_OUTPUT = 3001,

    EC_DETECT_SUCCESS = 9001,
    EC_FEATURE_SUCCESS = 9002,
    EC_MASK_FEATURE_SUCCESS = 9003,
    EC_GENDER_SUCCESS = 9004,
    EC_AGE_SUCCESS = 9005,
    EC_EMOTION_SUCCESS = 9006,
    EC_SCORE_SUCCESS = 9007,
    EC_LANDMARK_SUCCESS = 9008,
    EC_MASK_SUCCESS = 9009,
    EC_RACE_SUCCESS = 9010,
    EC_QUALITY_SUCCESS = 9011,
    EC_GLASS_SUCCESS = 9012,
    EC_BEARD_SUCCESS = 9013,
    EC_SPOOF_SUCCESS = 9014,
    EC_SHELTER_SUCCESS = 9015,
    EC_POSE_SUCCESS = 9016,
    EC_ATTRIBUTE_SUCCESS = 9017,
    EC_DBNET_SUCCESS = 9018,
    EC_AGEGENDER_SUCCESS = 9018,
    EC_EYECLOSENET_SUCCESS = 9020,
    EC_GAZETRACKING_SUCCESS = 9021,
    EC_LANDMARK68_SUCCESS = 9022,
    EC_GET_FEATURE_SUCCESS = 9022,

    EC_RECOGNITION_SUCCESS = 9101,
    EC_REGISTER_SUCCESS = 9101,
    EC_AUTH_FAIL = 9999
};

struct boeFaceLibrary
{
    void *library;
    enum ErrCode errCode;
};

struct boeModel
{
    void *model;
    char name[32];
    enum ErrCode errCode;
};

struct boePointF
{
    double x;
    double y;
};


struct boeImageData
{
    unsigned char *data;
    int width;
    int height;
    int channels;
    int step;
};

struct boeBox
{
    /* 
     * cx: x of box center
     * cy: y of box center
     * sx: width of box
     * sy: height of box
     */

    float cx;
    float cy;
    float sx;
    float sy;
};


struct boeRect
{
    int x;
    int y;
    int x1;
    int y1;
    int width;
    int height;
    float score;
    int classId;
};


struct boeBBox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    boePointF landmarks[5];
};

struct boeFaceInfo
{
    std::vector<boeRect> rects;                  //人脸检测框
    std::vector<std::vector<boePointF>> landmarks;  //人脸关键点
    std::vector<std::vector<float>> features;     //人脸特征
    std::vector<int> ages;
    std::vector<int> genders;
    std::vector<float> qualities;
};

#endif