#include "boeFaceLandmark.hpp"
#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#include "opencv2/opencv.hpp"

#if WINDOWS_SYSTEM
#include <io.h>
#else
#include <unistd.h>
#endif
#include <string>
using namespace MNN;
using namespace MNN::CV;
using namespace std;


#define SHOW_MODEL 0

boeFaceLandmark::boeFaceLandmark(const char *modelPath){

    m_errorCode = EC_LANDMARK_SUCCESS;
    MNN::ScheduleConfig mnnNetConfig;
    mnnNetConfig.type = MNN_FORWARD_CPU;
    mnnNetConfig.numThread = 2;
    m_mnnNet_landmark = NULL;
    m_mnnSession_landmark = NULL;
    m_pretreat_landmark = NULL;

    if (!modelPath){
        m_errorCode = EC_INPUT_ERROR;
        return;
    }
    strcpy(m_modelPath, modelPath);

    ImageProcess::Config processConfigLandmark;
    processConfigLandmark.filterType = BILINEAR;
    processConfigLandmark.sourceFormat = BGR;
    processConfigLandmark.destFormat   = BGR;

    ::memcpy(processConfigLandmark.mean, meanValues, sizeof(meanValues));
    ::memcpy(processConfigLandmark.normal, normValues, sizeof(normValues));

    m_mnnNet_landmark = MNN::Interpreter::createFromFile(modelPath);
    m_mnnSession_landmark = m_mnnNet_landmark->createSession(mnnNetConfig);
    m_pretreat_landmark = ImageProcess::create(processConfigLandmark);
    auto input = m_mnnNet_landmark->getSessionInput(m_mnnSession_landmark, nullptr);


    m_modelW = input->width();
    m_modelH = input->height();
    m_modelC = input->channel();


#if SHOW_MODEL
    int o_modelW = input->width();
    int o_modelH = input->height();
    int o_modelC = input->channel();
    cout<<"##############    iW iH iC oW oH oC    ##############"<<endl;
    cout<<m_modelW<<endl;
    cout<<m_modelH<<endl;
    cout<<m_modelC<<endl;
    cout<<o_modelW<<endl;
    cout<<o_modelH<<endl;
    cout<<o_modelC<<endl;
    cout<<"#####################################################"<<endl;
#endif
}

boeFaceLandmark::~boeFaceLandmark(){

    if (m_mnnSession_landmark)
    {
        m_mnnNet_landmark->releaseSession(m_mnnSession_landmark);
        m_mnnSession_landmark = NULL;
    }
    if (m_mnnNet_landmark)
    {
        m_mnnNet_landmark->releaseModel();
        m_mnnNet_landmark = NULL;
    }
    if (m_pretreat_landmark)
    {
        delete m_pretreat_landmark;
        m_pretreat_landmark = NULL;
    }

}

ErrCode boeFaceLandmark::landmarkInfer(cv::Mat& cropImg, const boeRect rect, std::vector<boePointF> &points){


    if (!m_mnnNet_landmark)
    {
        printf("error: CFaceDetection::FaceDetectImp(), m_mnnNet_det is null.\n");
        cout<< EC_MODEL_WRONG<<endl;;
    }

    auto input = m_mnnNet_landmark->getSessionInput(m_mnnSession_landmark, nullptr);
    auto output = m_mnnNet_landmark->getSessionOutput(m_mnnSession_landmark, nullptr);

    boeImageData image;
    image.width = cropImg.cols;
    image.height = cropImg.rows;
    image.channels = cropImg.channels();
    image.data = cropImg.data;
    image.step = cropImg.step[0];

    m_pretreat_landmark->convert(image.data, image.width, image.height, image.step, input);
    m_mnnNet_landmark->runSession(m_mnnSession_landmark);

    auto nchwTensor = new Tensor(output, Tensor::CAFFE);
    output->copyToHostTensor(nchwTensor);
    auto data = nchwTensor->host<float>();

    // for (int i=0; i<output->width()*output->height(); i++){
    //     cout<<data[i]<<endl;
    // }

    boePointF left_eye, right_eye, p3, p4, p5;
    left_eye.x = data[0];
    left_eye.y = data[1];
    right_eye.x = data[2];
    right_eye.y = data[3];
    p3.x = data[4];
    p3.y = data[5];
    p4.x = data[6];
    p4.y = data[7];
    p5.x = data[8];
    p5.y = data[9];

    std::vector<boePointF> pointsCrop;
    pointsCrop.push_back(left_eye);
    pointsCrop.push_back(right_eye);
    pointsCrop.push_back(p3);
    pointsCrop.push_back(p4);
    pointsCrop.push_back(p5);

    float ratio_x = (float)rect.width / (float)m_modelW;
    float ratio_y = (float)rect.height / (float)m_modelH;
    for (int j = 0; j < 5; j++)
    {
        boePointF pt;
        pt.x = (float)rect.x + ratio_x * pointsCrop[j].x;
        pt.y = (float)rect.y + ratio_y * pointsCrop[j].y;
        points.push_back(pt);
    }
    return EC_LANDMARK_SUCCESS;
}
