#include "boeFaceFeature.hpp"
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


#define WRITE 1
#define SHOW_MODEL 0

boeFaceFeature::boeFaceFeature(const char *modelPath){

    m_errorCode = EC_FEATURE_SUCCESS;
    MNN::ScheduleConfig mnnNetConfig;
    mnnNetConfig.type = MNN_FORWARD_CPU;
    mnnNetConfig.numThread = 2;
    m_mnnNet_feature = NULL;
    m_mnnSession_feature = NULL;
    m_pretreat_feature = NULL;

    if (!modelPath){
        m_errorCode = EC_INPUT_ERROR;
        return;
    }
    strcpy(m_modelPath, modelPath);

    ImageProcess::Config processConfigFeature;
    processConfigFeature.filterType = BILINEAR;
    processConfigFeature.sourceFormat = BGR;
    processConfigFeature.destFormat   = BGR;

    ::memcpy(processConfigFeature.mean, meanValues, sizeof(meanValues));
    ::memcpy(processConfigFeature.normal, normValues, sizeof(normValues));

    // Key:
    m_mnnNet_feature = MNN::Interpreter::createFromFile(modelPath);
    m_mnnSession_feature = m_mnnNet_feature->createSession(mnnNetConfig);
    m_pretreat_feature = ImageProcess::create(processConfigFeature);
    auto input = m_mnnNet_feature->getSessionInput(m_mnnSession_feature, nullptr);


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

boeFaceFeature::~boeFaceFeature(){

    if (m_mnnSession_feature)
    {
        m_mnnNet_feature->releaseSession(m_mnnSession_feature);
        m_mnnSession_feature = NULL;
    }
    if (m_mnnNet_feature)
    {
        m_mnnNet_feature->releaseModel();
        m_mnnNet_feature = NULL;
    }
    if (m_pretreat_feature)
    {
        delete m_pretreat_feature;
        m_pretreat_feature = NULL;
    }

}


ErrCode boeFaceFeature::featureInfer(cv::Mat& originImg, float* feature){

    if (!m_mnnNet_feature)
    {
        printf("error: CFaceDetection::FaceDetectImp(), m_mnnNet_det is null.\n");
        cout<< EC_MODEL_WRONG<<endl;;
    }

    auto input = m_mnnNet_feature->getSessionInput(m_mnnSession_feature, nullptr);
    auto output = m_mnnNet_feature->getSessionOutput(m_mnnSession_feature, nullptr);

    boeImageData image;
    image.width = originImg.cols;
    image.height = originImg.rows;
    image.channels = originImg.channels();
    image.data = originImg.data;
    image.step = originImg.step[0];

    m_pretreat_feature->convert(image.data, image.width, image.height, image.step, input);
    m_mnnNet_feature->runSession(m_mnnSession_feature);

    auto nchwTensor = new Tensor(output, Tensor::CAFFE);
    output->copyToHostTensor(nchwTensor);
    auto data = nchwTensor->host<float>();

    std::vector<float> feature_tmp;
    float sum = 0;
    for (int i = 0; i < 512; ++i){
        sum += data[i] * data[i];
    }
    sum = 1.0f / sqrt(sum);
    for (int j = 0; j < 512; ++j){
        feature[j] = data[j] * sum;
    }
    return EC_FEATURE_SUCCESS;
}

