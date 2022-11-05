#include "boeFaceQuality.hpp"
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

boeFaceQuality::boeFaceQuality(const char *modelPath){

    m_errorCode = EC_QUALITY_SUCCESS;
    MNN::ScheduleConfig mnnNetConfig;
    mnnNetConfig.type = MNN_FORWARD_CPU;
    mnnNetConfig.numThread = 2;
    m_mnnNet_quality = NULL;
    m_mnnSession_quality = NULL;
    m_pretreat_quality = NULL;

    if (!modelPath){
        m_errorCode = EC_INPUT_ERROR;
        return;
    }
    strcpy(m_modelPath, modelPath);

    ImageProcess::Config processConfigQuality;
    processConfigQuality.filterType = BILINEAR;
    processConfigQuality.sourceFormat = BGR;
    processConfigQuality.destFormat   = BGR;

    ::memcpy(processConfigQuality.mean, meanValues, sizeof(meanValues));
    ::memcpy(processConfigQuality.normal, normValues, sizeof(normValues));

    m_mnnNet_quality = MNN::Interpreter::createFromFile(modelPath);
    m_mnnSession_quality = m_mnnNet_quality->createSession(mnnNetConfig);
    m_pretreat_quality = ImageProcess::create(processConfigQuality);
    auto input = m_mnnNet_quality->getSessionInput(m_mnnSession_quality, "input.1");

    m_modelW = input->width();
    m_modelH = input->height();
    m_modelC = input->channel();


#if SHOW_MODEL
    int o_modelW = output->width();
    int o_modelH = output->height();
    int o_modelC = output->channel();
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

boeFaceQuality::~boeFaceQuality(){

    if (m_mnnSession_quality)
    {
        m_mnnNet_quality->releaseSession(m_mnnSession_quality);
        m_mnnSession_quality = NULL;
    }
    if (m_mnnNet_quality)
    {
        m_mnnNet_quality->releaseModel();
        m_mnnNet_quality = NULL;
    }
    if (m_pretreat_quality)
    {
        delete m_pretreat_quality;
        m_pretreat_quality = NULL;
    }

}



float sigmond(float x){
    return (1 / (1 + exp(-x)));
}


int activation_function_sigmond(std::vector<float> src, std::vector<float>& dst, int length)
{
    for (int i=0; i<length; i++){
        float tmp;
        tmp = 1.0f / (1.0f+exp(-1.0f * src[i]));
        dst.push_back(tmp);
    }
    // std::cout<<std::endl;
	return 0;
}
ErrCode boeFaceQuality::qualityInfer(cv::Mat& originImg, float& quality){

    cv::resize(originImg, originImg, cv::Size(112, 112));

    if (!m_mnnNet_quality)
    {
        printf("error: CFaceDetection::FaceDetectImp(), m_mnnNet_det is null.\n");
        cout<< EC_MODEL_WRONG<<endl;;
    }


    // auto input = m_mnnNet_quality->getSessionInput(m_mnnSession_quality, nullptr);
    // auto output = m_mnnNet_quality->getSessionOutput(m_mnnSession_quality, nullptr);
    auto input = m_mnnNet_quality->getSessionInput(m_mnnSession_quality, "input.1");
    auto output = m_mnnNet_quality->getSessionOutput(m_mnnSession_quality, nullptr);

    boeImageData image;
    image.width = originImg.cols;
    image.height = originImg.rows;
    image.channels = originImg.channels();
    image.data = originImg.data;
    image.step = originImg.step[0];

    m_pretreat_quality->convert(image.data, image.width, image.height, image.step, input);
    m_mnnNet_quality->runSession(m_mnnSession_quality);

    auto nchwTensor = new Tensor(output, Tensor::CAFFE);
    output->copyToHostTensor(nchwTensor);
    auto data = nchwTensor->host<float>();

    // std::vector<float> result;
    // float tmp;

    // for (int k = 0; k<11; k++){
    //     tmp = sigmond(data[k]);
    //     tmp = round(tmp * 100) / 100;
    //     result.push_back(tmp);
    // }

    std::vector<float> weight {-0.6, 0.9, -0.7, -0.6, -0.4, -0.5, -0.3, -0.1, 0.05, 0, 0, 0};
    float quality_score = 0.0;

    std::vector<float> quality_src;
	std::vector<float> quality_dst;

    for (int j = 0; j < 12; j++){
        quality_src.push_back(data[j]);
    }

    activation_function_sigmond(quality_src, quality_dst, weight.size());

    for (int j = 0; j < 12; j++){
        quality_score += quality_dst[j]*weight[j];
    }
    if (quality_score<=0.0) quality_score = 0.0;
    if (quality_score>=1.0) quality_score = 1.0;
    quality = quality_score;


    return EC_QUALITY_SUCCESS;
}

