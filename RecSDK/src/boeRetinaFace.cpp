#include "boeRetinaFace.hpp"
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

boeRetinaFace::boeRetinaFace(const char *modelPath){
    m_errorCode = EC_DETECT_SUCCESS;
    MNN::ScheduleConfig mnnNetConfig;
    mnnNetConfig.type = MNN_FORWARD_CPU;
    mnnNetConfig.numThread = 2;
    m_mnnNet_detect = NULL;
    m_mnnSession_detect = NULL;
    m_pretreat_detect = NULL;

    if (!modelPath){
        m_errorCode = EC_INPUT_ERROR;
        return;
    }
    strcpy(m_modelPath, modelPath);
    ImageProcess::Config processConfigDetect;
    processConfigDetect.filterType = BILINEAR;
    processConfigDetect.sourceFormat = BGR;
    processConfigDetect.destFormat   = BGR;

    ::memcpy(processConfigDetect.mean, meanValues, sizeof(meanValues));
    // ::memcpy(processConfigDetect.normal, normValues, sizeof(normValues));

    m_mnnNet_detect = MNN::Interpreter::createFromFile(modelPath);
    m_mnnSession_detect = m_mnnNet_detect->createSession(mnnNetConfig);
    m_pretreat_detect = ImageProcess::create(processConfigDetect);
    auto input = m_mnnNet_detect->getSessionInput(m_mnnSession_detect, nullptr);

    m_mnnNet_detect->resizeTensor(input, {1, 3, _in_h, _in_w});
    m_mnnNet_detect->resizeSession(m_mnnSession_detect);

    m_modelW = input->width();
    m_modelH = input->height();
    m_modelC = input->channel();
    this->create_anchors(anchors, _in_w, _in_h);


#if SHOW_MODEL
    auto output1 = m_mnnNet_detect->getSessionOutput(m_mnnSession_detect, "836");
    auto output2 = m_mnnNet_detect->getSessionOutput(m_mnnSession_detect, "762");
    auto output3 = m_mnnNet_detect->getSessionOutput(m_mnnSession_detect, "837");

    int o1_modelW = output1->width();
    int o1_modelH = output1->height();
    int o1_modelC = output1->channel();
    int o2_modelW = output2->width();
    int o2_modelH = output2->height();
    int o2_modelC = output2->channel();
    int o3_modelW = output3->width();
    int o3_modelH = output3->height();
    int o3_modelC = output3->channel();
    cout<<"##############    iW iH iC oW oH oC    ##############"<<endl;
    cout<<m_modelW<<endl;
    cout<<m_modelH<<endl;
    cout<<m_modelC<<endl;
    cout<<o1_modelW<<endl;
    cout<<o1_modelH<<endl;
    cout<<o1_modelC<<endl;
    cout<<o2_modelW<<endl;
    cout<<o2_modelH<<endl;
    cout<<o2_modelC<<endl;
    cout<<o3_modelW<<endl;
    cout<<o3_modelH<<endl;
    cout<<o3_modelC<<endl;
    cout<<"#####################################################"<<endl;
#endif
}

boeRetinaFace::~boeRetinaFace(){

    if (m_mnnSession_detect)
    {
        m_mnnNet_detect->releaseSession(m_mnnSession_detect);
        m_mnnSession_detect = NULL;
    }
    if (m_mnnNet_detect)
    {
        m_mnnNet_detect->releaseModel();
        m_mnnNet_detect = NULL;
    }
    if (m_pretreat_detect)
    {
        delete m_pretreat_detect;
        m_pretreat_detect = NULL;
    }

}


void boeRetinaFace::create_anchors(std::vector<boeBox>& anchors, int w, int h) const
{
    // create predefined anchors
    anchors.clear();
    std::vector<std::vector<int> > feature_map(3), anchor_sizes(3);
    float strides[3] = {8, 16, 32};
    for (uint16_t i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/strides[i]));
        feature_map[i].push_back(ceil(w/strides[i]));
    }
    std::vector<int> stage1_size = {8, 16};
    anchor_sizes[0] = stage1_size;
    std::vector<int> stage2_size = {32, 64};
    anchor_sizes[1] = stage2_size;
    std::vector<int> stage3_size = {128, 256};
    anchor_sizes[2] = stage3_size;

    for (uint16_t k = 0; k < feature_map.size(); ++k) {
        std::vector<int> anchor_size = anchor_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i) {
            for (int j = 0; j < feature_map[k][1]; ++j) {
                for (uint16_t l = 0; l < anchor_size.size(); ++l) {
                    float kx = anchor_size[l]* 1.0 / w;
                    float ky = anchor_size[l]* 1.0 / h;
                    float cx = (j + 0.5) * strides[k] / w;
                    float cy = (i + 0.5) * strides[k] / h;
                    anchors.push_back({cx, cy, kx, ky});
                }
            }
        }
    }
}


inline void boeRetinaFace::clip_bboxes(boeBBox& bbox, int w, int h) const
{
    if(bbox.x1 < 0) bbox.x1 = 0;
    if(bbox.y1 < 0) bbox.y1 = 0;
    if(bbox.x2 > w) bbox.x2 = w;
    if(bbox.y2 > h) bbox.y2 = h;
}

void boeRetinaFace::nms(std::vector<boeBBox>& bboxes, float nms_threshold) const
{
    std::vector<float> bbox_areas(bboxes.size());
    for (uint16_t i = 0; i < bboxes.size(); i++) {
        bbox_areas[i] = (bboxes.at(i).x2 - bboxes.at(i).x1 + 1) * (bboxes.at(i).y2 - bboxes.at(i).y1 + 1);
    }

    for (uint16_t i = 0; i < bboxes.size(); i++) {
        for (uint16_t j = i + 1; j < bboxes.size(); ) {
            float xx1 = std::max(bboxes[i].x1, bboxes[j].x1);
            float yy1 = std::max(bboxes[i].y1, bboxes[j].y1);
            float xx2 = std::min(bboxes[i].x2, bboxes[j].x2);
            float yy2 = std::min(bboxes[i].y2, bboxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float IoU = inter / (bbox_areas[i] + bbox_areas[j] - inter);
            if (IoU >= nms_threshold) {
                bboxes.erase(bboxes.begin() + j);
                bbox_areas.erase(bbox_areas.begin() + j);
            } else {
                j++;
            }
        }
    }
}

cv::Mat ResizeImageKeepASP(cv::Mat input, int widthNew, int heightNew)
{
    cv::Mat output = cv::Mat(cv::Size(widthNew, heightNew), CV_8UC3, cv::Scalar(114, 114, 114));
    int width = input.cols;
    int height = input.rows;
    int widthNew2;
    int heightNew2;
    cv::Mat temImage;
    cv::Mat imageROI;
    if (widthNew * height> heightNew * width){
        heightNew2 = heightNew;
        widthNew2 = width * heightNew / height;
        cv::resize(input, temImage, cv::Size(widthNew2, heightNew2), 0, 0, cv::INTER_LINEAR);
        imageROI = output(cv::Rect((widthNew - widthNew2) / 2, 0, temImage.cols, temImage.rows));
        temImage.copyTo(imageROI);
    } else {
        widthNew2 = widthNew;
        heightNew2 = height * widthNew / width;
        cv::resize(input, temImage, cv::Size(widthNew2, heightNew2), 0, 0, cv::INTER_LINEAR);
        imageROI = output(cv::Rect(0, (heightNew - heightNew2) / 2, temImage.cols, temImage.rows));
        temImage.copyTo(imageROI);
    }
    return output;
}


ErrCode boeRetinaFace::detectInfer(cv::Mat& originImg, std::vector<boeRect>& rects){

    std::vector<boeBBox> final_bboxes;
    if (!m_mnnNet_detect)
    {
        printf("error: CFaceDetection::FaceDetectImp(), m_mnnNet_det is null.\n");
        cout<< EC_MODEL_WRONG<<endl;;
    }
    auto input = m_mnnNet_detect->getSessionInput(m_mnnSession_detect, nullptr);
    auto loc = m_mnnNet_detect->getSessionOutput(m_mnnSession_detect, "378");
    auto conf = m_mnnNet_detect->getSessionOutput(m_mnnSession_detect, "414");

    cv::Mat orgImg;
    if(originImg.cols != m_modelW || originImg.rows != m_modelH) {
        orgImg = ResizeImageKeepASP(originImg, m_modelW, m_modelH);
    }
    else{
        orgImg = originImg.clone();
    }

    boeImageData image;
    image.width = orgImg.cols;
    image.height = orgImg.rows;
    image.channels = orgImg.channels();
    image.data = orgImg.data;
    image.step = orgImg.step[0];

    m_pretreat_detect->convert(image.data, image.width, image.height, image.step, input);
    m_mnnNet_detect->runSession(m_mnnSession_detect);

    auto nchwTensor_loc = new Tensor(loc, Tensor::CAFFE);
    auto nchwTensor_conf = new Tensor(conf, Tensor::CAFFE);
    loc->copyToHostTensor(nchwTensor_loc);
    conf->copyToHostTensor(nchwTensor_conf);


    auto offsets = nchwTensor_loc->host<float>();
    auto scores = nchwTensor_conf->host<float>();

    for (const boeBox& anchor : anchors) {
        boeBBox bbox;
        boeBox refined_box;
        if (scores[1] > this->_score_threshold) {
            // score
            bbox.score = scores[1];

            // bbox
            refined_box.cx = anchor.cx + offsets[0] * 0.1 * anchor.sx;
            refined_box.cy = anchor.cy + offsets[1] * 0.1 * anchor.sy;
            refined_box.sx = anchor.sx * exp(offsets[2] * 0.2);
            refined_box.sy = anchor.sy * exp(offsets[3] * 0.2);

            bbox.x1 = (refined_box.cx - refined_box.sx/2) * image.width;
            bbox.y1 = (refined_box.cy - refined_box.sy/2) * image.height;
            bbox.x2 = (refined_box.cx + refined_box.sx/2) * image.width;
            bbox.y2 = (refined_box.cy + refined_box.sy/2) * image.height;

            clip_bboxes(bbox, image.width, image.height);

            final_bboxes.push_back(bbox);
        }

        scores += 2;
        offsets += 4;
    }
    std::sort(final_bboxes.begin(), final_bboxes.end(), [](boeBBox &lsh, boeBBox &rsh) {
        return lsh.score > rsh.score;
    });
    nms(final_bboxes, this->_nms_threshold);

    // float scale_x = (float)org_width / (float)m_modelW;
    // float scale_y = (float)org_height / (float)m_modelH;
    // for (int r=0; r<final_bboxes.size(); r++){
    //     boeRect rect;
    //     rect.x = final_bboxes[r].x1 * scale_x;
    //     rect.y = final_bboxes[r].y1 * scale_y;
    //     rect.width = (final_bboxes[r].x2 - final_bboxes[r].x1) * scale_x;
    //     rect.height = (final_bboxes[r].y2 - final_bboxes[r].y1) * scale_y;
    //     rect.score = final_bboxes[r].score;
    //     rects.push_back(rect);
    // }
    cout<<"faces numbers: "<<final_bboxes.size()<<endl;

    for (uint16_t r = 0; r<final_bboxes.size(); r++){
        boeRect rect_;
        float ratio_x, ratio_y;

        float ori_img_width = (float) originImg.cols;
        float ori_img_height = (float) originImg.rows;
        if (originImg.cols * _in_h == originImg.rows * _in_w){
            ratio_x = ori_img_width / _in_w;
            ratio_y = ori_img_height / _in_h;
            rect_.x = final_bboxes[r].x1 * ratio_x;
            rect_.y = final_bboxes[r].y1 * ratio_y;
            rect_.x1 = final_bboxes[r].x2* ratio_x;
            rect_.y1 = final_bboxes[r].y2 * ratio_y;
        }
        else if (originImg.cols * _in_h > originImg.rows * _in_w)
        {
            ratio_x = ori_img_width / _in_w;
            ratio_y = ori_img_width / _in_w;


            rect_.x = final_bboxes[r].x1 * ratio_x;
            rect_.y = (final_bboxes[r].y1 - (_in_h / 2 - ori_img_height / ratio_y / 2.0f)) * ratio_y;

            rect_.x1 = final_bboxes[r].x2 * ratio_x;
            rect_.y1 = (final_bboxes[r].y2 - (_in_h / 2 - ori_img_height / ratio_y / 2.0f)) * ratio_y;

        }
        else{
            ratio_x = ori_img_height / _in_h;
            ratio_y = ori_img_height / _in_h;

            rect_.x = (final_bboxes[r].x1 - (_in_w / 2 - ori_img_width / ratio_x / 2.0f)) * ratio_x;
            rect_.y = final_bboxes[r].y1 * ratio_y;
            rect_.x1 = (final_bboxes[r].x2 - (_in_w / 2 - ori_img_width / ratio_x / 2.0f)) * ratio_x;
            rect_.y1 = final_bboxes[r].y2 * ratio_y;

        }

        rect_.width = rect_.x1 - rect_.x;
        rect_.height = rect_.y1 - rect_.y;
        rect_.score = final_bboxes[r].score;
        rects.push_back(rect_);
    }

    delete nchwTensor_loc;
    delete nchwTensor_conf;

    return EC_DETECT_SUCCESS;
}

