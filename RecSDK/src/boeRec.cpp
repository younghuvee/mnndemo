#include "boeRec.hpp"

int FaceCrop(const boeImageData image, const std::vector<boePointF> points, boeImageData &face)
{
    float mean_shape[10] = {
        38.2946f, 51.6963f,
        73.5318f, 51.5014f,
        56.0252f, 71.7366f,
        41.5493f, 92.3655f,
        70.7299f, 92.2041f,
    };
    float pt[10];
    if (points.size() != 5){
        printf("error: CFaceQuality::FaceCrop(), points.size is not 5.\n");
        return EC_INPUT_ERROR;
    }
    if (!image.data || image.width <= 0 || image.height <= 0 || image.channels <= 0 || image.step < image.width * image.channels){
        printf("error: CFaceQuality::FaceCrop(), input image is wrong.\n");
        return EC_INPUT_ERROR;
    }
    if (!face.data || face.width != 112 || face.height != 112 || face.channels != 3 || face.step != face.width * face.channels){
        printf("error: CFaceQuality::FaceCrop(), input face is wrong.\n");
        return EC_INPUT_ERROR;
    }
    for(int i = 0; i < 5; ++i){
        pt[2 * i] = (float)(points[i].x);
        pt[2 * i + 1] = (float)(points[i].y);
    }
    // cout<<"width: "<<image.width<<"channels: "<<image.channels<<"step: "<<image.step<<endl;
    if (image.width * image.channels != image.step) {
        printf("warning: image.width * image.channels != image.step, create new image for detection.\n");
        int width2 = (image.width >> 2) << 2;
        int height2 = (image.height >> 2) << 2;
        unsigned char *buffer = new unsigned char[height2 * width2 * image.channels];
        for (int i = 0; i < height2; i++){
            memcpy(buffer + i * width2 * image.channels, image.data + i * image.step, width2 * image.channels);
        }
        face_crop_core(buffer, width2, height2, image.channels, face.data, 112, 112, pt, 5, mean_shape, 112, 112, 0, 0, 0, 0, NULL);
        delete[] buffer;
        buffer = NULL;
    }
    else {
        face_crop_core(image.data, image.width, image.height, image.channels, face.data, 112, 112, pt, 5, mean_shape, 112, 112, 0, 0, 0, 0, NULL);
    }
    return EC_FACECROP_SUCCESS;
}

static int findMaxFace(std::vector<boeRect> faceRects)
{
    int index = 0;
    int maxFaceSize = 0;
    uint16_t i;
    boeRect rect;
    for (i = 0; i < faceRects.size(); i++){
        rect = faceRects[i];
        if (rect.width <= 0 || rect.height <= 0)
            continue;
        if (maxFaceSize < rect.width * rect.height){
            maxFaceSize = rect.width * rect.height;
            index = i;
        }
    }
    return index;
}


BoeRec::BoeRec(const char* faceLibPath, const char *detModelPath, const char *lmModelPath, const char *quaModelPath, const char *recModelPath){

    detObj = new boeRetinaFace(detModelPath);
    landmarkObj = new boeFaceLandmark(lmModelPath);
    qualityObj = new boeFaceQuality(quaModelPath);
    featureObj = new boeFaceFeature(recModelPath);
    facelibrary = boeFaceLibraryInit(faceLibPath, FEATURE_LENGTH);
    
}

BoeRec::~BoeRec(){
    delete detObj;
    delete landmarkObj;
    delete qualityObj;
    delete featureObj;
}

ErrCode BoeRec::getFeatureInfer(cv::Mat& inpImg, float* feature){

    std::vector<boeRect> rects;
    int maxFaceIndex;
    detObj->detectInfer(inpImg, rects);
    boeImageData image2;
    image2.width = inpImg.cols;
    image2.height = inpImg.rows;
    image2.channels = inpImg.channels();
    image2.data = inpImg.data;
    image2.step = inpImg.step[0];

    if (rects.size()==0){
        return EC_FACE_NOT_FOUND;
    }
    else if (rects.size() == 1)
    {
        maxFaceIndex = 0;
    }
    else{
        maxFaceIndex = findMaxFace(rects);
    }

    cv::Rect rectCv;
    rectCv.x = rects[maxFaceIndex].x;
    rectCv.y = rects[maxFaceIndex].y;
    rectCv.width = rects[maxFaceIndex].width;
    rectCv.height = rects[maxFaceIndex].height;

    cv::Mat imgCrop = inpImg(rectCv);
    cv::resize(imgCrop, imgCrop, cv::Size(112, 112), 0, 0, cv::INTER_AREA);
    std::vector<boePointF> points;
    landmarkObj->landmarkInfer(imgCrop, rects[maxFaceIndex], points);
    boeImageData cropFace;
    cropFace.width = 112;
    cropFace.height = 112;
    cropFace.channels = 3;
    cropFace.step = cropFace.width * cropFace.channels;
    cropFace.data = new unsigned char[cropFace.width * cropFace.height * cropFace.channels];
    FaceCrop(image2, points, cropFace);
    cv::Mat img_f(cropFace.height, cropFace.width, CV_8UC3, cropFace.data, cropFace.step);
    float quality_score;
    qualityObj->qualityInfer(img_f, quality_score);
    std::cout<<"quality_score: "<<quality_score<<std::endl;
    if (quality_score > QUALITY_THRESHOLD){
        featureObj->featureInfer(img_f, feature);
    }
    else{
        return EC_LOW_QUALITY;
    }
    return EC_GET_FEATURE_SUCCESS;
}

ErrCode BoeRec::registerInfer(cv::Mat& inp_img, int Id){

    float feature[512];
    getFeatureInfer(inp_img, feature);
    FaceLibrary *fl = (FaceLibrary *)facelibrary.library;
    fl->AddId(feature, Id);
    return EC_REGISTER_SUCCESS;
}


ErrCode BoeRec::recInfer(cv::Mat& inp_img, int& id, float& similarity){
        // featureIndex: 1: match face feature   2: match mask face feature
    float feature[512];
    getFeatureInfer(inp_img, feature);
    BCvFaceLibrarySearchTop1(facelibrary, feature, id, similarity);

    return EC_RECOGNITION_SUCCESS;
}


