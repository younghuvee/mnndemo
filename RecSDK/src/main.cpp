#include "boeRec.hpp"

int main(){
    const char* modelDtPath = "../data/model/retinaface_mbnet_0714.mnn";
    const char* modelLmPath = "../data/model/face_landmark_0707.mnn";
    const char* modelQlPath = "../data/model/face_quality_0707.mnn";
    const char* modelFtPath = "../data/model/rec_1020.mnn";

    const char* face_lib_path = "../data/";
    const char* reg_image_path = "../data/images/006553.jpg";
    const char* rec_image_path = "../data/images/006553.jpg";

    cv::Mat reg_image = cv::imread(reg_image_path);
    cv::Mat rec_image = cv::imread(rec_image_path);


    BoeRec recObj(face_lib_path, modelDtPath, modelLmPath, modelQlPath, modelFtPath);
    int reg_id = 1010;
    recObj.registerInfer(reg_image, reg_id);
    int id_ret;
    float sim_ret;
    recObj.recInfer(rec_image, id_ret, sim_ret);
    if (sim_ret > REC_THRESHOLD){
        cout << "id: " << id_ret << endl;
        cout << "similarity: " << sim_ret << endl;
    }
    else{
        cout<<"Not in lib!"<<endl;
    }

    return 0;
}