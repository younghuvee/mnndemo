#pragma warning(disable: 4819)
#pragma warning(disable: 4996)
#include <stdio.h>
#include <string.h>
#include "face_library.h"
#if WINDOWS_SYSTEM
#include <io.h>
#else
#include <unistd.h>
#endif
#include <dirent.h>
#include <vector>
#include "opencv2/opencv.hpp"
// #include "NumCpp.hpp"
#include <mutex>
using namespace std;
using namespace cv;
#define FACE_Library_MAX_NUM 100000
#define FACE_Library_TEMP_ID_MAX_NUM 1000

static float FeatureSimilarity(const float *feature1, const float *feature2, int featureLength)
{
    float sum = 0;
    int i;
    if (!feature1 || !feature2 || featureLength <= 0){
        printf("error: FeatureSimilarity(), input is wrong.\n");
        return -1;
    }
    for (i = 0; i < featureLength; ++i){
        sum += feature1[i] * feature2[i];
    }
    return sum;
}

FaceLibrary::FaceLibrary(const char* faceLibPath, int faceFeatureLength)
{
    if (!faceLibPath){
        printf("Error: FaceLibrary::FaceLibrary(), faceLibPath is NULL.\n");
        return;
    }
    if (faceFeatureLength <= 0){
        printf("Error: FaceLibrary::FaceLibrary(), faceFeatureLength <= 0 && maskFeatureLength <= 0.\n");
        return;
    }
    strcpy(m_path, faceLibPath);
    if (faceFeatureLength > 0){
        m_faceFeatureLength = faceFeatureLength;
    } else {
        m_faceFeatureLength = 0;
    }
    m_maxId = 0;
    m_maxTempId = 0;
    LoadLib();
}

FaceLibrary::~FaceLibrary()
{
    ClearLibMem();
}

int FaceLibrary::ClearLib()
{
    FILE *fp = NULL;
    m_lock.lock();
    int faceNum = 0;
    char str[256];
    ClearLibMem();
    sprintf(str, "%s/boe_face_lib.bin", m_path);
    fp = fopen(str, "wb");
    if (fp){
        faceNum = m_faces.size();
        fwrite(&faceNum, sizeof(unsigned int), 1, fp);
        fwrite(&m_faceFeatureLength, sizeof(unsigned int), 1, fp);
        fclose(fp);
        fp = NULL;
    }
    m_lock.unlock();
    return 0;
}

int FaceLibrary::ClearLibMem()
{
    int faceNum = m_faces.size();
    for (uint16_t i = 0; i < faceNum; i++){
        if (m_faces[i].faceFeature){
            delete[] m_faces[i].faceFeature;
            m_faces[i].faceFeature = NULL;
        }
    }
    m_faces.clear();
    m_maxId = 0;
    // Temp
    faceNum = m_tempFaces.size();
    for (uint16_t i = 0; i < faceNum; i++){
        if (m_tempFaces[i].faceFeature){
            delete[] m_tempFaces[i].faceFeature;
            m_tempFaces[i].faceFeature = NULL;
        }
    }
    m_tempFaces.clear();
    m_maxTempId = 0;
    return 0;
}

int FaceLibrary::AddId(const float* faceFeature, int id)
{
    unsigned int i;
    uint16_t faceNum;
    if (m_faceFeatureLength > 0 && !faceFeature) {
        printf("Error: FaceLibrary::AddId(), faceFeature is NULL!\n");
        return -1;
    }

    m_lock.lock();
    if (m_faces.size() >= FACE_Library_MAX_NUM) {
        printf("Error: FaceLibrary::AddId(), face library is full!\n");
        m_lock.unlock();
        return -1;
    }
    // Search id
    faceNum = m_faces.size();
    for (i = 0; i < faceNum; i++){
        if (m_faces[i].id == id){
            printf("Error: FaceLibrary::AddId(), already has this id!\n");
            m_lock.unlock();
            return -1;
        }
    }
    // memory
    FaceLibraryFeature faceLibraryFeature;
    faceLibraryFeature.id = id;
    if (faceFeature){
        faceLibraryFeature.faceFeature = new float[m_faceFeatureLength];
        memcpy(faceLibraryFeature.faceFeature, faceFeature, sizeof(float) * m_faceFeatureLength);
    } else {
        faceLibraryFeature.faceFeature = NULL;
    }

    m_faces.push_back(faceLibraryFeature);
    if (m_maxId < id){
        m_maxId = id;
    }
    // File
    char str[256];
    sprintf(str, "%s/boe_face_lib.bin", m_path);
    FILE *fp = fopen(str, "rb+");
    if (fp){
        unsigned int faceNum = m_faces.size();
        fseek(fp, 0, SEEK_SET);
        fwrite(&faceNum, sizeof(unsigned int), 1, fp);
        int offset = 3 * sizeof(unsigned int) + (faceNum - 1) * (sizeof(int) + sizeof(float) * (m_faceFeatureLength));
        fseek(fp, offset, SEEK_SET);
        fwrite(&m_faces[faceNum - 1].id, sizeof(int), 1, fp);
        if (m_faceFeatureLength > 0){
            fwrite(m_faces[faceNum - 1].faceFeature, sizeof(float), m_faceFeatureLength, fp);
        }
        printf("AddId Success!\n");
        fclose(fp);
    }
    m_lock.unlock();
    SaveLib();
    return 0;
}

int FaceLibrary::AddWithAutoId(const float *faceFeature, int *id)
{
    if (m_faceFeatureLength > 0 && !faceFeature) {
        printf("Error: FaceLibrary::AddWithAutoId(), faceFeature is NULL!\n");
        return -1;
    }
    m_lock.lock();
    if (m_faces.size() >= FACE_Library_MAX_NUM) {
        printf("Error: FaceLibrary::AddWithAutoId(), face library is full!\n");
        m_lock.unlock();
        return -1;
    }

    m_maxId++;
    *id = m_maxId;
    // Memory
    FaceLibraryFeature faceLibraryFeature;
    faceLibraryFeature.id = m_maxId;
    if (faceFeature){
        faceLibraryFeature.faceFeature = new float[m_faceFeatureLength];
        memcpy(faceLibraryFeature.faceFeature, faceFeature, sizeof(float) * m_faceFeatureLength);
    } else {
        faceLibraryFeature.faceFeature = NULL;
    }
    m_faces.push_back(faceLibraryFeature);
    // File
    char str[256];
    sprintf(str, "%s/boe_face_lib.bin", m_path);
    FILE *fp = fopen(str, "rb+");
    if (fp){
        unsigned int faceNum = m_faces.size();
        fseek(fp, 0, SEEK_SET);
        fwrite(&faceNum, sizeof(unsigned int), 1, fp);
        int offset = 3 * sizeof(unsigned int) + (faceNum - 1) * (sizeof(int) + sizeof(float) * (m_faceFeatureLength));
        fseek(fp, offset, SEEK_SET);
        fwrite(&m_faces[faceNum - 1].id, sizeof(int), 1, fp);
        if (m_faceFeatureLength > 0){
            fwrite(m_faces[faceNum - 1].faceFeature, sizeof(float), m_faceFeatureLength, fp);
        }
        fclose(fp);
    }
    m_lock.unlock();
    return 0;
}


int FaceLibrary::RemoveId(int id)
{
    unsigned int i;
    unsigned int faceNum;
    m_lock.lock();
    faceNum = m_faces.size();
    for (i = 0; i < faceNum; i++){
        if (m_faces[i].id == id){
            // Memory
            if (i < faceNum - 1){
                m_faces[i].id = m_faces[faceNum - 1].id;
                if (m_faceFeatureLength > 0){
                    memcpy(m_faces[i].faceFeature, m_faces[faceNum - 1].faceFeature, sizeof(float) * m_faceFeatureLength);
                }

            }
            if (m_faces[faceNum - 1].faceFeature){
                delete[] m_faces[faceNum - 1].faceFeature;
                m_faces[faceNum - 1].faceFeature = NULL;
            }

            m_faces.pop_back();
            // File
            char str[256];
            sprintf(str, "%s/boe_face_lib.bin", m_path);
            FILE *fp = fopen(str, "rb+");
            if (fp){
                unsigned int faceNum = m_faces.size();
                fseek(fp, 0, SEEK_SET);
                fwrite(&faceNum, sizeof(unsigned int), 1, fp);
                if (i < faceNum){
                    int offset = 3 * sizeof(unsigned int) + (i - 1) * (sizeof(int) + sizeof(float) * (m_faceFeatureLength));
                    fseek(fp, offset, SEEK_SET);
                    fwrite(&m_faces[i].id, sizeof(int), 1, fp);
                    if (m_faceFeatureLength > 0){
                        fwrite(m_faces[i].faceFeature, sizeof(float), m_faceFeatureLength, fp);
                    }
                }
                fclose(fp);
            }
            m_lock.unlock();
            return 0;
        }
    }
    m_lock.unlock();
    return -1;
}


int FaceLibrary::SaveLib()
{
    char str[256];
    sprintf(str, "%s/boe_face_lib.bin", m_path);
    m_lock.lock();
    FILE *fp = fopen(str, "wb");
    if (fp){
        printf("open boe_face_lib.bin finish!\n");
        unsigned int faceNum = m_faces.size();
        fwrite(&faceNum, sizeof(unsigned int), 1, fp);
        fwrite(&m_faceFeatureLength, sizeof(unsigned int), 1, fp);
        unsigned int i;
        for (i = 0; i < faceNum; i++){
            fwrite(&m_faces[i].id, sizeof(int), 1, fp);
            if (m_faceFeatureLength > 0){
                fwrite(m_faces[i].faceFeature, sizeof(float), m_faceFeatureLength, fp);
            }
        }
    }
    m_lock.unlock();
    fclose(fp);
    return 0;
}

int FaceLibrary::LoadLib()
{
    size_t len;
    char str[256];
    FILE *fp = NULL;
    unsigned int faceNum = 0;
    m_lock.lock();
    ClearLibMem();
    sprintf(str, "%s/boe_face_lib.bin", m_path);
    fp = fopen(str, "rb");
    if (!fp){
        fp = fopen(str, "wb");
        if (!fp){
            m_lock.unlock();
            return -1;
        }
        faceNum = m_faces.size();
        fwrite(&faceNum, sizeof(unsigned int), 1, fp);
        fwrite(&m_faceFeatureLength, sizeof(unsigned int), 1, fp);
        fclose(fp);
        m_lock.unlock();
        return 0;
    }
    printf("load boe_face_lib.bin!\n");
// printf("@@@@@@\n");
    len = fread(&faceNum, sizeof(unsigned int), 1, fp);
    if (len != 1){
        printf("error: Load library: %s\n", str);
        fclose(fp);
        m_lock.unlock();
        return -1;
    }
    if (faceNum > FACE_Library_MAX_NUM) {
        printf("error: Load library: %s, face num too large!\n", str);
        fclose(fp);
        m_lock.unlock();
        return -1;
    }
// printf("######, %d\n", m_faceFeatureLength);
    unsigned int faceFeatureLength = 0;
    len = fread(&faceFeatureLength, sizeof(unsigned int), 1, fp);
    if (len != 1){
        printf("error: Load library: %s\n", str);
        fclose(fp);
        m_lock.unlock();
        return -1;
    }
    if (faceFeatureLength != m_faceFeatureLength){
        printf("error: Load library: %s\n", str);
        fclose(fp);
        m_lock.unlock();
    }
// printf("%%%%%%\n");
    unsigned int i;
    for (i = 0; i < faceNum; i++){
        FaceLibraryFeature faceLibraryFeature;
        len = fread(&faceLibraryFeature.id, sizeof(int), 1, fp);
        if (len != 1){
            ClearLibMem();
            printf("error: Load library: %s\n", str);
            fclose(fp);
            m_lock.unlock();
            return -1;
        }
        if (m_maxId < faceLibraryFeature.id){
            m_maxId = faceLibraryFeature.id;
        }
        if (m_faceFeatureLength > 0){
            faceLibraryFeature.faceFeature = new float[m_faceFeatureLength];
            len = fread(faceLibraryFeature.faceFeature, sizeof(float), m_faceFeatureLength, fp);
            if (len != m_faceFeatureLength){
                if (faceLibraryFeature.faceFeature)
                    delete[] faceLibraryFeature.faceFeature;
                ClearLibMem();
                printf("error: Load library: %s\n", str);
                fclose(fp);
                m_lock.unlock();
                return -1;
            }
        } else {
            faceLibraryFeature.faceFeature = NULL;
        }
        m_faces.push_back(faceLibraryFeature);
    }
    fclose(fp);
    m_lock.unlock();
    return 0;
}

int FaceLibrary::SearchTop1(const float *feature, int &id, float &similarity)
{
    unsigned int i;
    unsigned int faceNum;
    float maxSimilarity = 0.0f;
    float tempSimilarity;
    int tempId = -1;
    m_lock.lock();
    faceNum = m_faces.size();
    for (i = 0; i < faceNum; i++){
        tempSimilarity = FeatureSimilarity(feature, m_faces[i].faceFeature, m_faceFeatureLength);   // no mask
        if (maxSimilarity < tempSimilarity){
            maxSimilarity = tempSimilarity;
            tempId = m_faces[i].id;
        }
    }
    id = tempId;
    similarity = maxSimilarity;
    m_lock.unlock();
    return 0;
}


void FaceLibrary::CheckFaceLibrarySimilarity()
{
    unsigned int i, j;
    unsigned int faceNum;
    float maxSimilarity1 = 0.0f;
    float maxSimilarity2 = 0.0f;
    float tempSimilarity;
    int total_count = 0;
    int count1 = 0;
    float th1 = 0.64f;
    m_lock.lock();
    faceNum = m_faces.size();
    for (i = 0; i < faceNum; i++) {
        //printf("i:%d\n", i);
        for (j = i + 1; j < faceNum; j++) {
            if (j == i)
                continue;

            tempSimilarity = FeatureSimilarity(m_faces[i].faceFeature, m_faces[j].faceFeature, m_faceFeatureLength);

            if (tempSimilarity > 0.64f) {
                printf("tempSimilarity:%f, i:%d, j:%d\n", tempSimilarity, m_faces[i].id, m_faces[j].id);
                continue;
            }
            if (tempSimilarity > maxSimilarity1) {
                maxSimilarity1 = tempSimilarity;
            }
            if (tempSimilarity > th1) {
                count1++;
            }

            total_count++;
        }
    }
    m_lock.unlock();
    printf("maxSimilarity1:%f\n", maxSimilarity1);
    printf("maxSimilarity2:%f\n", maxSimilarity2);
    printf("error rate1:%f\n", (float)count1 / (float)total_count);
    printf("error rate2:%f\n", (float)count1 / (float)total_count);
}

unsigned int FaceLibrary::Size()
{
    int faceNum;
    m_lock.lock();
    faceNum = m_faces.size();
    m_lock.unlock();
    return faceNum;
}

int FaceLibrary::GetMaxId()
{
    int maxId;
    m_lock.lock();
    maxId = m_maxId;
    m_lock.unlock();
    return maxId;
}

boeFaceLibrary boeFaceLibraryInit(const char* faceLibPath, int faceFeatureLength)
{
    boeFaceLibrary library;
    library.library = NULL;
    library.errCode = EC_FACE_LIBRARY_SUCCESS;
    if ((faceLibPath==NULL) || (access(faceLibPath, 0) != 0)){
        library.errCode = EC_FACE_LIBRARY_NOT_FOUND;
        return library;
    }
    FaceLibrary *fl = new FaceLibrary(faceLibPath, faceFeatureLength);
    library.library = (void *)fl;
    return library;
}

enum ErrCode BCvFaceLibraryRelease(boeFaceLibrary library)
{
    if (!library.library){
        FaceLibrary *fl = (FaceLibrary *)library.library;
        delete fl;
        memset(&library, 0, sizeof(boeFaceLibrary));
    }
    return EC_FACE_LIBRARY_SUCCESS;
}

// int SearchTop1(const float *feature, int featureIndex, int &id, float &similarity);
enum ErrCode BCvFaceLibrarySearchTop1(boeFaceLibrary library, const float *feature, int &id, float &similarity)
{

    FaceLibrary *fl = (FaceLibrary *)library.library;
    fl->SearchTop1(feature, id, similarity);
    return EC_FACE_LIBRARY_SUCCESS;
}