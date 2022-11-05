#ifndef FACE_LIBRARY_H
#define FACE_LIBRARY_H

#include "common_struct.hpp"
#include <mutex>


struct FaceFeature{
    boeRect rect;
    std::vector<float> faceFeature;
    std::vector<float> maskfaceFeature;
};

struct FaceLibraryFeature {
    int id;
    float* faceFeature;
    float* maskFaceFeature;
};

class FaceLibrary
{
public:
    FaceLibrary(const char* faceLibPath, int faceFeatureLength);
    ~FaceLibrary();

    int ClearLib();
    int AddId(const float* faceFeature, int id);
    int AddWithAutoId(const float *faceFeature, int *id);
    int RemoveId(int id);
    // featureIndex: 1: match face feature   2: match mask face feature
    int SearchTop1(const float *feature, int &id, float &similarity);
    unsigned int Size();
    int GetMaxId();
    void CheckFaceLibrarySimilarity();

private:
    int ClearLibMem();
    int LoadLib();
    int SaveLib();

private:
    std::mutex m_lock;
    char m_path[256];
    std::vector<FaceLibraryFeature> m_faces;
    std::vector<FaceLibraryFeature> m_tempFaces;
    unsigned int m_faceFeatureLength;
    int m_maxId;
    int m_maxTempId;
};


DLL_EXPORT boeFaceLibrary boeFaceLibraryInit(const char* faceLibPath, int faceFeatureLength);
DLL_EXPORT enum ErrCode BCvFaceLibraryRelease(boeFaceLibrary library);
DLL_EXPORT std::vector<RegisterFace> BCvAddNewRegisterFace(const boeFaceLibrary library, char *imagePath, int startId, int showImage, int imageHeight, int imageWidth, int minFaceSize);
DLL_EXPORT enum ErrCode BCvFaceLibrarySearchTop1(boeFaceLibrary library, const float *feature, int &id, float &similarity);

#endif // FACELIBRARY_H
