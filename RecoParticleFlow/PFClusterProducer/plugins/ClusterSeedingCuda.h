#ifndef ClusterSeedingCuda_h
#define ClusterSeedingCuda_h
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
#include <Eigen/Dense>

namespace ClusterSeedingCuda {

  struct pfRhForSeeding {
    float rho;
    float eta;
    float phi;
    float energy;
    float pt2;
    int   layer;
    int   depth;
    int   mask;
    Eigen::Vector4f neigh_Ens;   
    Eigen::Vector4i neigh_Index;
    int isSeed;
    int topoId;
  };


  void seedingWrapperXYZ(pfRhForSeeding* pfrh, int* seedBool, size_t size,  float* hbthresholds, float* hethresholds);

  void seedingWrapperXYZ_2(pfRhForSeeding* pfrh, int* seedBool, size_t size,  float* hbthresholds, float* hethresholds);

  void seedingWrapperXYZ_2ECAL(pfRhForSeeding* pfrh, int* seedBool, size_t size,  float* hbthresholds, float* hethresholds);


}  // namespace cudavectors

#endif  // ClusterSeedingCuda_h
