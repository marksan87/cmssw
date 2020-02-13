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
    int   layer;
    int   depth;
    int   mask;
//Eigen::Matrix<float, Dynamic, 1> neigh_Ens;   
    Eigen::Vector4f neigh_Ens;   
//    float energyN1=0.;
//    float energyN2=0.;
//    float energyN3=0.;
//    float energyN4=0.;
  };

  /*  struct pfrhToSeedBool {
    float x;
    float y;
    float z;
  };
  */
  struct theThresholds {
    Eigen::Vector4f HBThresh;
    Eigen::VectorXf HEThresh;
  };

  //Eigen::Vector4f hbthresh;
  //Eigen::VectorXf hethresh;

  void seedingWrapperXYZ(pfRhForSeeding const* pfrh, int* seedBool, size_t size,  float* hbthresholds, float* hethresholds);

//void seedingWrapperXYZ(int a, int b);



}  // namespace cudavectors

#endif  // ClusterSeedingCuda_h
