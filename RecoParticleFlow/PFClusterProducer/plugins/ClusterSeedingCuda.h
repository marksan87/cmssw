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

void seedingWrapperXYZ(pfRhForSeeding const* pfrh, int* seedBool, size_t size/*, std::vector<int> const HE_depths, std::vector<double> const HE_EnThresholds, std::vector<int> const HB_depths, std::vector<double> const HB_EnThresholds*/);

//void seedingWrapperXYZ(int a, int b);



}  // namespace cudavectors

#endif  // ClusterSeedingCuda_h
