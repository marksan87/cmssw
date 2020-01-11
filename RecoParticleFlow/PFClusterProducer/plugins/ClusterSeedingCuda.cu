// system include files
#include <cmath>
#include <iostream>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/ClusterSeedingCuda.h"
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
#include <Eigen/Dense>

namespace ClusterSeedingCuda {

  __host__ __device__ inline void seed(pfRhForSeeding const& pfrh, int& seedBool) {

      
      seedBool = (pfrh.energy>pfrh.neigh_Ens[0] && pfrh.energy>pfrh.neigh_Ens[1] && pfrh.energy>pfrh.neigh_Ens[2] && pfrh.energy>pfrh.neigh_Ens[3] ) ? 1 : 0;
  }

  __global__ void seedingKernel(pfRhForSeeding const* pfrh, int* seedBool, size_t size) {

    
    	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i<size) {
		seed(pfrh[i],seedBool[i]);
	}
  }

  //void seedingWrapper(CylindricalVector const* cylindrical, CartesianVector* cartesian, size_t size) {

  

void seedingWrapperXYZ(pfRhForSeeding const* pfrh, int* seedBool, size_t size/*, std::vector<int> const HE_depths, std::vector<double> const HE_EnThresholds, std::vector<int> const HB_depths, std::vector<double> const HB_EnThresholds*/)
//  void seedingWrapperXYZ(int a, int b)
	{

    //std::cout<<"I am in the wrapper"<<std::endl;
    
   seedingKernel<<<(size+512-1)/512, 512>>>(pfrh, seedBool, size);
   cudaDeviceSynchronize();
   cudaCheck(cudaGetLastError());
  }

}  // namespace cudavectors
