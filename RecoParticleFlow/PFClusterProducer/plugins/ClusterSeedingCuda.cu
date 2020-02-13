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

  __host__ __device__ inline void seed(pfRhForSeeding const& pfrh, int& seedBool, float* hbthresholds,float* hethresholds) {
     
     if( (pfrh.layer == 1 && pfrh.energy>hbthresholds[pfrh.depth-1]) || (pfrh.layer == 3 && pfrh.energy>hethresholds[pfrh.depth-1]) && pfrh.mask)
     {
	seedBool = ( (pfrh.energy>pfrh.neigh_Ens[0] && pfrh.energy>pfrh.neigh_Ens[1] && pfrh.energy>pfrh.neigh_Ens[2] && pfrh.energy>pfrh.neigh_Ens[3] ) )? 1 : 0;
      }
      else{ seedBool = 0; }

  }

  __global__ void seedingKernel(pfRhForSeeding const* pfrh, int* seedBool, size_t size, float* hbthresholds, float* hethresholds) {

    
    	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i<size) {
		seed(pfrh[i],seedBool[i], hbthresholds, hethresholds);
	}
  }

  

void seedingWrapperXYZ(pfRhForSeeding const* pfrh, int* seedBool, size_t size, float* hbthresholds, float* hethresholds)
	{
    
   seedingKernel<<<(size+512-1)/512, 512>>>(pfrh, seedBool, size, hbthresholds, hethresholds);
   cudaDeviceSynchronize();
   cudaCheck(cudaGetLastError());
  }

//|| (pfrh.layer == 3 && pfrh.energy>thresholds.HEThresh[pfrh.depth-1])

}  // namespace cudavectors
