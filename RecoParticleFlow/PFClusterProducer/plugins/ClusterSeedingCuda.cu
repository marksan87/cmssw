
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

  __host__ __device__ inline void seed(pfRhForSeeding& pfrh, int& seedBool, float* hbthresholds,float* hethresholds) {
     
     if( (pfrh.layer == 1 && pfrh.energy>hbthresholds[pfrh.depth-1]) || (pfrh.layer == 3 && pfrh.energy>hethresholds[pfrh.depth-1]) && pfrh.mask)
     {
	seedBool = ( (pfrh.energy>pfrh.neigh_Ens[0] && pfrh.energy>pfrh.neigh_Ens[1] && pfrh.energy>pfrh.neigh_Ens[2] && pfrh.energy>pfrh.neigh_Ens[3] ) )? 1 : 0;
      }
      else{ seedBool = 0; }

  }

  __global__ void seedingKernel(pfRhForSeeding* pfrh, int* seedBool, size_t size, float* hbthresholds, float* hethresholds) {

    
    	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i<size) {
		seed(pfrh[i],seedBool[i], hbthresholds, hethresholds);
	}
  }



  

void seedingWrapperXYZ(pfRhForSeeding* pfrh, int* seedBool, size_t size, float* hbthresholds, float* hethresholds)
	{
    
   seedingKernel<<<(size+512-1)/512, 512>>>(pfrh, seedBool, size, hbthresholds, hethresholds);
   cudaDeviceSynchronize();
   cudaCheck(cudaGetLastError());
  }




__host__ __device__ inline void seed_2(pfRhForSeeding& pfrh, int& seedBool, float* hbthresholds,float* hethresholds, pfRhForSeeding* rhs) {         
     if( ( (pfrh.layer == 1 || pfrh.layer == -1) && pfrh.energy>hbthresholds[pfrh.depth-1]) || ( (pfrh.layer == 3 || pfrh.layer == -2) && pfrh.energy>hethresholds[pfrh.depth-1]) && pfrh.mask)
     {
	pfrh.isSeed=1;
	seedBool = 1;	
        for(int i=0; i<4; i++){
		if( pfrh.neigh_Index[i]>-1 && pfrh.energy<=rhs[pfrh.neigh_Index[i]].energy){
			pfrh.isSeed=0;
			pfrh.topoId=-1;
			seedBool = 0;
			break;
		}
	}
		
      }
      else{ 
      	    pfrh.topoId=-1;
	    pfrh.isSeed=0;
	    seedBool = 0;	    
	}
  }


  __global__ void seedingKernel_2(pfRhForSeeding* pfrh, int* seedBool, size_t size, float* hbthresholds, float* hethresholds) {

    
    	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i<size) {
		seed_2(pfrh[i],seedBool[i], hbthresholds, hethresholds, pfrh);
	}
  }

   __global__ void topoKernel_2(pfRhForSeeding* pfrh, size_t size) {
    	int l = threadIdx.x+blockIdx.x*blockDim.x;
	if(l<size) {
		if(1==1)
		{
			for(int k=0; k<4; k++){
				if( pfrh[l].neigh_Index[k] > -1 && pfrh[l].topoId < pfrh[pfrh[l].neigh_Index[k]].topoId )
				{
						pfrh[l].topoId=pfrh[pfrh[l].neigh_Index[k]].topoId;
				}
			}
			
					
		}

	}//loop end

  }



void seedingWrapperXYZ_2(pfRhForSeeding* pfrh, int* seedBool, size_t size, float* hbthresholds, float* hethresholds)
	{
    
   seedingKernel_2<<<(size+512-1)/512, 512>>>(pfrh, seedBool, size, hbthresholds, hethresholds);
   //cudaDeviceSynchronize();
   //cudaCheck(cudaGetLastError());

   for(int j=0;j<16;j++){
   topoKernel_2<<<(size+512-1)/512, 512>>>(pfrh, size);
   cudaDeviceSynchronize();
   }
   
   cudaCheck(cudaGetLastError());
   
  }







__host__ __device__ inline void seed_2ECAL(pfRhForSeeding& pfrh, int& seedBool, float* hbthresholds,float* hethresholds, pfRhForSeeding* rhs) {  
       
     if( ( pfrh.layer == -1 && pfrh.energy>0.23 && pfrh.pt2>0.00) || ( (pfrh.layer == -2) && pfrh.energy>0.6 && pfrh.pt2>0.15) )
     {
	pfrh.isSeed=1;	
	seedBool=1;
        for(int i=0; i<4; i++){
		if(pfrh.neigh_Index[i]<0) continue; 
		if(  pfrh.energy<rhs[pfrh.neigh_Index[i]].energy){
			pfrh.isSeed=0;
			pfrh.topoId=-1;
			seedBool=0;
			break;
		}
	}		
      }
      else{ 
      	    pfrh.topoId=-1;
	    pfrh.isSeed=0;
	    seedBool=0;	    
	}
  }


  __global__ void seedingKernel_2ECAL(pfRhForSeeding* pfrh, int* seedBool, size_t size, float* hbthresholds, float* hethresholds) {

    
    	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i<size) {
		seed_2ECAL(pfrh[i],seedBool[i], hbthresholds, hethresholds, pfrh);
	}
  }

   __global__ void topoKernel_2ECAL(pfRhForSeeding* pfrh, size_t size) {
    	int l = threadIdx.x+blockIdx.x*blockDim.x;
	if(l<size) {
		if(1==1)
		{
			for(int k=0; k<4; k++){
				if( pfrh[l].neigh_Index[k] > -1 && pfrh[l].topoId < pfrh[pfrh[l].neigh_Index[k]].topoId && ( (pfrh[l].layer == -2 && pfrh[l].energy>0.3) || (pfrh[l].layer == -1 && pfrh[l].energy>0.08) ) )
				{
						pfrh[l].topoId=pfrh[pfrh[l].neigh_Index[k]].topoId;
				}
			}
			
					
		}

	}//loop end

  }



void seedingWrapperXYZ_2ECAL(pfRhForSeeding* pfrh, int* seedBool, size_t size, float* hbthresholds, float* hethresholds)
	{
    
   if(size>0) seedingKernel_2ECAL<<<(size+512-1)/512, 512>>>(pfrh, seedBool, size, hbthresholds, hethresholds);
   //cudaDeviceSynchronize();
   //cudaCheck(cudaGetLastError());

   for(int j=0;j<16;j++){
   if(size>0) topoKernel_2ECAL<<<(size+512-1)/512, 512>>>(pfrh, size);
   cudaDeviceSynchronize();
   }
   
   cudaCheck(cudaGetLastError());
   
  }


}  // namespace cudavectors
