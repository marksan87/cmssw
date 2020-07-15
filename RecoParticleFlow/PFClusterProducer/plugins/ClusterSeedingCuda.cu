
#include <cmath>
#include <iostream>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/ClusterSeedingCuda.h"
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
	  //seed_2ECALV2(pfrh[i],seedBool[i], hbthresholds, hethresholds, pfrh, neigh4_Ind, neigh4_Ens);
	}
  }


 __global__ void seedingKernel_2ECALV2(pfRhForSeeding* pfrh, int* seedBool, size_t size, float* hbthresholds, float* hethresholds, int* neigh4_Ind, float* neigh4_Ens) {

       	int i = threadIdx.x+blockIdx.x*blockDim.x;
	if(i<size) {
	  

	  if( ( pfrh[i].layer == -1 && pfrh[i].energy>0.23 && pfrh[i].pt2>0.00) || ( (pfrh[i].layer == -2) && pfrh[i].energy>0.6 && pfrh[i].pt2>(0.15*0.15)) )
	    {
	      pfrh[i].isSeed=1;	
	      seedBool[i]=1;
	      for(int k=0; k<8; k++){
		//if(pfrh[i].neigh_Index[i]<0) continue; 
		if(neigh4_Ind[8*i+k]<0) continue; 
		if(pfrh[i].energy<neigh4_Ens[8*i+k]){
		  pfrh[i].isSeed=0;
		  pfrh[i].topoId=-1;
		  seedBool[i]=0;
		  break;
		}
	      }		
	    }
	  else{ 
      	    pfrh[i].topoId=-1;
	    pfrh[i].isSeed=0;
	    seedBool[i]=0;	    
	  }

	}
 }



   __global__ void topoKernel_2ECAL(pfRhForSeeding* pfrh, size_t size) {
    	int l = threadIdx.x+blockIdx.x*blockDim.x;
	if(l<size) {
		
		        for(int k=0; k<4; k++){
				if( pfrh[l].neigh_Index[k] > -1 && pfrh[l].topoId < pfrh[pfrh[l].neigh_Index[k]].topoId && ( (pfrh[l].layer == -2 && pfrh[l].energy>0.3) || (pfrh[l].layer == -1 && pfrh[l].energy>0.08) ) )
				{
						pfrh[l].topoId=pfrh[pfrh[l].neigh_Index[k]].topoId;
				}
			}
			
					
		

	}//loop end

  }


  __constant__ float showerSigma = 1.5;
  __constant__ float recHitEnergyNormEB = 0.08;
  __constant__ float recHitEnergyNormEE = 0.3;
  __constant__ float minFracToKeep = 0.0000001;
  __device__ float d2[50*600];

  __global__ void pfClusterKernel_2ECAL_step1(pfRhForSeeding* pfrh, size_t size, pfClusterz* pfcl, float* pfrhfrac, int* pfrhfracind) {

    int l = threadIdx.x+blockIdx.x*blockDim.x;
	if(l<size) {
	  
          int countFracPerRh = 1;
	  float fracTot = 1.;
	  if(pfrh[l].isSeed == 1){//<-- if it is seed, create first entry in pfcl/frac and fr
	    
	    pfcl[l].energy = pfrh[l].energy;
	    pfcl[l].x = pfrh[l].x;
	    pfcl[l].y = pfrh[l].y;
	    pfcl[l].z = pfrh[l].z;
	    //countFracPerRh++;
	   
	  }//if it is seed

	  if(pfrh[l].isSeed != 1){ //<-- seeds are not part of other clusters
	    
	    //loop over rechits to find seeds the rechit could be part of the cluster 
	    for(int p=0; p<size; p++){
	      
	      if(pfrh[l].topoId == pfrh[p].topoId && pfrh[p].topoId>0. && pfrh[p].isSeed == 1){ //<-- only if rechits are in the same topocluster they should be part of a pfcluster, we record those which are seeds		
		//measure distance to seed
		float dist2 = 
		   (pfrh[l].x - pfrh[p].x)*(pfrh[l].x - pfrh[p].x)
		  +(pfrh[l].y - pfrh[p].y)*(pfrh[l].y - pfrh[p].y)
		  +(pfrh[l].z - pfrh[p].z)*(pfrh[l].z - pfrh[p].z);

		float d2 = dist2 / (showerSigma*showerSigma);	  
		//if(d2>100.) printf("the distance in units of showerSigma is larger than 100...");
		
		float fraction = -1.;
		if(pfrh[p].layer == -1) { fraction = pfrh[p].energy / recHitEnergyNormEB * expf(-0.5 * d2); }
		if(pfrh[p].layer == -2) { fraction = pfrh[p].energy / recHitEnergyNormEE * expf(-0.5 * d2); }

		if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

		if(d2 < 100 /*&&  fraction > minFracToKeep*/){ 
		  pfrhfrac[l*50+countFracPerRh] = fraction;
		  pfrhfracind[l*50+countFracPerRh] = p;
		  fracTot = fracTot + fraction;
		  countFracPerRh++;
		}//end of putting Seed to list of Seeds per rechit
	      }// if rechit in neighbourhood is seed
	    }//<== loop of rechits	    
	    //normalize the fractions

            //printf("seeds per : %d\n",countFracPerRh);
	    for(int m = 1; m<=countFracPerRh; m++){
	      pfrhfrac[l*50+m] = pfrhfrac[l*50+m]/(fracTot+1); 
	    }//<== end normalise fractions

	  }//<== if it is not seed
	}//<== if l<size 
  }//<== end of function

  __global__ void pfClusterKernel_2ECAL_step2(pfRhForSeeding* pfrh, size_t size, pfClusterz* pfcl, float* pfrhfrac, int* pfrhfracind) {

    int l = threadIdx.x+blockIdx.x*blockDim.x;
	if(l<size) {
	  
	  int nFracPerSeed=0;
	  for(int i=0; i<size*50; i++){
	    if(pfrhfracind[i] == l && i != l*50 && pfrhfrac[i]>minFracToKeep) {
	      pfcl[l].energy = pfcl[l].energy + pfrhfrac[i]*pfrh[pfrhfracind[i]].energy; 
	      nFracPerSeed++;
	    }
	  }
	  pfcl[l].nfrac=nFracPerSeed;
	  //printf("nFrac per Seed: %d\n",nFracPerSeed);
	}//end of l<size
}//end of function




  void seedingWrapperXYZ_2ECAL(pfRhForSeeding* pfrh, int* seedBool, size_t size, float* hbthresholds, float* hethresholds, int* neigh4_Ind, float* neigh4_Ens, pfClusterz* pfcl, float* pfrhfrac, int* pfrhfracind)
	{
    
	  //seeding
	  if(size>0) seedingKernel_2ECALV2<<<(size+512-1)/512, 512>>>(pfrh, seedBool, size, hbthresholds, hethresholds, neigh4_Ind, neigh4_Ens);
	  cudaDeviceSynchronize();
    
	  //topoclustering
	  for(int j=0;j<16;j++){
	    if(size>0) topoKernel_2ECAL<<<(size+512-1)/512, 512>>>(pfrh, size);
	    cudaDeviceSynchronize();
	  }

	  //pfclustering
	  if(size>0) pfClusterKernel_2ECAL_step1<<<(size+512-1)/512, 512>>>(pfrh, size, pfcl, pfrhfrac, pfrhfracind);
	  cudaDeviceSynchronize();
	  if(size>0) pfClusterKernel_2ECAL_step2<<<(size+512-1)/512, 512>>>(pfrh, size, pfcl, pfrhfrac, pfrhfracind);
	  cudaDeviceSynchronize();
   
	  cudaCheck(cudaGetLastError());
   
  }


}  // namespace cudavectors
