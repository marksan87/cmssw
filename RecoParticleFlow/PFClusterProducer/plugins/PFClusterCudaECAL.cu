
#include <cmath>
#include <iostream>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterCudaECAL.h"
#include <Eigen/Dense>

namespace PFClusterCudaECAL {

  // THE ART OF HARDCODING
  // these numbers should be copied over during initialization
  __constant__ float showerSigma = 1.5;
  __constant__ float recHitEnergyNormEB = 0.08;
  __constant__ float recHitEnergyNormEE = 0.3;
  __constant__ float minFracToKeep = 0.0000001;

  __constant__ float seedEThresholdEB = 0.23;
  __constant__ float seedEThresholdEE = 0.6;
  __constant__ float seedPt2ThresholdEB = 0.0*0.0;
  __constant__ float seedPt2hresholdEE = 0.15*0.15;

  __constant__ float topoEThresholdEB = 0.08;
  __constant__ float topoEThresholdEE = 0.3;

  __constant__ int nNeigh = 8;
  __constant__ int maxSize = 50;
   
 __global__ void seedingKernel_ECAL(
     				    size_t size, 
				    float* pfrh_energy,
				    float* pfrh_pt2,
				    int*   pfrh_isSeed,
				    int*   pfrh_topoId,
				    int*   pfrh_layer,
				    int*   neigh8_Ind
				    ) {

   int i = threadIdx.x+blockIdx.x*blockDim.x;

   if(i<size) {        
     if( ( pfrh_layer[i] == -1 && pfrh_energy[i]>seedEThresholdEB && pfrh_pt2[i]>seedPt2ThresholdEB) || ( (pfrh_layer[i] == -2) && pfrh_energy[i]>seedEThresholdEE && pfrh_pt2[i]>seedPt2hresholdEE) )
       {
	 pfrh_isSeed[i]=1;		 
	 for(int k=0; k<nNeigh; k++){
	   if(neigh8_Ind[nNeigh*i+k]<0) continue; 
	   if(pfrh_energy[i]<pfrh_energy[neigh8_Ind[nNeigh*i+k]]){
	     pfrh_isSeed[i]=0;
	     pfrh_topoId[i]=-1;	     
	     break;
	   }
	 }		
       }
     else{ 
       pfrh_topoId[i]=-1;
       pfrh_isSeed[i]=0;
       	    
     }     
   }
 }
  
  __global__ void topoKernel_ECAL( 
				  size_t size,
				  float* pfrh_energy,
				  int* pfrh_topoId,
				  int* pfrh_layer,
				  int* neigh8_Ind
				  ) {
    
    int l = threadIdx.x+blockIdx.x*blockDim.x;
	if(l<size) {
	  //printf("layer: %d",pfrh_layer[l]);
	  for(int k=0; k<nNeigh; k++){
	    if( neigh8_Ind[nNeigh*l+k] > -1 && 
		pfrh_topoId[l] < pfrh_topoId[neigh8_Ind[nNeigh*l+k]] && 
		( (pfrh_layer[l] == -2 && pfrh_energy[l]>topoEThresholdEE) || 
		  (pfrh_layer[l] == -1 && pfrh_energy[l]>topoEThresholdEB) ) )
	      {
		pfrh_topoId[l]=pfrh_topoId[neigh8_Ind[nNeigh*l+k]];
	      }
	  }				       
	}//loop end
  }
  
  
  __global__ void pfClusterKernel_ECAL_step1(
					      
					     size_t size,
					     float* pfrh_x,
					     float* pfrh_y,
					     float* pfrh_z,
					     float* pfrh_energy,
					     int* pfrh_topoId,
					     int* pfrh_isSeed,
					     int* pfrh_layer,
					     
					     float* pfrhfrac, 
					     int* pfrhfracind
					     ) {
    
    int l = threadIdx.x+blockIdx.x*blockDim.x;
    if(l<size) {
      
      int countFracPerRh = 0;
      float fracTot = 0.;
      if(pfrh_isSeed[l] == 1){//<-- if it is seed, create first entry in pfcl/frac and fr
	
	pfrhfrac[l*maxSize+countFracPerRh] = 1;
	pfrhfracind[l*maxSize+countFracPerRh] = l;
	fracTot = fracTot+1;
	countFracPerRh++;


	for(int p=0; p<size; p++){
	  
	  if(pfrh_topoId[l] == pfrh_topoId[p] && pfrh_topoId[p]>0. && pfrh_isSeed[p] != 1){ //<-- only if rechits are in the same topocluster they should be part of a pfcluster, if current rechit is seed we record those which are not seeds		
	    //measure distance 
	    float dist2 = 
	       (pfrh_x[l] - pfrh_x[p])*(pfrh_x[l] - pfrh_x[p])
	      +(pfrh_y[l] - pfrh_y[p])*(pfrh_y[l] - pfrh_y[p])
	      +(pfrh_z[l] - pfrh_z[p])*(pfrh_z[l] - pfrh_z[p]);
	    
	    float d2 = dist2 / (showerSigma*showerSigma);	  
	    //if(d2>100.) printf("the distance in units of showerSigma is larger than 100...");
		
	    float fraction = -1.;
	    if(pfrh_layer[p] == -1) { fraction = pfrh_energy[p] / recHitEnergyNormEB * expf(-0.5 * d2); }
	    if(pfrh_layer[p] == -2) { fraction = pfrh_energy[p] / recHitEnergyNormEE * expf(-0.5 * d2); }
	    
	    if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");
	    
	    if(d2 < 100 /*&&  fraction > minFracToKeep*/){ 
	      pfrhfracind[l*maxSize+countFracPerRh] = p;
	      countFracPerRh++;
	    }//end of putting Seed to list of Seeds per rechit
	  }// if rechit in neighbourhood is seed
	}//<== loop of rechits	    
      }//<== if it is seed
	
      
      
      if(pfrh_isSeed[l] != 1){ //<-- seeds are not part of other clusters
	
	//loop over rechits to find seeds the rechit could be part of the cluster 
	for(int p=0; p<size; p++){
	  
	  if(pfrh_topoId[l] == pfrh_topoId[p] && pfrh_topoId[p]>0. && pfrh_isSeed[p] == 1){ //<-- only if rechits are in the same topocluster they should be part of a pfcluster, if current rechit is not seed we record those which are seeds		
	    //measure distance to seed
	    float dist2 = 
	       (pfrh_x[l] - pfrh_x[p])*(pfrh_x[l] - pfrh_x[p])
	      +(pfrh_y[l] - pfrh_y[p])*(pfrh_y[l] - pfrh_y[p])
	      +(pfrh_z[l] - pfrh_z[p])*(pfrh_z[l] - pfrh_z[p]);
	    
	    float d2 = dist2 / (showerSigma*showerSigma);	  
	    //if(d2>100.) printf("the distance in units of showerSigma is larger than 100...");
		
	    float fraction = -1.;
	    if(pfrh_layer[p] == -1) { fraction = pfrh_energy[p] / recHitEnergyNormEB * expf(-0.5 * d2); }
	    if(pfrh_layer[p] == -2) { fraction = pfrh_energy[p] / recHitEnergyNormEE * expf(-0.5 * d2); }
	    
	    if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");
	    
	    if(d2 < 100 /*&&  fraction > minFracToKeep*/){ 
	      pfrhfrac[l*maxSize+countFracPerRh] = fraction;
	      pfrhfracind[l*maxSize+countFracPerRh] = p;
	      fracTot = fracTot + fraction;
	      countFracPerRh++;
	    }//end of putting Seed to list of Seeds per rechit
	  }// if rechit in neighbourhood is seed
	}//<== loop of rechits	    
      }//<== if it is not seed
      
      //normalize the fractions
      for(int m = 0; m<countFracPerRh; m++){
	pfrhfrac[l*maxSize+m] = pfrhfrac[l*maxSize+m]/(fracTot); 
      }//<== end normalise fractions
    }//<== if l<size 
  }//<== end of function
  
 

__global__ void pfClusterKernel_ECAL_step2_V2(					     
					     size_t size, 
					     int* pfrh_isSeed,
					     float* pfrh_energy,
					      
					     float* pfrhfrac, 
					     int* pfrhfracind, 
					     int* pcrhfracind,
					     float* pcrhfrac
					     ) {
    
    int l = threadIdx.x+blockIdx.x*blockDim.x;
    if(l<size) {
      
      int nFracPerSeed=0;

      if(pfrh_isSeed[l]==1)
	{
	  pcrhfracind[l*maxSize] = l;
	  pcrhfrac[l*maxSize] = 1;
	  nFracPerSeed++;
	for(int i=1; i<maxSize; i++)
	  {
	  if(pfrhfracind[l*maxSize+i] > -1)
	    {
	      for(int j=0; j<maxSize; j++)
		{
		  if(pfrhfracind[ pfrhfracind[l*maxSize+i]*maxSize + j ] == l && pfrhfrac[pfrhfracind[l*maxSize+i]*maxSize + j]>minFracToKeep)
		    {
		      
		      pcrhfracind[l*maxSize+nFracPerSeed]=pfrhfracind[l*maxSize+i];
		      pcrhfrac[l*maxSize+nFracPerSeed]=pfrhfrac[pfrhfracind[l*maxSize+i]*maxSize + j ];
		      nFracPerSeed++;
		      break;
		    }
		}
	    }
	  if(pfrhfracind[l*maxSize+i] < 0) break;
	}
	
      }

    }//end of l<size
  }//end of function
   
  void PFRechitToPFCluster_ECAL(size_t size, 
				float* pfrh_x, 
				float* pfrh_y, 
				float* pfrh_z, 
				float* pfrh_energy, 
				float* pfrh_pt2, 				 				
				int* pfrh_isSeed,
				int* pfrh_topoId, 
				int* pfrh_layer, 
				int* neigh8_Ind, 
				
				float* pfrhfrac, 
				int* pfrhfracind, 
				int* pcrhfracind,
				float* pcrhfrac
				)
  { 
    //seeding
    if(size>0) seedingKernel_ECAL<<<(size+512-1)/512, 512>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,  neigh8_Ind);
    cudaDeviceSynchronize();
    
    //topoclustering
    for(int j=0;j<16;j++){
      if(size>0) topoKernel_ECAL<<<(size+512-1)/512, 512>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, neigh8_Ind);	    
      cudaDeviceSynchronize();
    }
    
    //pfclustering
    if(size>0) pfClusterKernel_ECAL_step1<<<(size+512-1)/512, 512>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrhfrac, pfrhfracind);
    cudaDeviceSynchronize();
    
    if(size>0) pfClusterKernel_ECAL_step2_V2<<<(size+512-1)/512, 512>>>(size, pfrh_isSeed, pfrh_energy, pfrhfrac, pfrhfracind, pcrhfracind, pcrhfrac);
    cudaDeviceSynchronize();
    
    cudaCheck(cudaGetLastError());	  
  }
}  // namespace cudavectors
