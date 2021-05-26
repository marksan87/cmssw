
#include <cmath>
#include <iostream>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterCudaHCAL.h"
#include <Eigen/Dense>
#include <cuda_profiler_api.h>

// Uncomment for debugging
//#define DEBUG_GPU_HCAL

// Uncomment for verbose debugging output (including extra Cuda Memcpy steps)
//#define DEBUG_VERBOSE_OUTPUT

constexpr int sizeof_float = sizeof(float);
constexpr int sizeof_int = sizeof(int);



namespace PFClusterCudaHCAL {

  __constant__ float showerSigma;
  __constant__ float recHitEnergyNormEB_vec[4];
  __constant__ float recHitEnergyNormEE_vec[7];
  __constant__ float minFracToKeep;

  __constant__ float seedEThresholdEB_vec[4];
  __constant__ float seedEThresholdEE_vec[7];
  __constant__ float seedPt2ThresholdEB;
  __constant__ float seedPt2ThresholdEE;

  __constant__ float topoEThresholdEB_vec[4];
  __constant__ float topoEThresholdEE_vec[7];
  
  __constant__ int nNT = 8;  // Number of neighbors considered for topo clustering
  __constant__ int nNeigh;
  __constant__ int maxSize;
 
  //int nTopoLoops = 100;
  int nTopoLoops = 35;


  bool initializeCudaConstants(float h_showerSigma,
                               const float (&h_recHitEnergyNormEB_vec)[4],
                               const float (&h_recHitEnergyNormEE_vec)[7],
                               float h_minFracToKeep,
                               const float (&h_seedEThresholdEB_vec)[4],
                               const float (&h_seedEThresholdEE_vec)[7],
                               float h_seedPt2ThresholdEB,
                               float h_seedPt2ThresholdEE,
                               const float (&h_topoEThresholdEB_vec)[4],
                               const float (&h_topoEThresholdEE_vec)[7],
                               int   h_nNeigh,
                               int   h_maxSize
                           )
  {
     bool status = true;
     status &= cudaCheck(cudaMemcpyToSymbolAsync(showerSigma, &h_showerSigma, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     float val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, showerSigma, sizeof_float));
     std::cout<<"showerSigma read from symbol: "<<val<<std::endl;
#endif
     
     status &= cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormEB_vec, &h_recHitEnergyNormEB_vec, 4*sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     float val4[4];
     status &= cudaCheck(cudaMemcpyFromSymbol(&val4, recHitEnergyNormEB_vec, 4*sizeof_float));
     std::cout<<"recHitEnergyNormEB_vec read from symbol: ";
     for (int i = 0; i < 4; i++) {std::cout<<val4[i]<<" ";}
     std::cout<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormEE_vec, &h_recHitEnergyNormEE_vec, 7*sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     float val7[7];
     status &= cudaCheck(cudaMemcpyFromSymbol(&val7, recHitEnergyNormEE_vec, 7*sizeof_float));
     std::cout<<"recHitEnergyNormEE_vec read from symbol: ";
     for (int i = 0; i < 7; i++) {std::cout<<val7[i]<<" ";}
     std::cout<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(minFracToKeep, &h_minFracToKeep, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, minFracToKeep, sizeof_float));
     std::cout<<"minFracToKeep read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(seedEThresholdEB_vec, &h_seedEThresholdEB_vec, 4*sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     status &= cudaCheck(cudaMemcpyFromSymbol(&val4, seedEThresholdEB_vec, 4*sizeof_float));
     std::cout<<"seedEThresholdEB_vec read from symbol: ";
     for (int i = 0; i < 4; i++) {std::cout<<val4[i]<<" ";}
     std::cout<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(seedEThresholdEE_vec, &h_seedEThresholdEE_vec, 7*sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     status &= cudaCheck(cudaMemcpyFromSymbol(&val7, seedEThresholdEE_vec, 7*sizeof_float));
     std::cout<<"seedEThresholdEE_vec read from symbol: ";
     for (int i = 0; i < 7; i++) {std::cout<<val7[i]<<" ";}
     std::cout<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(seedPt2ThresholdEB, &h_seedPt2ThresholdEB, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, seedPt2ThresholdEB, sizeof_float));
     std::cout<<"seedPt2ThresholdEB read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(seedPt2ThresholdEE, &h_seedPt2ThresholdEE, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, seedPt2ThresholdEE, sizeof_float));
     std::cout<<"seedPt2ThresholdEE read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(topoEThresholdEB_vec, &h_topoEThresholdEB_vec, 4*sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     status &= cudaCheck(cudaMemcpyFromSymbol(&val4, topoEThresholdEB_vec, 4*sizeof_float));
     std::cout<<"topoEThresholdEB_vec read from symbol: ";
     for (int i = 0; i < 4; i++) {std::cout<<val4[i]<<" ";}
     std::cout<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(topoEThresholdEE_vec, &h_topoEThresholdEE_vec, 7*sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     status &= cudaCheck(cudaMemcpyFromSymbol(&val7, topoEThresholdEE_vec, 7*sizeof_float));
     std::cout<<"topoEThresholdEE_vec read from symbol: ";
     for (int i = 0; i < 7; i++) {std::cout<<val7[i]<<" ";}
     std::cout<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(nNeigh, &h_nNeigh, sizeof_int));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     int ival = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&ival, nNeigh, sizeof_int));
     std::cout<<"nNeigh read from symbol: "<<ival<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(maxSize, &h_maxSize, sizeof_int));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     ival = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&ival, maxSize, sizeof_int));
     std::cout<<"maxSize read from symbol: "<<ival<<std::endl;
#endif
     
     return status;
}
 
 __global__ void seedingKernel_HCAL(
     				size_t size, 
				    const double* __restrict__ pfrh_energy,
				    const double* __restrict__ pfrh_pt2,
				    int*   pfrh_isSeed,
				    int*   pfrh_topoId,
				    const int* __restrict__ pfrh_layer,
				    const int* __restrict__ pfrh_depth,
				    const int* __restrict__ neigh4_Ind
				    ) {

   int i = threadIdx.x+blockIdx.x*blockDim.x;

   if(i<size) {        
     if ( (pfrh_layer[i] == 1 && pfrh_energy[i]>seedEThresholdEB_vec[pfrh_depth[i]-1] && pfrh_pt2[i]>seedPt2ThresholdEB) || 
          (pfrh_layer[i] == 3 && pfrh_energy[i]>seedEThresholdEE_vec[pfrh_depth[i]-1] && pfrh_pt2[i]>seedPt2ThresholdEE) )
       {
	 pfrh_isSeed[i]=1;		 
	 for(int k=0; k<nNeigh; k++){
	   if(neigh4_Ind[nNeigh*i+k]<0) continue; 
	   if(pfrh_energy[i]<pfrh_energy[neigh4_Ind[nNeigh*i+k]]){
	     pfrh_isSeed[i]=0;
	     //pfrh_topoId[i]=-1;	     
	     break;
	   }
	 }		
       }
     else{ 
       // pfrh_topoId[i]=-1;
       pfrh_isSeed[i]=0;
       	    
     }     
   }
 }

  
 __global__ void seedingKernel_HCAL_serialize(
     				    size_t size, 
				    const double* __restrict__ pfrh_energy,
				    const double* __restrict__ pfrh_pt2,
				    int*   pfrh_isSeed,
				    int*   pfrh_topoId,
				    const int* __restrict__ pfrh_layer,
				    const int* __restrict__ pfrh_depth,
				    const int* __restrict__ neigh4_Ind
				    ) {

   //int i = threadIdx.x+blockIdx.x*blockDim.x;
   for (int i = 0; i < size; i++) {
       if(i<size) {        
         if ( (pfrh_layer[i] == 1 && pfrh_energy[i]>seedEThresholdEB_vec[pfrh_depth[i]-1] && pfrh_pt2[i]>seedPt2ThresholdEB) || 
              (pfrh_layer[i] == 3 && pfrh_energy[i]>seedEThresholdEE_vec[pfrh_depth[i]-1] && pfrh_pt2[i]>seedPt2ThresholdEE) )
           {
         pfrh_isSeed[i]=1;		 
         for(int k=0; k<nNeigh; k++){
           if(neigh4_Ind[nNeigh*i+k]<0) continue; 
           if(pfrh_energy[i]<pfrh_energy[neigh4_Ind[nNeigh*i+k]]){
             pfrh_isSeed[i]=0;
             //pfrh_topoId[i]=-1;	     
             break;
           }
         }		
           }
         else{ 
           // pfrh_topoId[i]=-1;
           pfrh_isSeed[i]=0;
                
         }     
       }
    }
 }
  

   __global__ void topoKernel_HCALV2( 
				  size_t size,
				  const double* __restrict__ pfrh_energy,
				  int* pfrh_topoId,
				  const int* __restrict__ pfrh_layer,
				  const int* __restrict__ pfrh_depth,
				  const int* __restrict__ neigh8_Ind
				  ) {
     
     int l = threadIdx.x+blockIdx.x*blockDim.x;
     //int k = threadIdx.y+blockIdx.y*blockDim.y;
     int k = (threadIdx.y+blockIdx.y*blockDim.y) % nNT;
           
      //if(l<size && k<nNT) {
      if(l<size) {

	while( neigh8_Ind[nNT*l+k] > -1 && pfrh_topoId[l] != pfrh_topoId[neigh8_Ind[nNT*l+k]] && 
	    ( (pfrh_layer[neigh8_Ind[nNT*l+k]] == 3 && pfrh_energy[neigh8_Ind[nNT*l+k]]>topoEThresholdEE_vec[pfrh_depth[neigh8_Ind[nNT*l+k]]-1]) ||
	      (pfrh_layer[neigh8_Ind[nNT*l+k]] == 1 && pfrh_energy[neigh8_Ind[nNT*l+k]]>topoEThresholdEB_vec[pfrh_depth[neigh8_Ind[nNT*l+k]]-1]) ) &&
	    ( (pfrh_layer[l] == 3 && pfrh_energy[l]>topoEThresholdEE_vec[pfrh_depth[l]-1]) ||
	      (pfrh_layer[l] == 1 && pfrh_energy[l]>topoEThresholdEB_vec[pfrh_depth[l]-1]))
	    )
	    {
	      if(pfrh_topoId[l] > pfrh_topoId[neigh8_Ind[nNT*l+k]]){
		atomicMax(&pfrh_topoId[neigh8_Ind[nNT*l+k]],pfrh_topoId[l]);
	      }
	      if(pfrh_topoId[l] < pfrh_topoId[neigh8_Ind[nNT*l+k]]){
		atomicMax(&pfrh_topoId[l], pfrh_topoId[neigh8_Ind[nNT*l+k]]);
	      }
	    }	
      }
   }
 

   __global__ void topoKernel_HCAL_serialize( 
				  size_t size,
				  const double* __restrict__ pfrh_energy,
				  int* pfrh_topoId,
				  const int* __restrict__ pfrh_layer,
				  const int* __restrict__ pfrh_depth,
				  const int* __restrict__ neigh8_Ind
				  ) {
     
     //int l = threadIdx.x+blockIdx.x*blockDim.x;
     //int k = threadIdx.y+blockIdx.y*blockDim.y;
     
     for (int l = 0; l < size; l++) {
        //for (int k = 0; k < size; k++) {
        for (int k = 0; k < 8; k++) {
           
            while( neigh8_Ind[nNT*l+k] > -1 && pfrh_topoId[l] != pfrh_topoId[neigh8_Ind[nNT*l+k]] && 
                ( (pfrh_layer[neigh8_Ind[nNT*l+k]] == 3 && pfrh_energy[neigh8_Ind[nNT*l+k]]>topoEThresholdEE_vec[pfrh_depth[neigh8_Ind[nNT*l+k]]-1]) ||
                  (pfrh_layer[neigh8_Ind[nNT*l+k]] == 1 && pfrh_energy[neigh8_Ind[nNT*l+k]]>topoEThresholdEB_vec[pfrh_depth[neigh8_Ind[nNT*l+k]]-1]) ) &&
                ( (pfrh_layer[l] == 3 && pfrh_energy[l]>topoEThresholdEE_vec[pfrh_depth[l]-1]) ||
                  (pfrh_layer[l] == 1 && pfrh_energy[l]>topoEThresholdEB_vec[pfrh_depth[l]-1]))
                )
                {
                  if(pfrh_topoId[l] > pfrh_topoId[neigh8_Ind[nNT*l+k]]){
                atomicMax(&pfrh_topoId[neigh8_Ind[nNT*l+k]],pfrh_topoId[l]);
                  }
                  if(pfrh_topoId[l] < pfrh_topoId[neigh8_Ind[nNT*l+k]]){
                atomicMax(&pfrh_topoId[l], pfrh_topoId[neigh8_Ind[nNT*l+k]]);
                  }
                }	
          }
        }
   }
 
__global__ void hcalFastCluster_step1( size_t size,
					     const float* __restrict__ pfrh_x,
					     const float* __restrict__ pfrh_y,
					     const float* __restrict__ pfrh_z,
					     const double* __restrict__ pfrh_energy,
					     int* pfrh_topoId,
					     int* pfrh_isSeed,
					     const int* __restrict__ pfrh_layer,
				         const int* __restrict__ pfrh_depth,
					     float* pcrhfrac,
					     int* pcrhfracind,
					     float* fracSum,
					     int* rhCount
					     ) {

    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int j = threadIdx.y+blockIdx.y*blockDim.y;
    //make sure topoID, Layer is the same, i is seed and j is not seed
    if( i<size && j<size){

      if( pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i]==1 ){

      float dist2 =
	       (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
	      +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
	      +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);

      float d2 = dist2 / (showerSigma*showerSigma);
      float fraction = -1.;

      if(pfrh_layer[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
      else if (pfrh_layer[j] == 3) { fraction = pfrh_energy[i] / recHitEnergyNormEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
	  
      if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

      if( pfrh_isSeed[j]!=1 && d2<100.)
	{
	  atomicAdd(&fracSum[j],fraction);
	}
      }
    }
}

__global__ void hcalFastCluster_step1( int size,
					     const float* __restrict__ pfrh_x,
					     const float* __restrict__ pfrh_y,
					     const float* __restrict__ pfrh_z,
					     const double* __restrict__ pfrh_energy,
					     int* pfrh_topoId,
					     int* pfrh_isSeed,
					     const int* __restrict__ pfrh_layer,
				         const int* __restrict__ pfrh_depth,
					     float* pcrhfrac,
					     int* pcrhfracind,
					     float* fracSum,
					     int* rhCount
					     ) {

    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int j = threadIdx.y+blockIdx.y*blockDim.y;
    //make sure topoID, Layer is the same, i is seed and j is not seed
    if( i<size && j<size){

      if( pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i]==1 ){

      float dist2 =
	       (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
	      +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
	      +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);

      float d2 = dist2 / (showerSigma*showerSigma);
      float fraction = -1.;

      if(pfrh_layer[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
      else if (pfrh_layer[j] == 3) { fraction = pfrh_energy[i] / recHitEnergyNormEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
	  
      if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

      if( pfrh_isSeed[j]!=1 && d2<100.)
	{
	  atomicAdd(&fracSum[j],fraction);
	}
      }
    }
}
 
__global__ void hcalFastCluster_step1_serialize( size_t size,
					     const float* __restrict__ pfrh_x,
					     const float* __restrict__ pfrh_y,
					     const float* __restrict__ pfrh_z,
					     const double* __restrict__ pfrh_energy,
					     int* pfrh_topoId,
					     int* pfrh_isSeed,
					     const int* __restrict__ pfrh_layer,
				         const int* __restrict__ pfrh_depth,
					     float* pcrhfrac,
					     int* pcrhfracind,
					     float* fracSum,
					     int* rhCount
					     ) {

    //int i = threadIdx.x+blockIdx.x*blockDim.x;
    //int j = threadIdx.y+blockIdx.y*blockDim.y;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            //make sure topoID, Layer is the same, i is seed and j is not seed
            if( i<size && j<size){

              if( pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i]==1 ){

              float dist2 =
                   (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
                  +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
                  +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);

              float d2 = dist2 / (showerSigma*showerSigma);
              float fraction = -1.;

              if(pfrh_layer[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
              else if (pfrh_layer[j] == 3) { fraction = pfrh_energy[i] / recHitEnergyNormEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
              
              if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

              if( pfrh_isSeed[j]!=1 && d2<100.)
            {
              atomicAdd(&fracSum[j],fraction);
            }
              }
            }
        }
    }
  }

__global__ void hcalFastCluster_step2( int size,
					     const float* __restrict__ pfrh_x,
					     const float* __restrict__ pfrh_y,
					     const float* __restrict__ pfrh_z,
					     const double* __restrict__ pfrh_energy,
					     int* pfrh_topoId,
					     int* pfrh_isSeed,
					     const int* __restrict__ pfrh_layer,
				         const int* __restrict__ pfrh_depth,
					     float* pcrhfrac,
					     int* pcrhfracind,
					     float* fracSum,
					     int* rhCount
					     ) {

    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int j = threadIdx.y+blockIdx.y*blockDim.y;
    //make sure topoID, Layer is the same, i is seed and j is not seed
    if( i<size && j<size){
      if( pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i]==1 ){
      if(i==j)
	{
	  pcrhfrac[i*maxSize]    = 1.;
	  pcrhfracind[i*maxSize] = j;
	}
      if( pfrh_isSeed[j]!=1 ){
	float dist2 =
	   (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
	  +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
	  +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);

	float d2 = dist2 / (showerSigma*showerSigma);
	float fraction = -1.;

    if(pfrh_layer[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
    else if (pfrh_layer[j] == 3) { fraction = pfrh_energy[i] / recHitEnergyNormEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
	
	if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");
	if(d2 < 100. )
	  {
	    if ((fraction/fracSum[j])>minFracToKeep){
	      int k = atomicAdd(&rhCount[i],1);
	      pcrhfrac[i*maxSize+k] = fraction/fracSum[j];
	      pcrhfracind[i*maxSize+k] = j;
	    }
	  }
      }
      }
    }
}

__global__ void hcalFastCluster_step2( size_t size,
					     const float* __restrict__ pfrh_x,
					     const float* __restrict__ pfrh_y,
					     const float* __restrict__ pfrh_z,
					     const double* __restrict__ pfrh_energy,
					     int* pfrh_topoId,
					     int* pfrh_isSeed,
					     const int* __restrict__ pfrh_layer,
				         const int* __restrict__ pfrh_depth,
					     float* pcrhfrac,
					     int* pcrhfracind,
					     float* fracSum,
					     int* rhCount
					     ) {

    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int j = threadIdx.y+blockIdx.y*blockDim.y;
    //make sure topoID, Layer is the same, i is seed and j is not seed
    if( i<size && j<size){
      if( pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i]==1 ){
      if(i==j)
	{
	  pcrhfrac[i*maxSize]    = 1.;
	  pcrhfracind[i*maxSize] = j;
	}
      if( pfrh_isSeed[j]!=1 ){
	float dist2 =
	   (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
	  +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
	  +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);

	float d2 = dist2 / (showerSigma*showerSigma);
	float fraction = -1.;

    if(pfrh_layer[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
    else if (pfrh_layer[j] == 3) { fraction = pfrh_energy[i] / recHitEnergyNormEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
	
	if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");
	if(d2 < 100. )
	  {
	    if ((fraction/fracSum[j])>minFracToKeep){
	      int k = atomicAdd(&rhCount[i],1);
	      pcrhfrac[i*maxSize+k] = fraction/fracSum[j];
	      pcrhfracind[i*maxSize+k] = j;
	      //printf("(i,j)=(%d,%d), rhCount=%d, fraction=%f, fracsum=%f \n",i,j,rhCount[i], fraction, fracSum[j]);
	    }
	  }
      }
      }
    }
}


__global__ void hcalFastCluster_step2_serialize( size_t size,
					     const float* __restrict__ pfrh_x,
					     const float* __restrict__ pfrh_y,
					     const float* __restrict__ pfrh_z,
					     const double* __restrict__ pfrh_energy,
					     int* pfrh_topoId,
					     int* pfrh_isSeed,
					     const int* __restrict__ pfrh_layer,
				         const int* __restrict__ pfrh_depth,
					     float* pcrhfrac,
					     int* pcrhfracind,
					     float* fracSum,
					     int* rhCount
					     ) {

    //int i = threadIdx.x+blockIdx.x*blockDim.x;
    //int j = threadIdx.y+blockIdx.y*blockDim.y;
    //make sure topoID, Layer is the same, i is seed and j is not seed
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if( i<size && j<size){
              if( pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i]==1 ){
              if(i==j)
            {
              pcrhfrac[i*maxSize]    = 1.;
              pcrhfracind[i*maxSize] = j;
            }
              if( pfrh_isSeed[j]!=1 ){
            float dist2 =
               (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
              +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
              +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);

            float d2 = dist2 / (showerSigma*showerSigma);
            float fraction = -1.;

            if(pfrh_layer[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
            else if (pfrh_layer[j] == 3) { fraction = pfrh_energy[i] / recHitEnergyNormEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
              
            
            if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");
            if(d2 < 100. )
              {
                if ((fraction/fracSum[j])>minFracToKeep){
                  int k = atomicAdd(&rhCount[i],1);
                  pcrhfrac[i*maxSize+k] = fraction/fracSum[j];
                  pcrhfracind[i*maxSize+k] = j;
                  //printf("(i,j)=(%d,%d), rhCount=%d, fraction=%f, fracsum=%f \n",i,j,rhCount[i], fraction, fracSum[j]);
                }
              }
              }
              }
            }
        }
    }
}

// Compute whether rechits pass topo clustering energy threshold
__global__ void passingTopoThreshold(size_t size,
                const int* __restrict__ pfrh_layer,
                const int* __restrict__ pfrh_depth,
                const double* __restrict__ pfrh_energy,
                bool* pfrh_passTopoThresh) {

    int i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i < size) {
        if ( (pfrh_layer[i] == 3 && pfrh_energy[i]>topoEThresholdEE_vec[pfrh_depth[i]-1]) ||
             (pfrh_layer[i] == 1 && pfrh_energy[i]>topoEThresholdEB_vec[pfrh_depth[i]-1])) {
            pfrh_passTopoThresh[i] = true;
        }
        else { pfrh_passTopoThresh[i] = false; }
    }
}

__global__ void passingTopoThreshold(int size,
                const int* __restrict__ pfrh_layer,
                const int* __restrict__ pfrh_depth,
                const double* __restrict__ pfrh_energy,
                bool* pfrh_passTopoThresh) {

    int i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i < size) {
        if ( (pfrh_layer[i] == 3 && pfrh_energy[i]>topoEThresholdEE_vec[pfrh_depth[i]-1]) ||
             (pfrh_layer[i] == 1 && pfrh_energy[i]>topoEThresholdEB_vec[pfrh_depth[i]-1])) {
            pfrh_passTopoThresh[i] = true;
        }
        else { pfrh_passTopoThresh[i] = false; }
    }
}

// Linking step on odd iterations
__global__ void oddLinkingParent(size_t nEdges,
    const int* __restrict__ pfrh_edgeId,
    const int* __restrict__ pfrh_edgeList,
    const int* __restrict__ pfrh_edgeLeft,
    const int* __restrict__ pfrh_edgeMask,
    int* pfrh_parent,
    const bool* __restrict__ pfrh_passTopoThresh) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < nEdges && pfrh_edgeMask[idx] > 0) {
        int i = pfrh_edgeId[idx];    // Get edge topo id
        if (pfrh_passTopoThresh[i] && (i == 0 || (i != pfrh_edgeLeft[idx]))) {
            pfrh_parent[i] = (int)min(i, pfrh_edgeList[idx]);
        }
    }
}

// Linking step on even iterations
__global__ void evenLinkingParent(size_t nEdges,
    const int* __restrict__ pfrh_edgeId,
    const int* __restrict__ pfrh_edgeList,
    const int* __restrict__ pfrh_edgeRight,
    const int* __restrict__ pfrh_edgeMask,
    int* pfrh_parent,
    const bool* __restrict__ pfrh_passTopoThresh) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < nEdges && pfrh_edgeMask[idx] > 0) {
        int i = pfrh_edgeId[idx];    // Get real edge index
        if (pfrh_passTopoThresh[i] && (i == (nEdges - 1) || (i != pfrh_edgeRight[idx]))) {
            pfrh_parent[i] = (int)max(i, pfrh_edgeList[idx]);
        }
    }
}

// Linking step on odd iterations
__global__ void oddLinkingParent_copyif(size_t nEdges,
                const int* __restrict__ pfrh_edgeId,
                const int* __restrict__ pfrh_edgeList,
                const int* __restrict__ pfrh_edgesLeft,
                int* pfrh_parent,
                const bool* __restrict__ pfrh_passTopoThresh) {

    int edge = threadIdx.x+blockIdx.x*blockDim.x;
    if (edge < nEdges) {
        int i = pfrh_edgesLeft[edge];    // Get real edge index
        if (pfrh_passTopoThresh[pfrh_edgeId[i]] && (edge == 0 || (pfrh_edgeId[i] != pfrh_edgeId[pfrh_edgesLeft[edge-1]])) ) {
            pfrh_parent[pfrh_edgeId[i]] = (int)min(pfrh_edgeId[i], pfrh_edgeList[i]);
        }
    }
}

// Linking step on even iterations
__global__ void evenLinkingParent_copyif(size_t nEdges,
                const int* __restrict__ pfrh_edgeId,
                const int* __restrict__ pfrh_edgeList,
                const int* __restrict__ pfrh_edgesLeft,
                int* pfrh_parent,
                const bool* __restrict__ pfrh_passTopoThresh) {

    int edge = threadIdx.x+blockIdx.x*blockDim.x;
    if (edge < nEdges) {
        int i = pfrh_edgesLeft[edge];    // Get real edge index
        if (pfrh_passTopoThresh[pfrh_edgeId[i]] && (edge == (nEdges-1) || (pfrh_edgeId[i] != pfrh_edgeId[pfrh_edgesLeft[edge+1]])) ) {
            pfrh_parent[pfrh_edgeId[i]] = (int)max(pfrh_edgeId[i], pfrh_edgeList[i]);
        }
    }
}

// Linking step on odd iterations (serialized)
__global__ void oddLinkingParentSerial(size_t nEdges,
                const int* __restrict__ pfrh_edgeId,
                const int* __restrict__ pfrh_edgeList,
                const int* __restrict__ pfrh_edgesLeft,
                int* pfrh_parent,
                const bool* __restrict__ pfrh_passTopoThresh) {

    for (int edge = 0; edge < nEdges; edge++) {
        int i = pfrh_edgesLeft[edge];   // Get real edge index
        if (pfrh_passTopoThresh[pfrh_edgeId[i]] && (edge == 0 || (pfrh_edgeId[i] != pfrh_edgeId[pfrh_edgesLeft[edge-1]])) ) {
            pfrh_parent[pfrh_edgeId[i]] = (int)min(pfrh_edgeId[i], pfrh_edgeList[i]);
        }
    }
}

// Linking step on even iterations (serialized)
__global__ void evenLinkingParentSerial(size_t nEdges,
                const int* __restrict__ pfrh_edgeId,
                const int* __restrict__ pfrh_edgeList,
                const int* __restrict__ pfrh_edgesLeft,
                int* pfrh_parent,
                const bool* __restrict__ pfrh_passTopoThresh) {

    for (int edge = 0; edge < nEdges; edge++) {
        int i = pfrh_edgesLeft[edge];   // Get real edge index
        if (pfrh_passTopoThresh[pfrh_edgeId[i]] && (edge == (nEdges-1) || (pfrh_edgeId[i] != pfrh_edgeId[pfrh_edgesLeft[edge+1]])) ) {
            pfrh_parent[pfrh_edgeId[i]] = (int)max(pfrh_edgeId[i], pfrh_edgeList[i]);
        }
    }
}

// Replace edgeId, edgeList values with their parents
// If there are still entries remaining, d_notDone flag is set to true
__global__ void edgeParent(size_t nEdges,
    int* pfrh_edgeId,
    int* pfrh_edgeList,
    int* pfrh_edgeMask,
    const int* __restrict__ pfrh_parent,
    bool* d_notDone) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < nEdges) {
        if (pfrh_edgeMask[idx] > 0) {
            int id = pfrh_edgeId[idx];   // Get edge topo id
            int neighbor = pfrh_edgeList[idx]; // Get neighbor topo id
            pfrh_edgeId[idx] = pfrh_parent[id];
            pfrh_edgeList[idx] = pfrh_parent[neighbor];

            // edgeMask set to true if elements of edgeId and edgeList are different
            if (pfrh_edgeId[idx] == pfrh_edgeList[idx])
            {
                pfrh_edgeMask[idx] = 0;
            }
            else { *d_notDone= true; }
        }
    }
}

__global__ void edgeParent_remaining(size_t nEdges,
    int* pfrh_edgeId,
    int* pfrh_edgeList,
    int* pfrh_edgeMask,
    const int* __restrict__ pfrh_parent,
    int* remaining) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < nEdges) {
        if (pfrh_edgeMask[idx] > 0) {
            int id = pfrh_edgeId[idx];   // Get edge topo id
            int neighbor = pfrh_edgeList[idx]; // Get neighbor topo id
            pfrh_edgeId[idx] = pfrh_parent[id];
            pfrh_edgeList[idx] = pfrh_parent[neighbor];

            // edgeMask set to true if elements of edgeId and edgeList are different
            //pfrh_edgeMask[idx] = (int)(pfrh_edgeId[idx] != pfrh_edgeList[idx]);
            if (pfrh_edgeId[idx] == pfrh_edgeList[idx])
            {
                pfrh_edgeMask[idx] = 0;
            }
            else { *remaining = 1; }
        }
    }
}

// Replace edgeId and edgeList with parents
__global__ void edgeParent_newMask(size_t nEdges,
    int* pfrh_edgeId,
    int* pfrh_edgeList,
    const int* __restrict__ pfrh_edgesLeft,
    bool* pfrh_edgeMask,
    const int* __restrict__ pfrh_parent) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < nEdges) {
        int i = pfrh_edgesLeft[idx];   // Get real edge index
        pfrh_edgeId[i] = pfrh_parent[pfrh_edgeId[i]];
        pfrh_edgeList[i] = pfrh_parent[pfrh_edgeList[i]];

        // edgeMask set to true if elements of edgeId and edgeList are different
        pfrh_edgeMask[idx] = (pfrh_edgeId[i] != pfrh_edgeList[i]);
    }
}


// Replace edgeId and edgeList with parents
__global__ void edgeParentSerial(size_t nEdges,
                int* pfrh_edgeId,
                int* pfrh_edgeList,
                bool* pfrh_edgeMask,
                const int* __restrict__ pfrh_parent) {

    for (int i = 0; i < nEdges; i++) {
        if (pfrh_edgeMask[i]) {
            pfrh_edgeId[i] = pfrh_parent[pfrh_edgeId[i]];
            pfrh_edgeList[i] = pfrh_parent[pfrh_edgeList[i]];

            // edgeMask set to true if elements of edgeId and edgeList are different
            pfrh_edgeMask[i] = (pfrh_edgeId[i] != pfrh_edgeList[i]);
        }
    }
}

// Replace edgeId and edgeList with parents
__global__ void edgeParentSerial(size_t nEdges,
                int* pfrh_edgeId,
                int* pfrh_edgeList,
                int* pfrh_edgeMask,
                const int* __restrict__ pfrh_parent,
                int* d_remaining) {

    *d_remaining = 0;
    for (int idx = 0; idx < nEdges; idx++) {
        if (pfrh_edgeMask[idx] > 0) {
            pfrh_edgeId[idx] = pfrh_parent[pfrh_edgeId[idx]];
            pfrh_edgeList[idx] = pfrh_parent[pfrh_edgeList[idx]];

            // edgeMask set to true if elements of edgeId and edgeList are different
            if (pfrh_edgeId[idx] == pfrh_edgeList[idx])
            {
                pfrh_edgeMask[idx] = 0;
            }
            else { *d_remaining += 1; }
        }
    }
}


__global__ void updateEdgesLeftSerial(size_t nEdges,
                int* pfrh_edgesLeft,
                const bool* __restrict__ pfrh_edgeMask,
                int* remaining) {

    int diff = 0;
    for (int edge = 0; edge < nEdges; edge++) {
        int i = pfrh_edgesLeft[edge];    //  real edge index
        if (pfrh_edgeMask[i]) {
            pfrh_edgesLeft[diff] = i;
            diff++;
        }
    }

    *remaining = diff;
}

struct is_true {
    __host__ __device__ bool operator()(const bool &x) {
        return x;
    }
};


struct stencil_true {
    const bool* mask;
    stencil_true(const bool* stencil) {
        mask = stencil;
    }
    __host__ __device__ bool operator()(const int &x) {
        return mask[x];
    }
};


void topoClusterLinking_copyif(size_t nRH,
                size_t nEdges,
                int* pfrh_edgeId,
                int* pfrh_edgeList,
                int* pfrh_edgesAll,
                int* pfrh_edgesLeft,
                bool* pfrh_edgeMask,
                int* pfrh_parent,
                bool* pfrh_passTopoThresh,
                float & timer) {
                //bool* pfrh_passTopoThresh) {
    if (nEdges < 1) return;
    int s = (int)nEdges;
    int iter = 0;
    bool* edgeMask = new bool[s];
    int* edgesLeft = new int[nEdges];
    int* d_remaining;
    cudaMalloc(&d_remaining, sizeof(int));
    cudaMemsetAsync(d_remaining, nEdges, sizeof(int));

#ifdef DEBUG_VERBOSE_OUTPUT
    int* edgeId = new int[s];
    int* edgeList = new int[s];
    int* parent = new int[nRH]; 
    bool* passTopoThresh = new bool[nRH]; 
    
    cudaMemcpy(edgeId, pfrh_edgeId, sizeof(int)*s, cudaMemcpyDeviceToHost);
    cudaMemcpy(edgeList, pfrh_edgeList, sizeof(int)*s, cudaMemcpyDeviceToHost);
    cudaMemcpy(edgeMask, pfrh_edgeMask, sizeof(int)*s, cudaMemcpyDeviceToHost);
    cudaMemcpy(parent, pfrh_parent, sizeof(int)*nRH, cudaMemcpyDeviceToHost);
    cudaMemcpy(passTopoThresh, pfrh_passTopoThresh, sizeof(bool)*nRH, cudaMemcpyDeviceToHost);
    std::cout<<"s = "<<s<<std::endl;
    
    std::cout<<"\n--- Edge ID ---\n";
    for (int c = 0; c < s; c++)
        std::cout<<edgeId[c]<<" ";
    std::cout<<std::endl<<std::flush;
    
    std::cout<<"\n--- Edge List ---\n";
    for (int c = 0; c < s; c++)
        std::cout<<edgeList[c]<<" ";
    std::cout<<std::endl<<std::flush;
    
    std::cout<<"\n--- Parent ---\n";
    for (int c = 0; c < nRH; c++)
        std::cout<<parent[c]<<" ";
    std::cout<<std::endl<<std::flush;
    
    for (int c = 0; c < nRH; c++) {
        if (passTopoThresh[c] != 1)
            std::cout<<"passTopoThresh["<<c<<"] = "<<passTopoThresh[c]<<std::endl;
    }
#endif 
    std::vector<float> loopTimers;
    int remaining = nEdges;
    
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);

    while (remaining > 0) {
        iter++;
        if (iter > 100) {
            std::cout<<"Too many iterations! Bailing out"<<std::endl;
            return;
        }
#ifdef DEBUG_VERBOSE_OUTPUT
        std::cout<<"\nNow on iteration: "<<iter<<"\t\tremaining = "<<remaining<<std::endl;
#endif
        
        // odd iterations use pfrh_edgesAll
        if (iter%2) { oddLinkingParent_copyif<<<(remaining+31)/32, 256>>>(remaining, pfrh_edgeId, pfrh_edgeList, pfrh_edgesAll, pfrh_parent, pfrh_passTopoThresh); } 
        //if (iter%2) { oddLinkingParent<<<(remaining+255)/256, 256>>>(remaining, pfrh_edgeId, pfrh_edgeList, pfrh_edgesLeft, pfrh_parent, pfrh_passTopoThresh); } 
        //if (iter%2) { oddLinkingParentSerial<<<1,1>>>(remaining, pfrh_edgeId, pfrh_edgeList, pfrh_edgesLeft, pfrh_parent, pfrh_passTopoThresh); } 
        
        // even iterations use pfrh_edgesLeft
        else { evenLinkingParent_copyif<<<(remaining+31)/32, 256>>>(remaining, pfrh_edgeId, pfrh_edgeList, pfrh_edgesLeft, pfrh_parent, pfrh_passTopoThresh); }
        //else { evenLinkingParent<<<(remaining+255)/256, 256>>>(remaining, pfrh_edgeId, pfrh_edgeList, pfrh_edgesLeft, pfrh_parent, pfrh_passTopoThresh); }
        //else { evenLinkingParentSerial<<<1,1>>>(remaining, pfrh_edgeId, pfrh_edgeList, pfrh_edgesLeft, pfrh_parent, pfrh_passTopoThresh); }

#ifdef DEBUG_VERBOSE_OUTPUT
        cudaMemcpy(edgeId, pfrh_edgeId, sizeof(int)*s, cudaMemcpyDeviceToHost);
        cudaMemcpy(edgeList, pfrh_edgeList, sizeof(int)*s, cudaMemcpyDeviceToHost);
        cudaMemcpy(parent, pfrh_parent, sizeof(int)*nRH, cudaMemcpyDeviceToHost);
        std::cout<<"\n--- Edge ID ---\n";
        for (int c = 0; c < s; c++)
            std::cout<<edgeId[c]<<" ";
        std::cout<<std::endl<<std::flush;
        
        std::cout<<"\n--- Edge List ---\n";
        for (int c = 0; c < s; c++)
            std::cout<<edgeList[c]<<" ";
        std::cout<<std::endl<<std::flush;
        
        std::cout<<"\n--- Parent ---\n";
        for (int c = 0; c < nRH; c++)
            std::cout<<parent[c]<<" ";
        std::cout<<std::endl<<std::flush;
#endif
        
        // Replace edge values with parents
        //edgeParent<<<(s+255)/256, 256>>>(remaining, pfrh_edgeId, pfrh_edgeList, pfrh_edgesLeft, pfrh_edgeMask, pfrh_parent);
        if (iter % 2) {
            edgeParent_newMask <<<(s + 31) / 32, 256 >> > (remaining, pfrh_edgeId, pfrh_edgeList, pfrh_edgesAll, pfrh_edgeMask, pfrh_parent);
        }
        else {
            edgeParent_newMask <<<(s + 31) / 32, 256 >> > (remaining, pfrh_edgeId, pfrh_edgeList, pfrh_edgesLeft, pfrh_edgeMask, pfrh_parent);
        }

//        cudaEventRecord(start);

        if (iter % 2) {
            auto len = thrust::copy_if(thrust::device, pfrh_edgesAll, pfrh_edgesAll + remaining, pfrh_edgeMask, pfrh_edgesLeft, is_true());

            remaining = len - pfrh_edgesLeft;
        }
        else {
            auto len = thrust::copy_if(thrust::device, pfrh_edgesLeft, pfrh_edgesLeft + remaining, pfrh_edgeMask, pfrh_edgesAll, is_true());

            remaining = len - pfrh_edgesAll;
        }

        //std::cout<<"Now on thrust::copy_if"<<std::endl;
        //auto len = thrust::copy_if(thrust::device, pfrh_edgesAll, pfrh_edgesAll + nEdges, pfrh_edgeMask, pfrh_edgesLeft, is_true());
        //remaining = len - pfrh_edgesLeft;
        //remaining = cuCompactor::compact<int>(pfrh_edgesAll, pfrh_edgesLeft, nEdges, stencil_true(pfrh_edgeMask), 128); 
        
        //std::cout<<"Now on cuCompactor::compact with nEdges = "<<nEdges<<std::endl;
        //remaining = cuCompactor::compactIndices<int>(pfrh_edgesLeft, nEdges, stencil_true(pfrh_edgeMask), 32); 
        //remaining = cuCompactor::compact<int>(pfrh_edgesAll, pfrh_edgesLeft, nEdges, stencil_true(pfrh_edgeMask), 32); 
        //std::cout<<"remaining = "<<remaining<<std::endl;
        // Determine how many edges left to process
        
        /*
        cudaMemcpyAsync(edgesLeft, pfrh_edgesLeft, sizeof(int)*remaining, cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(edgeMask, pfrh_edgeMask, sizeof(bool)*nEdges, cudaMemcpyDeviceToHost);
        int diff = 0;
        for (int edge = 0; edge < remaining; edge++) {
            int i = edgesLeft[edge];    //  real edge index
            if (edgeMask[i]) {
                edgesLeft[diff] = i;
                diff++;
            }
        }

        remaining = diff;
        cudaMemcpyAsync(pfrh_edgesLeft, edgesLeft, sizeof(int)*remaining, cudaMemcpyHostToDevice);
        */

        //updateEdgesLeftSerial<<<1,1>>>(remaining, pfrh_edgesLeft, pfrh_edgeMask, d_remaining);
        //cudaMemcpyAsync(&remaining, d_remaining, sizeof(int), cudaMemcpyDeviceToHost);
//        float time = 0.0;
//        cudaEventRecord(stop);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&time, start, stop);     
//        loopTimers.push_back(time);

#ifdef DEBUG_VERBOSE_OUTPUT
        cudaMemcpy(edgeId, pfrh_edgeId, sizeof(int)*nEdges, cudaMemcpyDeviceToHost);
        cudaMemcpy(edgeList, pfrh_edgeList, sizeof(int)*nEdges, cudaMemcpyDeviceToHost);
        std::cout<<"\nAfter reordering:"<<std::endl;
        std::cout<<"Edge ID:\t";
        for (int c = 0; c < nEdges; c++)
            std::cout<<edgeId[c]<<" ";
        std::cout<<std::endl;
        
        std::cout<<"Edge List:\t";
        for (int c = 0; c < nEdges; c++)
            std::cout<<edgeList[c]<<" ";
        std::cout<<std::endl;
        std::cout<<"New length: s = "<<s<<std::endl;
#endif
    }

//    timer = 0.0;
//    std::cout<<"Loop times: "; 
//    for (auto x : loopTimers) {
//        std::cout<<x<<" ";
//        timer += x;
//    }
//    std::cout<<"\nTotal time: "<<timer<<std::endl<<std::endl<<std::flush;
}

__global__ void graphContraction(volatile bool* d_notDone, size_t size, int* pfrh_parent) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < size) {
        volatile bool threadNotDone = false;
        int parent = pfrh_parent[idx];
        if (parent >= 0 && parent != pfrh_parent[parent]) {
            threadNotDone = true;
            pfrh_parent[idx] = pfrh_parent[parent];
        }
        if (threadNotDone) {
            // Set device flag to true
            *d_notDone = true;
        }
    }
}

__global__ void graphContraction_copyif(volatile bool* notDone, size_t size, int* pfrh_parent) {
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    
    if (idx < size) {
        volatile bool threadNotDone = false;
        int parent = pfrh_parent[idx];
        if (parent >= 0 && parent != pfrh_parent[parent]) {
            threadNotDone = true;
            pfrh_parent[idx] = pfrh_parent[parent];
        }
        if (threadNotDone) {
            // Set device flag to true
            *notDone = true;
        }
    }
}


__global__ void graphContractionSerial(volatile bool* notDone, size_t size, int* pfrh_parent) {
    bool imNotDone = false;
    for (int idx = 0; idx < size; idx++) {
        int parent = pfrh_parent[idx];
        if (parent >= 0 && parent != pfrh_parent[parent]) {
            imNotDone = true;
            pfrh_parent[idx] = pfrh_parent[parent];
        }
    }
    *notDone = imNotDone;
}

// Contraction in a single block
__global__ void graphContractionSingleBlock(size_t size, int* pfrh_parent) {
    //int idx = threadIdx.x + blockIdx.x*blockDim.x;
    __shared__ int notDone;
    if (threadIdx.x == 0) notDone = 0;
    __syncthreads();

    do {
        volatile bool threadNotDone = false;
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            int parent = pfrh_parent[i];
            if (parent >= 0 && parent != pfrh_parent[parent]) {
                threadNotDone = true;
                pfrh_parent[i] = pfrh_parent[parent];
            }
        }
        if (threadIdx.x == 0) notDone = 0;
        __syncthreads();
        
        atomicAdd(&notDone, (int)threadNotDone);
        //if (threadNotDone) notDone = true;
        //notDone |= threadNotDone;
        __syncthreads();

    } while (notDone);
}

void topoClusterContraction_singleBlock(int nRH,
    int* pfrh_parent) {

    graphContractionSingleBlock<<<1, 512>>>(nRH, pfrh_parent);
}


void topoClusterContraction(size_t nRH,
    int* pfrh_parent,
    volatile bool*  h_notDone,
    volatile bool*  d_notDone) {

    do {
        *h_notDone = false;
        graphContraction << <(nRH + 31) / 32, 128 >>> (d_notDone, nRH, pfrh_parent);
        //cudaDeviceSynchronize();
    } while (*h_notDone);
}



void topoClusterContraction_copyif(size_t nRH,
                int* pfrh_parent) {
    
    bool h_notDone = true;
    bool* d_notDone;
    cudaMalloc(&d_notDone, sizeof(bool));
    
    while (h_notDone) { 
        h_notDone = false;
        cudaMemcpyAsync(d_notDone, &h_notDone, sizeof(bool), cudaMemcpyHostToDevice);
        graphContraction_copyif<<<(nRH+31)/32, 128>>>(d_notDone, nRH, pfrh_parent); 
        //graphContractionSerial<<<1,1>>>(d_notDone, nRH, pfrh_parent); 
        cudaMemcpyAsync(&h_notDone, d_notDone, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    
#ifdef DEBUG_VERBOSE_OUTPUT
    int* parent = new int[nRH]; 
    cudaMemcpy(parent, pfrh_parent, sizeof(int)*nRH, cudaMemcpyDeviceToHost);
    std::cout<<"\n--- Parent after graph contraction ---\n";
    for (int c = 0; c < nRH; c++)
        std::cout<<parent[c]<<" ";
    std::cout<<std::endl<<std::flush;
#endif    
}

__global__ void initializeArrays(int size,
    const int* __restrict__ pfrh_edgeId,
    int* pfrh_edgeLeft,
    int* pfrh_edgeRight,
    int* pfrh_edgeMask) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < size) {
        pfrh_edgeMask[idx] = 1;
        pfrh_edgeLeft[idx]  = (idx > 0 ? pfrh_edgeId[idx-1] : pfrh_edgeId[0]);
        pfrh_edgeRight[idx] = (idx < size ? pfrh_edgeId[idx+1] : pfrh_edgeId[idx]);
    }
}


__global__ void updateAdjacentEdges(int nEdges,
    const int* __restrict__ pfrh_edgeId,
    int* pfrh_edgeLeft,
    int* pfrh_edgeRight,
    const int* __restrict__ pfrh_edgeMask) {

    int doubledidx = threadIdx.x + blockIdx.x*blockDim.x;
    int idx = doubledidx >> 1;  // Divide by 2. Even (odd) values update edgeLeft (edgeRight) arrays 
    if (doubledidx < nEdges && pfrh_edgeMask[idx] > 0) {
        if (doubledidx & 0x1) {    // even
            // Update left
            if (idx > 0) {

                int temp = idx - 1;
                int minVal = max(idx - 9, 0);   //  Only test up to 9 neighbors
                int tempId = 0;
                int edgeId = pfrh_edgeId[idx];
                //int minVal = 0;
                while (temp >= minVal) {
                    tempId = pfrh_edgeId[temp];
                    if (edgeId != tempId) {
                        // Different topo Id here!
                        pfrh_edgeLeft[idx] = -1;
                        break;
                    }
                    else if (pfrh_edgeMask[temp] > 0) {
                        // Found adjacent edge
                        pfrh_edgeLeft[idx] = tempId;
                        break;
                    }
                    temp--;
                }
            }
        }
        else {      // odd
            // Update right
            if (idx < (nEdges - 1)) {
                int temp = idx + 1;
                int maxVal = min(idx - 9, nEdges - 1);  //  Only test up to 9 neighbors
                int tempId = 0;
                int edgeId = pfrh_edgeId[idx];
                while (temp >= maxVal) {
                    tempId = pfrh_edgeId[temp];
                    if (edgeId != tempId) {
                        // Different topo Id here!
                        pfrh_edgeRight[idx] = -1;
                        break;
                    }
                    else if (pfrh_edgeMask[temp] > 0) {
                        // Found adjacent edge
                        pfrh_edgeRight[idx] = pfrh_edgeId[temp];
                        break;
                    }
                    temp++;
                }
            }
        }
    }
}

__global__ void updateAdjacentEdges_serial(int nEdges,
    const int* __restrict__ pfrh_edgeId,
    int* pfrh_edgeLeft,
    int* pfrh_edgeRight,
    const int* __restrict__ pfrh_edgeMask) {

    for (int idx = 0; idx < nEdges; idx++) {
        if (pfrh_edgeMask[idx] > 0) {
            // Update left
            if (idx > 0) {
                int temp = idx-1;
                while (temp >= 0) {
                    if (pfrh_edgeMask[temp] > 0) {
                        // Found adjacent edge
                        pfrh_edgeLeft[idx] = pfrh_edgeId[temp];
                        break;
                    }
                    temp--;
                }
            }

            // Update right
            if (idx < (nEdges-1)) {
                int temp = idx + 1;
                while (temp < nEdges) {
                    if (pfrh_edgeMask[temp] > 0) {
                        // Found adjacent edge
                        pfrh_edgeRight[idx] = pfrh_edgeId[temp];
                        break;
                    }
                    temp++;
                }
            }
        }   
    }
}


__global__ void countRemaining_serial(int nEdges,
    const int* __restrict__ pfrh_edgeMask,
    int* remaining) {
    int total = 0;
    for (int idx = 0; idx < nEdges; idx++) {
        total += (int)pfrh_edgeMask[idx];
    }
    *remaining = total;
}


void topoClusterLinkingFast(int nRH,
                int nEdges,
                int* pfrh_edgeId,
                int* pfrh_edgeList,
                int* pfrh_edgeLeft,
                int* pfrh_edgeRight,
                int* pfrh_edgeMask,
                int* pfrh_parent,
                bool* pfrh_passTopoThresh,
                bool* h_notDone,
                bool* d_notDone,
                int &nIter) {

    if (nEdges < 1) return;
    int iter = 0;

//    cudaSetDeviceFlags(cudaDeviceMapHost);
//    int* h_remaining = nullptr;
//    cudaHostAlloc((void**)&h_remaining, sizeof(int), cudaHostAllocMapped);
//    bool* h_flag = new bool(true);
//    bool* d_flag;
//    cudaMalloc(&d_flag, sizeof(bool));
//    cudaMemset(d_flag, false, sizeof(bool));

/*
    int* h_remaining = new int(nEdges);
    int* d_remaining;
    cudaMalloc(&d_remaining, sizeof(int));
    cudaMemset(d_remaining, 0, sizeof(int));
*/ 
    
    initializeArrays <<<64, 256 >>> (nEdges, pfrh_edgeId, pfrh_edgeLeft, pfrh_edgeRight, pfrh_edgeMask);

    do {
        *h_notDone = false;
        //*h_remaining = 0;
        iter++;
        if (iter > 100) {
            std::cout << "Too many iterations! Bailing out" << std::endl;
            return;
        }
 
        std::cout << "================================================================================";
        std::cout << "\nNow on iteration: " << iter << std::endl << std::endl;

        // odd iterations
        if (iter & 0x1) { oddLinkingParent <<<(nEdges + 63) / 64, 128 >>> (nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeLeft, pfrh_edgeMask, pfrh_parent, pfrh_passTopoThresh); }

        // even iterations
        else { evenLinkingParent <<<(nEdges + 63) / 64, 128 >> > (nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeRight, pfrh_edgeMask, pfrh_parent, pfrh_passTopoThresh); }


        // Replace edge values with parents
        edgeParent <<<(nEdges + 31) / 32, 256 >>> (nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_parent, h_notDone);
        //edgeParent_remaining <<<(nEdges + 31) / 32, 256 >>> (nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_parent, h_remaining);
        
        //edgeParent <<<(nEdges + 31) / 32, 256 >>> (nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_parent, d_notDone);
        //edgeParent <<<(nEdges + 31) / 32, 256 >>> (nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_parent, d_flag);
        //edgeParentSerial <<<1,1 >>> ((int)nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_parent, d_remaining);

        // Update adjacent edge lists
        updateAdjacentEdges<<<64, 256>>>(2*nEdges, pfrh_edgeId, pfrh_edgeLeft, pfrh_edgeRight, pfrh_edgeMask);

        cudaDeviceSynchronize();
        //std::cout<<"*h_remaining = "<<*h_remaining<<std::endl<<std::endl;

        //cudaMemcpy(h_flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost);
/*
        cudaMemcpy(h_remaining, d_remaining, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout<<"Remaining edges: "<<*h_remaining<<std::endl<<std::endl;
        cudaDeviceSynchronize();
*/
    } while (*h_notDone);
    //} while (*h_flag);
    //} while (*h_remaining > 0);
    nIter = iter;
    //cudaFreeHost(h_remaining);
}

void topoClusterLinking(int nRH,
                int nEdges,
                int* pfrh_edgeId,
                int* pfrh_edgeList,
                int* pfrh_edgeLeft,
                int* pfrh_edgeRight,
                int* pfrh_edgeMask,
                int* pfrh_parent,
                bool* pfrh_passTopoThresh,
                bool* h_notDone,
                bool* d_notDone) {

    if (nEdges < 1) return;
    int iter = 0;

//    cudaSetDeviceFlags(cudaDeviceMapHost);
//    int* h_remaining = nullptr;
//    cudaHostAlloc((void**)&h_remaining, sizeof(int), cudaHostAllocMapped);
//    bool* h_flag = new bool(true);
//    bool* d_flag;
//    cudaMalloc(&d_flag, sizeof(bool));
//    cudaMemset(d_flag, false, sizeof(bool));

/*
    int* h_remaining = new int(nEdges);
    int* d_remaining;
    cudaMalloc(&d_remaining, sizeof(int));
    cudaMemset(d_remaining, 0, sizeof(int));
*/ 

    initializeArrays <<<64, 256 >>> (nEdges, pfrh_edgeId, pfrh_edgeLeft, pfrh_edgeRight, pfrh_edgeMask);

    do {
        *h_notDone = false;
        //*h_remaining = 0;
        iter++;
        if (iter > 100) {
            std::cout << "Too many iterations! Bailing out" << std::endl;
            return;
        }
 
        std::cout << "================================================================================";
        std::cout << "\nNow on iteration: " << iter << std::endl << std::endl;

        // odd iterations
        if (iter & 0x1) { oddLinkingParent <<<(nEdges + 63) / 64, 128 >>> (nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeLeft, pfrh_edgeMask, pfrh_parent, pfrh_passTopoThresh); }

        // even iterations
        else { evenLinkingParent <<<(nEdges + 63) / 64, 128 >> > (nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeRight, pfrh_edgeMask, pfrh_parent, pfrh_passTopoThresh); }


        // Replace edge values with parents
        edgeParent <<<(nEdges + 31) / 32, 256 >>> (nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_parent, h_notDone);
        //edgeParent_remaining <<<(nEdges + 31) / 32, 256 >>> (nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_parent, h_remaining);
        
        //edgeParent <<<(nEdges + 31) / 32, 256 >>> (nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_parent, d_notDone);
        //edgeParent <<<(nEdges + 31) / 32, 256 >>> (nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_parent, d_flag);
        //edgeParentSerial <<<1,1 >>> ((int)nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_parent, d_remaining);

        // Update adjacent edge lists
        updateAdjacentEdges<<<64, 256>>>(2*nEdges, pfrh_edgeId, pfrh_edgeLeft, pfrh_edgeRight, pfrh_edgeMask);

        cudaDeviceSynchronize();
        //std::cout<<"*h_remaining = "<<*h_remaining<<std::endl<<std::endl;

        //cudaMemcpy(h_flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost);
/*
        cudaMemcpy(h_remaining, d_remaining, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout<<"Remaining edges: "<<*h_remaining<<std::endl<<std::endl;
        cudaDeviceSynchronize();
*/
    } while (*h_notDone);
    //} while (*h_flag);
    //} while (*h_remaining > 0);

    //cudaFreeHost(h_remaining);
}

__global__ void testMe (int* val) {
    *val = 2;
}

void LabelClustering(int nRH,
    int nEdges,
    int* pfrh_topoId,
    int* pfrh_edgeId,
    int* pfrh_edgeList,
    int* pfrh_edgeLeft,
    int* pfrh_edgeRight,
    int* pfrh_edgeMask,
    bool* pfrh_passTopoThresh,
    bool* h_notDone,
    bool* d_notDone) {
    if (nRH < 1) return;

    std::cout<<"Rec hits: "<<nRH<<std::endl;
    //cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaProfilerStart();
    
    // Linking
    //topoClusterLinking(nRH, nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeLeft, pfrh_edgeRight, pfrh_edgeMask, pfrh_topoId, pfrh_passTopoThresh, h_notDone, d_notDone);
    topoClusterLinking(nRH, nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeLeft, pfrh_edgeRight, pfrh_edgeMask, pfrh_topoId, pfrh_passTopoThresh, h_notDone, d_notDone);

/*
    int* h_val;
    int* d_val;
    cudaHostAlloc(reinterpret_cast<void**>(&h_val), sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_val), reinterpret_cast<void*>(h_val), 0);
    
    *h_val = 0;
    std::cout<<"Initial: *h_val = "<<*h_val<<std::endl;
    testMe<<<1,1>>>(d_val);
    
    bool* h_flag;
    bool* d_flag;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(reinterpret_cast<void**>(&h_flag), sizeof(bool), cudaHostAllocMapped);
    cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_flag), reinterpret_cast<void*>(h_flag), 0);
*/
    // Graph contraction
    //topoClusterContraction(nRH, pfrh_topoId, h_notDone, d_notDone);
    //topoClusterContraction(nRH, pfrh_topoId, h_flag, d_flag);
    //topoClusterContraction_copyif(nRH, pfrh_topoId);
    topoClusterContraction_singleBlock(nRH, pfrh_topoId);
    
 /*   
    std::cout<<"Result: *h_val = "<<*h_val<<std::endl;
    cudaDeviceSynchronize();
    std::cout<<"After sync: *h_val = "<<*h_val<<std::endl;

    
    cudaFreeHost(h_flag);
    cudaFreeHost(h_val);
 */   
    cudaProfilerStop();
}

void LabelClustering_nIter(int nRH,
    int nEdges,
    int* pfrh_topoId,
    int* pfrh_edgeId,
    int* pfrh_edgeList,
    int* pfrh_edgeLeft,
    int* pfrh_edgeRight,
    int* pfrh_edgeMask,
    bool* pfrh_passTopoThresh,
    bool* h_notDone,
    bool* d_notDone,
    int &nIter) {
    if (nRH < 1) return;

    std::cout<<"Rec hits: "<<nRH<<std::endl;
    //cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaProfilerStart();
    
    // Linking
    //topoClusterLinking(nRH, nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeLeft, pfrh_edgeRight, pfrh_edgeMask, pfrh_topoId, pfrh_passTopoThresh, h_notDone, d_notDone);
    topoClusterLinkingFast(nRH, nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeLeft, pfrh_edgeRight, pfrh_edgeMask, pfrh_topoId, pfrh_passTopoThresh, h_notDone, d_notDone, nIter);

/*
    int* h_val;
    int* d_val;
    cudaHostAlloc(reinterpret_cast<void**>(&h_val), sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_val), reinterpret_cast<void*>(h_val), 0);
    
    *h_val = 0;
    std::cout<<"Initial: *h_val = "<<*h_val<<std::endl;
    testMe<<<1,1>>>(d_val);
    
    bool* h_flag;
    bool* d_flag;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(reinterpret_cast<void**>(&h_flag), sizeof(bool), cudaHostAllocMapped);
    cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_flag), reinterpret_cast<void*>(h_flag), 0);
*/
    // Graph contraction
    //topoClusterContraction(nRH, pfrh_topoId, h_notDone, d_notDone);
    //topoClusterContraction(nRH, pfrh_topoId, h_flag, d_flag);
    //topoClusterContraction_copyif(nRH, pfrh_topoId);
    topoClusterContraction_singleBlock(nRH, pfrh_topoId);
    
 /*   
    std::cout<<"Result: *h_val = "<<*h_val<<std::endl;
    cudaDeviceSynchronize();
    std::cout<<"After sync: *h_val = "<<*h_val<<std::endl;

    
    cudaFreeHost(h_flag);
    cudaFreeHost(h_val);
 */   
    cudaProfilerStop();
}

void PFRechitToPFCluster_HCAL_LabelClustering_nIter(int nRH,
                int nEdges,
                const float* __restrict__ pfrh_x,
                const float* __restrict__ pfrh_y,
                const float* __restrict__ pfrh_z,
                const double* __restrict__ pfrh_energy,
                const double* __restrict__ pfrh_pt2,
                int* pfrh_isSeed,
                int* pfrh_topoId,
                const int* __restrict__ pfrh_layer,
                const int* __restrict__ pfrh_depth,
                const int* __restrict__ neigh8_Ind,
                const int* __restrict__ neigh4_Ind,
                int* pfrh_edgeId,
                int* pfrh_edgeList,
                int* pfrh_edgeLeft,
                int* pfrh_edgeRight,
                int* pfrh_edgeMask,
                bool* pfrh_passTopoThresh,
                bool* h_notDone,
                bool* d_notDone,
                int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
                int* rhCount,
                float (&timer)[8],
                int &nIter) {
    if (nRH < 1) return;
#ifdef DEBUG_VERBOSE_OUTPUT
    std::cout<<"nRH = "<<nRH<<"\tnEdges = "<<nEdges<<std::endl;
#endif 

#ifdef DEBUG_GPU_HCAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif
    //seeding
    seedingKernel_HCAL<<<(nRH+511)/512, 512>>>( nRH,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,pfrh_depth,  neigh4_Ind);

    // Passing topo clustering threshold
    passingTopoThreshold<<<(nRH+255)/256, 256>>>( nRH, pfrh_layer, pfrh_depth, pfrh_energy, pfrh_passTopoThresh);
#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&timer[0], start, stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif
    
    //topoclustering 
    LabelClustering_nIter(nRH, nEdges, pfrh_topoId, pfrh_edgeId, pfrh_edgeList, pfrh_edgeLeft, pfrh_edgeRight, pfrh_edgeMask, pfrh_passTopoThresh, h_notDone, d_notDone, nIter);

#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&timer[1], start, stop);
    cudaEventRecord(start);
#endif

    dim3 grid( (nRH+32-1)/32, (nRH+32-1)/32 );
    dim3 block( 32, 32);

    hcalFastCluster_step1<<<grid, block>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);

#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&timer[2], start, stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif

    hcalFastCluster_step2<<<grid, block>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);

#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&timer[3], start, stop);
#endif

}

void PFRechitToPFCluster_HCAL_LabelClustering(int nRH,
                int nEdges,
                const float* __restrict__ pfrh_x,
                const float* __restrict__ pfrh_y,
                const float* __restrict__ pfrh_z,
                const double* __restrict__ pfrh_energy,
                const double* __restrict__ pfrh_pt2,
                int* pfrh_isSeed,
                int* pfrh_topoId,
                const int* __restrict__ pfrh_layer,
                const int* __restrict__ pfrh_depth,
                const int* __restrict__ neigh8_Ind,
                const int* __restrict__ neigh4_Ind,
                int* pfrh_edgeId,
                int* pfrh_edgeList,
                int* pfrh_edgeLeft,
                int* pfrh_edgeRight,
                int* pfrh_edgeMask,
                bool* pfrh_passTopoThresh,
                bool* h_notDone,
                bool* d_notDone,
                int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
                int* rhCount,
                float (&timer)[8]) {
    if (nRH < 1) return;
#ifdef DEBUG_VERBOSE_OUTPUT
    std::cout<<"nRH = "<<nRH<<"\tnEdges = "<<nEdges<<std::endl;
#endif 

#ifdef DEBUG_GPU_HCAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif
    //seeding
    seedingKernel_HCAL<<<(nRH+511)/512, 512>>>( nRH,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,pfrh_depth,  neigh4_Ind);

    // Passing topo clustering threshold
    passingTopoThreshold<<<(nRH+255)/256, 256>>>( nRH, pfrh_layer, pfrh_depth, pfrh_energy, pfrh_passTopoThresh);
#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&timer[0], start, stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif
    
    //topoclustering 
    LabelClustering(nRH, nEdges, pfrh_topoId, pfrh_edgeId, pfrh_edgeList, pfrh_edgeLeft, pfrh_edgeRight, pfrh_edgeMask, pfrh_passTopoThresh, h_notDone, d_notDone);

#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&timer[1], start, stop);
    cudaEventRecord(start);
#endif

    dim3 grid( (nRH+32-1)/32, (nRH+32-1)/32 );
    dim3 block( 32, 32);

    hcalFastCluster_step1<<<grid, block>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);

#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&timer[2], start, stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif

    hcalFastCluster_step2<<<grid, block>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);

#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&timer[3], start, stop);
#endif

}

void PFRechitToPFCluster_HCAL_LabelClustering_copyif(size_t nRH,
                size_t nEdges,
				const float* __restrict__ pfrh_x, 
				const float* __restrict__ pfrh_y, 
				const float* __restrict__ pfrh_z, 
				const double* __restrict__ pfrh_energy, 
				const double* __restrict__ pfrh_pt2,    				
				int* pfrh_isSeed,
				int* pfrh_topoId, 
				const int* __restrict__ pfrh_layer, 
				const int* __restrict__ pfrh_depth, 
			    const int* __restrict__ neigh8_Ind,
                const int* __restrict__ neigh4_Ind,	
				int* pfrh_edgeId,
				int* pfrh_edgeList,
				int* pfrh_edgesAll,
				int* pfrh_edgesLeft,
				bool* pfrh_edgeMask,
                bool* pfrh_passTopoThresh,
                int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
				int* rhCount,
				float (&timer)[8]
                )
  {
    if (nRH < 1) return;
#ifdef DEBUG_VERBOSE_OUTPUT
    std::cout<<"nRH = "<<nRH<<"\tnEdges = "<<nEdges<<std::endl;
#endif 

#ifdef DEBUG_GPU_HCAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif
    //seeding
    seedingKernel_HCAL<<<(nRH+511)/512, 512>>>( nRH,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,pfrh_depth,  neigh4_Ind);

#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);   
      cudaEventElapsedTime(&timer[0], start, stop);
      cudaDeviceSynchronize();
      cudaEventRecord(start);
#endif
    
    //topoclustering 
      // Determine rechits passing topo clustering energy threshold and initialize parent array
      passingTopoThreshold<<<(nRH+255)/256, 256>>>( nRH, pfrh_layer, pfrh_depth, pfrh_energy, pfrh_passTopoThresh);

#ifdef DEBUG_GPU_HCAL
//      cudaEventRecord(stop);
//      cudaEventSynchronize(stop);   
//      cudaEventElapsedTime(&timer[4], start, stop);
//      cudaEventRecord(start);
#endif
//  Example from https://uca.edu/computerscience/files/2020/02/An-Efficient-Parallel-Implementation-of-Structural-Network-Clustering-in-Massively-Parallel-GPU.pdf
//      nRH = 10;
//      nEdges = 18;
//      int edgeId[18]   = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 5, 5, 6, 6, 7, 7, 8, 9};
//      int edgeList[18] = {1, 2, 0, 2, 3, 0, 1, 3, 1, 2, 6, 7, 5, 7, 5, 6, 9, 8};
//      cudaMemcpy(pfrh_edgeId, edgeId, sizeof(int)*18, cudaMemcpyHostToDevice);
//      cudaMemcpy(pfrh_edgeList, edgeList, sizeof(int)*18, cudaMemcpyHostToDevice);
      
      // Linking
      //topoClusterLinking(nRH, nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgesLeft, pfrh_edgeMask, pfrh_topoId, pfrh_passTopoThresh);      
      topoClusterLinking_copyif(nRH, nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgesAll, pfrh_edgesLeft, pfrh_edgeMask, pfrh_topoId, pfrh_passTopoThresh, timer[5]);      
      
#ifdef DEBUG_GPU_HCAL
//      cudaEventRecord(stop);
//      cudaEventSynchronize(stop);   
//      cudaEventElapsedTime(&timer[5], start, stop);
//      cudaEventRecord(start);
#endif
      
      // Graph contraction
      topoClusterContraction_copyif(nRH, pfrh_topoId);

#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(stop);
      cudaDeviceSynchronize();
      cudaEventSynchronize(stop);   
//      cudaEventElapsedTime(&timer[6], start, stop);
      cudaEventElapsedTime(&timer[1], start, stop);
//      timer[1] = timer[4] + timer[5] + timer[6];
      cudaEventRecord(start);
#endif

      dim3 grid( (nRH+32-1)/32, (nRH+32-1)/32 );
      dim3 block( 32, 32);

      hcalFastCluster_step1<<<grid, block>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);

#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);   
      cudaEventElapsedTime(&timer[2], start, stop);
      cudaDeviceSynchronize();
      cudaEventRecord(start);
#endif

      hcalFastCluster_step2<<<grid, block>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);

#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);   
      cudaEventElapsedTime(&timer[3], start, stop);
#endif

  }

void PFRechitToPFCluster_HCALV2(size_t size, 
				const float* __restrict__ pfrh_x, 
				const float* __restrict__ pfrh_y, 
				const float* __restrict__ pfrh_z, 
				const double* __restrict__ pfrh_energy, 
				const double* __restrict__ pfrh_pt2,    				
				int* pfrh_isSeed,
				int* pfrh_topoId, 
				const int* __restrict__ pfrh_layer, 
				const int* __restrict__ pfrh_depth, 
			    const int* __restrict__ neigh8_Ind,
                const int* __restrict__ neigh4_Ind,	
				int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
				int* rhCount,
				float (&timer)[8]
                )
  {
#ifdef DEBUG_GPU_HCAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    //seeding
    if(size>0) seedingKernel_HCAL<<<(size+512-1)/512, 512>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,pfrh_depth,  neigh4_Ind);

#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);   
      cudaEventElapsedTime(&timer[0], start, stop);
      cudaEventRecord(start);
#endif
    
    //topoclustering 
     
      //dim3 gridT( (size+64-1)/64, 1 );
      //dim3 blockT( 64, 8);
      dim3 gridT( (size+64-1)/64, 8 );
      dim3 blockT( 64, 16); // 16 threads in a half-warp
      for(int h=0;h<nTopoLoops; h++){
    
      if(size>0) topoKernel_HCALV2<<<gridT, blockT>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, pfrh_depth, neigh8_Ind);	     
      }
   
#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);   
      cudaEventElapsedTime(&timer[1], start, stop);
      cudaEventRecord(start);
#endif

      dim3 grid( (size+32-1)/32, (size+32-1)/32 );
      dim3 block( 32, 32);

      if(size>0) hcalFastCluster_step1<<<grid, block>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);

#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);   
      cudaEventElapsedTime(&timer[2], start, stop);
      cudaEventRecord(start);
#endif

      if(size>0) hcalFastCluster_step2<<<grid, block>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);

#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);   
      cudaEventElapsedTime(&timer[3], start, stop);
#endif

  }

void PFRechitToPFCluster_HCAL_serialize(size_t size, 
				const float* __restrict__ pfrh_x, 
				const float* __restrict__ pfrh_y, 
				const float* __restrict__ pfrh_z, 
				const double* __restrict__ pfrh_energy, 
				const double* __restrict__ pfrh_pt2,    				
				int* pfrh_isSeed,
				int* pfrh_topoId, 
				const int* __restrict__ pfrh_layer, 
				const int* __restrict__ pfrh_depth, 
			    const int* __restrict__ neigh8_Ind,
                const int* __restrict__ neigh4_Ind,	
				int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
				int* rhCount,
				float* timer
				)
  {
#ifdef DEBUG_GPU_HCAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif
    //seeding
    if(size>0) seedingKernel_HCAL_serialize<<<1,1>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,pfrh_depth,  neigh4_Ind);
    
    //topoclustering 
      
#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(start);
#endif
      for(int h=0;h<nTopoLoops; h++){
    
      if(size>0) topoKernel_HCAL_serialize<<<1,1>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, pfrh_depth, neigh8_Ind);	     
      }
#ifdef DEBUG_GPU_HCAL
      float milliseconds = 0;
      if (timer != nullptr)
      {
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&milliseconds, start, stop);
          *timer = milliseconds;
      }
#endif    

      if(size>0) hcalFastCluster_step1_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);

      if(size>0) hcalFastCluster_step2_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);


  }

} // namespace cudavectors
