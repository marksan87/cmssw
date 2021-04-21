
#include <cmath>
#include <iostream>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterCudaHCAL.h"
#include <Eigen/Dense>

#define DEBUG_GPU_HCAL

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


  void initializeCudaConstants(float h_showerSigma,
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
     cudaCheck(cudaMemcpyToSymbolAsync(showerSigma, &h_showerSigma, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     float val = 0.;
     cudaMemcpyFromSymbol(&val, showerSigma, sizeof_float);
     std::cout<<"showerSigma read from symbol: "<<val<<std::endl;
#endif
     
     cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormEB_vec, &h_recHitEnergyNormEB_vec, 4*sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     float val4[4];
     cudaMemcpyFromSymbol(&val4, recHitEnergyNormEB_vec, 4*sizeof_float);
     std::cout<<"recHitEnergyNormEB_vec read from symbol: ";
     for (int i = 0; i < 4; i++) {std::cout<<val4[i]<<" ";}
     std::cout<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormEE_vec, &h_recHitEnergyNormEE_vec, 7*sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     float val7[7];
     cudaMemcpyFromSymbol(&val7, recHitEnergyNormEE_vec, 7*sizeof_float);
     std::cout<<"recHitEnergyNormEE_vec read from symbol: ";
     for (int i = 0; i < 7; i++) {std::cout<<val7[i]<<" ";}
     std::cout<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(minFracToKeep, &h_minFracToKeep, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, minFracToKeep, sizeof_float);
     std::cout<<"minFracToKeep read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(seedEThresholdEB_vec, &h_seedEThresholdEB_vec, 4*sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     cudaMemcpyFromSymbol(&val4, seedEThresholdEB_vec, 4*sizeof_float);
     std::cout<<"seedEThresholdEB_vec read from symbol: ";
     for (int i = 0; i < 4; i++) {std::cout<<val4[i]<<" ";}
     std::cout<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(seedEThresholdEE_vec, &h_seedEThresholdEE_vec, 7*sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     cudaMemcpyFromSymbol(&val7, seedEThresholdEE_vec, 7*sizeof_float);
     std::cout<<"seedEThresholdEE_vec read from symbol: ";
     for (int i = 0; i < 7; i++) {std::cout<<val7[i]<<" ";}
     std::cout<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(seedPt2ThresholdEB, &h_seedPt2ThresholdEB, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, seedPt2ThresholdEB, sizeof_float);
     std::cout<<"seedPt2ThresholdEB read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(seedPt2ThresholdEE, &h_seedPt2ThresholdEE, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, seedPt2ThresholdEE, sizeof_float);
     std::cout<<"seedPt2ThresholdEE read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(topoEThresholdEB_vec, &h_topoEThresholdEB_vec, 4*sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     cudaMemcpyFromSymbol(&val4, topoEThresholdEB_vec, 4*sizeof_float);
     std::cout<<"topoEThresholdEB_vec read from symbol: ";
     for (int i = 0; i < 4; i++) {std::cout<<val4[i]<<" ";}
     std::cout<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(topoEThresholdEE_vec, &h_topoEThresholdEE_vec, 7*sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     cudaMemcpyFromSymbol(&val7, topoEThresholdEE_vec, 7*sizeof_float);
     std::cout<<"topoEThresholdEE_vec read from symbol: ";
     for (int i = 0; i < 7; i++) {std::cout<<val7[i]<<" ";}
     std::cout<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(nNeigh, &h_nNeigh, sizeof_int));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     int ival = 0;
     cudaMemcpyFromSymbol(&ival, nNeigh, sizeof_int);
     std::cout<<"nNeigh read from symbol: "<<ival<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(maxSize, &h_maxSize, sizeof_int));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     ival = 0;
     cudaMemcpyFromSymbol(&ival, maxSize, sizeof_int);
     std::cout<<"maxSize read from symbol: "<<ival<<std::endl;
#endif
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

// Linking step on odd iterations
__global__ void oddLinkingParent(size_t nEdges,
                const int* __restrict__ pfrh_edgeId,
                const int* __restrict__ pfrh_edgeList,
                int* pfrh_parent,
                const bool* __restrict__ pfrh_passTopoThresh) {

    int i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i < nEdges && pfrh_passTopoThresh[pfrh_edgeId[i]] && (i == 0 || (pfrh_edgeId[i] != pfrh_edgeId[i-1])) ) {
        pfrh_parent[pfrh_edgeId[i]] = (int)min(pfrh_edgeId[i], pfrh_edgeList[i]);
    }
}

// Linking step on even iterations
__global__ void evenLinkingParent(size_t nEdges,
                const int* __restrict__ pfrh_edgeId,
                const int* __restrict__ pfrh_edgeList,
                int* pfrh_parent,
                const bool* __restrict__ pfrh_passTopoThresh) {

    int i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i < nEdges && pfrh_passTopoThresh[pfrh_edgeId[i]] && (i == (nEdges-1) || (pfrh_edgeId[i] != pfrh_edgeId[i+1])) ) {
        pfrh_parent[pfrh_edgeId[i]] = (int)max(pfrh_edgeId[i], pfrh_edgeList[i]);
    }
}

// Linking step on odd iterations (serialized)
__global__ void oddLinkingParentSerial(size_t nEdges,
                const int* __restrict__ pfrh_edgeId,
                const int* __restrict__ pfrh_edgeList,
                int* pfrh_parent,
                const bool* __restrict__ pfrh_passTopoThresh) {

    for (int i = 0; i < nEdges; i++) {
        if (i >= 0 && i < nEdges && pfrh_passTopoThresh[pfrh_edgeId[i]] && (i == 0 || (pfrh_edgeId[i] != pfrh_edgeId[i-1])) ) {
            pfrh_parent[pfrh_edgeId[i]] = (int)min(pfrh_edgeId[i], pfrh_edgeList[i]);
        }
    }
}

// Linking step on even iterations (serialized)
__global__ void evenLinkingParentSerial(size_t nEdges,
                const int* __restrict__ pfrh_edgeId,
                const int* __restrict__ pfrh_edgeList,
                int* pfrh_parent,
                const bool* __restrict__ pfrh_passTopoThresh) {

    for (int i = 0; i < nEdges; i++) {
        if (i >= 0 && i < nEdges && pfrh_passTopoThresh[pfrh_edgeId[i]] && (i == (nEdges-1) || (pfrh_edgeId[i] != pfrh_edgeId[i+1])) ) {
            pfrh_parent[pfrh_edgeId[i]] = (int)max(pfrh_edgeId[i], pfrh_edgeList[i]);
        }
    }
}


// Replace edgeId and edgeList with parents
__global__ void edgeParent(size_t nEdges,
                int* pfrh_edgeId,
                int* pfrh_edgeList,
                bool* pfrh_edgeMask,
                const int* __restrict__ pfrh_parent) {

    int i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i >= 0 && i < nEdges) {
        pfrh_edgeId[i] = pfrh_parent[pfrh_edgeId[i]];
        pfrh_edgeList[i] = pfrh_parent[pfrh_edgeList[i]];

        // edgeMask set to true if elements of edgeId and edgeList are different
        pfrh_edgeMask[i] = (pfrh_edgeId[i] != pfrh_edgeList[i]);
    }
}

// Replace edgeId and edgeList with parents
__global__ void edgeParentSerial(size_t nEdges,
                int* pfrh_edgeId,
                int* pfrh_edgeList,
                bool* pfrh_edgeMask,
                const int* __restrict__ pfrh_parent) {

    for (int i = 0; i < nEdges; i++) {
        if (i >= 0 && i < nEdges) {
            pfrh_edgeId[i] = pfrh_parent[pfrh_edgeId[i]];
            pfrh_edgeList[i] = pfrh_parent[pfrh_edgeList[i]];

            // edgeMask set to true if elements of edgeId and edgeList are different
            pfrh_edgeMask[i] = (pfrh_edgeId[i] != pfrh_edgeList[i]);
        }
    }
}

struct is_true {
    __host__ __device__ bool operator()(const bool &x) {
        return x; 
    }
};

void topoClusterLinking(size_t nRH,
                size_t nEdges,
                int* pfrh_edgeId,
                int* pfrh_edgeList,
                bool* pfrh_edgeMask,
                int* pfrh_parent,
                bool* pfrh_passTopoThresh) {
    if (nEdges < 1) return;
    int s = (int)nEdges;
    int iter = 0;

#ifdef DEBUG_VERBOSE_OUTPUT
    int* edgeId = new int[s];
    int* edgeList = new int[s];
    bool* edgeMask = new bool[s];
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

    while (s > 0) {
        iter++;
#ifdef DEBUG_VERBOSE_OUTPUT
        std::cout<<"\nNow on iteration: "<<iter<<"\t\ts = "<<s<<std::endl;
#endif
        // odd iterations
        if (iter%2) { oddLinkingParent<<<(s+63)/64, 64>>>(s, pfrh_edgeId, pfrh_edgeList, pfrh_parent, pfrh_passTopoThresh); } 
        //if (iter%2) { oddLinkingParentSerial<<<1,1>>>(s, pfrh_edgeId, pfrh_edgeList, pfrh_parent, pfrh_passTopoThresh); } 
        // even iterations
        else { evenLinkingParent<<<(s+63)/64, 64>>>(s, pfrh_edgeId, pfrh_edgeList, pfrh_parent, pfrh_passTopoThresh); }
        //else { evenLinkingParentSerial<<<1,1>>>(s, pfrh_edgeId, pfrh_edgeList, pfrh_parent, pfrh_passTopoThresh); }

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
        edgeParent<<<(s+63)/64, 64>>>(s, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_parent);

#ifdef DEBUG_VERBOSE_OUTPUT
        cudaMemcpy(edgeId, pfrh_edgeId, sizeof(int)*s, cudaMemcpyDeviceToHost);
        cudaMemcpy(edgeList, pfrh_edgeList, sizeof(int)*s, cudaMemcpyDeviceToHost);
        cudaMemcpy(edgeMask, pfrh_edgeMask, sizeof(bool)*s, cudaMemcpyDeviceToHost);
        cudaMemcpy(parent, pfrh_parent, sizeof(int)*nRH, cudaMemcpyDeviceToHost);
        
        int nSwaps = 0;
        for (int c = 0; c < s; c++) {
            if (!edgeMask[c]){
                nSwaps++;
            }
        }
   
        std::cout<<"*** Number of swaps to make: "<<nSwaps<<std::endl;
#endif

        // Reorder edgeId, edgeList vectors so that values that are identical between the two are moved to the end
        // stable_partition preserves relative order of elements
        auto len = thrust::stable_partition(thrust::device, pfrh_edgeId, pfrh_edgeId + s, pfrh_edgeMask, is_true());
        thrust::stable_partition(thrust::device, pfrh_edgeList, pfrh_edgeList + s, pfrh_edgeMask, is_true());
        // New length of partitioned edgeId, edgeList arrays containing different elements
        s = len - pfrh_edgeId;

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
}


__global__ void graphContraction(volatile bool* notDone, size_t size, int* pfrh_parent) {
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


void topoClusterContraction(size_t nRH,
                int* pfrh_parent) {
    
    bool h_notDone = true;
    bool* d_notDone;
    cudaMalloc(&d_notDone, sizeof(bool));
    
    while (h_notDone) { 
        h_notDone = false;
        cudaMemcpy(d_notDone, &h_notDone, sizeof(bool), cudaMemcpyHostToDevice);
        graphContraction<<<(nRH+63)/64, 64>>>(d_notDone, nRH, pfrh_parent); 
        //graphContractionSerial<<<1,1>>>(d_notDone, nRH, pfrh_parent); 
        cudaMemcpy(&h_notDone, d_notDone, sizeof(bool), cudaMemcpyDeviceToHost);
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

void PFRechitToPFCluster_HCAL_LabelClustering(size_t nRH,
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
				bool* pfrh_edgeMask,
                bool* pfrh_passTopoThresh,
                int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
				int* rhCount,
				float (&timer)[4]
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
    cudaEventRecord(start);
#endif
    //seeding
    seedingKernel_HCAL<<<(nRH+511)/512, 512>>>( nRH,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,pfrh_depth,  neigh4_Ind);

#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);   
      cudaEventElapsedTime(&timer[0], start, stop);
      cudaEventRecord(start);
#endif
    
    //topoclustering 
      // Determine rechits passing topo clustering energy threshold and initialize parent array
      passingTopoThreshold<<<(nRH+127)/128, 128>>>( nRH, pfrh_layer, pfrh_depth, pfrh_energy, pfrh_passTopoThresh);

//  Example from https://uca.edu/computerscience/files/2020/02/An-Efficient-Parallel-Implementation-of-Structural-Network-Clustering-in-Massively-Parallel-GPU.pdf
//      nRH = 10;
//      nEdges = 18;
//      int edgeId[18]   = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 5, 5, 6, 6, 7, 7, 8, 9};
//      int edgeList[18] = {1, 2, 0, 2, 3, 0, 1, 3, 1, 2, 6, 7, 5, 7, 5, 6, 9, 8};
//      cudaMemcpy(pfrh_edgeId, edgeId, sizeof(int)*18, cudaMemcpyHostToDevice);
//      cudaMemcpy(pfrh_edgeList, edgeList, sizeof(int)*18, cudaMemcpyHostToDevice);
      
      // Linking
      topoClusterLinking(nRH, nEdges, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_topoId, pfrh_passTopoThresh);      
      
      // Graph contraction
      topoClusterContraction(nRH, pfrh_topoId);

#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(stop);
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
				float (&timer)[4]
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
