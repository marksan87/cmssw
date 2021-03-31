
#include <cmath>
#include <iostream>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterCudaHCAL.h"
#include <Eigen/Dense>

//#define GPU_DEBUG_HCAL

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
  
  int nTopoLoops = 100;
  //int nTopoLoops = 35;


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
#ifdef GPU_DEBUG_HCAL
     // Read back the value
     float val = 0.;
     cudaMemcpyFromSymbol(&val, showerSigma, sizeof_float);
     std::cout<<"showerSigma read from symbol: "<<val<<std::endl;
#endif
     
     cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormEB_vec, &h_recHitEnergyNormEB_vec, 4*sizeof_float));
#ifdef GPU_DEBUG_HCAL
     // Read back the value
     float val4[4];
     cudaMemcpyFromSymbol(&val4, recHitEnergyNormEB_vec, 4*sizeof_float);
     std::cout<<"recHitEnergyNormEB_vec read from symbol: ";
     for (int i = 0; i < 4; i++) {std::cout<<val4[i]<<" ";}
     std::cout<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormEE_vec, &h_recHitEnergyNormEE_vec, 7*sizeof_float));
#ifdef GPU_DEBUG_HCAL
     // Read back the value
     float val7[7];
     cudaMemcpyFromSymbol(&val7, recHitEnergyNormEE_vec, 7*sizeof_float);
     std::cout<<"recHitEnergyNormEE_vec read from symbol: ";
     for (int i = 0; i < 7; i++) {std::cout<<val7[i]<<" ";}
     std::cout<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(minFracToKeep, &h_minFracToKeep, sizeof_float));
#ifdef GPU_DEBUG_HCAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, minFracToKeep, sizeof_float);
     std::cout<<"minFracToKeep read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(seedEThresholdEB_vec, &h_seedEThresholdEB_vec, 4*sizeof_float));
#ifdef GPU_DEBUG_HCAL
     // Read back the value
     cudaMemcpyFromSymbol(&val4, seedEThresholdEB_vec, 4*sizeof_float);
     std::cout<<"seedEThresholdEB_vec read from symbol: ";
     for (int i = 0; i < 4; i++) {std::cout<<val4[i]<<" ";}
     std::cout<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(seedEThresholdEE_vec, &h_seedEThresholdEE_vec, 7*sizeof_float));
#ifdef GPU_DEBUG_HCAL
     // Read back the value
     cudaMemcpyFromSymbol(&val7, seedEThresholdEE_vec, 7*sizeof_float);
     std::cout<<"seedEThresholdEE_vec read from symbol: ";
     for (int i = 0; i < 7; i++) {std::cout<<val7[i]<<" ";}
     std::cout<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(seedPt2ThresholdEB, &h_seedPt2ThresholdEB, sizeof_float));
#ifdef GPU_DEBUG_HCAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, seedPt2ThresholdEB, sizeof_float);
     std::cout<<"seedPt2ThresholdEB read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(seedPt2ThresholdEE, &h_seedPt2ThresholdEE, sizeof_float));
#ifdef GPU_DEBUG_HCAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, seedPt2ThresholdEE, sizeof_float);
     std::cout<<"seedPt2ThresholdEE read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(topoEThresholdEB_vec, &h_topoEThresholdEB_vec, 4*sizeof_float));
#ifdef GPU_DEBUG_HCAL
     // Read back the value
     cudaMemcpyFromSymbol(&val4, topoEThresholdEB_vec, 4*sizeof_float);
     std::cout<<"topoEThresholdEB_vec read from symbol: ";
     for (int i = 0; i < 4; i++) {std::cout<<val4[i]<<" ";}
     std::cout<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(topoEThresholdEE_vec, &h_topoEThresholdEE_vec, 7*sizeof_float));
#ifdef GPU_DEBUG_HCAL
     // Read back the value
     cudaMemcpyFromSymbol(&val7, topoEThresholdEE_vec, 7*sizeof_float);
     std::cout<<"topoEThresholdEE_vec read from symbol: ";
     for (int i = 0; i < 7; i++) {std::cout<<val7[i]<<" ";}
     std::cout<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(nNeigh, &h_nNeigh, sizeof_int));
#ifdef GPU_DEBUG_HCAL
     // Read back the value
     int ival = 0;
     cudaMemcpyFromSymbol(&ival, nNeigh, sizeof_int);
     std::cout<<"nNeigh read from symbol: "<<ival<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(maxSize, &h_maxSize, sizeof_int));
#ifdef GPU_DEBUG_HCAL
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
//     if( ( pfrh_layer[i] == 1 && 
//	   pfrh_depth[i] == 1 &&
//	   pfrh_energy[i]>seedEThresholdEB_1 && 
//	   pfrh_pt2[i]>seedPt2ThresholdEB ) ||
//	 ( pfrh_layer[i] == 1 && 
//	   pfrh_depth[i] == 2 &&
//	   pfrh_energy[i]>seedEThresholdEB_2 && 
//	   pfrh_pt2[i]>seedPt2ThresholdEB ) ||
//	 ( pfrh_layer[i] == 1 && 
//	   pfrh_depth[i] == 3 &&
//	   pfrh_energy[i]>seedEThresholdEB_3 && 
//	   pfrh_pt2[i]>seedPt2ThresholdEB ) ||
//	 ( pfrh_layer[i] == 1 && 
//	   pfrh_depth[i] == 4 &&
//	   pfrh_energy[i]>seedEThresholdEB_4 && 
//	   pfrh_pt2[i]>seedPt2ThresholdEB ) ||
//	 ( pfrh_layer[i] == 3  && 
//	   pfrh_depth[i] == 1  &&
//	   pfrh_energy[i]>seedEThresholdEE_1 && 
//	   pfrh_pt2[i]>seedPt2ThresholdEE)   ||
//	 ( pfrh_layer[i] == 3  && 
//	   pfrh_depth[i] > 1   &&
//	   pfrh_energy[i]>seedEThresholdEE_2_7 && 
//	   pfrh_pt2[i]>seedPt2ThresholdEE))
     if ( (pfrh_layer[i] == 1 && pfrh_energy[i]>seedEThresholdEB_vec[pfrh_depth[i] - 1] && pfrh_pt2[i]>seedPt2ThresholdEB) || 
          (pfrh_layer[i] == 3 && pfrh_energy[i]>seedEThresholdEE_vec[pfrh_depth[i] - 1] && pfrh_pt2[i]>seedPt2ThresholdEE) )
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
//         if( ( pfrh_layer[i] == 1 && 
//           pfrh_depth[i] == 1 &&
//           pfrh_energy[i]>seedEThresholdEB_1 && 
//           pfrh_pt2[i]>seedPt2ThresholdEB ) ||
//         ( pfrh_layer[i] == 1 && 
//           pfrh_depth[i] == 2 &&
//           pfrh_energy[i]>seedEThresholdEB_2 && 
//           pfrh_pt2[i]>seedPt2ThresholdEB ) ||
//         ( pfrh_layer[i] == 1 && 
//           pfrh_depth[i] == 3 &&
//           pfrh_energy[i]>seedEThresholdEB_3 && 
//           pfrh_pt2[i]>seedPt2ThresholdEB ) ||
//         ( pfrh_layer[i] == 1 && 
//           pfrh_depth[i] == 4 &&
//           pfrh_energy[i]>seedEThresholdEB_4 && 
//           pfrh_pt2[i]>seedPt2ThresholdEB ) ||
//         ( pfrh_layer[i] == 3  && 
//           pfrh_depth[i] == 1  &&
//           pfrh_energy[i]>seedEThresholdEE_1 && 
//           pfrh_pt2[i]>seedPt2ThresholdEE)   ||
//         ( pfrh_layer[i] == 3  && 
//           pfrh_depth[i] > 1   &&
//           pfrh_energy[i]>seedEThresholdEE_2_7 && 
//           pfrh_pt2[i]>seedPt2ThresholdEE))
     if ( (pfrh_layer[i] == 1 && pfrh_energy[i]>seedEThresholdEB_vec[pfrh_depth[i] - 1] && pfrh_pt2[i]>seedPt2ThresholdEB) || 
          (pfrh_layer[i] == 3 && pfrh_energy[i]>seedEThresholdEE_vec[pfrh_depth[i] - 1] && pfrh_pt2[i]>seedPt2ThresholdEE) )
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
//      if(pfrh_layer[j] == 1 && pfrh_depth[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEB_1 * expf(-0.5 * d2); }
//      if(pfrh_layer[j] == 1 && pfrh_depth[j] == 2) { fraction = pfrh_energy[i] / recHitEnergyNormEB_2 * expf(-0.5 * d2); }
//      if(pfrh_layer[j] == 1 && pfrh_depth[j] == 3) { fraction = pfrh_energy[i] / recHitEnergyNormEB_3 * expf(-0.5 * d2); }
//      if(pfrh_layer[j] == 1 && pfrh_depth[j] == 4) { fraction = pfrh_energy[i] / recHitEnergyNormEB_4 * expf(-0.5 * d2); }
//      if(pfrh_layer[j] == 3 && pfrh_depth[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEE_1 * expf(-0.5 * d2); }
//      if(pfrh_layer[j] == 3 && pfrh_depth[j] > 1 ) { fraction = pfrh_energy[i] / recHitEnergyNormEE_2_7 * expf(-0.5 * d2); }
	  
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
//              if(pfrh_layer[j] == 1 && pfrh_depth[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEB_1 * expf(-0.5 * d2); }
//              if(pfrh_layer[j] == 1 && pfrh_depth[j] == 2) { fraction = pfrh_energy[i] / recHitEnergyNormEB_2 * expf(-0.5 * d2); }
//              if(pfrh_layer[j] == 1 && pfrh_depth[j] == 3) { fraction = pfrh_energy[i] / recHitEnergyNormEB_3 * expf(-0.5 * d2); }
//              if(pfrh_layer[j] == 1 && pfrh_depth[j] == 4) { fraction = pfrh_energy[i] / recHitEnergyNormEB_4 * expf(-0.5 * d2); }
//              if(pfrh_layer[j] == 3 && pfrh_depth[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEE_1 * expf(-0.5 * d2); }
//              if(pfrh_layer[j] == 3 && pfrh_depth[j] > 1 ) { fraction = pfrh_energy[i] / recHitEnergyNormEE_2_7 * expf(-0.5 * d2); }
              
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
//	if(pfrh_layer[j] == 1 && pfrh_depth[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEB_1 * expf(-0.5 * d2); }
//	if(pfrh_layer[j] == 1 && pfrh_depth[j] == 2) { fraction = pfrh_energy[i] / recHitEnergyNormEB_2 * expf(-0.5 * d2); }
//	if(pfrh_layer[j] == 1 && pfrh_depth[j] == 3) { fraction = pfrh_energy[i] / recHitEnergyNormEB_3 * expf(-0.5 * d2); }
//	if(pfrh_layer[j] == 1 && pfrh_depth[j] == 4) { fraction = pfrh_energy[i] / recHitEnergyNormEB_4 * expf(-0.5 * d2); }
//	if(pfrh_layer[j] == 3 && pfrh_depth[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEE_1 * expf(-0.5 * d2); }
//	if(pfrh_layer[j] == 3 && pfrh_depth[j] > 1 ) { fraction = pfrh_energy[i] / recHitEnergyNormEE_2_7 * expf(-0.5 * d2); }
	  
	
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
//            if(pfrh_layer[j] == 1 && pfrh_depth[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEB_1 * expf(-0.5 * d2); }
//            if(pfrh_layer[j] == 1 && pfrh_depth[j] == 2) { fraction = pfrh_energy[i] / recHitEnergyNormEB_2 * expf(-0.5 * d2); }
//            if(pfrh_layer[j] == 1 && pfrh_depth[j] == 3) { fraction = pfrh_energy[i] / recHitEnergyNormEB_3 * expf(-0.5 * d2); }
//            if(pfrh_layer[j] == 1 && pfrh_depth[j] == 4) { fraction = pfrh_energy[i] / recHitEnergyNormEB_4 * expf(-0.5 * d2); }
//            if(pfrh_layer[j] == 3 && pfrh_depth[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEE_1 * expf(-0.5 * d2); }
//            if(pfrh_layer[j] == 3 && pfrh_depth[j] > 1 ) { fraction = pfrh_energy[i] / recHitEnergyNormEE_2_7 * expf(-0.5 * d2); }
              
            
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
				float* timer
                )
  {
#ifdef GPU_DEBUG_HCAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif
    //seeding
    if(size>0) seedingKernel_HCAL<<<(size+512-1)/512, 512>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,pfrh_depth,  neigh4_Ind);
    //cudaDeviceSynchronize();
    
    //topoclustering 
     
      dim3 gridT( (size+64-1)/64, 1 );
      dim3 blockT( 64, 8);
      //dim3 gridT( (size+64-1)/64, 8 );
      //dim3 blockT( 64, 16); // 16 threads in a half-warp
#ifdef GPU_DEBUG_HCAL
      cudaEventRecord(start);
#endif
      for(int h=0;h<nTopoLoops; h++){
    
      if(size>0) topoKernel_HCALV2<<<gridT, blockT>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, pfrh_depth, neigh8_Ind);	     
      }
      //cudaDeviceSynchronize();
   
#ifdef GPU_DEBUG_HCAL
      float milliseconds = 0;
      if (timer != nullptr)
      {
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);   
          cudaEventElapsedTime(&milliseconds, start, stop);
          *timer = milliseconds;
      }
#endif

      dim3 grid( (size+32-1)/32, (size+32-1)/32 );
      dim3 block( 32, 32);

      if(size>0) hcalFastCluster_step1<<<grid, block>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);
     //cudaDeviceSynchronize();

      if(size>0) hcalFastCluster_step2<<<grid, block>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);


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
#ifdef GPU_DEBUG_HCAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif
    //seeding
    if(size>0) seedingKernel_HCAL_serialize<<<1,1>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,pfrh_depth,  neigh4_Ind);
    //cudaDeviceSynchronize();
    
    //topoclustering 
      
      //dim3 gridT( (size+64-1)/64, 1 );
      //dim3 blockT( 64, 8);
#ifdef GPU_DEBUG_HCAL
      cudaEventRecord(start);
#endif
      for(int h=0;h<nTopoLoops; h++){
    
      if(size>0) topoKernel_HCAL_serialize<<<1,1>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, pfrh_depth, neigh8_Ind);	     
      }
      //cudaDeviceSynchronize();
#ifdef GPU_DEBUG_HCAL
      float milliseconds = 0;
      if (timer != nullptr)
      {
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&milliseconds, start, stop);
          *timer = milliseconds;
      }
#endif    

      //dim3 grid( (size+32-1)/32, (size+32-1)/32 );
      //dim3 block( 32, 32);

      if(size>0) hcalFastCluster_step1_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);
     //cudaDeviceSynchronize();

      if(size>0) hcalFastCluster_step2_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);


  }

void PFRechitToPFCluster_HCAL_serialize_seedingParallel(size_t size, 
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
				int* rhCount
				)
  { 
    //seeding
    if(size>0) seedingKernel_HCAL<<<(size+512-1)/512, 512>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,pfrh_depth,  neigh4_Ind);
    //cudaDeviceSynchronize();
    
    //topoclustering 
      
      //dim3 gridT( (size+64-1)/64, 1 );
      //dim3 blockT( 64, 8);
      for(int h=0;h<nTopoLoops; h++){
      if(size>0) topoKernel_HCAL_serialize<<<1,1>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, pfrh_depth, neigh8_Ind);
      }
     cudaDeviceSynchronize();
    
    

      //dim3 grid( (size+32-1)/32, (size+32-1)/32 );
      //dim3 block( 32, 32);

      if(size>0) hcalFastCluster_step1_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);
     //cudaDeviceSynchronize();

      if(size>0) hcalFastCluster_step2_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);


  }

void PFRechitToPFCluster_HCAL_serialize_topoParallel(size_t size, 
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
				int* rhCount
				)
  { 
    //seeding
    if(size>0) seedingKernel_HCAL_serialize<<<1,1>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,pfrh_depth,  neigh4_Ind);
    //cudaDeviceSynchronize();
    
    //topoclustering 
      
      dim3 gridT( (size+64-1)/64, 1 );
      dim3 blockT( 64, 8);
      for(int h=0;h<nTopoLoops; h++){
   
      if(size>0) topoKernel_HCALV2<<<gridT, blockT>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, pfrh_depth, neigh8_Ind);
      }
     cudaDeviceSynchronize();
    
    

      //dim3 grid( (size+32-1)/32, (size+32-1)/32 );
      //dim3 block( 32, 32);

      if(size>0) hcalFastCluster_step1_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);
     //cudaDeviceSynchronize();

      if(size>0) hcalFastCluster_step2_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);


  }

void PFRechitToPFCluster_HCAL_serialize_step1Parallel(size_t size, 
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
				int* rhCount
				)
  { 
    //seeding
    if(size>0) seedingKernel_HCAL_serialize<<<1,1>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,pfrh_depth,  neigh4_Ind);
    //cudaDeviceSynchronize();
    
    //topoclustering 
      
      //dim3 gridT( (size+64-1)/64, 1 );
      //dim3 blockT( 64, 8);
      for(int h=0;h<nTopoLoops; h++){
      if(size>0) topoKernel_HCAL_serialize<<<1,1>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, pfrh_depth, neigh8_Ind); 
      }
     cudaDeviceSynchronize();
    
    

      dim3 grid( (size+32-1)/32, (size+32-1)/32 );
      dim3 block( 32, 32);

      if(size>0) hcalFastCluster_step1<<<grid, block>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);
     //cudaDeviceSynchronize();

      if(size>0) hcalFastCluster_step2_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);


  }

void PFRechitToPFCluster_HCAL_serialize_step2Parallel(size_t size, 
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
				int* rhCount
				)
  { 
    //seeding
    if(size>0) seedingKernel_HCAL_serialize<<<1,1>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,pfrh_depth,  neigh4_Ind);
    //cudaDeviceSynchronize();
    
    //topoclustering 
      
      //dim3 gridT( (size+64-1)/64, 1 );
      //dim3 blockT( 64, 8);
      for(int h=0;h<nTopoLoops; h++){
      if(size>0) topoKernel_HCAL_serialize<<<1,1>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, pfrh_depth, neigh8_Ind); 
      }
     cudaDeviceSynchronize();
    
    

      dim3 grid( (size+32-1)/32, (size+32-1)/32 );
      dim3 block( 32, 32);

      if(size>0) hcalFastCluster_step1_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);
     //cudaDeviceSynchronize();
      
      if(size>0) hcalFastCluster_step2<<<grid, block>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);

  }

} // namespace cudavectors
