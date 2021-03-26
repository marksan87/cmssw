#include <cmath>
#include <iostream>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterCudaECAL.h"
#include <Eigen/Dense>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

//#define GPU_DEBUG_ECAL

constexpr int sizeof_float = sizeof(float);
constexpr int sizeof_int = sizeof(int);

namespace PFClusterCudaECAL {

  __constant__ float showerSigma;
  __constant__ float recHitEnergyNormEB;
  __constant__ float recHitEnergyNormEE;
  __constant__ float minFracToKeep;

  __constant__ float seedEThresholdEB;
  __constant__ float seedEThresholdEE;
  __constant__ float seedPt2ThresholdEB;
  __constant__ float seedPt2ThresholdEE;

  __constant__ float topoEThresholdEB;
  __constant__ float topoEThresholdEE;

  __constant__ int nNeigh;
  __constant__ int maxSize;

  int nTopoLoops = 18; // Number of iterations for topo kernel 
  
  
  void initializeCudaConstants(float h_showerSigma,
                               float h_recHitEnergyNormEB,
                               float h_recHitEnergyNormEE,
                               float h_minFracToKeep,
                               float h_seedEThresholdEB,
                               float h_seedEThresholdEE,
                               float h_seedPt2ThresholdEB,
                               float h_seedPt2ThresholdEE,
                               float h_topoEThresholdEB,
                               float h_topoEThresholdEE,
                               int   h_nNeigh,
                               int   h_maxSize
                           )
  {
     cudaCheck(cudaMemcpyToSymbolAsync(showerSigma, &h_showerSigma, sizeof_float)); 
#ifdef GPU_DEBUG_ECAL
     // Read back the value
     float val = 0.;
     cudaMemcpyFromSymbol(&val, showerSigma, sizeof_float);
     std::cout<<"showerSigma read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormEB, &h_recHitEnergyNormEB, sizeof_float)); 
#ifdef GPU_DEBUG_ECAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, recHitEnergyNormEB, sizeof_float);
     std::cout<<"recHitEnergyNormEB read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormEE, &h_recHitEnergyNormEE, sizeof_float)); 
#ifdef GPU_DEBUG_ECAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, recHitEnergyNormEE, sizeof_float);
     std::cout<<"recHitEnergyNormEE read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(minFracToKeep, &h_minFracToKeep, sizeof_float)); 
#ifdef GPU_DEBUG_ECAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, minFracToKeep, sizeof_float);
     std::cout<<"minFracToKeep read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(seedEThresholdEB, &h_seedEThresholdEB, sizeof_float)); 
#ifdef GPU_DEBUG_ECAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, seedEThresholdEB, sizeof_float);
     std::cout<<"seedEThresholdEB read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(seedEThresholdEE, &h_seedEThresholdEE, sizeof_float)); 
#ifdef GPU_DEBUG_ECAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, seedEThresholdEE, sizeof_float);
     std::cout<<"seedEThresholdEE read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(seedPt2ThresholdEB, &h_seedPt2ThresholdEB, sizeof_float)); 
#ifdef GPU_DEBUG_ECAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, seedPt2ThresholdEB, sizeof_float);
     std::cout<<"seedPt2ThresholdEB read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(seedPt2ThresholdEE, &h_seedPt2ThresholdEE, sizeof_float)); 
#ifdef GPU_DEBUG_ECAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, seedPt2ThresholdEE, sizeof_float);
     std::cout<<"seedPt2ThresholdEE read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(topoEThresholdEB, &h_topoEThresholdEB, sizeof_float)); 
#ifdef GPU_DEBUG_ECAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, topoEThresholdEB, sizeof_float);
     std::cout<<"topoEThresholdEB read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(topoEThresholdEE, &h_topoEThresholdEE, sizeof_float)); 
#ifdef GPU_DEBUG_ECAL
     // Read back the value
     val = 0.;
     cudaMemcpyFromSymbol(&val, topoEThresholdEE, sizeof_float);
     std::cout<<"topoEThresholdEE read from symbol: "<<val<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(nNeigh, &h_nNeigh, sizeof_int)); 
#ifdef GPU_DEBUG_ECAL
     // Read back the value
     int ival = 0;
     cudaMemcpyFromSymbol(&ival, nNeigh, sizeof_int);
     std::cout<<"nNeigh read from symbol: "<<ival<<std::endl;
#endif

     cudaCheck(cudaMemcpyToSymbolAsync(maxSize, &h_maxSize, sizeof_int)); 
#ifdef GPU_DEBUG_ECAL
     // Read back the value
     ival = 0;
     cudaMemcpyFromSymbol(&ival, maxSize, sizeof_int);
     std::cout<<"maxSize read from symbol: "<<ival<<std::endl;
#endif  
  }
  
  /*
  void initializeConstants(PFClusterCudaECAL::CudaECALConstants* constants)
  {
     cudaMemcpyToSymbolAsync(showerSigma, &(constants->showerSigma), sizeof(float));
     // Read back the value
     float val = 0.;
     cudaMemcpyFromSymbolAsync(&val, showerSigma, sizeof(float));
     std::cout<<"Value read from symbol: "<<val<<std::endl;
  }
  */

 __global__ void seedingKernel_ECAL(
     				size_t size, 
				    const float* __restrict__ pfrh_energy,
				    const float* __restrict__ pfrh_pt2,
				    int*   pfrh_isSeed,
				    int*   pfrh_topoId,
				    const int* __restrict__ pfrh_layer,
				    const int* __restrict__ neigh8_Ind
				    ) {

   int i = threadIdx.x+blockIdx.x*blockDim.x;

   if(i<size) {        
     if( ( pfrh_layer[i] == -1 && pfrh_energy[i]>seedEThresholdEB && pfrh_pt2[i]>seedPt2ThresholdEB) || ( (pfrh_layer[i] == -2) && pfrh_energy[i]>seedEThresholdEE && pfrh_pt2[i]>seedPt2ThresholdEE) )
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
   
 __global__ void seedingKernel_ECAL_serialize(
     				size_t size, 
				    const float* __restrict__ pfrh_energy,
				    const float* __restrict__ pfrh_pt2,
				    int*   pfrh_isSeed,
				    int*   pfrh_topoId,
				    const int* __restrict__ pfrh_layer,
				    const int* __restrict__ neigh8_Ind
				    ) {

   //int i = threadIdx.x+blockIdx.x*blockDim.x;

   //if(i<size) {        
   for (int i=0; i<size; i++) {
     if( ( pfrh_layer[i] == -1 && pfrh_energy[i]>seedEThresholdEB && pfrh_pt2[i]>seedPt2ThresholdEB) || ( (pfrh_layer[i] == -2) && pfrh_energy[i]>seedEThresholdEE && pfrh_pt2[i]>seedPt2ThresholdEE) )
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
				  const float* __restrict__ pfrh_energy,
				  int* pfrh_topoId,
				  const int* __restrict__ pfrh_layer,
				  const int* __restrict__ neigh8_Ind
				  ) {
    for(int j=0;j<16;j++){
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
      }//loop over neighbours end
      
    }//loop over neumann neighbourhood clustering end
  }
  
  __global__ void topoKernel_ECALV2( 
                  size_t size,
                  const float* __restrict__ pfrh_energy,
                  int* pfrh_topoId,
                  const int* __restrict__ pfrh_layer,
                  const int* __restrict__ neigh8_Ind
                  ) {

      int l = threadIdx.x+blockIdx.x*blockDim.x;
      int k = (threadIdx.y+blockIdx.y*blockDim.y) % nNeigh;

      //if(l<size && k<8) {   
      if(l<size) {   
      while( neigh8_Ind[nNeigh*l+k] > -1 && pfrh_topoId[l] != pfrh_topoId[neigh8_Ind[nNeigh*l+k]] && 
         (
          ((pfrh_layer[l] == -2 && pfrh_energy[l]>topoEThresholdEE) || 
           (pfrh_layer[l] == -1 && pfrh_energy[l]>topoEThresholdEB) )
          &&
          ((pfrh_layer[neigh8_Ind[nNeigh*l+k]] == -2 && pfrh_energy[neigh8_Ind[nNeigh*l+k]]>topoEThresholdEE) || 
           (pfrh_layer[neigh8_Ind[nNeigh*l+k]] == -1 && pfrh_energy[neigh8_Ind[nNeigh*l+k]]>topoEThresholdEB) )
          )
         )
        {
          if(pfrh_topoId[l] > pfrh_topoId[neigh8_Ind[nNeigh*l+k]]){
        atomicMax(&pfrh_topoId[neigh8_Ind[nNeigh*l+k]],pfrh_topoId[l]);
          }
          if(pfrh_topoId[l] < pfrh_topoId[neigh8_Ind[nNeigh*l+k]]){
        atomicMax(&pfrh_topoId[l], pfrh_topoId[neigh8_Ind[nNeigh*l+k]]);
          }       
        }
      }                        
  }

  __global__ void topoKernel_ECAL_serialize( 
				  size_t size,
				  const float* __restrict__ pfrh_energy,
				  int* pfrh_topoId,
				  const int* __restrict__ pfrh_layer,
				  const int* __restrict__ neigh8_Ind
                  ) {

      for (int l = 0; l < size; l++) {
        for (int k = 0; k < 8; k++) {

          while( neigh8_Ind[nNeigh*l+k] > -1 && pfrh_topoId[l] != pfrh_topoId[neigh8_Ind[nNeigh*l+k]] && 
             (
              ((pfrh_layer[l] == -2 && pfrh_energy[l]>topoEThresholdEE) || 
               (pfrh_layer[l] == -1 && pfrh_energy[l]>topoEThresholdEB) )
              &&
              ((pfrh_layer[neigh8_Ind[nNeigh*l+k]] == -2 && pfrh_energy[neigh8_Ind[nNeigh*l+k]]>topoEThresholdEE) || 
               (pfrh_layer[neigh8_Ind[nNeigh*l+k]] == -1 && pfrh_energy[neigh8_Ind[nNeigh*l+k]]>topoEThresholdEB) )
              )
             )
            {
              if(pfrh_topoId[l] > pfrh_topoId[neigh8_Ind[nNeigh*l+k]]){
            atomicMax(&pfrh_topoId[neigh8_Ind[nNeigh*l+k]],pfrh_topoId[l]);
              }
              if(pfrh_topoId[l] < pfrh_topoId[neigh8_Ind[nNeigh*l+k]]){
            atomicMax(&pfrh_topoId[l], pfrh_topoId[neigh8_Ind[nNeigh*l+k]]);
              }       
            }
         }
      } 
  }

  
__global__ void fastCluster_step1( size_t size,
					     const float* __restrict__ pfrh_x,
					     const float* __restrict__ pfrh_y,
					     const float* __restrict__ pfrh_z,
					     const float* __restrict__ pfrh_energy,
					     int* pfrh_topoId,
					     int* pfrh_isSeed,
					     const int* __restrict__ pfrh_layer,
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

      if(pfrh_layer[j] == -1) { fraction = pfrh_energy[i] / recHitEnergyNormEB * expf(-0.5 * d2); }
      if(pfrh_layer[j] == -2) { fraction = pfrh_energy[i] / recHitEnergyNormEE * expf(-0.5 * d2); }
      if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");
      
      if( pfrh_isSeed[j]!=1 && d2<100)
	{ 
	  atomicAdd(&fracSum[j],fraction);	  
	}
      }
    }
  }


__global__ void fastCluster_step2( size_t size,
					     const float* __restrict__ pfrh_x,
					     const float* __restrict__ pfrh_y,
					     const float* __restrict__ pfrh_z,
					     const float* __restrict__ pfrh_energy,
					     int* pfrh_topoId,
					     int* pfrh_isSeed,
					     const int* __restrict__ pfrh_layer,
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

	if(pfrh_layer[j] == -1) { fraction = pfrh_energy[i] / recHitEnergyNormEB * expf(-0.5 * d2); }
	if(pfrh_layer[j] == -2) { fraction = pfrh_energy[i] / recHitEnergyNormEE * expf(-0.5 * d2); }
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

__global__ void fastCluster_step1_serialize( size_t size,
					     const float* __restrict__ pfrh_x,
					     const float* __restrict__ pfrh_y,
					     const float* __restrict__ pfrh_z,
					     const float* __restrict__ pfrh_energy,
					     int* pfrh_topoId,
					     int* pfrh_isSeed,
					     const int* __restrict__ pfrh_layer,
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

              float dist2 = 
                   (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
                  +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
                  +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);	

              float d2 = dist2 / (showerSigma*showerSigma);
              float fraction = -1.;

              if(pfrh_layer[j] == -1) { fraction = pfrh_energy[i] / recHitEnergyNormEB * expf(-0.5 * d2); }
              if(pfrh_layer[j] == -2) { fraction = pfrh_energy[i] / recHitEnergyNormEE * expf(-0.5 * d2); }
              if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");
              
              if( pfrh_isSeed[j]!=1 && d2<100)
            { 
              atomicAdd(&fracSum[j],fraction);	  
            }
              }
            }
        }
    }
  }


__global__ void fastCluster_step2_serialize( size_t size,
					     const float* __restrict__ pfrh_x,
					     const float* __restrict__ pfrh_y,
					     const float* __restrict__ pfrh_z,
					     const float* __restrict__ pfrh_energy,
					     int* pfrh_topoId,
					     int* pfrh_isSeed,
					     const int* __restrict__ pfrh_layer,
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

            if(pfrh_layer[j] == -1) { fraction = pfrh_energy[i] / recHitEnergyNormEB * expf(-0.5 * d2); }
            if(pfrh_layer[j] == -2) { fraction = pfrh_energy[i] / recHitEnergyNormEE * expf(-0.5 * d2); }
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
    }
}
  

   

  void PFRechitToPFCluster_ECALV2(size_t size, 
				const float* __restrict__ pfrh_x, 
				const float* __restrict__ pfrh_y, 
				const float* __restrict__ pfrh_z, 
				const float* __restrict__ pfrh_energy, 
				const float* __restrict__ pfrh_pt2,      				
				int* pfrh_isSeed,
				int* pfrh_topoId, 
				const int* __restrict__ pfrh_layer, 
				const int* __restrict__ neigh8_Ind, 				
				float* pfrhfrac, 
				int* pfrhfracind,
				int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
				int* rhCount,
                float* timer
				)
  { 
#ifdef GPU_DEBUG_ECAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif
    //seeding
    if(size>0) seedingKernel_ECAL<<<(size+512-1)/512, 512>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,  neigh8_Ind);
    //cudaDeviceSynchronize();
      
    // for(int a=0;a<16;a++){
    //if(size>0) topoKernel_ECAL<<<(size+512-1)/512, 512>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, neigh8_Ind);
    //}	    
    //cudaDeviceSynchronize();

    dim3 gridT( (size+64-1)/64, 1 );
    dim3 blockT( 64, 8);
    //dim3 gridT( (size+64-1)/64, 8 );
    //dim3 blockT( 64, 16);
#ifdef GPU_DEBUG_ECAL
    cudaEventRecord(start);
#endif
    //for(int h=0; h<18; h++){  
    for(int h=0; h<nTopoLoops; h++){  
      if(size>0) topoKernel_ECALV2<<<gridT, blockT>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, neigh8_Ind);        
    }

#ifdef GPU_DEBUG_ECAL
    float milliseconds = 0;
    if (timer != NULL)
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        *timer = milliseconds;
    }
#endif
    dim3 grid( (size+32-1)/32, (size+32-1)/32 );
    dim3 block( 32, 32);

    //if(size>0) std::cout<<std::endl<<"NEW EVENT !!"<<std::endl<<std::endl;

     if(size>0) fastCluster_step1<<<grid, block>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);
    //cudaDeviceSynchronize();

    if(size>0) fastCluster_step2<<<grid, block>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);  
    //cudaDeviceSynchronize();
    
   
  }

  void PFRechitToPFCluster_ECAL_serialize(size_t size, 
				const float* __restrict__ pfrh_x, 
				const float* __restrict__ pfrh_y, 
				const float* __restrict__ pfrh_z, 
				const float* __restrict__ pfrh_energy, 
				const float* __restrict__ pfrh_pt2,      				
				int* pfrh_isSeed,
				int* pfrh_topoId, 
				const int* __restrict__ pfrh_layer, 
				const int* __restrict__ neigh8_Ind, 				
				float* pfrhfrac, 
				int* pfrhfracind,
				int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
				int* rhCount,
                float* timer
				)
  { 
#ifdef GPU_DEBUG_ECAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif    
    //seeding
    if(size>0) seedingKernel_ECAL_serialize<<<1,1>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,  neigh8_Ind);
    //cudaDeviceSynchronize();
     
#ifdef GPU_DEBUG_ECAL
    cudaEventRecord(start);
#endif
    for(int h=0; h < nTopoLoops; h++){
        if(size>0) topoKernel_ECAL_serialize<<<1,1>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, neigh8_Ind);
    }	    
    //cudaDeviceSynchronize();

#ifdef GPU_DEBUG_ECAL
    float milliseconds = 0;
    if (timer != NULL)
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        *timer = milliseconds;
    }
#endif
    //dim3 grid( (size+32-1)/32, (size+32-1)/32 );
    //dim3 block( 32, 32);

    //if(size>0) std::cout<<std::endl<<"NEW EVENT !!"<<std::endl<<std::endl;

     if(size>0) fastCluster_step1_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);
     //cudaDeviceSynchronize();

    if(size>0) fastCluster_step2_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);  
    //cudaDeviceSynchronize();
    
   
  }
  
  void PFRechitToPFCluster_ECAL_serialize_seedingParallel(size_t size, 
				const float* __restrict__ pfrh_x, 
				const float* __restrict__ pfrh_y, 
				const float* __restrict__ pfrh_z, 
				const float* __restrict__ pfrh_energy, 
				const float* __restrict__ pfrh_pt2,      				
				int* pfrh_isSeed,
				int* pfrh_topoId, 
				const int* __restrict__ pfrh_layer, 
				const int* __restrict__ neigh8_Ind, 				
				float* pfrhfrac, 
				int* pfrhfracind,
				int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
				int* rhCount
				)
  { 
    //seeding
    if(size>0) seedingKernel_ECAL<<<(size+512-1)/512, 512>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,  neigh8_Ind);
    //cudaDeviceSynchronize();
      
    // for(int a=0;a<16;a++){
    if(size>0) topoKernel_ECAL_serialize<<<1,1>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, neigh8_Ind);
    //}	    
    cudaDeviceSynchronize();

    //dim3 grid( (size+32-1)/32, (size+32-1)/32 );
    //dim3 block( 32, 32);

    //if(size>0) std::cout<<std::endl<<"NEW EVENT !!"<<std::endl<<std::endl;

     if(size>0) fastCluster_step1_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);
     cudaDeviceSynchronize();

    if(size>0) fastCluster_step2_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);  
    cudaDeviceSynchronize();
    
   
  }
  
  void PFRechitToPFCluster_ECAL_serialize_topoParallel(size_t size, 
				const float* __restrict__ pfrh_x, 
				const float* __restrict__ pfrh_y, 
				const float* __restrict__ pfrh_z, 
				const float* __restrict__ pfrh_energy, 
				const float* __restrict__ pfrh_pt2,      				
				int* pfrh_isSeed,
				int* pfrh_topoId, 
				const int* __restrict__ pfrh_layer, 
				const int* __restrict__ neigh8_Ind, 				
				float* pfrhfrac, 
				int* pfrhfracind,
				int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
				int* rhCount
				)
  { 
    //seeding
    if(size>0) seedingKernel_ECAL_serialize<<<1,1>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,  neigh8_Ind);
    //cudaDeviceSynchronize();
      
    // for(int a=0;a<16;a++){
    if(size>0) topoKernel_ECAL<<<(size+512-1)/512, 512>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, neigh8_Ind);
    //}	    
    cudaDeviceSynchronize();

    //dim3 grid( (size+32-1)/32, (size+32-1)/32 );
    //dim3 block( 32, 32);

    //if(size>0) std::cout<<std::endl<<"NEW EVENT !!"<<std::endl<<std::endl;

     if(size>0) fastCluster_step1_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);
     cudaDeviceSynchronize();

    if(size>0) fastCluster_step2_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);  
    cudaDeviceSynchronize();
    
   
  }
  
  void PFRechitToPFCluster_ECAL_serialize_step1Parallel(size_t size, 
				const float* __restrict__ pfrh_x, 
				const float* __restrict__ pfrh_y, 
				const float* __restrict__ pfrh_z, 
				const float* __restrict__ pfrh_energy, 
				const float* __restrict__ pfrh_pt2,      				
				int* pfrh_isSeed,
				int* pfrh_topoId, 
				const int* __restrict__ pfrh_layer, 
				const int* __restrict__ neigh8_Ind, 				
				float* pfrhfrac, 
				int* pfrhfracind,
				int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
				int* rhCount
				)
  { 
    //seeding
    if(size>0) seedingKernel_ECAL_serialize<<<1,1>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,  neigh8_Ind);
    //cudaDeviceSynchronize();
      
    // for(int a=0;a<16;a++){
    if(size>0) topoKernel_ECAL_serialize<<<1,1>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, neigh8_Ind); 
    //}	    
    cudaDeviceSynchronize();

    dim3 grid( (size+32-1)/32, (size+32-1)/32 );
    dim3 block( 32, 32);

    //if(size>0) std::cout<<std::endl<<"NEW EVENT !!"<<std::endl<<std::endl;
     if(size>0) fastCluster_step1<<<grid, block>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);
     cudaDeviceSynchronize();

     if(size>0) fastCluster_step2_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);  
     cudaDeviceSynchronize();
    
   
  }
  
  void PFRechitToPFCluster_ECAL_serialize_step2Parallel(size_t size, 
				const float* __restrict__ pfrh_x, 
				const float* __restrict__ pfrh_y, 
				const float* __restrict__ pfrh_z, 
				const float* __restrict__ pfrh_energy, 
				const float* __restrict__ pfrh_pt2,      				
				int* pfrh_isSeed,
				int* pfrh_topoId, 
				const int* __restrict__ pfrh_layer, 
				const int* __restrict__ neigh8_Ind, 				
				float* pfrhfrac, 
				int* pfrhfracind,
				int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
				int* rhCount
				)
  { 
    //seeding
    if(size>0) seedingKernel_ECAL_serialize<<<1,1>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,  neigh8_Ind);
    //cudaDeviceSynchronize();
      
    // for(int a=0;a<16;a++){
    if(size>0) topoKernel_ECAL_serialize<<<1,1>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, neigh8_Ind); 
    //}	    
    cudaDeviceSynchronize();

    dim3 grid( (size+32-1)/32, (size+32-1)/32 );
    dim3 block( 32, 32);

    //if(size>0) std::cout<<std::endl<<"NEW EVENT !!"<<std::endl<<std::endl;

     if(size>0) fastCluster_step1_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);
     cudaDeviceSynchronize();
     
     if(size>0) fastCluster_step2<<<grid, block>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);
     cudaDeviceSynchronize();
    
   
  }
}  // namespace cudavectors
