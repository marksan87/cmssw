
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
#define DEBUG_GPU_HCAL

constexpr int sizeof_float = sizeof(float);
constexpr int sizeof_int = sizeof(int);


namespace PFClusterCudaHCAL {

  __constant__ float showerSigma2;
  __constant__ float recHitEnergyNormEB_vec[4];
  __constant__ float recHitEnergyNormEE_vec[7];
  __constant__ float minFracToKeep;
  __constant__ float minFracTot;
  __constant__ float stoppingTolerance;

  __constant__ float seedEThresholdEB_vec[4];
  __constant__ float seedEThresholdEE_vec[7];
  __constant__ float seedPt2ThresholdEB;
  __constant__ float seedPt2ThresholdEE;

  __constant__ float topoEThresholdEB_vec[4];
  __constant__ float topoEThresholdEE_vec[7];
 
  __constant__ int maxIterations;
  __constant__ bool excludeOtherSeeds;

  __constant__ int nNT = 8;  // Number of neighbors considered for topo clustering
  __constant__ int nNeigh;
  __constant__ int maxSize;
 
  //int nTopoLoops = 100;
  int nTopoLoops = 35;


  bool initializeCudaConstants(float h_showerSigma2,
                               const float (&h_recHitEnergyNormEB_vec)[4],
                               const float (&h_recHitEnergyNormEE_vec)[7],
                               float h_minFracToKeep,
                               float h_minFracTot,
                               int   h_maxIterations,
                               float h_stoppingTolerance,
                               bool  h_excludeOtherSeeds,
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
     status &= cudaCheck(cudaMemcpyToSymbolAsync(showerSigma2, &h_showerSigma2, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     std::cout<<"--- HCAL Cuda constant values ---"<<std::endl;
     float val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, showerSigma2, sizeof_float));
     std::cout<<"showerSigma2 read from symbol: "<<val<<std::endl;
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
    
     status &= cudaCheck(cudaMemcpyToSymbolAsync(minFracTot, &h_minFracTot, sizeof_float));
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, minFracTot, sizeof_float));
     std::cout<<"minFracTot read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(stoppingTolerance, &h_stoppingTolerance, sizeof_float));
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, stoppingTolerance, sizeof_float));
     std::cout<<"stoppingTolerance read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(excludeOtherSeeds, &h_excludeOtherSeeds, sizeof(bool)));
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     bool bval = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&bval, excludeOtherSeeds, sizeof(bool)));
     std::cout<<"excludeOtherSeeds read from symbol: "<<bval<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(maxIterations, &h_maxIterations, sizeof_int));
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     int ival = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&ival, maxIterations, sizeof_int));
     std::cout<<"maxIterations read from symbol: "<<ival<<std::endl;
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
 
 __global__ void seedingTopoThreshKernel_HCAL(
     				size_t size, 
				    const double* __restrict__ pfrh_energy,
				    const double* __restrict__ pfrh_pt2,
				    int*   pfrh_isSeed,
				    int*   pfrh_topoId,
                    bool*  pfrh_passTopoThresh,
				    const int* __restrict__ pfrh_layer,
				    const int* __restrict__ pfrh_depth,
				    const int* __restrict__ neigh4_Ind
				    ) {

   int i = threadIdx.x+blockIdx.x*blockDim.x;

   if(i<size) {
     // Seeding threshold test
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
    
     // Topo clustering threshold test
     if ( (pfrh_layer[i] == 3 && pfrh_energy[i]>topoEThresholdEE_vec[pfrh_depth[i]-1]) ||
             (pfrh_layer[i] == 1 && pfrh_energy[i]>topoEThresholdEB_vec[pfrh_depth[i]-1])) {
            pfrh_passTopoThresh[i] = true;
        }
     else { pfrh_passTopoThresh[i] = false; }
   }
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
 
 __global__ void topoKernel_HCAL_passTopoThresh(
    size_t size,
    const double* __restrict__ pfrh_energy,
    int* pfrh_topoId,
    const bool* __restrict__ pfrh_passTopoThresh,
    const int* __restrict__ neigh8_Ind
) {

    int l = threadIdx.x + blockIdx.x*blockDim.x;
    int k = (threadIdx.y + blockIdx.y*blockDim.y) % nNT;

    //if(l<size && k<nNT) {
    if (l < size) {

        while (pfrh_passTopoThresh[nNT*l + k] && neigh8_Ind[nNT*l + k] > -1 && pfrh_topoId[l] != pfrh_topoId[neigh8_Ind[nNT*l + k]])
        {
            if (pfrh_topoId[l] > pfrh_topoId[neigh8_Ind[nNT*l + k]]) {
                atomicMax(&pfrh_topoId[neigh8_Ind[nNT*l + k]], pfrh_topoId[l]);
            }
            if (pfrh_topoId[l] < pfrh_topoId[neigh8_Ind[nNT*l + k]]) {
                atomicMax(&pfrh_topoId[l], pfrh_topoId[neigh8_Ind[nNT*l + k]]);
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

      float d2 = dist2 / showerSigma2;
      float fraction = -1.;

      if(pfrh_layer[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
      else if (pfrh_layer[j] == 3) { fraction = pfrh_energy[i] / recHitEnergyNormEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
	  
      if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

      if( pfrh_isSeed[j]!=1 )
	{
	  atomicAdd(&fracSum[j],fraction);
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

        float d2 = dist2 / showerSigma2; 
        float fraction = -1.;

        if(pfrh_layer[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
        else if (pfrh_layer[j] == 3) { fraction = pfrh_energy[i] / recHitEnergyNormEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
        
        if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");
        
        if (fracSum[j] > minFracTot) {
            float fracpct = fraction / fracSum[j];
            if(fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep))
              {
                  int k = atomicAdd(&rhCount[i],1);
                  pcrhfrac[i*maxSize+k] = fracpct;
                  pcrhfracind[i*maxSize+k] = j;
              }
        }
        /*
        if(d2 < 100. )
          {
            if ((fraction/fracSum[j])>minFracToKeep){
              int k = atomicAdd(&rhCount[i],1);
              pcrhfrac[i*maxSize+k] = fraction/fracSum[j];
              pcrhfracind[i*maxSize+k] = j;
              //printf("(i,j)=(%d,%d), rhCount=%d, fraction=%f, fracsum=%f \n",i,j,rhCount[i], fraction, fracSum[j]);
            }
          }
        */
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

              float d2 = dist2 / showerSigma2; 
              float fraction = -1.;

              if(pfrh_layer[j] == 1) { fraction = pfrh_energy[i] / recHitEnergyNormEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
              else if (pfrh_layer[j] == 3) { fraction = pfrh_energy[i] / recHitEnergyNormEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
              
              if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

              if( pfrh_isSeed[j]!=1 )
            {
              atomicAdd(&fracSum[j],fraction);
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

            float d2 = dist2 / showerSigma2; 
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

// Contraction in a single block
__global__ void topoClusterContraction(size_t size, int* pfrh_parent) {
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


__device__ bool isLeftEdge(const int idx,
    const int nEdges,
    const int* __restrict__ pfrh_edgeId,
    const int* __restrict__ pfrh_edgeMask) {

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
                return true;
            }
            else if (pfrh_edgeMask[temp] > 0) {
                // Found adjacent edge
                return false;
            }
            temp--;
        }
    }
    else if (idx == 0) {
        return true;
    }

    // Invalid index
    return false;
}

__device__ bool isRightEdge(const int idx,
    const int nEdges,
    const int* __restrict__ pfrh_edgeId,
    const int* __restrict__ pfrh_edgeMask) {

    // Update right
    if (idx < (nEdges - 1)) {
        int temp = idx + 1;
        int maxVal = min(idx - 9, nEdges - 1);  //  Only test up to 9 neighbors
        //int maxVal = nEdges - 1;
        int tempId = 0;
        int edgeId = pfrh_edgeId[idx];
        while (temp >= maxVal) {
            tempId = pfrh_edgeId[temp];
            if (edgeId != tempId) {
                // Different topo Id here!
                return true;
            }
            else if (pfrh_edgeMask[temp] > 0) {
                // Found adjacent edge
                return false;
            }
            temp++;
        }
    }
    else if (idx == (nEdges - 1)) {
        return true;
    }

    // Overflow
    return false;
}


__global__ void topoClusterLinking(int nRH,
    int nEdges,
    int* pfrh_parent,
    int* pfrh_edgeId,
    int* pfrh_edgeList,
    int* pfrh_edgeMask,
    bool* pfrh_passTopoThresh,
    int* nIter) {

    __shared__ bool notDone;
    __shared__ int iter, gridStride;

    int start = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        *nIter = 0;
        iter = 0;
        gridStride = blockDim.x * gridDim.x; // For single block kernel this is the number of threads
    }
    __syncthreads();

    // Check if pairs in edgeId,edgeList contain a rh not passing topo threshold
    // If found, set the mask to 0
    for (int idx = start; idx < nEdges; idx += gridStride) {
        if (pfrh_passTopoThresh[pfrh_edgeId[idx]] && pfrh_passTopoThresh[pfrh_edgeList[idx]])
            pfrh_edgeMask[idx] = 1;
        else
            pfrh_edgeMask[idx] = 0;
    }

    do {
        if (threadIdx.x == 0) {
            notDone = false;
        }
        __syncthreads();

        // Odd linking
        for (int idx = start; idx < nEdges; idx += gridStride) {
            int i = pfrh_edgeId[idx];   // Get edge topo id
            //if (pfrh_edgeMask[idx] > 0 && pfrh_passTopoThresh[i] && isLeftEdge(idx, nEdges, pfrh_edgeId, pfrh_edgeMask)) {
            if (pfrh_edgeMask[idx] > 0 && isLeftEdge(idx, nEdges, pfrh_edgeId, pfrh_edgeMask)) {
                pfrh_parent[i] = (int)min(i, pfrh_edgeList[idx]);
            }
        }

        __syncthreads();

        // edgeParent
        for (int idx = start; idx < nEdges; idx += gridStride) {
            if (pfrh_edgeMask[idx] > 0) {
                int id = pfrh_edgeId[idx];   // Get edge topo id
                int neighbor = pfrh_edgeList[idx]; // Get neighbor topo id
                pfrh_edgeId[idx] = pfrh_parent[id];
                pfrh_edgeList[idx] = pfrh_parent[neighbor];

                // edgeMask set to true if elements of edgeId and edgeList are different
                if (pfrh_edgeId[idx] != pfrh_edgeList[idx]) {
                    pfrh_edgeMask[idx] = 1;
                    notDone = true;
                }
                else {
                    pfrh_edgeMask[idx] = 0;
                }
            }
        }
        if (threadIdx.x == 0)
            iter++;

        __syncthreads();

        if (!notDone) break;

        if (threadIdx.x == 0) {
            notDone = false;
        }

        __syncthreads();

        // Even linking
        for (int idx = start; idx < nEdges; idx += gridStride) {
            int i = pfrh_edgeId[idx];   // Get edge topo id
            //if (pfrh_edgeMask[idx] > 0 && pfrh_passTopoThresh[i] && isRightEdge(idx, nEdges, pfrh_edgeId, pfrh_edgeMask)) {
            if (pfrh_edgeMask[idx] > 0 && isRightEdge(idx, nEdges, pfrh_edgeId, pfrh_edgeMask)) {
                pfrh_parent[i] = (int)max(i, pfrh_edgeList[idx]);
            }
        }

        __syncthreads();

        // edgeParent
        for (int idx = start; idx < nEdges; idx += gridStride) {
            if (pfrh_edgeMask[idx] > 0) {
                int id = pfrh_edgeId[idx];   // Get edge topo id
                int neighbor = pfrh_edgeList[idx]; // Get neighbor topo id
                pfrh_edgeId[idx] = pfrh_parent[id];
                pfrh_edgeList[idx] = pfrh_parent[neighbor];

                // edgeMask set to true if elements of edgeId and edgeList are different
                if (pfrh_edgeId[idx] != pfrh_edgeList[idx]) {
                    pfrh_edgeMask[idx] = 1;
                    notDone = true;
                }
                else {
                    pfrh_edgeMask[idx] = 0;
                }
            }
        }
        if (threadIdx.x == 0)
            iter++;

        __syncthreads();

    } while (notDone);
    *nIter = iter;
}


void PFRechitToPFCluster_HCAL_CCLClustering(int nRH,
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
                int* pfrh_edgeMask,
                bool* pfrh_passTopoThresh,
                int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
                int* rhCount,
                float (&timer)[8],
                int* nIter) {
    if (nRH < 1) return;

#ifdef DEBUG_GPU_HCAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif
    cudaProfilerStart();
    // Combined seeding & topo clustering thresholds
    seedingTopoThreshKernel_HCAL<<<(nRH+63/64), 128>>>(nRH, pfrh_energy, pfrh_pt2, pfrh_isSeed, pfrh_topoId, pfrh_passTopoThresh, pfrh_layer, pfrh_depth, neigh4_Ind);

#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&timer[0], start, stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif
    
    //topoclustering 
    topoClusterLinking<<<1, 1024 >>>(nRH, nEdges, pfrh_topoId, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_passTopoThresh, nIter);
    topoClusterContraction<<<1, 512>>>(nRH, pfrh_topoId);

#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&timer[1], start, stop);
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
#endif
    cudaProfilerStop();
}


void PFRechitToPFCluster_HCALV2(size_t size, 
				const float* __restrict__ pfrh_x, 
				const float* __restrict__ pfrh_y, 
				const float* __restrict__ pfrh_z, 
				const double* __restrict__ pfrh_energy, 
				const double* __restrict__ pfrh_pt2,    				
				int* pfrh_isSeed,
				bool* pfrh_passTopoThresh,
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
    if (size <= 0) return;
#ifdef DEBUG_GPU_HCAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    cudaProfilerStart();
    //seeding
    seedingKernel_HCAL<<<(size+512-1)/512, 512>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,pfrh_depth,  neigh4_Ind);

    // Passing topo clustering threshold
    passingTopoThreshold<<<(size+255)/256, 256>>>( size, pfrh_layer, pfrh_depth, pfrh_energy, pfrh_passTopoThresh);
#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);   
      cudaEventElapsedTime(&timer[0], start, stop);
      cudaEventRecord(start);
#endif
    
    //topoclustering 
     
      //cudaProfilerStart();
      //dim3 gridT( (size+64-1)/64, 1 );
      //dim3 blockT( 64, 8);
      dim3 gridT( (size+64-1)/64, 8 );
      dim3 blockT( 64, 16); // 16 threads in a half-warp
      for(int h=0;h<nTopoLoops; h++){
        topoKernel_HCAL_passTopoThresh <<<gridT, blockT >>> (size, pfrh_energy, pfrh_topoId, pfrh_passTopoThresh, neigh8_Ind);
        //topoKernel_HCALV2<<<gridT, blockT>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, pfrh_depth, neigh8_Ind);	     
      }
   
      //cudaProfilerStop();
#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);   
      cudaEventElapsedTime(&timer[1], start, stop);
      cudaEventRecord(start);
#endif

      dim3 grid( (size+32-1)/32, (size+32-1)/32 );
      dim3 block( 32, 32);

      hcalFastCluster_step1<<<grid, block>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);

#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);   
      cudaEventElapsedTime(&timer[2], start, stop);
      cudaEventRecord(start);
#endif

      hcalFastCluster_step2<<<grid, block>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);

#ifdef DEBUG_GPU_HCAL
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);   
      cudaEventElapsedTime(&timer[3], start, stop);
#endif

      cudaProfilerStop();
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
