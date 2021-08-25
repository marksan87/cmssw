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
#include <cuda_profiler_api.h>

using PFClustering::common::PFLayer;

// Uncomment for debugging
//#define DEBUG_GPU_ECAL


constexpr int sizeof_float = sizeof(float);
constexpr int sizeof_int = sizeof(int);
constexpr const float PI_F = 3.141592654f;
constexpr const float preshowerStartEta = 1.653;
constexpr const float preshowerEndEta = 2.6;

namespace PFClusterCudaECAL {

  __constant__ float showerSigma2;
  __constant__ float recHitEnergyNormEB;
  __constant__ float recHitEnergyNormEE;
  __constant__ float minFracToKeep;
  __constant__ float minFracTot;
  __constant__ float stoppingTolerance;

  __constant__ float seedEThresholdEB;
  __constant__ float seedEThresholdEE;
  __constant__ float seedPt2ThresholdEB;
  __constant__ float seedPt2ThresholdEE;

  __constant__ float topoEThresholdEB;
  __constant__ float topoEThresholdEE;

  __constant__ int maxIterations;
  __constant__ bool excludeOtherSeeds;

  __constant__ int nNeigh;
  __constant__ int maxSize;

  // Generic position calc constants
  __constant__ float minAllowedNormalization;
  __constant__ float logWeightDenominator; 
  __constant__ float minFractionInCalc;

  // Convergence position calc constants
  __constant__ float conv_minAllowedNormalization;
  __constant__ float conv_T0_ES;
  __constant__ float conv_T0_EE;
  __constant__ float conv_T0_EB;
  __constant__ float conv_X0;
  __constant__ float conv_minFractionInCalc;
  __constant__ float conv_W0;

  int nTopoLoops = 18; // Number of iterations for topo kernel 
  
  
  bool initializeCudaConstants(const float h_showerSigma2,
                               const float h_recHitEnergyNormEB,
                               const float h_recHitEnergyNormEE,
                               const float h_minFracToKeep,
                               const float h_minFracTot,
                               const int   h_maxIterations,
                               const float h_stoppingTolerance,
                               const bool  h_excludeOtherSeeds,
                               const float h_seedEThresholdEB,
                               const float h_seedEThresholdEE,
                               const float h_seedPt2ThresholdEB,
                               const float h_seedPt2ThresholdEE,
                               const float h_topoEThresholdEB,
                               const float h_topoEThresholdEE,
                               const int   h_nNeigh,
                               const int   h_maxSize,
                               const PFClustering::common::PosCalcConfig h_posCalcConfig,
                               const PFClustering::common::ECALPosDepthCalcConfig h_convergencePosCalcConfig
                           )
  {
     bool status = true;
     status &= cudaCheck(cudaMemcpyToSymbolAsync(showerSigma2, &h_showerSigma2, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     std::cout<<"--- ECAL Cuda constant values ---"<<std::endl;
     float val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, showerSigma2, sizeof_float));
     std::cout<<"showerSigma2 read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormEB, &h_recHitEnergyNormEB, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, recHitEnergyNormEB, sizeof_float));
     std::cout<<"recHitEnergyNormEB read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormEE, &h_recHitEnergyNormEE, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, recHitEnergyNormEE, sizeof_float));
     std::cout<<"recHitEnergyNormEE read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(minFracToKeep, &h_minFracToKeep, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
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

     status &= cudaCheck(cudaMemcpyToSymbolAsync(seedEThresholdEB, &h_seedEThresholdEB, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, seedEThresholdEB, sizeof_float));
     std::cout<<"seedEThresholdEB read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(seedEThresholdEE, &h_seedEThresholdEE, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, seedEThresholdEE, sizeof_float));
     std::cout<<"seedEThresholdEE read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(seedPt2ThresholdEB, &h_seedPt2ThresholdEB, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, seedPt2ThresholdEB, sizeof_float));
     std::cout<<"seedPt2ThresholdEB read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(seedPt2ThresholdEE, &h_seedPt2ThresholdEE, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, seedPt2ThresholdEE, sizeof_float));
     std::cout<<"seedPt2ThresholdEE read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(topoEThresholdEB, &h_topoEThresholdEB, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, topoEThresholdEB, sizeof_float));
     std::cout<<"topoEThresholdEB read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(topoEThresholdEE, &h_topoEThresholdEE, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, topoEThresholdEE, sizeof_float));
     std::cout<<"topoEThresholdEE read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(nNeigh, &h_nNeigh, sizeof_int)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     ival = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&ival, nNeigh, sizeof_int));
     std::cout<<"nNeigh read from symbol: "<<ival<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(maxSize, &h_maxSize, sizeof_int)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     ival = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&ival, maxSize, sizeof_int));
     std::cout<<"maxSize read from symbol: "<<ival<<std::endl;
#endif  

     // Generic position calc constants
     status &= cudaCheck(cudaMemcpyToSymbolAsync(minAllowedNormalization, &h_posCalcConfig.minAllowedNormalization, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, minAllowedNormalization, sizeof_float));
     std::cout<<"minAllowedNormalization read from symbol: "<<val<<std::endl;
#endif  
     
     status &= cudaCheck(cudaMemcpyToSymbolAsync(logWeightDenominator, &h_posCalcConfig.logWeightDenominator, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, logWeightDenominator, sizeof_float));
     std::cout<<"logWeightDenominator read from symbol: "<<val<<std::endl;
#endif  
     
     status &= cudaCheck(cudaMemcpyToSymbolAsync(minFractionInCalc, &h_posCalcConfig.minFractionInCalc, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, minFractionInCalc, sizeof_float));
     std::cout<<"minFractionInCalc read from symbol: "<<val<<std::endl;
#endif  
     

     // Convergence position calc constants
     status &= cudaCheck(cudaMemcpyToSymbolAsync(conv_minAllowedNormalization, &h_convergencePosCalcConfig.minAllowedNormalization, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, conv_minAllowedNormalization, sizeof_float));
     std::cout<<"conv_minAllowedNormalization read from symbol: "<<val<<std::endl;
#endif  

     status &= cudaCheck(cudaMemcpyToSymbolAsync(conv_T0_ES, &h_convergencePosCalcConfig.T0_ES, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, conv_T0_ES, sizeof_float));
     std::cout<<"conv_T0_ES read from symbol: "<<val<<std::endl;
#endif  

     status &= cudaCheck(cudaMemcpyToSymbolAsync(conv_T0_EE, &h_convergencePosCalcConfig.T0_EE, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, conv_T0_EE, sizeof_float));
     std::cout<<"conv_T0_EE read from symbol: "<<val<<std::endl;
#endif  

     status &= cudaCheck(cudaMemcpyToSymbolAsync(conv_T0_EB, &h_convergencePosCalcConfig.T0_EB, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, conv_T0_EB, sizeof_float));
     std::cout<<"conv_T0_EB read from symbol: "<<val<<std::endl;
#endif  
     
     status &= cudaCheck(cudaMemcpyToSymbolAsync(conv_X0, &h_convergencePosCalcConfig.X0, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, conv_X0, sizeof_float));
     std::cout<<"conv_X0 read from symbol: "<<val<<std::endl;
#endif  

     status &= cudaCheck(cudaMemcpyToSymbolAsync(conv_minFractionInCalc, &h_convergencePosCalcConfig.minFractionInCalc, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, conv_minFractionInCalc, sizeof_float));
     std::cout<<"conv_minFractionInCalc read from symbol: "<<val<<std::endl;
#endif  

     status &= cudaCheck(cudaMemcpyToSymbolAsync(conv_W0, &h_convergencePosCalcConfig.W0, sizeof_float)); 
#ifdef DEBUG_GPU_ECAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, conv_W0, sizeof_float));
     std::cout<<"conv_W0 read from symbol: "<<val<<std::endl;
#endif  


     return status;
  }

__device__ __forceinline__ float mag(float xpos, float ypos, float zpos) {
    return sqrtf(xpos*xpos + ypos*ypos + zpos*zpos);
}

__device__ __forceinline__ float mag(float4 pos) {
    return sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);
}

__device__ __forceinline__ float phiFromCartesian(float4 pos) {
    return atan2f(pos.y, pos.x);
}

__device__ __forceinline__ float etaFromCartesian(float posX, float posY, float posZ) {
    float m = mag(posX, posY, posZ); 
    float cosTheta = m > 0.0 ? posZ / m : 1.0;
    return (0.5 * logf( (1.0+cosTheta) / (1.0-cosTheta) ));
}

__device__ __forceinline__ float etaFromCartesian(float4 pos) {
    float m = mag(pos); 
    float cosTheta = m > 0.0 ? pos.z / m : 1.0;
    return (0.5 * logf( (1.0+cosTheta) / (1.0-cosTheta) ));
}


__device__ __forceinline__ float dR2(float4 pos1, float4 pos2) {
//    float mag1 = sqrtf(pos1.x*pos1.x + pos1.y*pos1.y + pos1.z*pos1.z);
//    float cosTheta1 = mag1 > 0.0 ? pos1.z / mag1 : 1.0;
//    float eta1 = 0.5 * logf( (1.0+cosTheta1) / (1.0-cosTheta1) );
//    float phi1 = atan2f(pos1.y, pos1.x);
//    float mag2 = sqrtf(pos2.x*pos2.x + pos2.y*pos2.y + pos2.z*pos2.z);
//    float cosTheta2 = mag2 > 0.0 ? pos2.z / mag2 : 1.0;
//    float eta2 = 0.5 * logf( (1.0+cosTheta2) / (1.0-cosTheta2) );
//    float phi2 = atan2f(pos2.y, pos2.x);
    
    float eta1 = etaFromCartesian(pos1);
    float phi1 = phiFromCartesian(pos1);

    float eta2 = etaFromCartesian(pos2);
    float phi2 = phiFromCartesian(pos2);

    float deta = eta2-eta1;
    float dphi = fabsf(fabsf(phi2 - phi1) - PI_F) - PI_F;
    return (deta*deta + dphi*dphi);
}


__global__ void seedingTopoThreshKernel_ECAL(
                    size_t size,
                    int*  rhCount,
                    float* fracSum,
                    const float* __restrict__ pfrh_energy,
                    const float* __restrict__ pfrh_pt2,
                    int*   pfrh_isSeed,
                    int*   pfrh_topoId,
                    bool*  pfrh_passTopoThresh,
                    const int* __restrict__ pfrh_layer,
                    const int* __restrict__ neigh8_Ind
                    ) {

   int i = threadIdx.x+blockIdx.x*blockDim.x;

   if(i<size) {
     // Initialize rhCount
     rhCount[i] = 1;
     fracSum[i] = 0.;

     // Seeding threshold test
     if ( (pfrh_layer[i] == PFLayer::ECAL_BARREL && pfrh_energy[i]>seedEThresholdEB && pfrh_pt2[i]>seedPt2ThresholdEB) || 
          (pfrh_layer[i] == PFLayer::ECAL_ENDCAP && pfrh_energy[i]>seedEThresholdEE && pfrh_pt2[i]>seedPt2ThresholdEE) ) {
     pfrh_isSeed[i]=1;
     for(int k=0; k<nNeigh; k++){
       if(neigh8_Ind[nNeigh*i+k]<0) continue;
       if(pfrh_energy[i]<pfrh_energy[neigh8_Ind[nNeigh*i+k]]){
         pfrh_isSeed[i]=0;
         //pfrh_topoId[i]=-1;
         break;
       }
     }
       }
     else{
       //pfrh_topoId[i]=-1;
       pfrh_isSeed[i]=0;
     }
   

     // Topo clustering threshold test
     if ( (pfrh_layer[i] == PFLayer::ECAL_ENDCAP && pfrh_energy[i]>topoEThresholdEE) ||
          (pfrh_layer[i] == PFLayer::ECAL_BARREL && pfrh_energy[i]>topoEThresholdEB) ) {
            pfrh_passTopoThresh[i] = true;
        }
     //else { pfrh_passTopoThresh[i] = false; }
     else { pfrh_passTopoThresh[i] = false; pfrh_topoId[i] = -1; }
   }
}

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
     if( ( pfrh_layer[i] == PFLayer::ECAL_BARREL && pfrh_energy[i]>seedEThresholdEB && pfrh_pt2[i]>seedPt2ThresholdEB) || ( (pfrh_layer[i] == PFLayer::ECAL_ENDCAP) && pfrh_energy[i]>seedEThresholdEE && pfrh_pt2[i]>seedPt2ThresholdEE) )
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
     if( ( pfrh_layer[i] == PFLayer::ECAL_BARREL && pfrh_energy[i]>seedEThresholdEB && pfrh_pt2[i]>seedPt2ThresholdEB) || ( (pfrh_layer[i] == PFLayer::ECAL_ENDCAP) && pfrh_energy[i]>seedEThresholdEE && pfrh_pt2[i]>seedPt2ThresholdEE) )
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
	      ( (pfrh_layer[l] == PFLayer::ECAL_ENDCAP && pfrh_energy[l]>topoEThresholdEE) || 
		(pfrh_layer[l] == PFLayer::ECAL_BARREL && pfrh_energy[l]>topoEThresholdEB) ) )
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
          ((pfrh_layer[l] == PFLayer::ECAL_ENDCAP && pfrh_energy[l]>topoEThresholdEE) || 
           (pfrh_layer[l] == PFLayer::ECAL_BARREL && pfrh_energy[l]>topoEThresholdEB) )
          &&
          ((pfrh_layer[neigh8_Ind[nNeigh*l+k]] == PFLayer::ECAL_ENDCAP && pfrh_energy[neigh8_Ind[nNeigh*l+k]]>topoEThresholdEE) || 
           (pfrh_layer[neigh8_Ind[nNeigh*l+k]] == PFLayer::ECAL_BARREL && pfrh_energy[neigh8_Ind[nNeigh*l+k]]>topoEThresholdEB) )
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
              ((pfrh_layer[l] == PFLayer::ECAL_ENDCAP && pfrh_energy[l]>topoEThresholdEE) || 
               (pfrh_layer[l] == PFLayer::ECAL_BARREL && pfrh_energy[l]>topoEThresholdEB) )
              &&
              ((pfrh_layer[neigh8_Ind[nNeigh*l+k]] == PFLayer::ECAL_ENDCAP && pfrh_energy[neigh8_Ind[nNeigh*l+k]]>topoEThresholdEE) || 
               (pfrh_layer[neigh8_Ind[nNeigh*l+k]] == PFLayer::ECAL_BARREL && pfrh_energy[neigh8_Ind[nNeigh*l+k]]>topoEThresholdEB) )
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


__global__ void fastCluster_serialize(size_t nRH,
                            const float* __restrict__ pfrh_x,
                            const float* __restrict__ pfrh_y,
                            const float* __restrict__ pfrh_z,
                            const float* __restrict__ geomAxis_x,
                            const float* __restrict__ geomAxis_y,
                            const float* __restrict__ geomAxis_z,
                            const float* __restrict__ pfrh_energy,
                            int* pfrh_topoId,
                            int* pfrh_isSeed,
                            const int* __restrict__ pfrh_layer,
                            const int* __restrict__ neigh8_Ind,
                            float* pcrhfrac,
                            int* pcrhfracind,
                            float* fracSum,
                            int* rhCount
                            ) {

  for (int topoId = 0; topoId < (int)nRH; topoId++) {
    int iter = 0;
    int nSeeds = 0;
    int nRHTopo = 0;
    if (topoId >-1 && topoId < nRH) {
        //int seeds[25];
        int seeds[75];
        int rechits[150];
        // First determine how many rechits are in this topo cluster
        for (int r = 0; r < nRH; r++) {
            if (pfrh_topoId[r] == topoId) {
                // Found a rechit belonging to this topo cluster
                rechits[nRHTopo] = r;
                nRHTopo++;
                if (pfrh_isSeed[r]) {
                    // This rechit is a seed
                    seeds[nSeeds] = r;
                    nSeeds++;
                }
            }
        }
        if (nSeeds == 0) continue;  // No seeds found for this topoId. Skip it

        //bool debug = true;
        bool debug = false;
        
        if (debug) {
            printf("\n===========================================================================================\n");
            printf("Processing topo cluster %d with seeds (", topoId);
                for (int s = 0; s < nSeeds; s++) {
                    if (s != 0) printf(", ");
                    printf("%d", seeds[s]);
                }
                printf(") and rechits (");
                for (int r = 0; r < nRHTopo; r++) {
                    if (r != 0) printf(", ");
                    printf("%d", rechits[r]);
                }
                printf(")");
        }

        //float tolScaling2 = std::pow(std::max(1.0, nSeeds - 1.0), 4.0);     // Tolerance scaling squared
        float tolScaling = powf(fmaxf(1.0, nSeeds - 1.0), 2.0);     // Tolerance scaling

        float4 prevClusterPos[75], linearClusterPos[75], clusterPos[75], convergenceClusterPos[75];  //  W component is position norm
        float clusterEnergy[75];
        //float prevClusterEnergy[75];
        
        auto computeDepthPosFromArrays = [&] (float4& pos4, float4& linear_pos, float _frac, int rhInd, float _clusterT0, float maxDepthFront, float totalClusterEnergy, float logETotInv, bool isDebug) {
            float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
            float4 rechitAxis = make_float4(geomAxis_x[rhInd], geomAxis_y[rhInd], geomAxis_z[rhInd], 1.0);
            float weight = 0.0; 

            const auto rh_energy = pfrh_energy[rhInd] * _frac;
            if (rh_energy > 0.0)
                weight = fmaxf(0.0, conv_W0 + logf(rh_energy) + logETotInv);
            const float depth = maxDepthFront - mag(rechitPos);
            
            if (isDebug)
                printf("\t\t\trechit %d: w=%f\tfrac=%f\tdepth=%f\trh_energy=%f\trhPos=(%f, %f, %f)\tdeltaPos=(%f, %f, %f)\n", rhInd, weight, _frac, depth, rh_energy, rechitPos.x, rechitPos.y, rechitPos.z, weight * (rechitPos.x + depth * geomAxis_x[rhInd]), weight * (rechitPos.y + depth * geomAxis_y[rhInd]), weight * (rechitPos.z + depth * geomAxis_z[rhInd]));

            pos4.x += weight * (rechitPos.x + depth * geomAxis_x[rhInd]);
            pos4.y += weight * (rechitPos.y + depth * geomAxis_y[rhInd]);
            pos4.z += weight * (rechitPos.z + depth * geomAxis_z[rhInd]);
            pos4.w += weight;     //  position_norm

            if (pos4.w > 0) return; // No need to compute position with linear weights if position_norm > 0
            // Compute linear weights
            float lin_weight = 0.0;
            if (rh_energy > 0.0)
                lin_weight = rh_energy / totalClusterEnergy;

            linear_pos.x += lin_weight * rechitPos.x;
            linear_pos.y += lin_weight * rechitPos.y;
            linear_pos.z += lin_weight * rechitPos.z;
            linear_pos.w += lin_weight;
        };

        auto computeFromArrays = [&] (float4& pos4, float _frac, int rhInd, bool isDebug) {
            float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
            float threshold = 1. / logWeightDenominator;
//            float threshold = 0.0;
//            if(pfrh_layer[rhInd] == PFLayer::ECAL_BARREL) {
//                threshold = 1. / recHitEnergyNormEB; // This number needs to be inverted
//            }
//            else if (pfrh_layer[rhInd] == PFLayer::ECAL_ENDCAP) { threshold = 1. / recHitEnergyNormEE; }

            const auto rh_energy = pfrh_energy[rhInd] * _frac;
            const auto norm =
                (_frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * threshold)));
            if (isDebug)
                printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n", rhInd, norm, _frac, rh_energy, rechitPos.x, rechitPos.y, rechitPos.z);

            pos4.x += rechitPos.x * norm;
            pos4.y += rechitPos.y * norm;
            pos4.z += rechitPos.z * norm;
            pos4.w += norm;     //  position_norm
        };
        /*
        auto compute = [&] (float4& pos4, float& clusterEn, int seedInd, int rhInd) {
            float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
            float threshold = 0.0;
            // Search for this rechit index in fraction arrays
            float _frac = -1.0;
            for (int _n = seedInd*maxSize; _n < ((seedInd+1)*maxSize); _n++) {
                if (pcrhfracind[_n] == rhInd) {
                    // Found it
                    _frac = pcrhfrac[_n];
                    break;
                }
            }
            if (_frac < 0)
                printf("Warning: negative rechitfrac found for seed %d rechit %d!\n", seedInd, rhInd);
            if(pfrh_layer[rhInd] == 1) {
                threshold = 1. / recHitEnergyNormEB; // This number needs to be inverted
            }
            else if (pfrh_layer[rhInd] == 3) { threshold = 1. / recHitEnergyNormEE; }

            const auto rh_energy = pfrh_energy[rhInd] * _frac;
            const auto norm =
                (_frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * threshold)));
            pos4.x += rechitPos.x * norm;
            pos4.y += rechitPos.y * norm;
            pos4.z += rechitPos.z * norm;
            pos4.w += norm;     //  position_norm

            clusterEn += rh_energy;
        };
        */
        float diff = -1.0;
        while (iter < maxIterations) {
            if (debug) {
                printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
            }
            // Reset fracSum and rhCount
            for (int r = 0; r < (int)nRH; r++) {
                fracSum[r] = 0.0;
                rhCount[r] = 1;
            }

            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = seeds[s];
                if (iter == 0) {
                    // Set initial cluster position to seed rechit position
                    clusterPos[s] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
                    prevClusterPos[s] = make_float4(0.0, 0.0, 0.0, 0.0);
                    //prevClusterPos[s] = clusterPos[s];

                    float clusterT0 = 0.0;
                    if (pfrh_layer[i] == PFLayer::ECAL_BARREL) clusterT0 = conv_T0_EB;
                    else if (pfrh_layer[i] == PFLayer::ECAL_ENDCAP) clusterT0 = conv_T0_EE;


                    float seedEta = etaFromCartesian(pfrh_x[i], pfrh_y[i], pfrh_z[i]);
                    float absSeedEta = fabsf(seedEta);
                    if (absSeedEta > preshowerStartEta && absSeedEta < preshowerEndEta) {
                        if (seedEta > 0) {
                            clusterT0 = conv_T0_ES;
                        }
                        else if (seedEta < 0) {
                            clusterT0 = conv_T0_ES;
                        }
                        else {
                            printf("SOMETHING WRONG WITH THIS CLUSTER ETA!\n");
                        }
                    }

                    float logETot_inv = -logf(pfrh_energy[i]);
                    float maxDepth = conv_X0 * (clusterT0 - logETot_inv);
                    float maxToFront = mag(pfrh_x[i], pfrh_y[i], pfrh_z[i]);
                    float maxDepthPlusFront = maxDepth + maxToFront;
                    computeDepthPosFromArrays(prevClusterPos[s], linearClusterPos[s], 1.0, i, clusterT0, maxDepthPlusFront, pfrh_energy[i], logETot_inv, debug);
                    prevClusterPos[s].x /= prevClusterPos[s].w;
                    prevClusterPos[s].y /= prevClusterPos[s].w;
                    prevClusterPos[s].z /= prevClusterPos[s].w;
                    // Set initial cluster energy to seed energy
                    clusterEnergy[s] = pfrh_energy[i];
                    //prevClusterEnergy[s] = 0.0;
                }
                else {
                    prevClusterPos[s] = convergenceClusterPos[s];
                    //prevClusterEnergy[s] = clusterEnergy[s];

                    // Reset cluster indices and fractions
                    for (int _n = i*maxSize; _n < (i+1)*maxSize; _n++) {
                        pcrhfrac[_n] = -1.0;
                        pcrhfracind[_n] = -1.0;
                    }
                }
                if (debug) {
                    printf("\tCluster %d (seed %d) using prev convergence pos = (%f, %f, %f) and cluster position = (%f, %f, %f)\n", s, i, prevClusterPos[s].x, prevClusterPos[s].y, prevClusterPos[s].z, clusterPos[s].x, clusterPos[s].y, clusterPos[s].z);
                }

                for (int r = 0; r < nRHTopo; r++) {
                    int j = rechits[r];
                    if( pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i]==1 ){
                        float dist2 =
                           (clusterPos[s].x - pfrh_x[j])*(clusterPos[s].x - pfrh_x[j])
                          +(clusterPos[s].y - pfrh_y[j])*(clusterPos[s].y - pfrh_y[j])
                          +(clusterPos[s].z - pfrh_z[j])*(clusterPos[s].z - pfrh_z[j]);
                        float d2 = dist2 / showerSigma2;
                        float fraction = -1.;

                        if(pfrh_layer[j] == PFLayer::ECAL_BARREL) { fraction = clusterEnergy[s] / recHitEnergyNormEB * expf(-0.5 * d2); }
                        else if (pfrh_layer[j] == PFLayer::ECAL_ENDCAP) { fraction = clusterEnergy[s] / recHitEnergyNormEE * expf(-0.5 * d2); }
                        if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                        if( pfrh_isSeed[j]!=1) {
                            atomicAdd(&fracSum[j],fraction);
                        }
                    }
                }
            }
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = seeds[s];
                for (int r = 0; r < nRHTopo; r++) {
                    int j = rechits[r];
                    if( pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i]==1 ){
                        if(i==j)
                        {
                            pcrhfrac[i*maxSize]    = 1.;
                            pcrhfracind[i*maxSize] = j;
                        }
                        if( pfrh_isSeed[j]!=1 ){
                            float dist2 =
                               (clusterPos[s].x - pfrh_x[j])*(clusterPos[s].x - pfrh_x[j])
                              +(clusterPos[s].y - pfrh_y[j])*(clusterPos[s].y - pfrh_y[j])
                              +(clusterPos[s].z - pfrh_z[j])*(clusterPos[s].z - pfrh_z[j]);
                            float d2 = dist2 / showerSigma2;
                            float fraction = -1.;

                            if(pfrh_layer[j] == PFLayer::ECAL_BARREL) { fraction = clusterEnergy[s] / recHitEnergyNormEB * expf(-0.5 * d2); }
                            else if (pfrh_layer[j] == PFLayer::ECAL_ENDCAP) { fraction = clusterEnergy[s] / recHitEnergyNormEE * expf(-0.5 * d2); }
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
                        }
                    }
                }
            }

            if (debug)
                printf("Computing cluster position for topoId %d\n", topoId);
            // Recalculate position
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = seeds[s];

                if (debug) {
                    printf("\tNow on seed %d\t\tneigh8Ind = [", i);
                    for(int k=0; k<nNeigh; k++){
                        if (k != 0) printf(", ");
                        printf("%d", neigh8_Ind[nNeigh*i+k]);
                    }
                    printf("]\n");

                }
                // Zero out cluster position and energy
                clusterPos[s] = make_float4(0.0, 0.0, 0.0, 0.0);
                linearClusterPos[s] = make_float4(0.0, 0.0, 0.0, 0.0);
                convergenceClusterPos[s] = make_float4(0.0, 0.0, 0.0, 0.0);
                clusterEnergy[s] = 0;
                float clusterT0 = 0.0, maxDepth = 0.0, maxToFront = 0.0, maxDepthPlusFront = 0.0, logETot_inv = 0.0;
                if (pfrh_layer[i] == PFLayer::ECAL_BARREL) clusterT0 = conv_T0_EB;
                else if (pfrh_layer[i] == PFLayer::ECAL_ENDCAP) clusterT0 = conv_T0_EE;

                
                float seedEta = etaFromCartesian(pfrh_x[i], pfrh_y[i], pfrh_z[i]);
                float absSeedEta = fabsf(seedEta);
                if (absSeedEta > preshowerStartEta && absSeedEta < preshowerEndEta) {
                    if (seedEta > 0) {
                        clusterT0 = conv_T0_ES;
                        if (debug) printf("\t\t## This cluster is in esPlus! ##\n");
                    }
                    else if (seedEta < 0) {
                        clusterT0 = conv_T0_ES;
                        if (debug) printf("\t\t## This cluster is in esMinus! ##\n");
                    }
                    else {
                        printf("SOMETHING WRONG WITH THIS CLUSTER ETA!\n");
                    }
                }

                // Calculate cluster energy by summing rechit fractional energies
                for (int r = 0; r < nRHTopo; r++) {
                    int j = rechits[r];
                    float frac = -1.0;
                    int _n = -1;
                    if (j == i) {
                        // This is the seed
                        frac = 1.0;
                        _n = i*maxSize;
                    }
                    else {
                        for (_n = i*maxSize; _n < ((i+1)*maxSize); _n++) {
                            if (pcrhfracind[_n] == j) {
                                // Found it
                                frac = pcrhfrac[_n];
                                break;
                            }
                        }
                    }
                    if (frac > -0.5) {
                        //if (debug)
                            //printf("\t\tRechit %d (position %d) in this PF cluster with frac = %f\n", j, _n, frac);
                        clusterEnergy[s] += frac * pfrh_energy[j];
                        
                        // Do generic cluster position calculation
                        if (nSeeds == 1) {
                            if (debug)
                                printf("\t\tThis topo cluster has a single seed.\n");
                            computeFromArrays(clusterPos[s], frac, j, debug);
                        }
                        else {
                            if (j == i) {
                                // This is the seed
                                computeFromArrays(clusterPos[s], frac, j, debug);
                            }
                            else {
                                // Check if this is one of the neighboring rechits
                                for(int k=0; k<nNeigh; k++){
                                    if(neigh8_Ind[nNeigh*i+k]<0) continue;
                                    if(neigh8_Ind[nNeigh*i+k] == j) {
                                        // Found it
                                        if (debug) printf("\t\tRechit %d is one of the 8 neighbors of seed %d\n", j, i);
                                        computeFromArrays(clusterPos[s], frac, j, debug);
                                    }
                                }
                            }
                        }
                    }
                }
                
                logETot_inv = -logf(clusterEnergy[s]);
                maxDepth = conv_X0 * (clusterT0 - logETot_inv);
                maxToFront = mag(pfrh_x[i], pfrh_y[i], pfrh_z[i]);
                maxDepthPlusFront = maxDepth + maxToFront;

                // ECAL 2D depth cluster position calculation
                for (int r = 0; r < nRHTopo; r++) {
                    int j = rechits[r];
                    float frac = -1.0;
                    int _n = -1;
                    if (j == i) {
                        // This is the seed
                        frac = 1.0;
                        _n = i*maxSize;
                    }
                    else {
                        for (_n = i*maxSize; _n < ((i+1)*maxSize); _n++) {
                            if (pcrhfracind[_n] == j) {
                                // Found it
                                frac = pcrhfrac[_n];
                                break;
                            }
                        }
                    }
                    if (frac > -0.5) {
                        computeDepthPosFromArrays(convergenceClusterPos[s], linearClusterPos[s], frac, j, clusterT0, maxDepthPlusFront, clusterEnergy[s], logETot_inv, debug);
                    }
                    //else if (debug)
                    //    printf("Can't find rechit fraction for cluster %d (seed %d) rechit %d!\n", s, i, j);

                }

                // Generic position calculation
                if (clusterPos[s].w >= minAllowedNormalization) {
                    // Divide by position norm
                    clusterPos[s].x /= clusterPos[s].w;
                    clusterPos[s].y /= clusterPos[s].w;
                    clusterPos[s].z /= clusterPos[s].w;
                    if (debug)
                        printf("\tCluster %d (seed %d) energy = %f\tgeneric pos = (%f, %f, %f)\n", s, i, clusterEnergy[s], clusterPos[s].x, clusterPos[s].y, clusterPos[s].z);
                }
                else {
                    if (debug)
                        printf("\tGeneric pos calc: Cluster %d (seed %d) position norm (%f) less than minimum (%f)\n", s, i, clusterPos[s].w, minAllowedNormalization);
                    clusterPos[s].x = 0.0;
                    clusterPos[s].y = 0.0;
                    clusterPos[s].z = 0.0;
                }

                // ECAL depth corrected position
                if (convergenceClusterPos[s].w >= conv_minAllowedNormalization && convergenceClusterPos[s].w >= 0.0)
                {
                    // Divide by position norm
                    convergenceClusterPos[s].x /= convergenceClusterPos[s].w;
                    convergenceClusterPos[s].y /= convergenceClusterPos[s].w;
                    convergenceClusterPos[s].z /= convergenceClusterPos[s].w;

                    if (debug)
                        printf("\tCluster %d (seed %d) energy = %f\t2D depth cor pos = (%f, %f, %f)\n", s, i, clusterEnergy[s], convergenceClusterPos[s].x, convergenceClusterPos[s].y, convergenceClusterPos[s].z);
                }
                else if (fabsf(convergenceClusterPos[s].w) < 1e-5 && linearClusterPos[s].w >= conv_minAllowedNormalization) {
                    convergenceClusterPos[s].x = linearClusterPos[s].x / linearClusterPos[s].w;
                    convergenceClusterPos[s].y = linearClusterPos[s].y / linearClusterPos[s].w;
                    convergenceClusterPos[s].z = linearClusterPos[s].z / linearClusterPos[s].w;
                    if (debug) printf("\tCluster %d (seed %d) falling back to linear weights!\tenergy = %f\tpos = (%f, %f, %f)\n", s, i, clusterEnergy[s], convergenceClusterPos[s].x, convergenceClusterPos[s].y, convergenceClusterPos[s].z);
                }
                else {
                    if (debug)
                        printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n", s, i, linearClusterPos[s].w, conv_minAllowedNormalization);
                    convergenceClusterPos[s].x = 0.0;
                    convergenceClusterPos[s].y = 0.0;
                    convergenceClusterPos[s].z = 0.0;
                    //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);

                }
            }

            float diff2 = 0.0;
            for (int s = 0; s < nSeeds; s++) {
                //float delta2 = dR2(prevClusterPos[s], clusterPos[s]);
                float delta2 = dR2(prevClusterPos[s], convergenceClusterPos[s]);
                if (debug) printf("\tCluster %d (seed %d) has delta2 = %f\n", s, seeds[s], delta2);
                if (delta2 > diff2) {
                    diff2 = delta2;
                    if (debug) printf("\t\tNew diff2 = %f\n", diff2);
                }
            }
            //float diff = sqrtf(diff2);
            diff = sqrtf(diff2);
            iter++;
            //if (iter >= maxIterations || diff2 <= stoppingTolerance2 * tolScaling2) break;
            if (diff <= stoppingTolerance * tolScaling) {
                if (debug) printf("\tTopoId %d has diff = %f LESS than tolerance (terminating!)\n", topoId, diff);
                break;
            }
            else if (debug) {
                printf("\tTopoId %d has diff = %f greater than tolerance (continuing)\n", topoId, diff);
            }
        }
    }
  }
}



__global__ void fastCluster_original(size_t nRH,
                            const float* __restrict__ pfrh_x,
                            const float* __restrict__ pfrh_y,
                            const float* __restrict__ pfrh_z,
                            const float* __restrict__ geomAxis_x,
                            const float* __restrict__ geomAxis_y,
                            const float* __restrict__ geomAxis_z,
                            const float* __restrict__ pfrh_energy,
                            int* pfrh_topoId,
                            int* pfrh_isSeed,
                            const int* __restrict__ pfrh_layer,
                            float* pcrhfrac,
                            int* pcrhfracind,
                            float* fracSum,
                            int* rhCount
                            ) {

    int topoId = threadIdx.x+blockIdx.x*blockDim.x; // TopoId
    int iter = 0;
    float tol = 0.0;
    int nSeeds = 0;
    int nRHTopo = 0;
    if (topoId >-1 && topoId < nRH) {
        int seeds[25];
        int rechits[60];
        // First determine how many rechits are in this topo cluster
        for (int r = 0; r < nRH; r++) {
            if (pfrh_topoId[r] == topoId) {
                // Found a rechit belonging to this topo cluster
                rechits[nRHTopo] = r;
                nRHTopo++;
                if (pfrh_isSeed[r]) {
                    // This rechit is a seed
                    seeds[nSeeds] = r;
                    nSeeds++;
                }
            }
        }

        
        while (iter < 1) {
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = seeds[s];
                for (int r = 0; r < nRHTopo; r++) {
                    int j = rechits[r];
                    if( pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i]==1 ){
                        float dist2 =
                           (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
                          +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
                          +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);

                        float d2 = dist2 / showerSigma2;
                        float fraction = -1.;

                        if(pfrh_layer[j] == PFLayer::ECAL_BARREL) { fraction = pfrh_energy[i] / recHitEnergyNormEB * expf(-0.5 * d2); }
                        else if(pfrh_layer[j] == PFLayer::ECAL_ENDCAP) { fraction = pfrh_energy[i] / recHitEnergyNormEE * expf(-0.5 * d2); }
                        if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                        if( pfrh_isSeed[j]!=1) {
                            atomicAdd(&fracSum[j],fraction);
                        }            
                    }
                }
            }
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = seeds[s];
                for (int r = 0; r < nRHTopo; r++) {
                    int j = rechits[r];
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

                            if(pfrh_layer[j] == PFLayer::ECAL_BARREL) { fraction = pfrh_energy[i] / recHitEnergyNormEB * expf(-0.5 * d2); }
                            if(pfrh_layer[j] == PFLayer::ECAL_ENDCAP) { fraction = pfrh_energy[i] / recHitEnergyNormEE * expf(-0.5 * d2); }
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
                        }
                    }
                }
            }
            iter++;
            if (abs(tol) < stoppingTolerance) break;
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

      float d2 = dist2 / showerSigma2; 
      float fraction = -1.;

      if(pfrh_layer[j] == PFLayer::ECAL_BARREL) { fraction = pfrh_energy[i] / recHitEnergyNormEB * expf(-0.5 * d2); }
      else if(pfrh_layer[j] == PFLayer::ECAL_ENDCAP) { fraction = pfrh_energy[i] / recHitEnergyNormEE * expf(-0.5 * d2); }
      if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");
      
      if( pfrh_isSeed[j]!=1)
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

        float d2 = dist2 / showerSigma2; 
        float fraction = -1.;

        if(pfrh_layer[j] == PFLayer::ECAL_BARREL) { fraction = pfrh_energy[i] / recHitEnergyNormEB * expf(-0.5 * d2); }
        if(pfrh_layer[j] == PFLayer::ECAL_ENDCAP) { fraction = pfrh_energy[i] / recHitEnergyNormEE * expf(-0.5 * d2); }
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
            }
          }
        */
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

              float d2 = dist2 / showerSigma2; 
              float fraction = -1.;

              if(pfrh_layer[j] == PFLayer::ECAL_BARREL) { fraction = pfrh_energy[i] / recHitEnergyNormEB * expf(-0.5 * d2); }
              if(pfrh_layer[j] == PFLayer::ECAL_ENDCAP) { fraction = pfrh_energy[i] / recHitEnergyNormEE * expf(-0.5 * d2); }
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

            float d2 = dist2 / showerSigma2; 
            float fraction = -1.;

            if(pfrh_layer[j] == PFLayer::ECAL_BARREL) { fraction = pfrh_energy[i] / recHitEnergyNormEB * expf(-0.5 * d2); }
            if(pfrh_layer[j] == PFLayer::ECAL_ENDCAP) { fraction = pfrh_energy[i] / recHitEnergyNormEE * expf(-0.5 * d2); }
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


__device__ __forceinline__ bool isLeftEdge(const int idx,
    const int nEdges,
    const int* __restrict__ pfrh_edgeId,
    const int* __restrict__ pfrh_edgeMask) {

    if (idx > 0) {
        int temp = idx - 1;
        int minVal = max(idx - 9, 0);   //  Only test up to 9 neighbors
        int tempId = 0;
        int edgeId = pfrh_edgeId[idx];
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

__device__ __forceinline__ bool isRightEdge(const int idx,
    const int nEdges,
    const int* __restrict__ pfrh_edgeId,
    const int* __restrict__ pfrh_edgeMask) {

    // Update right
    if (idx < (nEdges - 1)) {
        int temp = idx + 1;
        int maxVal = min(idx + 9, nEdges - 1);  //  Only test up to 9 neighbors
        int tempId = 0;
        int edgeId = pfrh_edgeId[idx];
        while (temp <= maxVal) {
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


    // Begin linking loop
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



void PFRechitToPFCluster_ECAL_CCLClustering(int nRH,
                int nEdges,
                const float* __restrict__ pfrh_x,
                const float* __restrict__ pfrh_y,
                const float* __restrict__ pfrh_z,
                const float* __restrict__ geomAxis_x,
                const float* __restrict__ geomAxis_y,
                const float* __restrict__ geomAxis_z,
                const float* __restrict__ pfrh_energy,
                const float* __restrict__ pfrh_pt2,
                int* pfrh_isSeed,
                int* pfrh_topoId,
                const int* __restrict__ pfrh_layer,
                const int* __restrict__ neigh8_Ind,
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

#ifdef DEBUG_GPU_ECAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif
    cudaProfilerStart();
    // Combined seeding & topo clustering thresholds
    seedingTopoThreshKernel_ECAL<<<(nRH+63)/64, 128>>>(nRH, rhCount, fracSum, pfrh_energy, pfrh_pt2, pfrh_isSeed, pfrh_topoId, pfrh_passTopoThresh, pfrh_layer, neigh8_Ind);

#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[0], start, stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif

    //topoclustering 
    //topoClusterLinking<<<1, 1024 >>>(nRH, nEdges, pfrh_topoId, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_passTopoThresh, nIter);
    topoClusterLinking<<<1, 512>>>(nRH, nEdges, pfrh_topoId, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_passTopoThresh, nIter);
    topoClusterContraction<<<1, 512>>>(nRH, pfrh_topoId);

#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[1], start, stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif

    //fastCluster_serialize<<<1, 1>>>(nRH, pfrh_x,  pfrh_y,  pfrh_z,  geomAxis_x, geomAxis_y, geomAxis_z, pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, neigh8_Ind, pcrhfrac, pcrhfracind, fracSum, rhCount);

    //fastCluster_original<<<1, 1>>>(nRH, pfrh_x,  pfrh_y,  pfrh_z,  geomAxis_x, geomAxis_y, geomAxis_z, pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);


    dim3 grid( (nRH+32-1)/32, (nRH+32-1)/32 );
    dim3 block( 32, 32);

    fastCluster_step1<<<grid, block>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);

#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[2], start, stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif

    fastCluster_step2<<<grid, block>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);

#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[3], start, stop);
    cudaDeviceSynchronize();
#endif
    cudaProfilerStop();
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
				int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
				int* rhCount,
                float (&timer)[8]
				)
  { 
#ifdef DEBUG_GPU_ECAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    cudaMemsetAsync(rhCount, 1, sizeof(int)*size);
    cudaMemsetAsync(fracSum, 0., sizeof(float)*size);
    //seeding
    if(size>0) seedingKernel_ECAL<<<(size+512-1)/512, 512>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,  neigh8_Ind);
    
#ifdef DEBUG_GPU_ECAL
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&timer[0], start, stop);
      cudaEventRecord(start);
#endif

    // for(int a=0;a<16;a++){
    //if(size>0) topoKernel_ECAL<<<(size+512-1)/512, 512>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, neigh8_Ind);
    //}	    
    //cudaDeviceSynchronize();

    dim3 gridT( (size+64-1)/64, 1 );
    dim3 blockT( 64, 8);
    //dim3 gridT( (size+64-1)/64, 8 );
    //dim3 blockT( 64, 16);
    //for(int h=0; h<18; h++){  
    for(int h=0; h<nTopoLoops; h++){  
      if(size>0) topoKernel_ECALV2<<<gridT, blockT>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, neigh8_Ind);        
    }

#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[1], start, stop);
    
    cudaEventRecord(start);
#endif
    dim3 grid( (size+32-1)/32, (size+32-1)/32 );
    dim3 block( 32, 32);

    //if(size>0) std::cout<<std::endl<<"NEW EVENT !!"<<std::endl<<std::endl;

    if(size>0) fastCluster_step1<<<grid, block>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);
#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[2], start, stop);
    
    cudaEventRecord(start);
#endif

    if(size>0) fastCluster_step2<<<grid, block>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);  
   
#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer[3], start, stop);
#endif
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
				int* pcrhfracind,
				float* pcrhfrac,
				float* fracSum,
				int* rhCount,
                float* timer
				)
  { 
#ifdef DEBUG_GPU_ECAL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif    
    //seeding
    if(size>0) seedingKernel_ECAL_serialize<<<1,1>>>( size,  pfrh_energy,   pfrh_pt2,   pfrh_isSeed,  pfrh_topoId,  pfrh_layer,  neigh8_Ind);
    //cudaDeviceSynchronize();
     
#ifdef DEBUG_GPU_ECAL
    cudaEventRecord(start);
#endif
    for(int h=0; h < nTopoLoops; h++){
        if(size>0) topoKernel_ECAL_serialize<<<1,1>>>( size, pfrh_energy,  pfrh_topoId,  pfrh_layer, neigh8_Ind);
    }	    
    //cudaDeviceSynchronize();

#ifdef DEBUG_GPU_ECAL
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

    if(size>0) fastCluster_step2_serialize<<<1,1>>>( size, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pcrhfrac, pcrhfracind, fracSum, rhCount);  
   
  }
  
}  // namespace cudavectors
