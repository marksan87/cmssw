
#include <cmath>
#include <iostream>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterCudaHCAL.h"
#include <Eigen/Dense>
#include <cuda_profiler_api.h>

using PFClustering::common::PFLayer;

// Uncomment for debugging
//#define DEBUG_GPU_HCAL


constexpr int sizeof_float = sizeof(float);
constexpr int sizeof_int = sizeof(int);
constexpr const float PI_F = 3.141592654f;

namespace PFClusterCudaHCAL {

  __constant__ float showerSigma2;
  __constant__ float recHitEnergyNormInvEB_vec[4];
  __constant__ float recHitEnergyNormInvEE_vec[7];
  __constant__ float minFracToKeep;
  __constant__ float minFracTot;
  __constant__ float minFractionInCalc;
  __constant__ float minAllowedNormalization;
  __constant__ float stoppingTolerance;

  __constant__ float seedEThresholdEB_vec[4];
  __constant__ float seedEThresholdEE_vec[7];
  __constant__ float seedPt2ThresholdEB;
  __constant__ float seedPt2ThresholdEE;

  __constant__ float topoEThresholdEB_vec[4];
  __constant__ float topoEThresholdEE_vec[7];
  __constant__ int   maxIterations;
  __constant__ bool  excludeOtherSeeds;

  // Endcap timing constants
  __constant__ float corrTermLowEE;
  __constant__ float threshLowEE;
  __constant__ float noiseTermE;
  __constant__ float constantTermLowE2E;
  __constant__ float noiseTermLowEE;
  __constant__ float threshHighEE;
  __constant__ float constantTerm2E;
  __constant__ float resHighE2E;

  // Barrel timing constants
  __constant__ float corrTermLowEB;
  __constant__ float threshLowEB;
  __constant__ float noiseTermB;
  __constant__ float constantTermLowE2B;
  __constant__ float noiseTermLowEB;
  __constant__ float threshHighEB;
  __constant__ float constantTerm2B;
  __constant__ float resHighE2B;

  __constant__ int nNT = 8;  // Number of neighbors considered for topo clustering
  __constant__ int nNeigh;
 
  //int nTopoLoops = 100;
  int nTopoLoops = 35;


  bool initializeCudaConstants(const float h_showerSigma2,
                               const float (&h_recHitEnergyNormInvEB_vec)[4],
                               const float (&h_recHitEnergyNormInvEE_vec)[7],
                               const float h_minFracToKeep,
                               const float h_minFracTot,
                               const float h_minFractionInCalc,
                               const float h_minAllowedNormalization,
                               const int   h_maxIterations,
                               const float h_stoppingTolerance,
                               const bool  h_excludeOtherSeeds,
                               const float (&h_seedEThresholdEB_vec)[4],
                               const float (&h_seedEThresholdEE_vec)[7],
                               const float h_seedPt2ThresholdEB,
                               const float h_seedPt2ThresholdEE,
                               const float (&h_topoEThresholdEB_vec)[4],
                               const float (&h_topoEThresholdEE_vec)[7],
                               const PFClustering::common::TimeResConsts endcapTimeResConsts,
                               const PFClustering::common::TimeResConsts barrelTimeResConsts,
                               const int   h_nNeigh
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
     
     status &= cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormInvEB_vec, &h_recHitEnergyNormInvEB_vec, 4*sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     float val4[4];
     status &= cudaCheck(cudaMemcpyFromSymbol(&val4, recHitEnergyNormInvEB_vec, 4*sizeof_float));
     std::cout<<"recHitEnergyNormInvEB_vec read from symbol: ";
     for (int i = 0; i < 4; i++) {std::cout<<val4[i]<<" ";}
     std::cout<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(recHitEnergyNormInvEE_vec, &h_recHitEnergyNormInvEE_vec, 7*sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     float val7[7];
     status &= cudaCheck(cudaMemcpyFromSymbol(&val7, recHitEnergyNormInvEE_vec, 7*sizeof_float));
     std::cout<<"recHitEnergyNormInvEE_vec read from symbol: ";
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
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, minFracTot, sizeof_float));
     std::cout<<"minFracTot read from symbol: "<<val<<std::endl;
#endif
     
     status &= cudaCheck(cudaMemcpyToSymbolAsync(minFractionInCalc, &h_minFractionInCalc, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, minFractionInCalc, sizeof_float));
     std::cout<<"minFractionInCalc read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(minAllowedNormalization, &h_minAllowedNormalization, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, minAllowedNormalization, sizeof_float));
     std::cout<<"minAllowedNormalization read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(stoppingTolerance, &h_stoppingTolerance, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, stoppingTolerance, sizeof_float));
     std::cout<<"stoppingTolerance read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(excludeOtherSeeds, &h_excludeOtherSeeds, sizeof(bool)));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     bool bval = 0.;
     status &= cudaCheck(cudaMemcpyFromSymbol(&bval, excludeOtherSeeds, sizeof(bool)));
     std::cout<<"excludeOtherSeeds read from symbol: "<<bval<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(maxIterations, &h_maxIterations, sizeof_int));
#ifdef DEBUG_GPU_HCAL
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
     ival = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&ival, nNeigh, sizeof_int));
     std::cout<<"nNeigh read from symbol: "<<ival<<std::endl;
#endif

     // Endcap time resolution
     status &= cudaCheck(cudaMemcpyToSymbolAsync(corrTermLowEE, &endcapTimeResConsts.corrTermLowE, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, corrTermLowEE, sizeof_float));
     std::cout<<"corrTermLowEE read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(threshLowEE, &endcapTimeResConsts.threshLowE, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, threshLowEE, sizeof_float));
     std::cout<<"threshLowEE read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(noiseTermE, &endcapTimeResConsts.noiseTerm, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, noiseTermE, sizeof_float));
     std::cout<<"noiseTermE read from symbol: "<<val<<std::endl;
#endif
     
     status &= cudaCheck(cudaMemcpyToSymbolAsync(constantTermLowE2E, &endcapTimeResConsts.constantTermLowE2, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, constantTermLowE2E, sizeof_float));
     std::cout<<"constantTermLowE2E read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(noiseTermLowEE, &endcapTimeResConsts.noiseTermLowE, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, noiseTermLowEE, sizeof_float));
     std::cout<<"noiseTermLowEE read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(threshHighEE, &endcapTimeResConsts.threshHighE, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, threshHighEE, sizeof_float));
     std::cout<<"threshHighEE read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(constantTerm2E, &endcapTimeResConsts.constantTerm2, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, constantTerm2E, sizeof_float));
     std::cout<<"constantTerm2E read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(resHighE2E, &endcapTimeResConsts.resHighE2, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, resHighE2E, sizeof_float));
     std::cout<<"resHighE2E read from symbol: "<<val<<std::endl;
#endif

     // Barrel time resolution
     status &= cudaCheck(cudaMemcpyToSymbolAsync(corrTermLowEB, &barrelTimeResConsts.corrTermLowE, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, corrTermLowEB, sizeof_float));
     std::cout<<"corrTermLowEB read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(threshLowEB, &barrelTimeResConsts.threshLowE, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, threshLowEB, sizeof_float));
     std::cout<<"threshLowEB read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(noiseTermB, &barrelTimeResConsts.noiseTerm, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, noiseTermB, sizeof_float));
     std::cout<<"noiseTermB read from symbol: "<<val<<std::endl;
#endif
     
     status &= cudaCheck(cudaMemcpyToSymbolAsync(constantTermLowE2B, &barrelTimeResConsts.constantTermLowE2, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, constantTermLowE2B, sizeof_float));
     std::cout<<"constantTermLowE2B read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(noiseTermLowEB, &barrelTimeResConsts.noiseTermLowE, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, noiseTermLowEB, sizeof_float));
     std::cout<<"noiseTermLowEB read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(threshHighEB, &barrelTimeResConsts.threshHighE, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, threshHighEB, sizeof_float));
     std::cout<<"threshHighEB read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(constantTerm2B, &barrelTimeResConsts.constantTerm2, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, constantTerm2B, sizeof_float));
     std::cout<<"constantTerm2B read from symbol: "<<val<<std::endl;
#endif

     status &= cudaCheck(cudaMemcpyToSymbolAsync(resHighE2B, &barrelTimeResConsts.resHighE2, sizeof_float));
#ifdef DEBUG_GPU_HCAL
     // Read back the value
     val = 0;
     status &= cudaCheck(cudaMemcpyFromSymbol(&val, resHighE2B, sizeof_float));
     std::cout<<"resHighE2B read from symbol: "<<val<<std::endl;
#endif

     return status;
}


__device__ __forceinline__ float timeResolution2Endcap(const float energy) {
  float res2 = 10000.;
  
  if (energy <= 0.)
    return res2;
  else if (energy < threshLowEE) {
    if (corrTermLowEE > 0.) {  // different parametrisation
      const float res = noiseTermLowEE / energy + corrTermLowEE / (energy * energy);
      res2 = res * res;
    } else {
      const float noiseDivE = noiseTermLowEE / energy;
      res2 = noiseDivE * noiseDivE + constantTermLowE2E;
    }
  } else if (energy < threshHighEE) {
    const float noiseDivE = noiseTermE / energy;
    res2 = noiseDivE * noiseDivE + constantTerm2E;
  } else  // if (energy >=threshHighE_)
    res2 = resHighE2E;

  if (res2 > 10000.)
    return 10000.;
  return res2;
}

__device__ __forceinline__ float timeResolution2Barrel(const float energy) {
  float res2 = 10000.;

  if (energy <= 0.)
    return res2;
  else if (energy < threshLowEB) {
    if (corrTermLowEB > 0.) {  // different parametrisation
      const float res = noiseTermLowEB / energy + corrTermLowEB / (energy * energy);
      res2 = res * res;
    } else {
      const float noiseDivE = noiseTermLowEB / energy;
      res2 = noiseDivE * noiseDivE + constantTermLowE2B;
    }
  } else if (energy < threshHighEB) {
    const float noiseDivE = noiseTermB / energy;
    res2 = noiseDivE * noiseDivE + constantTerm2B;
  } else  // if (energy >=threshHighE_)
    res2 = resHighE2B;

  if (res2 > 10000.)
    return 10000.;
  return res2;
}

__device__ __forceinline__ float dR2(float4 pos1, float4 pos2) {
    float mag1 = sqrtf(pos1.x*pos1.x + pos1.y*pos1.y + pos1.z*pos1.z);
    float cosTheta1 = mag1 > 0.0 ? pos1.z / mag1 : 1.0;
    float eta1 = 0.5 * logf( (1.0+cosTheta1) / (1.0-cosTheta1) );
    float phi1 = atan2f(pos1.y, pos1.x);

    float mag2 = sqrtf(pos2.x*pos2.x + pos2.y*pos2.y + pos2.z*pos2.z);
    float cosTheta2 = mag2 > 0.0 ? pos2.z / mag2 : 1.0;
    float eta2 = 0.5 * logf( (1.0+cosTheta2) / (1.0-cosTheta2) );
    float phi2 = atan2f(pos2.y, pos2.x);

    float deta = eta2-eta1;
    float dphi = fabsf(fabsf(phi2 - phi1) - PI_F) - PI_F;
    return (deta*deta + dphi*dphi);
}

// https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
/*static __device__ __forceinline__ float atomicMinF(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val < __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}*/

static __device__ __forceinline__ float atomicMaxF(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

 __global__ void seedingTopoThreshKernel_HCAL(
     				size_t size, 
				    const float* __restrict__ pfrh_energy,
				    const float* __restrict__ pfrh_pt2,
				    int*   pfrh_isSeed,
				    int*   pfrh_topoId,
                    bool*  pfrh_passTopoThresh,
				    const int* __restrict__ pfrh_layer,
				    const int* __restrict__ pfrh_depth,
				    const int* __restrict__ neigh4_Ind,
				    int* rhCount,
                    int* topoSeedCount,
                    int* topoRHCount,
                    int* seedFracOffsets,
                    int* topoSeedOffsets,
                    int* topoSeedList,
                    int* pfcIter
                    ) {

   int i = threadIdx.x+blockIdx.x*blockDim.x;

   if(i<size) {
     // Initialize arrays
     pfrh_topoId[i] = i;
     pfrh_isSeed[i] = 0;
     rhCount[i] = 0;
     topoSeedCount[i] = 0;
     topoRHCount[i] = 0;
     seedFracOffsets[i] = -1;
     topoSeedOffsets[i] = -1;
     topoSeedList[i] = -1;
     pfcIter[i] = -1;

     // Seeding threshold test
     if ( (pfrh_layer[i] == PFLayer::HCAL_BARREL1 && pfrh_energy[i]>seedEThresholdEB_vec[pfrh_depth[i]-1] && pfrh_pt2[i]>seedPt2ThresholdEB) || 
          (pfrh_layer[i] == PFLayer::HCAL_ENDCAP && pfrh_energy[i]>seedEThresholdEE_vec[pfrh_depth[i]-1] && pfrh_pt2[i]>seedPt2ThresholdEE) )
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
     if ( (pfrh_layer[i] == PFLayer::HCAL_ENDCAP && pfrh_energy[i]>topoEThresholdEE_vec[pfrh_depth[i]-1]) ||
             (pfrh_layer[i] == PFLayer::HCAL_BARREL1 && pfrh_energy[i]>topoEThresholdEB_vec[pfrh_depth[i]-1])) {
            pfrh_passTopoThresh[i] = true;
        }
     //else { pfrh_passTopoThresh[i] = false; }
     else { pfrh_passTopoThresh[i] = false; pfrh_topoId[i] = -1; }
   }
 }
 __global__ void seedingKernel_HCAL(
     				size_t size, 
				    const float* __restrict__ pfrh_energy,
				    const float* __restrict__ pfrh_pt2,
				    int*   pfrh_isSeed,
				    int*   pfrh_topoId,
				    const int* __restrict__ pfrh_layer,
				    const int* __restrict__ pfrh_depth,
				    const int* __restrict__ neigh4_Ind
				    ) {

   int i = threadIdx.x+blockIdx.x*blockDim.x;

   if(i<size) {        
     if ( (pfrh_layer[i] == PFLayer::HCAL_BARREL1 && pfrh_energy[i]>seedEThresholdEB_vec[pfrh_depth[i]-1] && pfrh_pt2[i]>seedPt2ThresholdEB) || 
          (pfrh_layer[i] == PFLayer::HCAL_ENDCAP && pfrh_energy[i]>seedEThresholdEE_vec[pfrh_depth[i]-1] && pfrh_pt2[i]>seedPt2ThresholdEE) )
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
				    const float* __restrict__ pfrh_energy,
				    const float* __restrict__ pfrh_pt2,
				    int*   pfrh_isSeed,
				    int*   pfrh_topoId,
				    const int* __restrict__ pfrh_layer,
				    const int* __restrict__ pfrh_depth,
				    const int* __restrict__ neigh4_Ind
				    ) {

   //int i = threadIdx.x+blockIdx.x*blockDim.x;
   for (int i = 0; i < size; i++) {
       if(i<size) {        
         if ( (pfrh_layer[i] == PFLayer::HCAL_BARREL1 && pfrh_energy[i]>seedEThresholdEB_vec[pfrh_depth[i]-1] && pfrh_pt2[i]>seedPt2ThresholdEB) || 
              (pfrh_layer[i] == PFLayer::HCAL_ENDCAP && pfrh_energy[i]>seedEThresholdEE_vec[pfrh_depth[i]-1] && pfrh_pt2[i]>seedPt2ThresholdEE) )
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
    const float* __restrict__ pfrh_energy,
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
				  const float* __restrict__ pfrh_energy,
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
	    ( (pfrh_layer[neigh8_Ind[nNT*l+k]] == PFLayer::HCAL_ENDCAP && pfrh_energy[neigh8_Ind[nNT*l+k]]>topoEThresholdEE_vec[pfrh_depth[neigh8_Ind[nNT*l+k]]-1]) ||
	      (pfrh_layer[neigh8_Ind[nNT*l+k]] == PFLayer::HCAL_BARREL1 && pfrh_energy[neigh8_Ind[nNT*l+k]]>topoEThresholdEB_vec[pfrh_depth[neigh8_Ind[nNT*l+k]]-1]) ) &&
	    ( (pfrh_layer[l] == PFLayer::HCAL_ENDCAP && pfrh_energy[l]>topoEThresholdEE_vec[pfrh_depth[l]-1]) ||
	      (pfrh_layer[l] == PFLayer::HCAL_BARREL1 && pfrh_energy[l]>topoEThresholdEB_vec[pfrh_depth[l]-1]))
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
				  const float* __restrict__ pfrh_energy,
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
                ( (pfrh_layer[neigh8_Ind[nNT*l+k]] == PFLayer::HCAL_ENDCAP && pfrh_energy[neigh8_Ind[nNT*l+k]]>topoEThresholdEE_vec[pfrh_depth[neigh8_Ind[nNT*l+k]]-1]) ||
                  (pfrh_layer[neigh8_Ind[nNT*l+k]] == PFLayer::HCAL_BARREL1 && pfrh_energy[neigh8_Ind[nNT*l+k]]>topoEThresholdEB_vec[pfrh_depth[neigh8_Ind[nNT*l+k]]-1]) ) &&
                ( (pfrh_layer[l] == PFLayer::HCAL_ENDCAP && pfrh_energy[l]>topoEThresholdEE_vec[pfrh_depth[l]-1]) ||
                  (pfrh_layer[l] == PFLayer::HCAL_BARREL1 && pfrh_energy[l]>topoEThresholdEB_vec[pfrh_depth[l]-1]))
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


__global__ void hcalFastCluster_noLambdaPosCalc(size_t nRH,
                                const float* __restrict__ pfrh_x,
                                const float* __restrict__ pfrh_y,
                                const float* __restrict__ pfrh_z,
                                const float* __restrict__ pfrh_energy,
                                int* pfrh_topoId,
                                int* pfrh_isSeed,
                                const int* __restrict__ pfrh_layer,
                                const int* __restrict__ pfrh_depth,
                                const int* __restrict__ neigh4_Ind,
                                float* pcrhfrac,
                                int* pcrhfracind,
                                float* fracSum,
                                int* rhCount,
                                int* topoSeedCount,
                                int* topoRHCount,
                                int* seedFracOffsets,
                                int* topoSeedOffsets,
                                int* topoSeedList,
                                float4* clusterPos,
                                float4* prevClusterPos,
                                float* clusterEnergy,
                                int* pfcIter
                                ) {

  int topoId = blockIdx.x;
  if (topoId < nRH && topoRHCount[topoId] > 1 && topoSeedCount[topoId] > 0 && topoRHCount[topoId] != topoSeedCount[topoId]) {
    //printf("Now on topoId %d\tthreadIdx.x = %d\n", topoId, threadIdx.x);
    __shared__ int nSeeds, nRHTopo, nRHNotSeed, topoSeedBegin, gridStride, iter;
    __shared__ float tol, diff, diff2;
    __shared__ bool notDone, debug, noPosCalc;
    if (threadIdx.x == 0) {
        nSeeds = topoSeedCount[topoId];
        nRHTopo = topoRHCount[topoId];
        nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
        topoSeedBegin = topoSeedOffsets[topoId];
        tol = stoppingTolerance * powf(fmaxf(1.0, nSeeds - 1.0), 2.0);     // stopping tolerance * tolerance scaling
        gridStride = blockDim.x * gridDim.x;
        iter = 0;
        notDone = true;
        debug = false;
        noPosCalc = false; 
        //debug = (topoId == 432 || topoId == 438 || topoId == 439) ? true : false;
    }
    __syncthreads();

    auto getSeedRhIdx = [&] (int seedNum) {
        if (seedNum > topoSeedCount[topoId]) {
            printf("PROBLEM with seedNum = %d > nSeeds = %d", seedNum, nSeeds); 
            return -1;
        }
        return topoSeedList[topoSeedBegin + seedNum];
    };

    auto getRhFracIdx = [&] (int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfracind[seedFracOffsets[seedIdx] + rhNum];
    };
    
    auto getRhFrac = [&] (int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfrac[seedFracOffsets[seedIdx] + rhNum];
    };

    if (debug) {
        if (threadIdx.x == 0) {
            printf("\n===========================================================================================\n");
            printf("Processing topo cluster %d with nSeeds = %d nRHTopo = %d and seeds (", topoId, nSeeds, nRHTopo);
            for (int s = 0; s < nSeeds; s++) {
                if (s != 0) printf(", ");
                printf("%d", getSeedRhIdx(s));
            }
            if (nRHTopo == nSeeds) {
                printf(")\n\n");
            }
            else {
                printf(") and other rechits (");
                for (int r = 1; r < nRHNotSeed; r++) {
                    if (r != 1) printf(", ");
                    printf("%d", getRhFracIdx(0, r));
                }
                printf(")\n\n");
            }
        }
        __syncthreads();
    }
    

    /*
    auto computeClusterPos = [&] (float4& pos4, float _frac, int rhInd, bool isDebug) {
        float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
        float threshold = 0.0;
        if(pfrh_layer[rhInd] == PFLayer::HCAL_BARREL1) {
            threshold = recHitEnergyNormInvEB_vec[pfrh_depth[rhInd] - 1]; 
        }
        else if (pfrh_layer[rhInd] == PFLayer::HCAL_ENDCAP) { threshold = recHitEnergyNormInvEE_vec[pfrh_depth[rhInd] - 1]; }

        const auto rh_energy = pfrh_energy[rhInd] * _frac;
        const auto norm =
            (_frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * threshold)));
        if (isDebug)
            printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n", rhInd, norm, _frac, rh_energy, rechitPos.x, rechitPos.y, rechitPos.z);
        
        atomicAdd(&pos4.x, rechitPos.x * norm);
        atomicAdd(&pos4.y, rechitPos.y * norm);
        atomicAdd(&pos4.z, rechitPos.z * norm);
        atomicAdd(&pos4.w, norm);   // position_norm
    };
    */

    // Set initial cluster position (energy) to seed rechit position (energy)
    for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
        int i = getSeedRhIdx(s);
        clusterPos[i] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
        prevClusterPos[i] = clusterPos[i];
        clusterEnergy[i] = pfrh_energy[i];
    }
    __syncthreads();
    
    while (notDone) {
        if (debug && threadIdx.x == 0) {
            printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
        }

        // Reset fracSum and rhCount
        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {
            int j = getRhFracIdx(0, r);
            fracSum[j] = 0.;
            rhCount[j] = 1;

            for (int s = 0; s < nSeeds; s++) {
                int i = getSeedRhIdx(s);
                pcrhfrac[seedFracOffsets[i] + r] = -1.;
            }
        }
        __syncthreads();

        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {    // One thread for each (non-seed) rechit 
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = getSeedRhIdx(s);
                int j = getRhFracIdx(s, r);
                
                if (debug) {
                    printf("\tCluster %d (seed %d) has position: (%.4f, %.4f, %4f)\n", s, i, clusterPos[i].x, clusterPos[i].y, clusterPos[i].z);
                }

                float dist2 =
                   (clusterPos[i].x - pfrh_x[j])*(clusterPos[i].x - pfrh_x[j])
                  +(clusterPos[i].y - pfrh_y[j])*(clusterPos[i].y - pfrh_y[j])
                  +(clusterPos[i].z - pfrh_z[j])*(clusterPos[i].z - pfrh_z[j]);
                
                float d2 = dist2 / showerSigma2;
                float fraction = -1.;
                
                if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                if( pfrh_isSeed[j]!=1) {
                    atomicAdd(&fracSum[j],fraction);
                }
            }
        }
        __syncthreads();
        
        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {    // One thread for each (non-seed) rechit 
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = getSeedRhIdx(s); 
                int j = getRhFracIdx(s, r);
                
                if( pfrh_isSeed[j]!=1 ){
                    float dist2 =
                       (clusterPos[i].x - pfrh_x[j])*(clusterPos[i].x - pfrh_x[j])
                      +(clusterPos[i].y - pfrh_y[j])*(clusterPos[i].y - pfrh_y[j])
                      +(clusterPos[i].z - pfrh_z[j])*(clusterPos[i].z - pfrh_z[j]);
                    
                    float d2 = dist2 / showerSigma2;
                    float fraction = -1.;
                    
                    if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                    else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                    if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                    if (fracSum[j] > minFracTot) {
                        float fracpct = fraction / fracSum[j];
                        if(fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
                            pcrhfrac[seedFracOffsets[i]+r] = fracpct;
                        }
                        else {
                            pcrhfrac[seedFracOffsets[i]+r] = -1;
                        }
                    }
                    else {
                        pcrhfrac[seedFracOffsets[i]+r] = -1;
                    }
                }
            }
        }
        __syncthreads();
    if (!noPosCalc) {
        if (debug && threadIdx.x == 0)
            printf("Computing cluster position for topoId %d\n", topoId);
        
        // Reset cluster position and energy
        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);
            clusterPos[i] = make_float4(0.0, 0.0, 0.0, 0.0);
            clusterEnergy[i] = 0;
        }
        __syncthreads();

        // Recalculate position
        for (int r = threadIdx.x; r < nRHNotSeed; r += gridStride) {    // One thread for each (non-seed) rechit 
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = getSeedRhIdx(s);    // Seed index
                
                if (debug) {
                    printf("\tNow on seed %d\t\tneigh4Ind = [", i);
                    for(int k=0; k<nNeigh; k++){
                        if (k != 0) printf(", ");
                        printf("%d", neigh4_Ind[nNeigh*i+k]);
                    }
                    printf("]\n");
                
                }

                // Calculate cluster energy by summing rechit fractional energies
                int j = getRhFracIdx(s,r);
                float frac = getRhFrac(s,r); 
                
                if (frac > -0.5) {
                    //if (debug)
                        //printf("\t\tRechit %d (position %d) in this PF cluster with frac = %f\n", j, _n, frac);
                    //clusterEnergy[i] += frac * pfrh_energy[j];
                    atomicAdd(&clusterEnergy[i], frac * pfrh_energy[j]);
                    //bool updateClusterPos = (nSeeds == 1 || j == i) ? true : false;
                    bool updateClusterPos = false; 
                    if (nSeeds == 1) {
                        if (debug)
                            printf("\t\tThis topo cluster has a single seed.\n");
                        //computeClusterPos(clusterPos[i], frac, j, debug);
                        updateClusterPos = true;
                    }
                    else {
                        if (j == i) {
                            // This is the seed
                            //computeClusterPos(clusterPos[i], frac, j, debug);
                            updateClusterPos = true;
                        }
                        else {
                            // Check if this is one of the neighboring rechits
                            for(int k=0; k<nNeigh; k++){
                                if(neigh4_Ind[nNeigh*i+k]<0) continue;
                                if(neigh4_Ind[nNeigh*i+k] == j) {
                                    // Found it
                                    if (debug) printf("\t\tRechit %d is one of the 4 neighbors of seed %d\n", j, i);
                                    //computeClusterPos(clusterPos[i], frac, j, debug);
                                    updateClusterPos = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (updateClusterPos) {
                        float4 rechitPos = make_float4(pfrh_x[j], pfrh_y[j], pfrh_z[j], 1.0);
                        float threshold = 0.0;
                        if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) {
                            threshold = recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1]; 
                        }
                        else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { threshold = recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1]; }

                        const auto rh_energy = pfrh_energy[j] * frac;
                        const auto norm =
                            (frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * threshold)));
                        if (debug)
                            printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n", j, norm, frac, rh_energy, rechitPos.x, rechitPos.y, rechitPos.z);
                        
                        atomicAdd(&clusterPos[i].x, rechitPos.x * norm);
                        atomicAdd(&clusterPos[i].y, rechitPos.y * norm);
                        atomicAdd(&clusterPos[i].z, rechitPos.z * norm);
                        atomicAdd(&clusterPos[i].w, norm);  // position_norm
                    }
                }
                //else if (debug)
                //    printf("Can't find rechit fraction for cluster %d (seed %d) rechit %d!\n", s, i, j);
                    
            }
        }
        __syncthreads();
        
        // Normalize the seed postiions
        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);    // Seed index
            if (clusterPos[i].w >= minAllowedNormalization)
            {
                // Divide by position norm
                clusterPos[i].x /= clusterPos[i].w;
                clusterPos[i].y /= clusterPos[i].w;
                clusterPos[i].z /= clusterPos[i].w;

                if (debug)
                    printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n", s, i, clusterEnergy[i], clusterPos[i].x, clusterPos[i].y, clusterPos[i].z);
            }
            else {
                if (debug)
                    printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n", s, i, clusterPos[i].w, minAllowedNormalization);
                clusterPos[i].x = 0.0;
                clusterPos[i].y = 0.0;
                clusterPos[i].z = 0.0;
                //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
                
            }
        }

        // Reset diff2
        if (threadIdx.x == 0) {
            diff2 = -1.;
        }
        __syncthreads();
       
        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);
            float delta2 = dR2(prevClusterPos[i], clusterPos[i]);
            if (debug) printf("\tCluster %d (seed %d) has delta2 = %f\n", s, i, delta2);
            atomicMaxF(&diff2, delta2); 
//            if (delta2 > diff2) {
//                diff2 = delta2;
//                if (debug) printf("\t\tNew diff2 = %f\n", diff2);
//            }

            prevClusterPos[i] = clusterPos[i];  // Save clusterPos 
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            diff = sqrtf(diff2);
            iter++;
            notDone = (diff > tol) && (iter < maxIterations);
            if (debug) {
                if (diff > tol)
                    printf("\tTopoId %d has diff = %f greater than tolerance %f (continuing)\n", topoId, diff, tol);
                else
                    if (debug) printf("\tTopoId %d has diff = %f LESS than tolerance %f (terminating!)\n", topoId, diff, tol);
            }
        }
        __syncthreads();
     }  // if (!noPosCalc)
     else {
        if (threadIdx.x == 0) notDone = false;
        __syncthreads();
     }
    }
    if (threadIdx.x == 0) pfcIter[topoId] = iter;
  }
  else if (threadIdx.x == 0 && (topoRHCount[topoId] == 1 || (topoRHCount[topoId] > 1 && topoRHCount[topoId] == topoSeedCount[topoId]))) {
    // Single rh cluster or all rechits in this topo cluster are seeds. No iterations needed
    pfcIter[topoId] = 0;
  }
}

__global__ void hcalFastCluster_sharedMem(size_t nRH,
                                const float* __restrict__ pfrh_x,
                                const float* __restrict__ pfrh_y,
                                const float* __restrict__ pfrh_z,
                                const float* __restrict__ pfrh_energy,
                                int* pfrh_topoId,
                                int* pfrh_isSeed,
                                const int* __restrict__ pfrh_layer,
                                const int* __restrict__ pfrh_depth,
                                const int* __restrict__ neigh4_Ind,
                                float* pcrhfrac,
                                int* pcrhfracind,
                                float* fracSum,
                                int* rhCount,
                                int* topoSeedCount,
                                int* topoRHCount,
                                int* seedFracOffsets,
                                int* topoSeedOffsets,
                                int* topoSeedList,
                                float4* clusterPos,
                                float4* prevClusterPos,
                                float* clusterEnergy,
                                int* pfcIter
                                ) {

  int topoId = blockIdx.x;
  
  // Exclude topo clusters with >= 3 seeds for testing
  //if (topoId < nRH && topoRHCount[topoId] > 1 && topoSeedCount[topoId] > 0 && topoRHCount[topoId] != topoSeedCount[topoId] && (blockDim.x <= 32 ? (topoSeedCount[topoId] < 3) : (topoSeedCount[topoId] >= 3) ))  {
  
  if (topoId < nRH && topoRHCount[topoId] > 1 && topoSeedCount[topoId] > 0 && topoRHCount[topoId] != topoSeedCount[topoId]) {
    //printf("Now on topoId %d\tthreadIdx.x = %d\n", topoId, threadIdx.x);
    __shared__ int nSeeds, nRHTopo, nRHNotSeed, topoSeedBegin, gridStride, iter;
    __shared__ float tol, diff, diff2;
    __shared__ bool notDone, debug, noPosCalc;

    __shared__ int rechits[256];

    if (threadIdx.x == 0) {
        nSeeds = topoSeedCount[topoId];
        nRHTopo = topoRHCount[topoId];
        nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
        topoSeedBegin = topoSeedOffsets[topoId];
        tol = stoppingTolerance * powf(fmaxf(1.0, nSeeds - 1.0), 2.0);     // stopping tolerance * tolerance scaling
        gridStride = blockDim.x * gridDim.x;
        iter = 0;
        notDone = true;
        debug = false;
        noPosCalc = false; 
        //debug = (topoId == 432 || topoId == 438 || topoId == 439) ? true : false;
    }
    __syncthreads();

    auto getSeedRhIdx = [&] (int seedNum) {
        if (seedNum > topoSeedCount[topoId]) {
            printf("PROBLEM with seedNum = %d > nSeeds = %d", seedNum, nSeeds); 
            return -1;
        }
        return topoSeedList[topoSeedBegin + seedNum];
    };

    auto getRhFracIdx = [&] (int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfracind[seedFracOffsets[seedIdx] + rhNum];
    };
    
    auto getRhFrac = [&] (int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfrac[seedFracOffsets[seedIdx] + rhNum];
    };

    if (debug) {
        if (threadIdx.x == 0) {
            printf("\n===========================================================================================\n");
            printf("Processing topo cluster %d with nSeeds = %d nRHTopo = %d and seeds (", topoId, nSeeds, nRHTopo);
            for (int s = 0; s < nSeeds; s++) {
                if (s != 0) printf(", ");
                printf("%d", getSeedRhIdx(s));
            }
            if (nRHTopo == nSeeds) {
                printf(")\n\n");
            }
            else {
                printf(") and other rechits (");
                for (int r = 1; r < nRHNotSeed; r++) {
                    if (r != 1) printf(", ");
                    printf("%d", getRhFracIdx(0, r));
                }
                printf(")\n\n");
            }
        }
        __syncthreads();
    }
    


    auto computeClusterPos = [&] (float4& pos4, float _frac, int rhInd, bool isDebug) {
        float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
        float threshold = 0.0;
        if(pfrh_layer[rhInd] == PFLayer::HCAL_BARREL1) {
            threshold = recHitEnergyNormInvEB_vec[pfrh_depth[rhInd] - 1]; // This number needs to be inverted
        }
        else if (pfrh_layer[rhInd] == PFLayer::HCAL_ENDCAP) { threshold = recHitEnergyNormInvEE_vec[pfrh_depth[rhInd] - 1]; }

        const auto rh_energy = pfrh_energy[rhInd] * _frac;
        const auto norm =
            (_frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * threshold)));
        if (isDebug)
            printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n", rhInd, norm, _frac, rh_energy, rechitPos.x, rechitPos.y, rechitPos.z);
        
        atomicAdd(&pos4.x, rechitPos.x * norm);
        atomicAdd(&pos4.y, rechitPos.y * norm);
        atomicAdd(&pos4.z, rechitPos.z * norm);
        atomicAdd(&pos4.w, norm);   // position_norm
//        pos4.x += rechitPos.x * norm;
//        pos4.y += rechitPos.y * norm;
//        pos4.z += rechitPos.z * norm;
//        pos4.w += norm;     //  position_norm
    };
    

    // Set initial cluster position (energy) to seed rechit position (energy)
    for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
        int i = getSeedRhIdx(s);
        clusterPos[i] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
        prevClusterPos[i] = clusterPos[i];
        clusterEnergy[i] = pfrh_energy[i];
    }
    
    for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {
        rechits[r] = getRhFracIdx(0, r);
    }
    __syncthreads();
    
    while (notDone) {
        if (debug && threadIdx.x == 0) {
            printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
        }

        // Reset fracSum and rhCount
        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {
            int j = getRhFracIdx(0, r);
            fracSum[j] = 0.;
            rhCount[j] = 1;

            for (int s = 0; s < nSeeds; s++) {
                int i = getSeedRhIdx(s);
                pcrhfrac[seedFracOffsets[i] + r] = -1.;
            }
        }
        __syncthreads();

        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {    // One thread for each (non-seed) rechit 
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = getSeedRhIdx(s);
                int j = getRhFracIdx(s, r);
                
                if (debug) {
                    printf("\tCluster %d (seed %d) has position: (%.4f, %.4f, %4f)\n", s, i, clusterPos[i].x, clusterPos[i].y, clusterPos[i].z);
                }

                float dist2 =
                   (clusterPos[i].x - pfrh_x[j])*(clusterPos[i].x - pfrh_x[j])
                  +(clusterPos[i].y - pfrh_y[j])*(clusterPos[i].y - pfrh_y[j])
                  +(clusterPos[i].z - pfrh_z[j])*(clusterPos[i].z - pfrh_z[j]);
                
                float d2 = dist2 / showerSigma2;
                float fraction = -1.;
                
                if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                if( pfrh_isSeed[j]!=1) {
                    atomicAdd(&fracSum[j],fraction);
                }
            }
        }
        __syncthreads();
        
        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {    // One thread for each (non-seed) rechit 
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = getSeedRhIdx(s); 
                int j = getRhFracIdx(s, r);
                
                if( pfrh_isSeed[j]!=1 ){
                    float dist2 =
                       (clusterPos[i].x - pfrh_x[j])*(clusterPos[i].x - pfrh_x[j])
                      +(clusterPos[i].y - pfrh_y[j])*(clusterPos[i].y - pfrh_y[j])
                      +(clusterPos[i].z - pfrh_z[j])*(clusterPos[i].z - pfrh_z[j]);
                    
                    float d2 = dist2 / showerSigma2;
                    float fraction = -1.;
                    
                    if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                    else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                    if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                    if (fracSum[j] > minFracTot) {
                        float fracpct = fraction / fracSum[j];
                        if(fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
                            pcrhfrac[seedFracOffsets[i]+r] = fracpct;
                        }
                        else {
                            pcrhfrac[seedFracOffsets[i]+r] = -1;
                        }
                    }
                    else {
                        pcrhfrac[seedFracOffsets[i]+r] = -1;
                    }
                }
            }
        }
        __syncthreads();
    if (!noPosCalc) {
        if (debug && threadIdx.x == 0)
            printf("Computing cluster position for topoId %d\n", topoId);
        
        // Reset cluster position and energy
        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);
            clusterPos[i] = make_float4(0.0, 0.0, 0.0, 0.0);
            clusterEnergy[i] = 0;
        }
        __syncthreads();

        // Recalculate position
        for (int r = threadIdx.x; r < nRHNotSeed; r += gridStride) {    // One thread for each (non-seed) rechit 
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = getSeedRhIdx(s);    // Seed index
                
                if (debug) {
                    printf("\tNow on seed %d\t\tneigh4Ind = [", i);
                    for(int k=0; k<nNeigh; k++){
                        if (k != 0) printf(", ");
                        printf("%d", neigh4_Ind[nNeigh*i+k]);
                    }
                    printf("]\n");
                
                }

                // Calculate cluster energy by summing rechit fractional energies
                int j = getRhFracIdx(s,r);
                float frac = getRhFrac(s,r); 
                
                if (frac > -0.5) {
                    //if (debug)
                        //printf("\t\tRechit %d (position %d) in this PF cluster with frac = %f\n", j, _n, frac);
                    //clusterEnergy[i] += frac * pfrh_energy[j];
                    atomicAdd(&clusterEnergy[i], frac * pfrh_energy[j]);

                    bool updateClusterPos = false; 
                    if (nSeeds == 1) {
                        if (debug)
                            printf("\t\tThis topo cluster has a single seed.\n");
                        //computeClusterPos(clusterPos[i], frac, j, debug);
                        updateClusterPos = true;
                    }
                    else {
                        if (j == i) {
                            // This is the seed
                            //computeClusterPos(clusterPos[i], frac, j, debug);
                            updateClusterPos = true;
                        }
                        else {
                            // Check if this is one of the neighboring rechits
                            for(int k=0; k<nNeigh; k++){
                                if(neigh4_Ind[nNeigh*i+k]<0) continue;
                                if(neigh4_Ind[nNeigh*i+k] == j) {
                                    // Found it
                                    if (debug) printf("\t\tRechit %d is one of the 4 neighbors of seed %d\n", j, i);
                                    //computeClusterPos(clusterPos[i], frac, j, debug);
                                    updateClusterPos = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (updateClusterPos) computeClusterPos(clusterPos[i], frac, j, debug);
                    
                }
                //else if (debug)
                //    printf("Can't find rechit fraction for cluster %d (seed %d) rechit %d!\n", s, i, j);
                    
            }
        }
        __syncthreads();
        
        // Normalize the seed postiions
        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);    // Seed index
            if (clusterPos[i].w >= minAllowedNormalization)
            {
                // Divide by position norm
                clusterPos[i].x /= clusterPos[i].w;
                clusterPos[i].y /= clusterPos[i].w;
                clusterPos[i].z /= clusterPos[i].w;

                if (debug)
                    printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n", s, i, clusterEnergy[i], clusterPos[i].x, clusterPos[i].y, clusterPos[i].z);
            }
            else {
                if (debug)
                    printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n", s, i, clusterPos[i].w, minAllowedNormalization);
                clusterPos[i].x = 0.0;
                clusterPos[i].y = 0.0;
                clusterPos[i].z = 0.0;
                //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
                
            }
        }

        // Reset diff2
        if (threadIdx.x == 0) {
            diff2 = -1.;
        }
        __syncthreads();
       
        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);
            float delta2 = dR2(prevClusterPos[i], clusterPos[i]);
            if (debug) printf("\tCluster %d (seed %d) has delta2 = %f\n", s, i, delta2);
            atomicMaxF(&diff2, delta2); 
//            if (delta2 > diff2) {
//                diff2 = delta2;
//                if (debug) printf("\t\tNew diff2 = %f\n", diff2);
//            }

            prevClusterPos[i] = clusterPos[i];  // Save clusterPos 
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            diff = sqrtf(diff2);
            iter++;
            notDone = (diff > tol) && (iter < maxIterations);
            if (debug) {
                if (diff > tol)
                    printf("\tTopoId %d has diff = %f greater than tolerance %f (continuing)\n", topoId, diff, tol);
                else
                    if (debug) printf("\tTopoId %d has diff = %f LESS than tolerance %f (terminating!)\n", topoId, diff, tol);
            }
        }
        __syncthreads();
     }  // if (!noPosCalc)
     else {
        if (threadIdx.x == 0) notDone = false;
        __syncthreads();
     }
    }
    if (threadIdx.x == 0) pfcIter[topoId] = iter;
  }
  else if (threadIdx.x == 0 && (topoRHCount[topoId] == 1 || (topoRHCount[topoId] > 1 && topoRHCount[topoId] == topoSeedCount[topoId]))) {
    // Single rh cluster or all rechits in this topo cluster are seeds. No iterations needed
    pfcIter[topoId] = 0;
  }
}


__global__ void hcalFastCluster_optimizedLambdas(size_t nRH,
                                const float* __restrict__ pfrh_x,
                                const float* __restrict__ pfrh_y,
                                const float* __restrict__ pfrh_z,
                                const float* __restrict__ pfrh_energy,
                                int* pfrh_topoId,
                                int* pfrh_isSeed,
                                const int* __restrict__ pfrh_layer,
                                const int* __restrict__ pfrh_depth,
                                const int* __restrict__ neigh4_Ind,
                                float* pcrhfrac,
                                int* pcrhfracind,
                                float* fracSum,
                                int* rhCount,
                                int* topoSeedCount,
                                int* topoRHCount,
                                int* seedFracOffsets,
                                int* topoSeedOffsets,
                                int* topoSeedList,
                                float4* clusterPos,
                                float4* prevClusterPos,
                                float* clusterEnergy,
                                int* pfcIter
                                ) {

  int topoId = blockIdx.x;
  
  // Exclude topo clusters with >= 3 seeds for testing
  //if (topoId < nRH && topoRHCount[topoId] > 1 && topoSeedCount[topoId] > 0 && topoRHCount[topoId] != topoSeedCount[topoId] && (blockDim.x <= 32 ? (topoSeedCount[topoId] < 3) : (topoSeedCount[topoId] >= 3) ))  {
  
  if (topoId < nRH && topoRHCount[topoId] > 1 && topoSeedCount[topoId] > 0 && topoRHCount[topoId] != topoSeedCount[topoId]) {
    //printf("Now on topoId %d\tthreadIdx.x = %d\n", topoId, threadIdx.x);
    __shared__ int nSeeds, nRHTopo, nRHNotSeed, topoSeedBegin, gridStride, iter;
    __shared__ float tol, diff, diff2;
    __shared__ bool notDone, debug, noPosCalc;
    if (threadIdx.x == 0) {
        nSeeds = topoSeedCount[topoId];
        nRHTopo = topoRHCount[topoId];
        nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
        topoSeedBegin = topoSeedOffsets[topoId];
        tol = stoppingTolerance * powf(fmaxf(1.0, nSeeds - 1.0), 2.0);     // stopping tolerance * tolerance scaling
        gridStride = blockDim.x * gridDim.x;
        iter = 0;
        notDone = true;
        debug = false;
        noPosCalc = false; 
        //debug = (topoId == 432 || topoId == 438 || topoId == 439) ? true : false;
    }
    __syncthreads();

    auto getSeedRhIdx = [&] (int seedNum) {
        if (seedNum > topoSeedCount[topoId]) {
            printf("PROBLEM with seedNum = %d > nSeeds = %d", seedNum, nSeeds); 
            return -1;
        }
        return topoSeedList[topoSeedBegin + seedNum];
    };

    auto getRhFracIdx = [&] (int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfracind[seedFracOffsets[seedIdx] + rhNum];
    };
    
    auto getRhFrac = [&] (int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfrac[seedFracOffsets[seedIdx] + rhNum];
    };

    if (debug) {
        if (threadIdx.x == 0) {
            printf("\n===========================================================================================\n");
            printf("Processing topo cluster %d with nSeeds = %d nRHTopo = %d and seeds (", topoId, nSeeds, nRHTopo);
            for (int s = 0; s < nSeeds; s++) {
                if (s != 0) printf(", ");
                printf("%d", getSeedRhIdx(s));
            }
            if (nRHTopo == nSeeds) {
                printf(")\n\n");
            }
            else {
                printf(") and other rechits (");
                for (int r = 1; r < nRHNotSeed; r++) {
                    if (r != 1) printf(", ");
                    printf("%d", getRhFracIdx(0, r));
                }
                printf(")\n\n");
            }
        }
        __syncthreads();
    }
    


    auto computeClusterPos = [&] (float4& pos4, float _frac, int rhInd, bool isDebug) {
        float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
        float threshold = 0.0;
        if(pfrh_layer[rhInd] == PFLayer::HCAL_BARREL1) {
            threshold = recHitEnergyNormInvEB_vec[pfrh_depth[rhInd] - 1]; // This number needs to be inverted
        }
        else if (pfrh_layer[rhInd] == PFLayer::HCAL_ENDCAP) { threshold = recHitEnergyNormInvEE_vec[pfrh_depth[rhInd] - 1]; }

        const auto rh_energy = pfrh_energy[rhInd] * _frac;
        const auto norm =
            (_frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * threshold)));
        if (isDebug)
            printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n", rhInd, norm, _frac, rh_energy, rechitPos.x, rechitPos.y, rechitPos.z);
        
        atomicAdd(&pos4.x, rechitPos.x * norm);
        atomicAdd(&pos4.y, rechitPos.y * norm);
        atomicAdd(&pos4.z, rechitPos.z * norm);
        atomicAdd(&pos4.w, norm);   // position_norm
//        pos4.x += rechitPos.x * norm;
//        pos4.y += rechitPos.y * norm;
//        pos4.z += rechitPos.z * norm;
//        pos4.w += norm;     //  position_norm
    };
    

    // Set initial cluster position (energy) to seed rechit position (energy)
    for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
        int i = getSeedRhIdx(s);
        clusterPos[i] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
        prevClusterPos[i] = clusterPos[i];
        clusterEnergy[i] = pfrh_energy[i];
    }
    __syncthreads();
    
    while (notDone) {
        if (debug && threadIdx.x == 0) {
            printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
        }

        // Reset fracSum and rhCount
        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {
            int j = getRhFracIdx(0, r);
            fracSum[j] = 0.;
            rhCount[j] = 1;

            for (int s = 0; s < nSeeds; s++) {
                int i = getSeedRhIdx(s);
                pcrhfrac[seedFracOffsets[i] + r] = -1.;
            }
        }
        __syncthreads();

        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {    // One thread for each (non-seed) rechit 
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = getSeedRhIdx(s);
                int j = getRhFracIdx(s, r);
                
                if (debug) {
                    printf("\tCluster %d (seed %d) has position: (%.4f, %.4f, %4f)\n", s, i, clusterPos[i].x, clusterPos[i].y, clusterPos[i].z);
                }

                float dist2 =
                   (clusterPos[i].x - pfrh_x[j])*(clusterPos[i].x - pfrh_x[j])
                  +(clusterPos[i].y - pfrh_y[j])*(clusterPos[i].y - pfrh_y[j])
                  +(clusterPos[i].z - pfrh_z[j])*(clusterPos[i].z - pfrh_z[j]);
                
                float d2 = dist2 / showerSigma2;
                float fraction = -1.;
                
                if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                if( pfrh_isSeed[j]!=1) {
                    atomicAdd(&fracSum[j],fraction);
                }
            }
        }
        __syncthreads();
        
        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {    // One thread for each (non-seed) rechit 
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = getSeedRhIdx(s); 
                int j = getRhFracIdx(s, r);
                
                if( pfrh_isSeed[j]!=1 ){
                    float dist2 =
                       (clusterPos[i].x - pfrh_x[j])*(clusterPos[i].x - pfrh_x[j])
                      +(clusterPos[i].y - pfrh_y[j])*(clusterPos[i].y - pfrh_y[j])
                      +(clusterPos[i].z - pfrh_z[j])*(clusterPos[i].z - pfrh_z[j]);
                    
                    float d2 = dist2 / showerSigma2;
                    float fraction = -1.;
                    
                    if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                    else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                    if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                    if (fracSum[j] > minFracTot) {
                        float fracpct = fraction / fracSum[j];
                        if(fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
                            pcrhfrac[seedFracOffsets[i]+r] = fracpct;
                        }
                        else {
                            pcrhfrac[seedFracOffsets[i]+r] = -1;
                        }
                    }
                    else {
                        pcrhfrac[seedFracOffsets[i]+r] = -1;
                    }
                }
            }
        }
        __syncthreads();
    if (!noPosCalc) {
        if (debug && threadIdx.x == 0)
            printf("Computing cluster position for topoId %d\n", topoId);
        
        // Reset cluster position and energy
        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);
            clusterPos[i] = make_float4(0.0, 0.0, 0.0, 0.0);
            clusterEnergy[i] = 0;
        }
        __syncthreads();

        // Recalculate position
        for (int r = threadIdx.x; r < nRHNotSeed; r += gridStride) {    // One thread for each (non-seed) rechit 
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = getSeedRhIdx(s);    // Seed index
                
                if (debug) {
                    printf("\tNow on seed %d\t\tneigh4Ind = [", i);
                    for(int k=0; k<nNeigh; k++){
                        if (k != 0) printf(", ");
                        printf("%d", neigh4_Ind[nNeigh*i+k]);
                    }
                    printf("]\n");
                
                }

                // Calculate cluster energy by summing rechit fractional energies
                int j = getRhFracIdx(s,r);
                float frac = getRhFrac(s,r); 
                
                if (frac > -0.5) {
                    //if (debug)
                        //printf("\t\tRechit %d (position %d) in this PF cluster with frac = %f\n", j, _n, frac);
                    //clusterEnergy[i] += frac * pfrh_energy[j];
                    atomicAdd(&clusterEnergy[i], frac * pfrh_energy[j]);

                    bool updateClusterPos = false; 
                    if (nSeeds == 1) {
                        if (debug)
                            printf("\t\tThis topo cluster has a single seed.\n");
                        //computeClusterPos(clusterPos[i], frac, j, debug);
                        updateClusterPos = true;
                    }
                    else {
                        if (j == i) {
                            // This is the seed
                            //computeClusterPos(clusterPos[i], frac, j, debug);
                            updateClusterPos = true;
                        }
                        else {
                            // Check if this is one of the neighboring rechits
                            for(int k=0; k<nNeigh; k++){
                                if(neigh4_Ind[nNeigh*i+k]<0) continue;
                                if(neigh4_Ind[nNeigh*i+k] == j) {
                                    // Found it
                                    if (debug) printf("\t\tRechit %d is one of the 4 neighbors of seed %d\n", j, i);
                                    //computeClusterPos(clusterPos[i], frac, j, debug);
                                    updateClusterPos = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (updateClusterPos) computeClusterPos(clusterPos[i], frac, j, debug);
                    
                }
                //else if (debug)
                //    printf("Can't find rechit fraction for cluster %d (seed %d) rechit %d!\n", s, i, j);
                    
            }
        }
        __syncthreads();
        
        // Normalize the seed postiions
        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);    // Seed index
            if (clusterPos[i].w >= minAllowedNormalization)
            {
                // Divide by position norm
                clusterPos[i].x /= clusterPos[i].w;
                clusterPos[i].y /= clusterPos[i].w;
                clusterPos[i].z /= clusterPos[i].w;

                if (debug)
                    printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n", s, i, clusterEnergy[i], clusterPos[i].x, clusterPos[i].y, clusterPos[i].z);
            }
            else {
                if (debug)
                    printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n", s, i, clusterPos[i].w, minAllowedNormalization);
                clusterPos[i].x = 0.0;
                clusterPos[i].y = 0.0;
                clusterPos[i].z = 0.0;
                //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
                
            }
        }

        // Reset diff2
        if (threadIdx.x == 0) {
            diff2 = -1.;
        }
        __syncthreads();
       
        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);
            float delta2 = dR2(prevClusterPos[i], clusterPos[i]);
            if (debug) printf("\tCluster %d (seed %d) has delta2 = %f\n", s, i, delta2);
            atomicMaxF(&diff2, delta2); 
//            if (delta2 > diff2) {
//                diff2 = delta2;
//                if (debug) printf("\t\tNew diff2 = %f\n", diff2);
//            }

            prevClusterPos[i] = clusterPos[i];  // Save clusterPos 
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            diff = sqrtf(diff2);
            iter++;
            notDone = (diff > tol) && (iter < maxIterations);
            if (debug) {
                if (diff > tol)
                    printf("\tTopoId %d has diff = %f greater than tolerance %f (continuing)\n", topoId, diff, tol);
                else
                    if (debug) printf("\tTopoId %d has diff = %f LESS than tolerance %f (terminating!)\n", topoId, diff, tol);
            }
        }
        __syncthreads();
     }  // if (!noPosCalc)
     else {
        if (threadIdx.x == 0) notDone = false;
        __syncthreads();
     }
    }
    if (threadIdx.x == 0) pfcIter[topoId] = iter;
  }
  else if (threadIdx.x == 0 && (topoRHCount[topoId] == 1 || (topoRHCount[topoId] > 1 && topoRHCount[topoId] == topoSeedCount[topoId]))) {
    // Single rh cluster or all rechits in this topo cluster are seeds. No iterations needed
    pfcIter[topoId] = 0;
  }
}


__global__ void hcalFastCluster_withLambdas(size_t nRH,
                                const float* __restrict__ pfrh_x,
                                const float* __restrict__ pfrh_y,
                                const float* __restrict__ pfrh_z,
                                const float* __restrict__ pfrh_energy,
                                int* pfrh_topoId,
                                int* pfrh_isSeed,
                                const int* __restrict__ pfrh_layer,
                                const int* __restrict__ pfrh_depth,
                                const int* __restrict__ neigh4_Ind,
                                float* pcrhfrac,
                                int* pcrhfracind,
                                float* fracSum,
                                int* rhCount,
                                int* topoSeedCount,
                                int* topoRHCount,
                                int* seedFracOffsets,
                                int* topoSeedOffsets,
                                int* topoSeedList,
                                float4* clusterPos,
                                float4* prevClusterPos,
                                float* clusterEnergy,
                                int* pfcIter
                                ) {

  int topoId = blockIdx.x;
  if (topoId < nRH && topoRHCount[topoId] > 1 && topoSeedCount[topoId] > 0 && topoRHCount[topoId] != topoSeedCount[topoId]) {
    //printf("Now on topoId %d\tthreadIdx.x = %d\n", topoId, threadIdx.x);
    __shared__ int nSeeds, nRHTopo, nRHNotSeed, topoSeedBegin, gridStride, iter;
    __shared__ float tol, diff, diff2;
    __shared__ bool notDone, debug, noPosCalc;
    if (threadIdx.x == 0) {
        nSeeds = topoSeedCount[topoId];
        nRHTopo = topoRHCount[topoId];
        nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
        topoSeedBegin = topoSeedOffsets[topoId];
        tol = stoppingTolerance * powf(fmaxf(1.0, nSeeds - 1.0), 2.0);     // stopping tolerance * tolerance scaling
        gridStride = blockDim.x * gridDim.x;
        iter = 0;
        notDone = true;
        debug = false;
        noPosCalc = false; 
        //debug = (topoId == 432 || topoId == 438 || topoId == 439) ? true : false;
    }
    __syncthreads();

    auto getSeedRhIdx = [&] (int seedNum) {
        if (seedNum > topoSeedCount[topoId]) {
            printf("PROBLEM with seedNum = %d > nSeeds = %d", seedNum, nSeeds); 
            return -1;
        }
        return topoSeedList[topoSeedBegin + seedNum];
    };

    auto getRhFracIdx = [&] (int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfracind[seedFracOffsets[seedIdx] + rhNum];
    };
    
    auto getRhFrac = [&] (int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfrac[seedFracOffsets[seedIdx] + rhNum];
    };

    if (debug) {
        if (threadIdx.x == 0) {
            printf("\n===========================================================================================\n");
            printf("Processing topo cluster %d with nSeeds = %d nRHTopo = %d and seeds (", topoId, nSeeds, nRHTopo);
            for (int s = 0; s < nSeeds; s++) {
                if (s != 0) printf(", ");
                printf("%d", getSeedRhIdx(s));
            }
            if (nRHTopo == nSeeds) {
                printf(")\n\n");
            }
            else {
                printf(") and other rechits (");
                for (int r = 1; r < nRHNotSeed; r++) {
                    if (r != 1) printf(", ");
                    printf("%d", getRhFracIdx(0, r));
                }
                printf(")\n\n");
            }
        }
        __syncthreads();
    }
    


    auto computeClusterPos = [&] (float4& pos4, float _frac, int rhInd, bool isDebug) {
        float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
        float threshold = 0.0;
        if(pfrh_layer[rhInd] == PFLayer::HCAL_BARREL1) {
            threshold = recHitEnergyNormInvEB_vec[pfrh_depth[rhInd] - 1]; // This number needs to be inverted
        }
        else if (pfrh_layer[rhInd] == PFLayer::HCAL_ENDCAP) { threshold = recHitEnergyNormInvEE_vec[pfrh_depth[rhInd] - 1]; }

        const auto rh_energy = pfrh_energy[rhInd] * _frac;
        const auto norm =
            (_frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * threshold)));
        if (isDebug)
            printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n", rhInd, norm, _frac, rh_energy, rechitPos.x, rechitPos.y, rechitPos.z);
        
        atomicAdd(&pos4.x, rechitPos.x * norm);
        atomicAdd(&pos4.y, rechitPos.y * norm);
        atomicAdd(&pos4.z, rechitPos.z * norm);
        atomicAdd(&pos4.w, norm);   // position_norm
//        pos4.x += rechitPos.x * norm;
//        pos4.y += rechitPos.y * norm;
//        pos4.z += rechitPos.z * norm;
//        pos4.w += norm;     //  position_norm
    };
    

    // Set initial cluster position (energy) to seed rechit position (energy)
    for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
        int i = getSeedRhIdx(s);
        clusterPos[i] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
        prevClusterPos[i] = clusterPos[i];
        clusterEnergy[i] = pfrh_energy[i];
    }
    __syncthreads();
    
    while (notDone) {
        if (debug && threadIdx.x == 0) {
            printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
        }

        // Reset fracSum and rhCount
        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {
            int j = getRhFracIdx(0, r);
            fracSum[j] = 0.;
            rhCount[j] = 1;

            for (int s = 0; s < nSeeds; s++) {
                int i = getSeedRhIdx(s);
                pcrhfrac[seedFracOffsets[i] + r] = -1.;
            }
        }
        __syncthreads();

        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {    // One thread for each (non-seed) rechit 
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = getSeedRhIdx(s);
                int j = getRhFracIdx(s, r);
                
                if (debug) {
                    printf("\tCluster %d (seed %d) has position: (%.4f, %.4f, %4f)\n", s, i, clusterPos[i].x, clusterPos[i].y, clusterPos[i].z);
                }

                float dist2 =
                   (clusterPos[i].x - pfrh_x[j])*(clusterPos[i].x - pfrh_x[j])
                  +(clusterPos[i].y - pfrh_y[j])*(clusterPos[i].y - pfrh_y[j])
                  +(clusterPos[i].z - pfrh_z[j])*(clusterPos[i].z - pfrh_z[j]);
                
                float d2 = dist2 / showerSigma2;
                float fraction = -1.;
                
                if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                if( pfrh_isSeed[j]!=1) {
                    atomicAdd(&fracSum[j],fraction);
                }
            }
        }
        __syncthreads();
        
        for (int r = threadIdx.x + 1; r < nRHNotSeed; r += gridStride) {    // One thread for each (non-seed) rechit 
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = getSeedRhIdx(s); 
                int j = getRhFracIdx(s, r);
                
                if( pfrh_isSeed[j]!=1 ){
                    float dist2 =
                       (clusterPos[i].x - pfrh_x[j])*(clusterPos[i].x - pfrh_x[j])
                      +(clusterPos[i].y - pfrh_y[j])*(clusterPos[i].y - pfrh_y[j])
                      +(clusterPos[i].z - pfrh_z[j])*(clusterPos[i].z - pfrh_z[j]);
                    
                    float d2 = dist2 / showerSigma2;
                    float fraction = -1.;
                    
                    if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                    else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                    if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                    if (fracSum[j] > minFracTot) {
                        float fracpct = fraction / fracSum[j];
                        if(fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
                            pcrhfrac[seedFracOffsets[i]+r] = fracpct;
                        }
                        else {
                            pcrhfrac[seedFracOffsets[i]+r] = -1;
                        }
                    }
                    else {
                        pcrhfrac[seedFracOffsets[i]+r] = -1;
                    }
                }
            }
        }
        __syncthreads();
    if (!noPosCalc) {
        if (debug && threadIdx.x == 0)
            printf("Computing cluster position for topoId %d\n", topoId);
        
        // Reset cluster position and energy
        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);
            clusterPos[i] = make_float4(0.0, 0.0, 0.0, 0.0);
            clusterEnergy[i] = 0;
        }
        __syncthreads();

        // Recalculate position
        for (int r = threadIdx.x; r < nRHNotSeed; r += gridStride) {    // One thread for each (non-seed) rechit 
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = getSeedRhIdx(s);    // Seed index
                
                if (debug) {
                    printf("\tNow on seed %d\t\tneigh4Ind = [", i);
                    for(int k=0; k<nNeigh; k++){
                        if (k != 0) printf(", ");
                        printf("%d", neigh4_Ind[nNeigh*i+k]);
                    }
                    printf("]\n");
                
                }

                // Calculate cluster energy by summing rechit fractional energies
                int j = getRhFracIdx(s,r);
                float frac = getRhFrac(s,r); 
                
                if (frac > -0.5) {
                    //if (debug)
                        //printf("\t\tRechit %d (position %d) in this PF cluster with frac = %f\n", j, _n, frac);
                    //clusterEnergy[i] += frac * pfrh_energy[j];
                    atomicAdd(&clusterEnergy[i], frac * pfrh_energy[j]);

                    if (nSeeds == 1) {
                        if (debug)
                            printf("\t\tThis topo cluster has a single seed.\n");
                        computeClusterPos(clusterPos[i], frac, j, debug);
                    }
                    else {
                        if (j == i) {
                            // This is the seed
                            computeClusterPos(clusterPos[i], frac, j, debug);
                        }
                        else {
                            // Check if this is one of the neighboring rechits
                            for(int k=0; k<nNeigh; k++){
                                if(neigh4_Ind[nNeigh*i+k]<0) continue;
                                if(neigh4_Ind[nNeigh*i+k] == j) {
                                    // Found it
                                    if (debug) printf("\t\tRechit %d is one of the 4 neighbors of seed %d\n", j, i);
                                    computeClusterPos(clusterPos[i], frac, j, debug);
                                    break;
                                }
                            }
                        }
                    }
                }
                //else if (debug)
                //    printf("Can't find rechit fraction for cluster %d (seed %d) rechit %d!\n", s, i, j);
                    
            }
        }
        __syncthreads();
        
        // Normalize the seed postiions
        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);    // Seed index
            if (clusterPos[i].w >= minAllowedNormalization)
            {
                // Divide by position norm
                clusterPos[i].x /= clusterPos[i].w;
                clusterPos[i].y /= clusterPos[i].w;
                clusterPos[i].z /= clusterPos[i].w;

                if (debug)
                    printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n", s, i, clusterEnergy[i], clusterPos[i].x, clusterPos[i].y, clusterPos[i].z);
            }
            else {
                if (debug)
                    printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n", s, i, clusterPos[i].w, minAllowedNormalization);
                clusterPos[i].x = 0.0;
                clusterPos[i].y = 0.0;
                clusterPos[i].z = 0.0;
                //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
                
            }
        }

        // Reset diff2
        if (threadIdx.x == 0) {
            diff2 = -1.;
        }
        __syncthreads();
       
        for (int s = threadIdx.x; s < nSeeds; s += gridStride) {
            int i = getSeedRhIdx(s);
            float delta2 = dR2(prevClusterPos[i], clusterPos[i]);
            if (debug) printf("\tCluster %d (seed %d) has delta2 = %f\n", s, i, delta2);
            atomicMaxF(&diff2, delta2); 
//            if (delta2 > diff2) {
//                diff2 = delta2;
//                if (debug) printf("\t\tNew diff2 = %f\n", diff2);
//            }

            prevClusterPos[i] = clusterPos[i];  // Save clusterPos 
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            diff = sqrtf(diff2);
            iter++;
            notDone = (diff > tol) && (iter < maxIterations);
            if (debug) {
                if (diff > tol)
                    printf("\tTopoId %d has diff = %f greater than tolerance %f (continuing)\n", topoId, diff, tol);
                else
                    if (debug) printf("\tTopoId %d has diff = %f LESS than tolerance %f (terminating!)\n", topoId, diff, tol);
            }
        }
        __syncthreads();
     }  // if (!noPosCalc)
     else {
        if (threadIdx.x == 0) notDone = false;
        __syncthreads();
     }
    }
    if (threadIdx.x == 0) pfcIter[topoId] = iter;
  }
  else if (threadIdx.x == 0 && (topoRHCount[topoId] == 1 || (topoRHCount[topoId] > 1 && topoRHCount[topoId] == topoSeedCount[topoId]))) {
    // Single rh cluster or all rechits in this topo cluster are seeds. No iterations needed
    pfcIter[topoId] = 0;
  }
}


__global__ void hcalFastCluster_topoBlocks(size_t nRH,
                                const float* __restrict__ pfrh_x,
                                const float* __restrict__ pfrh_y,
                                const float* __restrict__ pfrh_z,
                                const float* __restrict__ pfrh_energy,
                                int* pfrh_topoId,
                                int* pfrh_isSeed,
                                const int* __restrict__ pfrh_layer,
                                const int* __restrict__ pfrh_depth,
                                const int* __restrict__ neigh4_Ind,
                                float* pcrhfrac,
                                int* pcrhfracind,
                                float* fracSum,
                                int* rhCount,
                                int* topoSeedCount,
                                int* topoRHCount,
                                int* seedFracOffsets,
                                int* topoSeedOffsets,
                                int* topoSeedList,
                                float4* clusterPos,
                                float4* prevClusterPos,
                                float* clusterEnergy
                                ) {

  int topoId = blockIdx.x;
  if (topoId < nRH) {
    //printf("Now on topoId %d\tthreadIdx.x = %d\n", topoId, threadIdx.x);
    int nSeeds = topoSeedCount[topoId];
    if (nSeeds < 1) return;  // No seeds found for this topoId. Skip it
    int topoSeedBegin = topoSeedOffsets[topoId];

    int nRHTopo = topoRHCount[topoId];
    int iter = 0;
    
    auto getSeedRhIdx = [&] (int seedNum) {
        if (seedNum > topoSeedCount[topoId]) {
            printf("PROBLEM with seedNum = %d > nSeeds = %d", seedNum, nSeeds); 
            return -1;
        }
        return topoSeedList[topoSeedBegin + seedNum];
    };

    auto getRhFracIdx = [&] (int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfracind[seedFracOffsets[seedIdx] + rhNum];
    };
    
    auto getRhFrac = [&] (int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfrac[seedFracOffsets[seedIdx] + rhNum];
    };

    //bool debug = true;
    bool debug = false;
    if (debug) {
        printf("\n===========================================================================================\n");
        printf("Processing topo cluster %d with nSeeds = %d nRHTopo = %d and seeds (", topoId, nSeeds, nRHTopo);
            for (int s = 0; s < nSeeds; s++) {
                if (s != 0) printf(", ");
                printf("%d", getSeedRhIdx(s));
            }
            if (nRHTopo == nSeeds) {
                printf(")\n\n");
            }
            else {
                printf(") and other rechits (");
                for (int r = 1; r < (nRHTopo-nSeeds+1); r++) {
                    if (r != 1) printf(", ");
                    printf("%d", getRhFracIdx(0, r));
                }
                printf(")\n\n");
            }
    }
    
    float tolScaling = powf(fmaxf(1.0, nSeeds - 1.0), 2.0);     // Tolerance scaling


    auto computeClusterPos = [&] (float4& pos4, float _frac, int rhInd, bool isDebug) {
        float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
        float threshold = 0.0;
        if(pfrh_layer[rhInd] == PFLayer::HCAL_BARREL1) {
            threshold = recHitEnergyNormInvEB_vec[pfrh_depth[rhInd] - 1]; // This number needs to be inverted
        }
        else if (pfrh_layer[rhInd] == PFLayer::HCAL_ENDCAP) { threshold = recHitEnergyNormInvEE_vec[pfrh_depth[rhInd] - 1]; }

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
    
    float diff = -1.0;
    while (iter < maxIterations) {
        if (debug) {
            printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
        }
        // Reset fracSum and rhCount
        for (int r = 1; r < (nRHTopo-nSeeds+1); r++) {
            int j = getRhFracIdx(0, r);
            fracSum[j] = 0.;
            rhCount[j] = 1;
        }
//        for (int r = 0; r < (int)nRH; r++) {
//            fracSum[r] = 0.0;
//            rhCount[r] = 1;
//        }

        for (int s = 0; s < nSeeds; s++) {      // PF clusters
            int i = getSeedRhIdx(s);
 
            if (iter == 0) {
                // Set initial cluster position to seed rechit position
                clusterPos[i] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
                prevClusterPos[i] = clusterPos[i];

                // Set initial cluster energy to seed energy
                clusterEnergy[i] = pfrh_energy[i];
            }
            else {
                prevClusterPos[i] = clusterPos[i];
                
                if (debug) {
                    printf("\tCluster %d (seed %d) has position: (%.4f, %.4f, %4f)\n", s, i, clusterPos[i].x, clusterPos[i].y, clusterPos[i].z);
                }

                // Reset cluster indices and fractions
                for (int n = (seedFracOffsets[i]+1); n < (seedFracOffsets[i] + topoRHCount[topoId]); n++) {
                    pcrhfrac[n] = -1.0;           
                }
            }
            for (int r = 1; r < (nRHTopo-nSeeds+1); r++) {
                int j = getRhFracIdx(s, r);
                
                float dist2 =
                   (clusterPos[i].x - pfrh_x[j])*(clusterPos[i].x - pfrh_x[j])
                  +(clusterPos[i].y - pfrh_y[j])*(clusterPos[i].y - pfrh_y[j])
                  +(clusterPos[i].z - pfrh_z[j])*(clusterPos[i].z - pfrh_z[j]);
                
                float d2 = dist2 / showerSigma2;
                float fraction = -1.;
                
                if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                if( pfrh_isSeed[j]!=1) {
                    atomicAdd(&fracSum[j],fraction);
                }
            }
        }
        for (int s = 0; s < nSeeds; s++) {      // PF clusters
            int i = getSeedRhIdx(s); 
            
            for (int r = 1; r < (nRHTopo-nSeeds+1); r++) {
                int j = getRhFracIdx(s, r);
                
                if( pfrh_isSeed[j]!=1 ){
                    float dist2 =
                       (clusterPos[i].x - pfrh_x[j])*(clusterPos[i].x - pfrh_x[j])
                      +(clusterPos[i].y - pfrh_y[j])*(clusterPos[i].y - pfrh_y[j])
                      +(clusterPos[i].z - pfrh_z[j])*(clusterPos[i].z - pfrh_z[j]);
                    
                    float d2 = dist2 / showerSigma2;
                    float fraction = -1.;
                    
                    if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                    else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                    if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                    if (fracSum[j] > minFracTot) {
                        float fracpct = fraction / fracSum[j];
                        if(fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
                            pcrhfrac[seedFracOffsets[i]+r] = fracpct;
                        }
                        else {
                            pcrhfrac[seedFracOffsets[i]+r] = -1;
                        }
                    }
                    else {
                        pcrhfrac[seedFracOffsets[i]+r] = -1;
                    }
                }
            }
        }

        if (debug)
            printf("Computing cluster position for topoId %d\n", topoId);
        // Recalculate position
        for (int s = 0; s < nSeeds; s++) {      // PF clusters
            int i = getSeedRhIdx(s);    // Seed index
            
            if (debug) {
                printf("\tNow on seed %d\t\tneigh4Ind = [", i);
                for(int k=0; k<nNeigh; k++){
                    if (k != 0) printf(", ");
                    printf("%d", neigh4_Ind[nNeigh*i+k]);
                }
                printf("]\n");
            
            }
            // Zero out cluster position and energy
            clusterPos[i] = make_float4(0.0, 0.0, 0.0, 0.0);
            clusterEnergy[i] = 0;

            // Calculate cluster energy by summing rechit fractional energies
            for (int r = 0; r < (nRHTopo-nSeeds+1); r++) {
                int j = getRhFracIdx(s,r);
                float frac = getRhFrac(s, r); 
                
                if (frac > -0.5) {
                    //if (debug)
                        //printf("\t\tRechit %d (position %d) in this PF cluster with frac = %f\n", j, _n, frac);
                    clusterEnergy[i] += frac * pfrh_energy[j];

                    if (nSeeds == 1) {
                        if (debug)
                            printf("\t\tThis topo cluster has a single seed.\n");
                        computeClusterPos(clusterPos[i], frac, j, debug);
                    }
                    else {
                        if (j == i) {
                            // This is the seed
                            computeClusterPos(clusterPos[i], frac, j, debug);
                        }
                        else {
                            // Check if this is one of the neighboring rechits
                            for(int k=0; k<nNeigh; k++){
                                if(neigh4_Ind[nNeigh*i+k]<0) continue;
                                if(neigh4_Ind[nNeigh*i+k] == j) {
                                    // Found it
                                    if (debug) printf("\t\tRechit %d is one of the 4 neighbors of seed %d\n", j, i);
                                    computeClusterPos(clusterPos[i], frac, j, debug);
                                }
                            }
                        }
                    }
                }
                //else if (debug)
                //    printf("Can't find rechit fraction for cluster %d (seed %d) rechit %d!\n", s, i, j);
                    
            }
            if (clusterPos[i].w >= minAllowedNormalization)
            {
                // Divide by position norm
                clusterPos[i].x /= clusterPos[i].w;
                clusterPos[i].y /= clusterPos[i].w;
                clusterPos[i].z /= clusterPos[i].w;

                if (debug)
                    printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n", s, i, clusterEnergy[i], clusterPos[i].x, clusterPos[i].y, clusterPos[i].z);
            }
            else {
                if (debug)
                    printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n", s, i, clusterPos[i].w, minAllowedNormalization);
                clusterPos[i].x = 0.0;
                clusterPos[i].y = 0.0;
                clusterPos[i].z = 0.0;
                //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
                
            }
        }
        
        float diff2 = 0.0;
        for (int s = 0; s < nSeeds; s++) {
            int i = getSeedRhIdx(s);
            float delta2 = dR2(prevClusterPos[i], clusterPos[i]);
            if (debug) printf("\tCluster %d (seed %d) has delta2 = %f\n", s, getSeedRhIdx(s), delta2);
            if (delta2 > diff2) {
                diff2 = delta2;
                if (debug) printf("\t\tNew diff2 = %f\n", diff2);
            }
        }
        
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


__global__ void hcalFastCluster_serialize(size_t nRH,
                                const float* __restrict__ pfrh_x,
                                const float* __restrict__ pfrh_y,
                                const float* __restrict__ pfrh_z,
                                const float* __restrict__ pfrh_energy,
                                int* pfrh_topoId,
                                int* pfrh_isSeed,
                                const int* __restrict__ pfrh_layer,
                                const int* __restrict__ pfrh_depth,
                                const int* __restrict__ neigh4_Ind,
                                float* pcrhfrac,
                                int* pcrhfracind,
                                float* fracSum,
                                int* rhCount,
                                int* topoSeedCount,
                                int* topoRHCount,
                                int* seedFracOffsets,
                                int* topoSeedOffsets,
                                int* topoSeedList,
                                float4* clusterPos,
                                float4* prevClusterPos,
                                float* clusterEnergy
                                ) {

  for (int topoId = 0; topoId < (int)nRH; topoId++) {
    int nSeeds = topoSeedCount[topoId];
    if (nSeeds < 1) continue;  // No seeds found for this topoId. Skip it
    int topoSeedBegin = topoSeedOffsets[topoId];

    int nRHTopo = topoRHCount[topoId];
    int iter = 0;
    
    auto getSeedRhIdx = [&] (int seedNum) {
        if (seedNum > topoSeedCount[topoId]) {
            printf("PROBLEM with seedNum = %d > nSeeds = %d", seedNum, nSeeds); 
            return -1;
        }
        return topoSeedList[topoSeedBegin + seedNum];
    };

    auto getRhFracIdx = [&] (int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfracind[seedFracOffsets[seedIdx] + rhNum];
    };
    
    auto getRhFrac = [&] (int seedNum, int rhNum) {
        int seedIdx = topoSeedList[topoSeedBegin + seedNum];
        return pcrhfrac[seedFracOffsets[seedIdx] + rhNum];
    };

    //bool debug = true;
    bool debug = false;
    if (debug) {
        printf("\n===========================================================================================\n");
        printf("Processing topo cluster %d with nSeeds = %d nRHTopo = %d and seeds (", topoId, nSeeds, nRHTopo);
            for (int s = 0; s < nSeeds; s++) {
                if (s != 0) printf(", ");
                printf("%d", getSeedRhIdx(s));
            }
            if (nRHTopo == nSeeds) {
                printf(")\n\n");
            }
            else {
                printf(") and other rechits (");
                for (int r = 1; r < (nRHTopo-nSeeds+1); r++) {
                    if (r != 1) printf(", ");
                    printf("%d", getRhFracIdx(0, r));
                }
                printf(")\n\n");
            }
    }
    
    float tolScaling = powf(fmaxf(1.0, nSeeds - 1.0), 2.0);     // Tolerance scaling


    auto computeClusterPos = [&] (float4& pos4, float _frac, int rhInd, bool isDebug) {
        float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
        float threshold = 0.0;
        if(pfrh_layer[rhInd] == PFLayer::HCAL_BARREL1) {
            threshold = recHitEnergyNormInvEB_vec[pfrh_depth[rhInd] - 1]; // This number needs to be inverted
        }
        else if (pfrh_layer[rhInd] == PFLayer::HCAL_ENDCAP) { threshold = recHitEnergyNormInvEE_vec[pfrh_depth[rhInd] - 1]; }

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
            int i = getSeedRhIdx(s);
 
            if (iter == 0) {
                // Set initial cluster position to seed rechit position
                clusterPos[i] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
                prevClusterPos[i] = clusterPos[i];

                // Set initial cluster energy to seed energy
                clusterEnergy[i] = pfrh_energy[i];
            }
            else {
                prevClusterPos[i] = clusterPos[i];
                
                if (debug) {
                    printf("\tCluster %d (seed %d) has position: (%.4f, %.4f, %4f)\n", s, i, clusterPos[i].x, clusterPos[i].y, clusterPos[i].z);
                }

                // Reset cluster indices and fractions
                for (int n = (seedFracOffsets[i]+1); n < (seedFracOffsets[i] + topoRHCount[topoId]); n++) {
                    pcrhfrac[n] = -1.0;           
                }
            }
            for (int r = 1; r < (nRHTopo-nSeeds+1); r++) {
                int j = getRhFracIdx(s, r);
                
                float dist2 =
                   (clusterPos[i].x - pfrh_x[j])*(clusterPos[i].x - pfrh_x[j])
                  +(clusterPos[i].y - pfrh_y[j])*(clusterPos[i].y - pfrh_y[j])
                  +(clusterPos[i].z - pfrh_z[j])*(clusterPos[i].z - pfrh_z[j]);
                
                float d2 = dist2 / showerSigma2;
                float fraction = -1.;
                
                if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                if( pfrh_isSeed[j]!=1) {
                    atomicAdd(&fracSum[j],fraction);
                }
            }
        }
        for (int s = 0; s < nSeeds; s++) {      // PF clusters
            int i = getSeedRhIdx(s); 
            
            for (int r = 1; r < (nRHTopo-nSeeds+1); r++) {
                int j = getRhFracIdx(s, r);
                
                if( pfrh_isSeed[j]!=1 ){
                    float dist2 =
                       (clusterPos[i].x - pfrh_x[j])*(clusterPos[i].x - pfrh_x[j])
                      +(clusterPos[i].y - pfrh_y[j])*(clusterPos[i].y - pfrh_y[j])
                      +(clusterPos[i].z - pfrh_z[j])*(clusterPos[i].z - pfrh_z[j]);
                    
                    float d2 = dist2 / showerSigma2;
                    float fraction = -1.;
                    
                    if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                    else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                    if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                    if (fracSum[j] > minFracTot) {
                        float fracpct = fraction / fracSum[j];
                        if(fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep)) {
                            pcrhfrac[seedFracOffsets[i]+r] = fracpct;
                        }
                        else {
                            pcrhfrac[seedFracOffsets[i]+r] = -1;
                        }
                    }
                    else {
                        pcrhfrac[seedFracOffsets[i]+r] = -1;
                    }
                }
            }
        }

        if (debug)
            printf("Computing cluster position for topoId %d\n", topoId);
        // Recalculate position
        for (int s = 0; s < nSeeds; s++) {      // PF clusters
            int i = getSeedRhIdx(s);    // Seed index
            
            if (debug) {
                printf("\tNow on seed %d\t\tneigh4Ind = [", i);
                for(int k=0; k<nNeigh; k++){
                    if (k != 0) printf(", ");
                    printf("%d", neigh4_Ind[nNeigh*i+k]);
                }
                printf("]\n");
            
            }
            // Zero out cluster position and energy
            clusterPos[i] = make_float4(0.0, 0.0, 0.0, 0.0);
            clusterEnergy[i] = 0;

            // Calculate cluster energy by summing rechit fractional energies
            for (int r = 0; r < (nRHTopo-nSeeds+1); r++) {
                int j = getRhFracIdx(s,r);
                float frac = getRhFrac(s, r); 
                
                if (frac > -0.5) {
                    //if (debug)
                        //printf("\t\tRechit %d (position %d) in this PF cluster with frac = %f\n", j, _n, frac);
                    clusterEnergy[i] += frac * pfrh_energy[j];

                    if (nSeeds == 1) {
                        if (debug)
                            printf("\t\tThis topo cluster has a single seed.\n");
                        computeClusterPos(clusterPos[i], frac, j, debug);
                    }
                    else {
                        if (j == i) {
                            // This is the seed
                            computeClusterPos(clusterPos[i], frac, j, debug);
                        }
                        else {
                            // Check if this is one of the neighboring rechits
                            for(int k=0; k<nNeigh; k++){
                                if(neigh4_Ind[nNeigh*i+k]<0) continue;
                                if(neigh4_Ind[nNeigh*i+k] == j) {
                                    // Found it
                                    if (debug) printf("\t\tRechit %d is one of the 4 neighbors of seed %d\n", j, i);
                                    computeClusterPos(clusterPos[i], frac, j, debug);
                                }
                            }
                        }
                    }
                }
                //else if (debug)
                //    printf("Can't find rechit fraction for cluster %d (seed %d) rechit %d!\n", s, i, j);
                    
            }
            if (clusterPos[i].w >= minAllowedNormalization)
            {
                // Divide by position norm
                clusterPos[i].x /= clusterPos[i].w;
                clusterPos[i].y /= clusterPos[i].w;
                clusterPos[i].z /= clusterPos[i].w;

                if (debug)
                    printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n", s, i, clusterEnergy[i], clusterPos[i].x, clusterPos[i].y, clusterPos[i].z);
            }
            else {
                if (debug)
                    printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n", s, i, clusterPos[i].w, minAllowedNormalization);
                clusterPos[i].x = 0.0;
                clusterPos[i].y = 0.0;
                clusterPos[i].z = 0.0;
                //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
                
            }
        }
        
        float diff2 = 0.0;
        for (int s = 0; s < nSeeds; s++) {
            int i = getSeedRhIdx(s);
            float delta2 = dR2(prevClusterPos[i], clusterPos[i]);
            if (debug) printf("\tCluster %d (seed %d) has delta2 = %f\n", s, getSeedRhIdx(s), delta2);
            if (delta2 > diff2) {
                diff2 = delta2;
                if (debug) printf("\t\tNew diff2 = %f\n", diff2);
            }
        }
        
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
    if (iter > 1) {
        if (iter >= maxIterations) printf("topoId %d (nSeeds = %d  nRHTopo = %d) hit maxIterations (%d) with diff (%f) > tol (%f)\n", topoId, nSeeds, nRHTopo, iter, diff, stoppingTolerance * tolScaling);
        else printf("topoId %d converged in %d iterations\n", topoId, iter);
    }
  }
}


__global__ void hcalFastCluster_old_serialize(size_t nRH,
                                const float* __restrict__ pfrh_x,
                                const float* __restrict__ pfrh_y,
                                const float* __restrict__ pfrh_z,
                                const float* __restrict__ pfrh_energy,
                                int* pfrh_topoId,
                                int* pfrh_isSeed,
                                const int* __restrict__ pfrh_layer,
                                const int* __restrict__ pfrh_depth,
                                const int* __restrict__ neigh4_Ind,
                                float* pcrhfrac,
                                int* pcrhfracind,
                                float* fracSum,
                                int* rhCount,
                                int* topoSeedCount,
                                int* topoRHCount,
                                int* seedFracOffsets
                                ) {

  for (int topoId = 0; topoId < (int)nRH; topoId++) {
    int iter = 0;
    //int nSeeds = topoSeedCount[topoId];
    //int nRHTopo = topoRHCount[topoId];
    int nSeeds = 0;
    int nRHTopo = 0;
    if (topoId >-1 && topoId < nRH) {
        int seeds[75];
        int rechits[250];
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

        if (nSeeds != topoSeedCount[topoId]) printf("WARNING: nSeeds (%d) doesn't match topoSeedCount (%d)!\n", nSeeds, topoSeedCount[topoId]);
        if (nRHTopo != topoRHCount[topoId]) printf("WARNING: nRHTopo (%d) doesn't match topoRHCount (%d)!\n", nRHTopo, topoRHCount[topoId]);
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

        float tolScaling = powf(fmaxf(1.0, nSeeds - 1.0), 2.0);     // Tolerance scaling

        float4 prevClusterPos[75], clusterPos[75];  //  W component is position norm
        float clusterEnergy[75];

        auto computeClusterPos = [&] (float4& pos4, float _frac, int rhInd, bool isDebug) {
            float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
            float threshold = 0.0;
            if(pfrh_layer[rhInd] == PFLayer::HCAL_BARREL1) {
                threshold = recHitEnergyNormInvEB_vec[pfrh_depth[rhInd] - 1]; // This number needs to be inverted
            }
            else if (pfrh_layer[rhInd] == PFLayer::HCAL_ENDCAP) { threshold = recHitEnergyNormInvEE_vec[pfrh_depth[rhInd] - 1]; }

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
                    prevClusterPos[s] = clusterPos[s];

                    // Set initial cluster energy to seed energy
                    clusterEnergy[s] = pfrh_energy[i];
                    //prevClusterEnergy[s] = 0.0;
                }
                else {
                    prevClusterPos[s] = clusterPos[s];
                    //prevClusterEnergy[s] = clusterEnergy[s];
                    
                    if (debug) {
                        printf("\tCluster %d (seed %d) has position: (%.4f, %.4f, %4f)\n", s, i, clusterPos[s].x, clusterPos[s].y, clusterPos[s].z);
                    }

                    // Reset cluster indices and fractions
                    for (int _n = seedFracOffsets[i]; _n < (seedFracOffsets[i] + topoRHCount[topoId]); _n++) {
                        pcrhfrac[_n] = -1.0;           
                        pcrhfracind[_n] = -1.0;           
                    }
//                    for (int _n = i*maxSize; _n < (i+1)*maxSize; _n++) {
//                        pcrhfrac[_n] = -1.0;           
//                        pcrhfracind[_n] = -1.0;           
//                    }
                }
                for (int r = 0; r < nRHTopo; r++) {
                    int j = rechits[r];
                    if( pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i]==1 ){
                        float dist2 =
                           (clusterPos[s].x - pfrh_x[j])*(clusterPos[s].x - pfrh_x[j])
                          +(clusterPos[s].y - pfrh_y[j])*(clusterPos[s].y - pfrh_y[j])
                          +(clusterPos[s].z - pfrh_z[j])*(clusterPos[s].z - pfrh_z[j]);
                        /*
                        float dist2 =
                           (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
                          +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
                          +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);
                        */
                        float d2 = dist2 / showerSigma2;
                        float fraction = -1.;
                        
                        if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[s] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                        else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[s] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
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
                            pcrhfrac[seedFracOffsets[i]]    = 1.;
                            pcrhfracind[seedFracOffsets[i]] = j;
//                            pcrhfrac[i*maxSize]    = 1.;
//                            pcrhfracind[i*maxSize] = j;
                        }
                        if( pfrh_isSeed[j]!=1 ){
                            float dist2 =
                               (clusterPos[s].x - pfrh_x[j])*(clusterPos[s].x - pfrh_x[j])
                              +(clusterPos[s].y - pfrh_y[j])*(clusterPos[s].y - pfrh_y[j])
                              +(clusterPos[s].z - pfrh_z[j])*(clusterPos[s].z - pfrh_z[j]);
                            /*
                            float dist2 =
                               (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
                              +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
                              +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);
                            */
                            float d2 = dist2 / showerSigma2;
                            float fraction = -1.;
                            
                            if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[s] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                            else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[s] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                            if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                            if (fracSum[j] > minFracTot) {
                                float fracpct = fraction / fracSum[j];
                                if(fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep))
                                  {
                                      int k = atomicAdd(&rhCount[i],1);
                                      pcrhfrac[seedFracOffsets[i]+k] = fracpct;
                                      pcrhfracind[seedFracOffsets[i]+k] = j;
//                                      pcrhfrac[i*maxSize+k] = fracpct;
//                                      pcrhfracind[i*maxSize+k] = j;
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
                    printf("\tNow on seed %d\t\tneigh4Ind = [", i);
                    for(int k=0; k<nNeigh; k++){
                        if (k != 0) printf(", ");
                        printf("%d", neigh4_Ind[nNeigh*i+k]);
                    }
                    printf("]\n");
                
                }
                // Zero out cluster position and energy
                clusterPos[s] = make_float4(0.0, 0.0, 0.0, 0.0);
                clusterEnergy[s] = 0;

                // Calculate cluster energy by summing rechit fractional energies
                for (int r = 0; r < nRHTopo; r++) {
                    int j = rechits[r];
                    float frac = -1.0;
                    int _n = -1;
                    if (j == i) {
                        // This is the seed
                        frac = 1.0;
                        _n = seedFracOffsets[i]; 
                    }
                    else {
                        for (_n = seedFracOffsets[i]; _n < (seedFracOffsets[i]+topoRHCount[topoId]); _n++) {
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

                        if (nSeeds == 1) {
                            if (debug)
                                printf("\t\tThis topo cluster has a single seed.\n");
                            computeClusterPos(clusterPos[s], frac, j, debug);
                        }
                        else {
                            if (j == i) {
                                // This is the seed
                                computeClusterPos(clusterPos[s], frac, j, debug);
                            }
                            else {
                                // Check if this is one of the neighboring rechits
                                for(int k=0; k<nNeigh; k++){
                                    if(neigh4_Ind[nNeigh*i+k]<0) continue;
                                    if(neigh4_Ind[nNeigh*i+k] == j) {
                                        // Found it
                                        if (debug) printf("\t\tRechit %d is one of the 4 neighbors of seed %d\n", j, i);
                                        computeClusterPos(clusterPos[s], frac, j, debug);
                                    }
                                }
                            }
                        }
                    }
                    //else if (debug)
                    //    printf("Can't find rechit fraction for cluster %d (seed %d) rechit %d!\n", s, i, j);
                        
                }
                if (clusterPos[s].w >= minAllowedNormalization)
                {
                    // Divide by position norm
                    clusterPos[s].x /= clusterPos[s].w;
                    clusterPos[s].y /= clusterPos[s].w;
                    clusterPos[s].z /= clusterPos[s].w;

                    if (debug)
                        printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n", s, i, clusterEnergy[s], clusterPos[s].x, clusterPos[s].y, clusterPos[s].z);
                }
                else {
                    if (debug)
                        printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n", s, i, clusterPos[s].w, minAllowedNormalization);
                    clusterPos[s].x = 0.0;
                    clusterPos[s].y = 0.0;
                    clusterPos[s].z = 0.0;
                    //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
                    
                }
            }
            
            float diff2 = 0.0;
            for (int s = 0; s < nSeeds; s++) {
                float delta2 = dR2(prevClusterPos[s], clusterPos[s]);
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

__global__ void hcalFastCluster_older_serialize(size_t nRH,
                                const float* __restrict__ pfrh_x,
                                const float* __restrict__ pfrh_y,
                                const float* __restrict__ pfrh_z,
                                const float* __restrict__ pfrh_energy,
                                int* pfrh_topoId,
                                int* pfrh_isSeed,
                                const int* __restrict__ pfrh_layer,
                                const int* __restrict__ pfrh_depth,
                                const int* __restrict__ neigh4_Ind,
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
        int seeds[75];
        int rechits[250];
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

        float tolScaling = powf(fmaxf(1.0, nSeeds - 1.0), 2.0);     // Tolerance scaling

        float4 prevClusterPos[75], clusterPos[75];  //  W component is position norm
        float clusterEnergy[75];

        auto computeClusterPos = [&] (float4& pos4, float _frac, int rhInd, bool isDebug) {
            float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
            float threshold = 0.0;
            if(pfrh_layer[rhInd] == PFLayer::HCAL_BARREL1) {
                threshold = recHitEnergyNormInvEB_vec[pfrh_depth[rhInd] - 1]; // This number needs to be inverted
            }
            else if (pfrh_layer[rhInd] == PFLayer::HCAL_ENDCAP) { threshold = recHitEnergyNormInvEE_vec[pfrh_depth[rhInd] - 1]; }

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
                    prevClusterPos[s] = clusterPos[s];

                    // Set initial cluster energy to seed energy
                    clusterEnergy[s] = pfrh_energy[i];
                    //prevClusterEnergy[s] = 0.0;
                }
                else {
                    prevClusterPos[s] = clusterPos[s];
                    //prevClusterEnergy[s] = clusterEnergy[s];
                    
                    if (debug) {
                        printf("\tCluster %d (seed %d) has position: (%.4f, %.4f, %4f)\n", s, i, clusterPos[s].x, clusterPos[s].y, clusterPos[s].z);
                    }

                    // Reset cluster indices and fractions
                    for (int _n = i*100; _n < (i+1)*100; _n++) {
                        pcrhfrac[_n] = -1.0;           
                        pcrhfracind[_n] = -1.0;           
                    }
                }
                for (int r = 0; r < nRHTopo; r++) {
                    int j = rechits[r];
                    if( pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i]==1 ){
                        float dist2 =
                           (clusterPos[s].x - pfrh_x[j])*(clusterPos[s].x - pfrh_x[j])
                          +(clusterPos[s].y - pfrh_y[j])*(clusterPos[s].y - pfrh_y[j])
                          +(clusterPos[s].z - pfrh_z[j])*(clusterPos[s].z - pfrh_z[j]);
                        /*
                        float dist2 =
                           (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
                          +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
                          +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);
                        */
                        float d2 = dist2 / showerSigma2;
                        float fraction = -1.;
                        
                        if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[s] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                        else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[s] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
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
                            pcrhfrac[i*100]    = 1.;
                            pcrhfracind[i*100] = j;
                        }
                        if( pfrh_isSeed[j]!=1 ){
                            float dist2 =
                               (clusterPos[s].x - pfrh_x[j])*(clusterPos[s].x - pfrh_x[j])
                              +(clusterPos[s].y - pfrh_y[j])*(clusterPos[s].y - pfrh_y[j])
                              +(clusterPos[s].z - pfrh_z[j])*(clusterPos[s].z - pfrh_z[j]);
                            /*
                            float dist2 =
                               (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
                              +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
                              +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);
                            */
                            float d2 = dist2 / showerSigma2;
                            float fraction = -1.;
                            
                            if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[s] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                            else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[s] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                            if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                            if (fracSum[j] > minFracTot) {
                                float fracpct = fraction / fracSum[j];
                                if(fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep))
                                  {
                                      int k = atomicAdd(&rhCount[i],1);
                                      pcrhfrac[i*100+k] = fracpct;
                                      pcrhfracind[i*100+k] = j;
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
                    printf("\tNow on seed %d\t\tneigh4Ind = [", i);
                    for(int k=0; k<nNeigh; k++){
                        if (k != 0) printf(", ");
                        printf("%d", neigh4_Ind[nNeigh*i+k]);
                    }
                    printf("]\n");
                
                }
                // Zero out cluster position and energy
                clusterPos[s] = make_float4(0.0, 0.0, 0.0, 0.0);
                clusterEnergy[s] = 0;

                // Calculate cluster energy by summing rechit fractional energies
                for (int r = 0; r < nRHTopo; r++) {
                    int j = rechits[r];
                    float frac = -1.0;
                    int _n = -1;
                    if (j == i) {
                        // This is the seed
                        frac = 1.0;
                        _n = i*100;
                    }
                    else {
                        for (_n = i*100; _n < ((i+1)*100); _n++) {
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

                        if (nSeeds == 1) {
                            if (debug)
                                printf("\t\tThis topo cluster has a single seed.\n");
                            computeClusterPos(clusterPos[s], frac, j, debug);
                        }
                        else {
                            if (j == i) {
                                // This is the seed
                                computeClusterPos(clusterPos[s], frac, j, debug);
                            }
                            else {
                                // Check if this is one of the neighboring rechits
                                for(int k=0; k<nNeigh; k++){
                                    if(neigh4_Ind[nNeigh*i+k]<0) continue;
                                    if(neigh4_Ind[nNeigh*i+k] == j) {
                                        // Found it
                                        if (debug) printf("\t\tRechit %d is one of the 4 neighbors of seed %d\n", j, i);
                                        computeClusterPos(clusterPos[s], frac, j, debug);
                                    }
                                }
                            }
                        }
                    }
                    //else if (debug)
                    //    printf("Can't find rechit fraction for cluster %d (seed %d) rechit %d!\n", s, i, j);
                        
                }
                if (clusterPos[s].w >= minAllowedNormalization)
                {
                    // Divide by position norm
                    clusterPos[s].x /= clusterPos[s].w;
                    clusterPos[s].y /= clusterPos[s].w;
                    clusterPos[s].z /= clusterPos[s].w;

                    if (debug)
                        printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n", s, i, clusterEnergy[s], clusterPos[s].x, clusterPos[s].y, clusterPos[s].z);
                }
                else {
                    if (debug)
                        printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n", s, i, clusterPos[s].w, minAllowedNormalization);
                    clusterPos[s].x = 0.0;
                    clusterPos[s].y = 0.0;
                    clusterPos[s].z = 0.0;
                    //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
                    
                }
            }
            
            float diff2 = 0.0;
            for (int s = 0; s < nSeeds; s++) {
                float delta2 = dR2(prevClusterPos[s], clusterPos[s]);
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



__global__ void hcalFastCluster_original(size_t nRH,
                                const float* __restrict__ pfrh_x,
                                const float* __restrict__ pfrh_y,
                                const float* __restrict__ pfrh_z,
                                const float* __restrict__ pfrh_energy,
                                int* pfrh_topoId,
                                int* pfrh_isSeed,
                                const int* __restrict__ pfrh_layer,
                                const int* __restrict__ pfrh_depth,
                                const int* __restrict__ neigh4_Ind,
                                float* pcrhfrac,
                                int* pcrhfracind,
                                float* fracSum,
                                int* rhCount
                                ) {

    int topoId = threadIdx.x+blockIdx.x*blockDim.x; // TopoId
    int iter = 0;
    int nSeeds = 0;
    int nRHTopo = 0;
    if (topoId >-1 && topoId < nRH) {
        int seeds[75];
        int rechits[250];
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
        //float tolScaling2 = std::pow(std::max(1.0, nSeeds - 1.0), 4.0);     // Tolerance scaling squared
        float tolScaling = std::pow(std::max(1.0, nSeeds - 1.0), 2.0);     // Tolerance scaling squared

        float4 prevClusterPos[75], clusterPos[75];  //  W component is position norm
        float clusterEnergy[75];
        //float prevClusterEnergy[75];

        auto computeClusterPos = [&] (float4 pos4, float _frac, int rhInd) {
            float4 rechitPos = make_float4(pfrh_x[rhInd], pfrh_y[rhInd], pfrh_z[rhInd], 1.0);
            float threshold = 0.0;
            if(pfrh_layer[rhInd] == PFLayer::HCAL_BARREL1) {
                threshold = recHitEnergyNormInvEB_vec[pfrh_depth[rhInd] - 1]; // This number needs to be inverted
            }
            else if (pfrh_layer[rhInd] == PFLayer::HCAL_ENDCAP) { threshold = recHitEnergyNormInvEE_vec[pfrh_depth[rhInd] - 1]; }

            const auto rh_energy = pfrh_energy[rhInd] * _frac;
            const auto norm =
                (_frac < minFractionInCalc ? 0.0f : max(0.0f, logf(rh_energy * threshold)));
            pos4.x += rechitPos.x * norm;
            pos4.y += rechitPos.y * norm;
            pos4.z += rechitPos.z * norm;
            pos4.w += norm;     //  position_norm
        };
        /*
        auto compute = [&] (float4 pos4, float clusterEn, int seedInd, int rhInd) {
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
                threshold = 1. / recHitEnergyNormInvEB_vec[pfrh_depth[rhInd] - 1]; // This number needs to be inverted
            }
            else if (pfrh_layer[rhInd] == 3) { threshold = 1. / recHitEnergyNormInvEE_vec[pfrh_depth[rhInd] - 1]; }

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
        while (iter < maxIterations) {
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = seeds[s];
                if (iter == 0) {
                    // Set initial cluster position to seed rechit position
                    clusterPos[s] = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.0);
                    prevClusterPos[s] = clusterPos[s];

                    // Set initial cluster energy to seed energy
                    clusterEnergy[s] = pfrh_energy[i];
                    //prevClusterEnergy[s] = 0.0;
                }
                else {
                    prevClusterPos[s] = clusterPos[s];
                    //prevClusterEnergy[s] = clusterEnergy[s];
                    
                    // Reset cluster indices and fractions
                    for (int _n = i*100; _n < (i+1)*100; _n++) {
                        pcrhfrac[_n] = -1.0;           
                        pcrhfracind[_n] = -1.0;           
                    }
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
                        
                        if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[s] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                        else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[s] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
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
                            pcrhfrac[i*100]    = 1.;
                            pcrhfracind[i*100] = j;
                        }
                        if( pfrh_isSeed[j]!=1 ){
                            float dist2 =
                               (clusterPos[s].x - pfrh_x[j])*(clusterPos[s].x - pfrh_x[j])
                              +(clusterPos[s].y - pfrh_y[j])*(clusterPos[s].y - pfrh_y[j])
                              +(clusterPos[s].z - pfrh_z[j])*(clusterPos[s].z - pfrh_z[j]);
                            float d2 = dist2 / showerSigma2;
                            float fraction = -1.;
                            
                            if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = clusterEnergy[s] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                            else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = clusterEnergy[s] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
                            if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");

                            if (fracSum[j] > minFracTot) {
                                float fracpct = fraction / fracSum[j];
                                if(fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep))
                                  {
                                      int k = atomicAdd(&rhCount[i],1);
                                      pcrhfrac[i*100+k] = fracpct;
                                      pcrhfracind[i*100+k] = j;
                                  }
                            }
                        }
                    }
                }
            }

            // Recalculate position
            for (int s = 0; s < nSeeds; s++) {      // PF clusters
                int i = seeds[s];
                    
                // Zero out cluster position and energy
                clusterPos[s] = make_float4(0.0, 0.0, 0.0, 0.0);
                clusterEnergy[s] = 0;

                // Calculate cluster energy by summing rechit fractional energies
                for (int r = 0; r < nRHTopo; r++) {
                    int j = rechits[r];
                    float frac = -1.0;
                    for (int _n = i*100; _n < ((i+1)*100); _n++) {
                        if (pcrhfracind[_n] == j) {
                            // Found it
                            frac = pcrhfrac[_n];
                            break;
                        }
                    }
                    if (frac > -0.5) {
                        clusterEnergy[s] += frac * pfrh_energy[j];
                        
                        if (nSeeds == 1)
                            computeClusterPos(clusterPos[s], frac, j);
                        else {
                            // Check if this is one of the neighboring rechits
                            for(int k=0; k<nNeigh; k++){
                                if(neigh4_Ind[nNeigh*i+k]<0) continue;
                                if(neigh4_Ind[nNeigh*i+k] == j) {
                                    // Found it
                                    computeClusterPos(clusterPos[s], frac, j);
                                }
                            }
                        }
                    }
                    //else
                    //    printf("Can't find rechit fraction for seed %d rechit %d!\n", i, j);
                        
                }
                if (clusterPos[s].w >= minAllowedNormalization)
                {
                    // Divide by position norm
                    clusterPos[s].x /= clusterPos[s].w;
                    clusterPos[s].y /= clusterPos[s].w;
                    clusterPos[s].z /= clusterPos[s].w;

                }
                else { 
                    clusterPos[s].x = 0.0;
                    clusterPos[s].y = 0.0;
                    clusterPos[s].z = 0.0;
                    //printf("PFCluster for seed rechit %d has position norm less than allowed minimum!\n", i);
                    
                }
            }
            
            float diff2 = 0.0;
            for (int s = 0; s < nSeeds; s++) {
                float delta2 = dR2(prevClusterPos[s], clusterPos[s]);
                if (delta2 > diff2)
                    diff2 = delta2;
            }
            float diff = sqrtf(diff2);
            iter++;
            //if (iter >= maxIterations || diff2 <= stoppingTolerance2 * tolScaling2) break;
            if (diff <= stoppingTolerance * tolScaling) break;
        }
    }
}


__global__ void hcalFastCluster_step1( size_t size,
					     const float* __restrict__ pfrh_x,
					     const float* __restrict__ pfrh_y,
					     const float* __restrict__ pfrh_z,
					     const float* __restrict__ pfrh_energy,
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

      if(pfrh_layer[j] == PFLayer::HCAL_BARREL1) { fraction = pfrh_energy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
      else if (pfrh_layer[j] == PFLayer::HCAL_ENDCAP) { fraction = pfrh_energy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
	  
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
					     const float* __restrict__ pfrh_energy,
					     int* pfrh_topoId,
					     int* pfrh_isSeed,
					     const int* __restrict__ pfrh_layer,
				         const int* __restrict__ pfrh_depth,
					     float* pcrhfrac,
					     int* pcrhfracind,
					     float* fracSum,
					     int* rhCount,
                         int* topoSeedCount,
                         int* topoRHCount,
                         int* seedFracOffsets,
                         int* topoSeedOffsets,
                         int* topoSeedList
                         ) {

    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int j = threadIdx.y+blockIdx.y*blockDim.y;
    //make sure topoID, Layer is the same, i is seed and j is not seed
    if( i<size && j<size){
      if( pfrh_topoId[i] == pfrh_topoId[j] && pfrh_isSeed[i]==1 ){
      if(i==j)
	{
	  pcrhfrac[i*100]    = 1.;
	  pcrhfracind[i*100] = j;
	}
      if( pfrh_isSeed[j]!=1 ){
        float dist2 =
           (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
          +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
          +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);

        float d2 = dist2 / showerSigma2; 
        float fraction = -1.;

        if(pfrh_layer[j] == 1) { fraction = pfrh_energy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
        else if (pfrh_layer[j] == 3) { fraction = pfrh_energy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
        
        if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");
        
        if (fracSum[j] > minFracTot) {
            float fracpct = fraction / fracSum[j];
            if(fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep))
              {
                  int k = atomicAdd(&rhCount[i],1);
                  pcrhfrac[seedFracOffsets[i]+k] = fracpct;
                  pcrhfracind[seedFracOffsets[i]+k] = j;
                  //pcrhfrac[i*100+k] = fracpct;
                  //pcrhfracind[i*100+k] = j;
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


__global__ void hcalFastCluster_step2( size_t size,
					     const float* __restrict__ pfrh_x,
					     const float* __restrict__ pfrh_y,
					     const float* __restrict__ pfrh_z,
					     const float* __restrict__ pfrh_energy,
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
	  pcrhfrac[i*100]    = 1.;
	  pcrhfracind[i*100] = j;
	}
      if( pfrh_isSeed[j]!=1 ){
        float dist2 =
           (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
          +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
          +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);

        float d2 = dist2 / showerSigma2; 
        float fraction = -1.;

        if(pfrh_layer[j] == 1) { fraction = pfrh_energy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
        else if (pfrh_layer[j] == 3) { fraction = pfrh_energy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
        
        if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");
        
        if (fracSum[j] > minFracTot) {
            float fracpct = fraction / fracSum[j];
            if(fracpct > 0.9999 || (d2 < 100. && fracpct > minFracToKeep))
              {
                  int k = atomicAdd(&rhCount[i],1);
                  pcrhfrac[i*100+k] = fracpct;
                  pcrhfracind[i*100+k] = j;
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
					     const float* __restrict__ pfrh_energy,
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

              if(pfrh_layer[j] == 1) { fraction = pfrh_energy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
              else if (pfrh_layer[j] == 3) { fraction = pfrh_energy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
              
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
					     const float* __restrict__ pfrh_energy,
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
              pcrhfrac[i*100]    = 1.;
              pcrhfracind[i*100] = j;
            }
              if( pfrh_isSeed[j]!=1 ){
            float dist2 =
               (pfrh_x[i] - pfrh_x[j])*(pfrh_x[i] - pfrh_x[j])
              +(pfrh_y[i] - pfrh_y[j])*(pfrh_y[i] - pfrh_y[j])
              +(pfrh_z[i] - pfrh_z[j])*(pfrh_z[i] - pfrh_z[j]);

            float d2 = dist2 / showerSigma2; 
            float fraction = -1.;

            if(pfrh_layer[j] == 1) { fraction = pfrh_energy[i] * recHitEnergyNormInvEB_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
            else if (pfrh_layer[j] == 3) { fraction = pfrh_energy[i] * recHitEnergyNormInvEE_vec[pfrh_depth[j] - 1] * expf(-0.5 * d2); }
              
            
            if(fraction==-1.) printf("FRACTION is NEGATIVE!!!");
            if(d2 < 100. )
              {
                if ((fraction/fracSum[j])>minFracToKeep){
                  int k = atomicAdd(&rhCount[i],1);
                  pcrhfrac[i*100+k] = fraction/fracSum[j];
                  pcrhfracind[i*100+k] = j;
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
                const float* __restrict__ pfrh_energy,
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
                const float* __restrict__ pfrh_energy,
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
__global__ void topoClusterContraction(size_t size,
                                       int* pfrh_parent,
                                       int* pfrh_isSeed) {
    
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


// Contraction in a single block
__global__ void topoClusterContraction(size_t size,
                                       int* pfrh_parent,
                                       int* pfrh_isSeed,
                                       int* rhCount,
                                       int* topoSeedCount,
                                       int* topoRHCount,
                                       int* seedFracOffsets,
                                       int* topoSeedOffsets,
                                       int* topoSeedList,
                                       int* pcrhfracind,
                                       float* pcrhfrac,
                                       int* pcrhFracSize
                                       ) {
    __shared__ int notDone, totalSeedOffset, totalSeedFracOffset;
    if (threadIdx.x == 0) {
        notDone = 0;
        totalSeedOffset = 0;
        totalSeedFracOffset = 0;
        *pcrhFracSize = 0;
        //atomicSub_system(pcrhFracSize, *pcrhFracSize);
    }
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

    // Now determine the number of seeds and rechits in each topo cluster
    for (int rhIdx = threadIdx.x; rhIdx < size; rhIdx += blockDim.x) {
        int topoId = pfrh_parent[rhIdx];
        if (topoId > -1) {
            // Valid topo cluster
            atomicAdd(&topoRHCount[topoId], 1);
            if (pfrh_isSeed[rhIdx]) {
                atomicAdd(&topoSeedCount[topoId], 1);
            }
        }
    }
    __syncthreads();
    
    
    
    // Determine offsets for topo ID seed array
    for (int topoId = threadIdx.x; topoId < size; topoId += blockDim.x) {
        if (topoSeedCount[topoId] > 0) {
            // This is a valid topo ID
            int offset = atomicAdd(&totalSeedOffset, topoSeedCount[topoId]);
            topoSeedOffsets[topoId] = offset;
        }
    }
    __syncthreads();
    //if (threadIdx.x == 0) printf("Total seeds found: %d\n", totalSeedOffset); 
    

    // Fill arrays of seed indicies per topo ID
    for (int rhIdx = threadIdx.x; rhIdx < size; rhIdx += blockDim.x) {
        int topoId = pfrh_parent[rhIdx];
        if (topoId > -1 && pfrh_isSeed[rhIdx]) {
            // Valid topo cluster
            int k = atomicAdd(&rhCount[topoId], 1);
            topoSeedList[topoSeedOffsets[topoId]+k] = rhIdx;
        }
    }
    __syncthreads();

    /*
    if (threadIdx.x == 0) {
        printf("topoSeedOffsets = \n[");
        for (int i = 0; i < size; i++) {
            if (i != 0) printf(", ");
            printf("%d", topoSeedOffsets[i]);
        }
        printf("]\n\n");
        
        printf("topoSeedList = \n[");
        for (int i = 0; i < size; i++) {
            if (i != 0) printf(", ");
            printf("%d", topoSeedList[i]);
        }
        printf("]\n\n");
    }
    */
    /* 
    if (threadIdx.x == 0) {
        for (int rhIdx = 0; rhIdx < size; rhIdx++) {
            int topoId = pfrh_parent[rhIdx];
            if (pfrh_isSeed[rhIdx] && topoId > -1) {
                // Add offset for this PF cluster seed
                seedFracOffsets[rhIdx] = totalSeedOffset;
                // Allot the total number of rechits for this topo cluster for rh fractions
                totalSeedOffset += topoRHCount[topoId];
            }
        }
        printf("--------> totalSeedOffset = %d\n", totalSeedOffset);
    }
    */
    
    // Determine seed offsets for rechit fraction array
    for (int rhIdx = threadIdx.x; rhIdx < size; rhIdx += blockDim.x) {
        rhCount[rhIdx] = 1; // Reset this counter array

        int topoId = pfrh_parent[rhIdx];
        if (pfrh_isSeed[rhIdx] && topoId > -1) {
            // Allot the total number of rechits for this topo cluster for rh fractions
            //printf("rechit %d is a seed for topoId %d with topoRHCount = %d\n", rhIdx, topoId, topoRHCount[topoId]);
            int offset = atomicAdd(&totalSeedFracOffset, topoRHCount[topoId]);
            
            // Add offset for this PF cluster seed
            seedFracOffsets[rhIdx] = offset; 
            pcrhfracind[offset] = rhIdx;
            pcrhfrac[offset] = 1.;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        //printf("--------> totalSeedFracOffset = %d\n", totalSeedFracOffset);
        *pcrhFracSize = totalSeedFracOffset;

        //atomicAdd_system(pcrhFracSize, totalSeedFracOffset); 
    }
    
}

__global__ void fillRhfIndex(size_t nRH,
                             int* pfrh_parent,
                             int* pfrh_isSeed,
                             int* topoSeedCount,
                             int* topoRHCount,
                             int* seedFracOffsets,
                             int* rhCount,
                             int* pcrhfracind) {

    int i = threadIdx.x+blockIdx.x*blockDim.x;  // i is the seed index
    int j = threadIdx.y+blockIdx.y*blockDim.y;  // j is NOT a seed

    if (i < nRH && j < nRH) {
        //if (i == debugSeedIdx) printf("This is fillRhfIndex with i = %d and j = %d\n", i, j);
        int topoId = pfrh_parent[i];
        if (topoId == pfrh_parent[j] && topoId > -1 && pfrh_isSeed[i] && !pfrh_isSeed[j]) {
            int k = atomicAdd(&rhCount[i], 1);  // Increment the number of rechit fractions for this seed
            pcrhfracind[seedFracOffsets[i] + k] = j;    // Save this rechit index
        }
    }
}

__global__ void fillRhfIndex_serialize(size_t nRH,
                             int* pfrh_parent,
                             int* pfrh_isSeed,
                             int* topoSeedCount,
                             int* topoRHCount,
                             int* seedFracOffsets,
                             int* rhCount,
                             int* pcrhfracind) {

    //int debugSeedIdx = 500;
    
    /* 
    printf("rhCount = \n[");
    for (int i = 0; i < (int)nRH; i++) {
        if (i != 0) printf(", ");
        printf("%d", rhCount[i]);
    }
    printf("]\n");
    */

    for (int i = 0; i < (int)nRH; i++) { 
        for (int j = 0; j < (int)nRH; j++) { 
            //if (i == debugSeedIdx) printf("This is fillRhfIndex with i = %d and j = %d\n", i, j);
            int topoId = pfrh_parent[i];
            if (topoId == pfrh_parent[j] && topoId > -1 && pfrh_isSeed[i] && !pfrh_isSeed[j]) {
                //if (i == debugSeedIdx) printf("This is seed %d with topoId %d and rechit %d\n", i, topoId, j);
                int k = atomicAdd(&rhCount[i], 1);  // Increment the number of rechit fractions for this seed
                if (seedFracOffsets[i] < 0) printf("WARNING: seed %d has offset %d!\n", i, seedFracOffsets[i]);
                //printf("seed %d: rechit %d index with k = %d and seed offset = %d\n", i, j, k, seedFracOffsets[i]);
                pcrhfracind[seedFracOffsets[i] + k] = j;    // Save this rechit index
            }
        }
    }
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
    int* topoIter) {

    __shared__ bool notDone;
    __shared__ int iter, gridStride;

    int start = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        *topoIter = 0;
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
    *topoIter = iter;
}


__global__ void printRhfIndex(int* pfrh_topoId, int* topoRHCount, int* seedFracOffsets, int* pcrhfracind) {
    int seedIdx = 500;
    int offset = seedFracOffsets[seedIdx];
    int topoId = pfrh_topoId[seedIdx];
    int nRHF = topoRHCount[topoId];
    if (offset > -1 && topoId > -1) {
        printf("seed %d has topoId %d and offset %d with %d expected rechit fractions:\n[", seedIdx, topoId, offset, nRHF);
        for (int r = offset; r < (offset+nRHF); r++) {
            if (r != offset) printf(", ");
            printf("%d", pcrhfracind[r]);
        }
        printf("]\n\n");
    }
}

void PFRechitToPFCluster_HCAL_CCLClustering(int nRH,
                int nEdges,
                const float* __restrict__ pfrh_x,
                const float* __restrict__ pfrh_y,
                const float* __restrict__ pfrh_z,
                const float* __restrict__ pfrh_energy,
                const float* __restrict__ pfrh_pt2,
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
                int* topoSeedCount,
                int* topoRHCount,
                int* seedFracOffsets,
                int* topoSeedOffsets,
                int* topoSeedList,
                float4* pfc_pos,
                float4* pfc_prevPos,
                float* pfc_energy,
                float (&timer)[8],
                int* topoIter,
                int* pfcIter,
                int* pcrhFracSize
                ) {
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
    seedingTopoThreshKernel_HCAL<<<(nRH+63)/64, 128>>>(nRH, pfrh_energy, pfrh_pt2, pfrh_isSeed, pfrh_topoId, pfrh_passTopoThresh, pfrh_layer, pfrh_depth, neigh4_Ind, rhCount, topoSeedCount, topoRHCount, seedFracOffsets, topoSeedOffsets, topoSeedList, pfcIter);
#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&timer[0], start, stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif
    
    //topoclustering 
    //topoClusterLinking<<<1, 1024 >>>(nRH, nEdges, pfrh_topoId, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_passTopoThresh, topoIter);
    topoClusterLinking<<<1, 512 >>>(nRH, nEdges, pfrh_topoId, pfrh_edgeId, pfrh_edgeList, pfrh_edgeMask, pfrh_passTopoThresh, topoIter);
    topoClusterContraction<<<1, 512>>>(nRH, pfrh_topoId, pfrh_isSeed, rhCount, topoSeedCount, topoRHCount, seedFracOffsets, topoSeedOffsets, topoSeedList, pcrhfracind, pcrhfrac, pcrhFracSize);

#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&timer[1], start, stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif

    

    dim3 grid((nRH+31)/32, (nRH+31)/32);
    dim3 block(32, 32);
    //printf("grid = (%d, %d, %d)\tblock = (%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    //printf("About to call fillRhfIndex with nRH = %d\n", nRH);
    
    fillRhfIndex<<<grid, block>>>(nRH, pfrh_topoId, pfrh_isSeed, topoSeedCount, topoRHCount, seedFracOffsets, rhCount, pcrhfracind);
    //fillRhfIndex_serialize<<<1, 1>>>(nRH, pfrh_topoId, pfrh_isSeed, topoSeedCount, topoRHCount, seedFracOffsets, rhCount, pcrhfracind);
   
#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&timer[2], start, stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif
   
    hcalFastCluster_optimizedLambdas<<<nRH, 256>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, neigh4_Ind, pcrhfrac, pcrhfracind, fracSum, rhCount, topoSeedCount, topoRHCount, seedFracOffsets, topoSeedOffsets, topoSeedList, pfc_pos, pfc_prevPos, pfc_energy, pfcIter);


    //printRhfIndex<<<1,1>>>(pfrh_topoId, topoRHCount, seedFracOffsets, pcrhfracind);
    //hcalFastCluster_sharedMem<<<nRH, 256>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, neigh4_Ind, pcrhfrac, pcrhfracind, fracSum, rhCount, topoSeedCount, topoRHCount, seedFracOffsets, topoSeedOffsets, topoSeedList, pfc_pos, pfc_prevPos, pfc_energy, pfcIter);
    //hcalFastCluster_optimizedLambdas<<<dim3(nRH,nRH), dim3(32,256)>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, neigh4_Ind, pcrhfrac, pcrhfracind, fracSum, rhCount, topoSeedCount, topoRHCount, seedFracOffsets, topoSeedOffsets, topoSeedList, pfc_pos, pfc_prevPos, pfc_energy, pfcIter);
    
    //hcalFastCluster_optimizedLambdas<<<nRH, 32>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, neigh4_Ind, pcrhfrac, pcrhfracind, fracSum, rhCount, topoSeedCount, topoRHCount, seedFracOffsets, topoSeedOffsets, topoSeedList, pfc_pos, pfc_prevPos, pfc_energy, pfcIter);
    //hcalFastCluster_optimizedLambdas<<<nRH, 256>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, neigh4_Ind, pcrhfrac, pcrhfracind, fracSum, rhCount, topoSeedCount, topoRHCount, seedFracOffsets, topoSeedOffsets, topoSeedList, pfc_pos, pfc_prevPos, pfc_energy, pfcIter);
    
   // hcalFastCluster_noLambdaPosCalc<<<nRH, 256>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, neigh4_Ind, pcrhfrac, pcrhfracind, fracSum, rhCount, topoSeedCount, topoRHCount, seedFracOffsets, topoSeedOffsets, topoSeedList, pfc_pos, pfc_prevPos, pfc_energy, pfcIter);
    //hcalFastCluster_withLambdas<<<nRH, 256>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, neigh4_Ind, pcrhfrac, pcrhfracind, fracSum, rhCount, topoSeedCount, topoRHCount, seedFracOffsets, topoSeedOffsets, topoSeedList, pfc_pos, pfc_prevPos, pfc_energy, pfcIter);
    //hcalFastCluster_optimizedLambdas<<<nRH, 256>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, neigh4_Ind, pcrhfrac, pcrhfracind, fracSum, rhCount, topoSeedCount, topoRHCount, seedFracOffsets, topoSeedOffsets, topoSeedList, pfc_pos, pfc_prevPos, pfc_energy, pfcIter);
    //hcalFastCluster<<<nRH, 256>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, neigh4_Ind, pcrhfrac, pcrhfracind, fracSum, rhCount, topoSeedCount, topoRHCount, seedFracOffsets, topoSeedOffsets, topoSeedList, pfc_pos, pfc_prevPos, pfc_energy, pfcIter);
    
    
    //hcalFastCluster_serialize<<<1,1>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, neigh4_Ind, pcrhfrac, pcrhfracind, fracSum, rhCount, topoSeedCount, topoRHCount, seedFracOffsets, topoSeedOffsets, topoSeedList, pfc_pos, pfc_prevPos, pfc_energy);
    //hcalFastCluster_old_serialize<<<1,1>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, neigh4_Ind, pcrhfrac, pcrhfracind, fracSum, rhCount, topoSeedCount, topoRHCount, seedFracOffsets);
    
    //hcalFastCluster_older_serialize<<<1,1>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, neigh4_Ind, pcrhfrac, pcrhfracind, fracSum, rhCount);

/*
    dim3 grid2( (nRH+32-1)/32, (nRH+32-1)/32 );
    dim3 block2( 32, 32);

    hcalFastCluster_step1<<<grid2, block2>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);

#ifdef DEBUG_GPU_HCAL
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&timer[2], start, stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
#endif

    //hcalFastCluster_step2<<<grid2, block2>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount);
    hcalFastCluster_step2<<<grid2, block2>>>( nRH, pfrh_x,  pfrh_y,  pfrh_z,  pfrh_energy, pfrh_topoId,  pfrh_isSeed,  pfrh_layer, pfrh_depth, pcrhfrac, pcrhfracind, fracSum, rhCount, topoSeedCount, topoRHCount, seedFracOffsets, topoSeedOffsets, topoSeedList);
*/
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
				const float* __restrict__ pfrh_energy, 
				const float* __restrict__ pfrh_pt2,    				
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
				const float* __restrict__ pfrh_energy, 
				const float* __restrict__ pfrh_pt2,    				
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
