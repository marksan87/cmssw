#ifndef PFClusterCudaHCAL_h
#define PFClusterCudaHCAL_h
//#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <Eigen/Dense>
#include "RecoParticleFlow/PFClusterProducer/plugins/cuCompactor.h"
//#include "cuStreamComp/src/cuCompactor.cuh"
//#include "cuStreamComp/src/cuda_error_check.cu"

namespace PFClusterCudaHCAL { 
  
  bool initializeCudaConstants(float h_showerSigma = 0.f,
                               const float (&h_recHitEnergyNormEB_vec)[4] = {0.f,0.f,0.f,0.f},
                               const float (&h_recHitEnergyNormEE_vec)[7] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f},
                               float h_minFracToKeep = 0.,
                               const float (&h_seedEThresholdEB_vec)[4] = {0.f,0.f,0.f,0.f}, 
                               const float (&h_seedEThresholdEE_vec)[7] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f},
                               float h_seedPt2ThresholdEB = 0.,
                               float h_seedPt2hresholdEE = 0.,
                               const float (&h_topoEThresholdEB_vec)[4] = {0.f,0.f,0.f,0.f},
                               const float (&h_topoEThresholdEE_vec)[7] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f},
                               int   h_nNeigh = 0,
                               int   h_maxSize = 100
                               );


  void PFRechitToPFCluster_HCALV1(size_t size, 
				float* pfrh_x,
				float* pfrh_y, 
				float* pfrh_z,	
				float* pfrh_energy,	
				float* pfrh_pt2, 
				int* pfrh_isSeed, 
				int* pfrh_topoId, 
				int* pfrh_layer, 
				int* pfrh_depth, 
				int* neigh8_Ind, 				
				int* neigh4_Ind, 				
				float* pfrhfrac, 
				int* pfrhind, 
				int* pcrhind,
				float* pcrhfracind
				);

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
				int* pcrhind,
				float* pcrhfracind,
				float* fracSum,
				int* rhCount,
				float (&timer)[8]  
                );

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
    bool* d_notDone);

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
                bool* d_notDone);

void topoClusterLinking_copyif(size_t nRH,
                size_t nEdges,
                int* pfrh_edgeId,
                int* pfrh_edgeList,
                int* pfrh_edgesLeft,
                bool* pfrh_edgeMask,
                int* pfrh_parent,
                bool* pfrh_passTopoThresh);

void topoClusterContraction_singleBlock(size_t nRH,
                int* pfrh_parent);

void topoClusterContraction_copyif(size_t nRH,
                int* pfrh_parent,
                bool*  h_notDone,
                bool*  d_notDone);

void topoClusterContraction_copyif(size_t nRH,
                int* pfrh_parent);

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
                int &nIter
                );

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
				float (&timer)[8]  
                );

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
                int* pcrhind,
				float* pcrhfracind,
				float* fracSum,
				int* rhCount,
				float (&timer)[8]  
                );

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
				int* pcrhind,
				float* pcrhfracind,
				float* fracSum,
				int* rhCount,
				float* timer = nullptr 
				);

}  // namespace cudavectors

#endif  // ClusterCudaECAL_h
