#ifndef PFClusterCudaECAL_h
#define PFClusterCudaECAL_h
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
#include <Eigen/Dense>
#include <cuda.h>

namespace PFClusterCudaECAL {
  
  bool initializeCudaConstants(float h_showerSigma = 0.,
                               float h_recHitEnergyNormEB = 0.,
                               float h_recHitEnergyNormEE = 0., 
                               float h_minFracToKeep = 0.,
                               float h_minFracTot = 0.,
                               int   h_maxIterations = 0,
                               float h_stoppingTolerance = 0.,
                               bool  h_excludeOtherSeeds = false, 
                               float h_seedEThresholdEB = 0.,
                               float h_seedEThresholdEE = 0.,
                               float h_seedPt2ThresholdEB = 0.,
                               float h_seedPt2hresholdEE = 0., 
                               float h_topoEThresholdEB = 0., 
                               float h_topoEThresholdEE = 0.,
                               int   h_nNeigh = 0,
                               int   h_maxSize = 50
                               );


  void PFRechitToPFCluster_ECAL_CCLClustering(int nRH,
                int nEdges,
                const float* __restrict__ pfrh_x,
                const float* __restrict__ pfrh_y,
                const float* __restrict__ pfrh_z,
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
                int* nIter
                );

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
				int* pcrhind,
				float* pcrhfracind,
				float* fracSum,
				int* rhCount,
				float (&timer)[8]
                );

 void PFRechitToPFCluster_ECALV1(size_t size, 
				float* pfrh_x,
				float* pfrh_y, 
				float* pfrh_z,	
				float* pfrh_energy,	
				float* pfrh_pt2, 
				int* pfrh_isSeed, 
				int* pfrh_topoId, 
				int* pfrh_layer, 
				int* neigh8_Ind, 				
				int* pcrhind,
				float* pcrhfracind);

  
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
				int* pcrhind,
				float* pcrhfracind,
				float* fracSum,
				int* rhCount,
				float* timer = nullptr
                );

}  // namespace cudavectors

#endif  // ClusterCudaECAL_h
