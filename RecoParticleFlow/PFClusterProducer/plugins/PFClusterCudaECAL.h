#ifndef PFClusterCudaECAL_h
#define PFClusterCudaECAL_h
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
#include <Eigen/Dense>

namespace PFClusterCudaECAL {

  

  

  void PFRechitToPFCluster_ECAL(size_t size, float* pfrh_x,float* pfrh_y, float* pfrh_z,	float* pfrh_energy,	float* pfrh_pt2, int* pfrh_isSeed, 
				int* pfrh_topoId, 
				int* pfrh_layer, 
				int* neigh8_Ind, 
				
				float* pfrhfrac, 
				int* pfrhind, 
				int* pcrhind,
				float* pcrhfracind
				);

}  // namespace cudavectors

#endif  // ClusterCudaECAL_h
