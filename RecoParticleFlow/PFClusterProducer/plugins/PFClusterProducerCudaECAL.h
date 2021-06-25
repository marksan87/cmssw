#ifndef __newpf_PFClusterProducerCudaECAL_H__
#define __newpf_PFClusterProducerCudaECAL_H__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"

#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"

#include <memory>
#include <array>
namespace PFClustering {
  struct ConfigurationParameters {
    uint32_t maxRH = 2000;
    uint32_t maxPFCSize = 75;
    uint32_t maxNeighbors = 8;
  };

  struct InputDataCPU {
      cms::cuda::host::unique_ptr<float[]> pfrh_x;
      cms::cuda::host::unique_ptr<float[]> pfrh_y;
      cms::cuda::host::unique_ptr<float[]> pfrh_z;
      cms::cuda::host::unique_ptr<float[]> pfrh_energy;
      cms::cuda::host::unique_ptr<float[]> pfrh_pt2;
      cms::cuda::host::unique_ptr<float[]> pcrh_frac;
      cms::cuda::host::unique_ptr<float[]> pcrh_fracSum;

      //cms::cuda::host::unique_ptr<int[]> rhcount;
      cms::cuda::host::unique_ptr<int[]> pfrh_topoId;
      cms::cuda::host::unique_ptr<int[]> pfrh_isSeed;
      cms::cuda::host::unique_ptr<int[]> pfrh_layer;
      cms::cuda::host::unique_ptr<int[]> pfNeighEightInd;
      cms::cuda::host::unique_ptr<int[]> pcrh_fracInd;
      cms::cuda::host::unique_ptr<int[]> pfrh_edgeId;
      cms::cuda::host::unique_ptr<int[]> pfrh_edgeList;
  
      void allocate(ConfigurationParameters const& config, cudaStream_t cudaStream = 0 /* default Cuda stream */) {
        pfrh_x = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_y = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_z = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_energy = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_pt2 = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pcrh_frac = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH*config.maxPFCSize, cudaStream);
        pcrh_fracSum = cms::cuda::make_host_unique<float[]>(sizeof(float)*config.maxRH*config.maxPFCSize, cudaStream);
        //rhcount = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH*config.maxPFCSize, cudaStream);
        pfrh_topoId = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        pfrh_isSeed = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        pfrh_layer = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        pfNeighEightInd = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH*config.maxNeighbors, cudaStream);
        pcrh_fracInd = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH*config.maxPFCSize, cudaStream);
        pfrh_edgeId = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH*config.maxNeighbors, cudaStream);
        pfrh_edgeList = cms::cuda::make_host_unique<int[]>(sizeof(int)*config.maxRH*config.maxNeighbors, cudaStream);
      }
  };
  
  struct InputDataGPU {
      cms::cuda::device::unique_ptr<float[]> pfrh_x; 
      cms::cuda::device::unique_ptr<float[]> pfrh_y; 
      cms::cuda::device::unique_ptr<float[]> pfrh_z; 
      cms::cuda::device::unique_ptr<float[]> pfrh_energy; 
      cms::cuda::device::unique_ptr<float[]> pfrh_pt2; 
      cms::cuda::device::unique_ptr<float[]> pcrh_frac;
      cms::cuda::device::unique_ptr<float[]> pcrh_fracSum;

      cms::cuda::device::unique_ptr<int[]> rhcount;
      cms::cuda::device::unique_ptr<int[]> pfrh_topoId;
      cms::cuda::device::unique_ptr<int[]> pfrh_isSeed;
      cms::cuda::device::unique_ptr<int[]> pfrh_layer;
      cms::cuda::device::unique_ptr<int[]> pfNeighEightInd;
      cms::cuda::device::unique_ptr<int[]> pcrh_fracInd;
      cms::cuda::device::unique_ptr<int[]> pfrh_edgeId;
      cms::cuda::device::unique_ptr<int[]> pfrh_edgeList;
      cms::cuda::device::unique_ptr<int[]> pfrh_edgeMask;
      
      cms::cuda::device::unique_ptr<bool[]> pfrh_passTopoThresh;
      
      void allocate(ConfigurationParameters const& config, cudaStream_t cudaStream = 0 /* default Cuda stream */) {
        pfrh_x = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_y = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_z = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_energy = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pfrh_pt2 = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH, cudaStream);
        pcrh_frac = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH*config.maxPFCSize, cudaStream);
        pcrh_fracSum = cms::cuda::make_device_unique<float[]>(sizeof(float)*config.maxRH*config.maxPFCSize, cudaStream);
        rhcount = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH*config.maxPFCSize, cudaStream);
        pfrh_topoId = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        pfrh_isSeed = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        pfrh_layer = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH, cudaStream);
        pfNeighEightInd = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH*config.maxNeighbors, cudaStream);
        pcrh_fracInd = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH*config.maxPFCSize, cudaStream);
        pfrh_edgeId = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH*config.maxNeighbors, cudaStream);
        pfrh_edgeList = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH*config.maxNeighbors, cudaStream);
        pfrh_edgeMask = cms::cuda::make_device_unique<int[]>(sizeof(int)*config.maxRH*config.maxNeighbors, cudaStream);
        pfrh_passTopoThresh = cms::cuda::make_device_unique<bool[]>(sizeof(bool)*config.maxRH, cudaStream);
      }
  };
} // namespace PFClustering

class PFClusterProducerCudaECAL : public edm::stream::EDProducer<> {
  typedef RecHitTopologicalCleanerBase RHCB;
  typedef InitialClusteringStepBase ICSB;
  typedef PFClusterBuilderBase PFCBB;
  typedef PFCPositionCalculatorBase PosCalc;

public:
  PFClusterProducerCudaECAL(const edm::ParameterSet&);
  ~PFClusterProducerCudaECAL();

  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  //void endJob();
  //  void beginStream(edm::StreamID);
 
  bool initializeCudaMemory(int cudaDevice=0);
  void freeCudaMemory(int cudaDevice=0); 
  
  // inputs
  std::vector< std::vector<double> > theThresh;
  edm::EDGetTokenT<reco::PFRecHitCollection> _rechitsLabel;
  edm::EDGetTokenT< std::vector<double> > _trash;
 

  // options
  const bool _prodInitClusters;
  // the actual algorithm
  std::vector<std::unique_ptr<RecHitTopologicalCleanerBase> > _cleaners;
  std::unique_ptr<SeedFinderBase> _seedFinder;
  std::unique_ptr<InitialClusteringStepBase> _initialClustering;
  std::unique_ptr<PFClusterBuilderBase> _pfClusterBuilder;
  std::unique_ptr<PFCPositionCalculatorBase> _positionReCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPosCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _convergencePosCalc;
  std::unique_ptr<PFClusterEnergyCorrectorBase> _energyCorrector;

  TFile *MyFile = new TFile("EventECAL.root","recreate");
  
  reco::PFClusterCollection __initialClusters;
  reco::PFClusterCollection __pfClusters;
  reco::PFClusterCollection __pfClustersFromCuda;
  reco::PFRecHitCollection  __rechits;

  // rechit physics quantities 
  std::vector<int>    __rh_mask;
  std::vector<int>    __rh_isSeed;
  std::vector<double>  __rh_x;
  std::vector<double>  __rh_y;
  std::vector<double>  __rh_z;
  std::vector<double>  __rh_eta;
  std::vector<double>  __rh_phi;
  std::vector<double> __rh_pt2;
  // rechit neighbours4, neighbours8 vectors
  std::vector<std::vector<unsigned int>> __rh_neighbours8;

  TTree *clusterTree = new TTree("clusterTree", "clusterTree");
  
  TH1F *nIterations = new TH1F("nIter","nIterations Topo Clustering", 26,-0.5,25.5);
  TH2F *nIter_vs_nRH = new TH2F("nIternRH","nIterations vs num rechits Topo Clustering", 3001, -0.5, 3000.5, 26,-0.5,25.5);
  TH1F *nTopo_CPU = new TH1F("nTopo_CPU","nTopo_CPU",500,0,500);
  TH1F *nTopo_GPU = new TH1F("nTopo_GPU","nTopo_GPU",500,0,500);

  TH1F *sumSeed_CPU = new TH1F("sumSeed_CPU", "sumSeed_CPU",201, -0.5, 200.5);
  TH1F *sumSeed_GPU = new TH1F("sumSeed_GPU", "sumSeed_GPU",201, -0.5, 200.5);

  TH1F *topoEn_CPU = new TH1F("topoEn_CPU", "topoEn_CPU", 500, 0, 500);
  TH1F *topoEn_GPU = new TH1F("topoEn_GPU", "topoEn_GPU", 500, 0, 500);
  
  TH1F *topoEta_CPU = new TH1F("topoEta_CPU", "topoEta_CPU", 100, -3, 3);
  TH1F *topoEta_GPU = new TH1F("topoEta_GPU", "topoEta_GPU", 100, -3, 3);
  
  TH1F *topoPhi_CPU = new TH1F("topoPhi_CPU", "topoPhi_CPU", 100, -3.1415926, 3.1415926);
  TH1F *topoPhi_GPU = new TH1F("topoPhi_GPU", "topoPhi_GPU", 100, -3.1415926, 3.1415926);

  TH1F *nPFCluster_CPU = new TH1F("nPFCluster_CPU","nPFCluster_CPU",501,-0.5,500.5);
  TH1F *nPFCluster_GPU = new TH1F("nPFCluster_GPU","nPFCluster_GPU",501,-0.5,500.5);

  TH1F *enPFCluster_CPU = new TH1F("enPFCluster_CPU","enPFCluster_CPU",500,0,500);
  TH1F *enPFCluster_GPU = new TH1F("enPFCluster_GPU","enPFCluster_GPU",500,0,500);

  TH1F *pfcEta_CPU = new TH1F("pfcEta_CPU", "pfcEta_CPU", 100, -3, 3);
  TH1F *pfcEta_GPU = new TH1F("pfcEta_GPU", "pfcEta_GPU", 100, -3, 3);

  TH1F *pfcPhi_CPU = new TH1F("pfcPhi_CPU", "pfcPhi_CPU", 100, -3.1415926, 3.1415926);
  TH1F *pfcPhi_GPU = new TH1F("pfcPhi_GPU", "pfcPhi_GPU", 100, -3.1415926, 3.1415926);

  TH1F *nRH_perPFCluster_CPU = new TH1F("nRH_perPFCluster_CPU","nRH_perPFCluster_CPU",101,-0.5,100.5);
  TH1F *nRH_perPFCluster_GPU = new TH1F("nRH_perPFCluster_GPU","nRH_perPFCluster_GPU",101,-0.5,100.5);

  TH1F *matched_pfcRh_CPU = new TH1F("matched_pfcRh_CPU", "matching seed pfcRh_CPU", 101,-0.5,100.5);
  TH1F *matched_pfcRh_GPU = new TH1F("matched_pfcRh_GPU", "matching seed pfcRh_GPU", 101,-0.5,100.5);

  TH1F *matched_pfcEn_CPU = new TH1F("matched_pfcEn_CPU", "matching seed pfcEn_CPU", 500,0,500);  
  TH1F *matched_pfcEn_GPU = new TH1F("matched_pfcEn_GPU", "matching seed pfcEn_GPU", 500,0,500);

  TH1F *matched_pfcEta_CPU = new TH1F("matched_pfcEta_CPU", "matching seed pfcEta_CPU", 100, -3, 3);
  TH1F *matched_pfcEta_GPU = new TH1F("matched_pfcEta_GPU", "matching seed pfcEta_GPU", 100, -3, 3);

  TH1F *matched_pfcPhi_CPU = new TH1F("matched_pfcPhi_CPU", "matching seed pfcPhi_CPU", 100, -3.1415926, 3.1415926);
  TH1F *matched_pfcPhi_GPU = new TH1F("matched_pfcPhi_GPU", "matching seed pfcPhi_GPU", 100, -3.1415926, 3.1415926);

  TH2F *nRh_CPUvsGPU = new TH2F("nRh_CPUvsGPU","nRh_CPUvsGPU",101,-0.5,100.5,101,-0.5,100.5);
  TH2F *enPFCluster_CPUvsGPU = new TH2F("enPFCluster_CPUvsGPU","enPFCluster_CPUvsGPU",50,0,500,50,0,500);

  bool doComparison=true;
  //bool doComparison=false;

  TH1F *deltaSumSeed  = new TH1F("deltaSumSeed", "sumSeed_{GPU} - sumSeed_{CPU}", 201, -100.5, 100.5);
  TH1F *deltaRH  = new TH1F("deltaRH", "nRH_{GPU} - nRH_{CPU}", 41, -20.5, 20.5);
  TH1F *deltaEn  = new TH1F("deltaEn", "E_{GPU} - E_{CPU}", 200, -10, 10);
  TH1F *deltaEta = new TH1F("deltaEta", "#eta_{GPU} - #eta_{CPU}", 200, -0.2, 0.2);
  TH1F *deltaPhi = new TH1F("deltaPhi", "#phi_{GPU} - #phi_{CPU}", 200, -0.2, 0.2);

  TH2F *coordinate = new TH2F("coordinate","coordinate",100,-3,3,100,-3.1415926,3.14159);
  TH1F *layer = new TH1F("layer","layer",7,0,7);

  TH1F *timer = new TH1F("timer", "GPU kernel timer", 1000, 0.0, 1.0);
  std::array<float,9> GPU_timers;
  Int_t numEvents = 0;
  Int_t nIter = 0;
  Int_t nEdges = 0;

//  int maxRH = 2000;     // Max number of rechits
//  int maxPFCSize = 75;  // Max number of rechits per pf cluster
//  int maxNeighbors = 8; // Max number of rechit neighbors for edge list

  PFClustering::ConfigurationParameters cudaConfig_;
  PFClustering::InputDataCPU inputCPU;
  PFClustering::InputDataGPU inputGPU;
  

  int *h_nIter = nullptr;
  int *d_nIter = nullptr;
};

DEFINE_FWK_MODULE(PFClusterProducerCudaECAL);

#endif
