#ifndef __newpf_PFClusterProducerCudaHCAL_H__
#define __newpf_PFClusterProducerCudaHCAL_H__

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

#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <memory>
#include <array>
#include <algorithm>

class PFClusterProducerCudaHCAL : public edm::stream::EDProducer<> {
  typedef RecHitTopologicalCleanerBase RHCB;
  typedef InitialClusteringStepBase ICSB;
  typedef PFClusterBuilderBase PFCBB;
  typedef PFCPositionCalculatorBase PosCalc;

public:
  PFClusterProducerCudaHCAL(const edm::ParameterSet&);
  ~PFClusterProducerCudaHCAL();

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
  std::unique_ptr<PFCPositionCalculatorBase> _positionCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPositionCalc;
  std::unique_ptr<PFClusterEnergyCorrectorBase> _energyCorrector;

  TFile *MyFile = new TFile("EventHCAL.root","recreate");

  reco::PFClusterCollection __initialClusters;
  reco::PFClusterCollection __pfClusters;
  reco::PFClusterCollection __pfClustersFromCuda;
  reco::PFRecHitCollection  __rechits;
  
  // rechit pt^2
  std::vector<int>    __rh_mask;
  std::vector<int>    __rh_isSeed;
  std::vector<float>  __rh_x;
  std::vector<float>  __rh_y;
  std::vector<float>  __rh_z;
  std::vector<float>  __rh_eta;
  std::vector<float>  __rh_phi;
  std::vector<double> __rh_pt2;
  // rechit neighbours4, neighbours8 vectors
  std::vector<std::vector<int>> __rh_neighbours4;
  std::vector<std::vector<int>> __rh_neighbours8;


  TTree *clusterTree = new TTree("clusterTree", "clusterTree");

  TH1F *nIterations = new TH1F("nIter","nIterations Topo Clustering", 26,-0.5,25.5);
  TH2F *nIter_vs_nRH = new TH2F("nIternRH","nIterations vs num rechits Topo Clustering", 3001, -0.5, 3000.5, 26,-0.5,25.5);
  TH1F *nTopo_CPU = new TH1F("nTopo_CPU","nTopo_CPU",501,-0.5,500.5);
  TH1F *nTopo_GPU = new TH1F("nTopo_GPU","nTopo_GPU",501,-0.5,500.5);

  TH1F *sumSeed_CPU = new TH1F("sumSeed_CPU", "sumSeed_CPU",201, -0.5, 200.5);
  TH1F *sumSeed_GPU = new TH1F("sumSeed_GPU", "sumSeed_GPU",201, -0.5, 200.5);

  TH1F *topoEn_CPU = new TH1F("topoEn_CPU", "topoEn_CPU", 500, 0, 500);
  TH1F *topoEn_GPU = new TH1F("topoEn_GPU", "topoEn_GPU", 500, 0, 500);

  TH1F *topoEta_CPU = new TH1F("topoEta_CPU", "topoEta_CPU", 100, -3, 3);
  TH1F *topoEta_GPU = new TH1F("topoEta_GPU", "topoEta_GPU", 100, -3, 3);

  TH1F *topoPhi_CPU = new TH1F("topoPhi_CPU", "topoPhi_CPU", 100, -3.1415926, 3.1415926);
  TH1F *topoPhi_GPU = new TH1F("topoPhi_GPU", "topoPhi_GPU", 100, -3.1415926, 3.1415926);

  TH1F *nPFCluster_CPU = new TH1F("nPFCluster_CPU","nPFCluster_CPU",1001,-0.5,1000.5);
  TH1F *nPFCluster_GPU = new TH1F("nPFCluster_GPU","nPFCluster_GPU",1001,-0.5,1000.5);

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
  TH1F *enPFCluster_CPUvsGPU_1d = new TH1F("enPFCluster_CPUvsGPU_1d","enPFCluster_CPUvsGPU_1d",400,-2,2);

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

  bool *h_notDone = nullptr;    // Host pointer to flag in pinned memory
  bool *d_notDone = nullptr;    // Device pointer to flag in pinned memory

  int *h_nIter = nullptr;
  int *d_nIter = nullptr;
};

DEFINE_FWK_MODULE(PFClusterProducerCudaHCAL);

#endif
