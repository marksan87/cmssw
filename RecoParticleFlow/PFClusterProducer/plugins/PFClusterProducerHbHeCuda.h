#ifndef __newpf_PFClusterProducer_H__
#define __newpf_PFClusterProducer_H__

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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <memory>

class PFClusterProducerHbHeCuda : public edm::stream::EDProducer<> {
  typedef RecHitTopologicalCleanerBase RHCB;
  typedef InitialClusteringStepBase ICSB;
  typedef PFClusterBuilderBase PFCBB;
  typedef PFCPositionCalculatorBase PosCalc;

public:
  PFClusterProducerHbHeCuda(const edm::ParameterSet&);
  ~PFClusterProducerHbHeCuda();

  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  //void endJob();
  //  void beginStream(edm::StreamID);
 
  TFile *MyFile = new TFile("Event.root","recreate");
  TH1F *nRecHits = new TH1F("nRecHits","nRecHits",5000,0,5000);
  TH1F *nTopoCluster = new TH1F("nTopoCluster","nTopoCluster",5000,0,5000);
  TH1F *nSeeds = new TH1F("nSeeds","nSeeds",5000,0,5000);
  TH1F *nSeedsPerTopoCluster = new TH1F("nSeedsPerTopoCluster","nSeedsPerTopoCluster",200,0,200);
  TH1F *nRecHitsPerTopoCluster = new TH1F("nRecHitsPerTopoCluster","nRecHitsPerTopoCluster",5000,0,5000);
  TH1F *nRecHitsPerPfCluster = new TH1F("nRecHitsPerPfCluster","nRecHitsPerPfCluster",5000,0,5000);

  TH2F *theMap = new TH2F("theMap","theMap",65,-2.825,2.825,72,-3.141592,3.1415926);

  TH2F *PFvsTopo = new TH2F("PFvsTopo","PFvsTopo",1200,0,1200,1200,0,1200);

  TH1F *nRhDiff = new TH1F("nRhDiff","nRhDiff",20,0,20);
  
  // inputs
  std::vector< std::vector<double> > theThresh;
  edm::EDGetTokenT<reco::PFRecHitCollection> _rechitsLabel;
  edm::EDGetTokenT< std::vector<double> > _trash;
  const std::vector<int> HE_depths;
  std::vector<float> HE_seedingThreshold;
  const std::vector<float> HE_seedingThresholdPt;
  const std::vector<int> HB_depths;
  std::vector<float> HB_seedingThreshold;
  const std::vector<float> HB_seedingThresholdPt;
  // options
  const bool _prodInitClusters;
  // the actual algorithm
  std::vector<std::unique_ptr<RecHitTopologicalCleanerBase> > _cleaners;
  std::unique_ptr<SeedFinderBase> _seedFinder;
  std::unique_ptr<InitialClusteringStepBase> _initialClustering;
  std::unique_ptr<PFClusterBuilderBase> _pfClusterBuilder;
  std::unique_ptr<PFCPositionCalculatorBase> _positionReCalc;
  std::unique_ptr<PFClusterEnergyCorrectorBase> _energyCorrector;
};

DEFINE_FWK_MODULE(PFClusterProducerHbHeCuda);

#endif
