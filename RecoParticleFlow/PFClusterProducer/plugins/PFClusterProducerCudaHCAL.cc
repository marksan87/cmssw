#include "PFClusterProducerCudaHCAL.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterCudaHCAL.h"
#include <TFile.h>
#include <TH1F.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"



#ifdef PFLOW_DEBUG
#define LOGVERB(x) edm::LogVerbatim(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) edm::LogInfo(x)
#else
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

// Uncomment to enable GPU debugging
//#define DEBUG_GPU_HCAL

// Uncomment to fill TTrees
//#define DEBUG_HCAL_TREES

// Uncomment to save cluster collections in TTree
//#define DEBUG_SAVE_CLUSTERS


PFClusterProducerCudaHCAL::PFClusterProducerCudaHCAL(const edm::ParameterSet& conf)
  : _prodInitClusters(conf.getUntrackedParameter<bool>("prodInitialClusters", false)) {
  _rechitsLabel = consumes<reco::PFRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsSource"));
  edm::ConsumesCollector cc = consumesCollector();

  pfcIterations->GetXaxis()->SetTitle("PF clustering iterations");
  pfcIterations->GetYaxis()->SetTitle("Entries");

  pfcIter_vs_nRHTopo->GetXaxis()->SetTitle("Num rechits in topo cluster");
  pfcIter_vs_nRHTopo->GetYaxis()->SetTitle("PF clustering iterations");
  
  pfcIter_vs_nSeedsTopo->GetXaxis()->SetTitle("Num seeds in topo cluster");
  pfcIter_vs_nSeedsTopo->GetYaxis()->SetTitle("PF clustering iterations");
  
  pfcIter_vs_nFracsTopo->GetXaxis()->SetTitle("Num rechit fractions in topo cluster");
  pfcIter_vs_nFracsTopo->GetYaxis()->SetTitle("PF clustering iterations");
  
  topoIterations->GetXaxis()->SetTitle("Topo clustering iterations");
  topoIterations->GetYaxis()->SetTitle("Entries");

  topoIter_vs_nRH->GetXaxis()->SetTitle("Num rechits");
  topoIter_vs_nRH->GetYaxis()->SetTitle("Topo clustering iterations");

#ifdef DEBUG_HCAL_TREES
  //setup TTree
  clusterTree->Branch("Event", &numEvents);
  clusterTree->Branch("topoIter", &topoIter, "topoIter/I");
  clusterTree->Branch("nEdges", &nEdges, "nEdges/I");
  clusterTree->Branch("nFracs", &nFracs, "nFracs/I");
  clusterTree->Branch("nRHperPFCTotal_CPU", &nRHperPFCTotal_CPU, "nRHperPFCTotal_CPU/I");
  clusterTree->Branch("nRHperPFCTotal_GPU", &nRHperPFCTotal_GPU, "nRHperPFCTotal_GPU/I");
  clusterTree->Branch("timers", &GPU_timers);
  clusterTree->Branch("pfcIter", &__pfcIter);
  clusterTree->Branch("nRHTopo", &__nRHTopo);
  clusterTree->Branch("nSeedsTopo", &__nSeedsTopo);
  clusterTree->Branch("nFracsTopo", &__nFracsTopo);
  clusterTree->Branch("rechits", "PFRecHitCollection", &__rechits);
  clusterTree->Branch("rechits_mask", &__rh_mask);
  clusterTree->Branch("rechits_isSeed", &__rh_isSeed);
  clusterTree->Branch("rechits_x", &__rh_x);
  clusterTree->Branch("rechits_y", &__rh_y);
  clusterTree->Branch("rechits_z", &__rh_z);
  clusterTree->Branch("rechits_eta", &__rh_eta);
  clusterTree->Branch("rechits_phi", &__rh_phi);
  clusterTree->Branch("rechits_pt2", &__rh_pt2);
  clusterTree->Branch("rechits_neighbours4", &__rh_neighbours4);
  clusterTree->Branch("rechits_neighbours8", &__rh_neighbours8);
#endif
#if defined DEBUG_HCAL_TREES && defined DEBUG_SAVE_CLUSTERS
  clusterTree->Branch("initialClusters", "PFClusterCollection", &__initialClusters);
  clusterTree->Branch("pfClusters", "PFClusterCollection", &__pfClusters);
  clusterTree->Branch("pfClustersFromCuda", "PFClusterCollection", &__pfClustersFromCuda);
#endif

  //setup rechit cleaners
  const edm::VParameterSet& cleanerConfs = conf.getParameterSetVector("recHitCleaners");

  for (const auto& conf : cleanerConfs) {
    const std::string& cleanerName = conf.getParameter<std::string>("algoName");
    _cleaners.emplace_back(RecHitTopologicalCleanerFactory::get()->create(cleanerName, conf, cc));
  }

  // setup seed finding
  const edm::ParameterSet& sfConf = conf.getParameterSet("seedFinder");
  const std::string& sfName = sfConf.getParameter<std::string>("algoName");
  _seedFinder = SeedFinderFactory::get()->create(sfName, sfConf);

  const edm::VParameterSet& seedFinderConfs = sfConf.getParameterSetVector("thresholdsByDetector"); 
  
  float minFracInCalc = 0.0, minAllowedNormalization = 0.0;

  //setup topo cluster builder
  const edm::ParameterSet& initConf = conf.getParameterSet("initialClusteringStep");
  const std::string& initName = initConf.getParameter<std::string>("algoName");
  _initialClustering = InitialClusteringStepFactory::get()->create(initName, initConf, cc);
  //setup pf cluster builder if requested
  const edm::ParameterSet& pfcConf = conf.getParameterSet("pfClusterBuilder");
  if (!pfcConf.empty()) {
    const std::string& pfcName = pfcConf.getParameter<std::string>("algoName");
    _pfClusterBuilder = PFClusterBuilderFactory::get()->create(pfcName, pfcConf, cc);
    /*if (pfcConf.exists("allCellsPositionCalc")) {
    const edm::ParameterSet& acConf = pfcConf.getParameterSet("allCellsPositionCalc");
    const std::string& algoac = acConf.getParameter<std::string>("algoName");
    _allCellsPosCalcCuda = PFCPositionCalculatorFactory::get()->create(algoac, acConf);*/
    
    if (pfcConf.exists("positionCalc")) {
        const edm::ParameterSet& acConf = pfcConf.getParameterSet("positionCalc");
        const std::string& algoac = acConf.getParameter<std::string>("algoName");
        _positionCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf, cc);
        minFracInCalc = (float)acConf.getParameter<double>("minFractionInCalc");
        minAllowedNormalization = (float)acConf.getParameter<double>("minAllowedNormalization");
    }

    if (pfcConf.exists("allCellsPositionCalc")) {
        const edm::ParameterSet& acConf = pfcConf.getParameterSet("allCellsPositionCalc");
        const std::string& algoac = acConf.getParameter<std::string>("algoName");
        _allCellsPositionCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf, cc);
    }
  }
  //setup (possible) recalcuation of positions
  const edm::ParameterSet& pConf = conf.getParameterSet("positionReCalc");
  if (!pConf.empty()) {
    const std::string& pName = pConf.getParameter<std::string>("algoName");
    _positionReCalc = PFCPositionCalculatorFactory::get()->create(pName, pConf, cc);
  }
  // see if new need to apply corrections, setup if there.
  const edm::ParameterSet& cConf = conf.getParameterSet("energyCorrector");
  if (!cConf.empty()) {
    const std::string& cName = cConf.getParameter<std::string>("algoName");
    _energyCorrector = PFClusterEnergyCorrectorFactory::get()->create(cName, cConf);
  }

  
  float showerSigma2 = (float)std::pow(pfcConf.getParameter<double>("showerSigma"), 2.0);
  float recHitEnergyNormInvEB_vec[4], recHitEnergyNormInvEE_vec[7];
  const auto recHitEnergyNormConf = pfcConf.getParameterSetVector("recHitEnergyNorms");
  for (const auto& pset : recHitEnergyNormConf)
  {
    const std::string& det = pset.getParameter<std::string>("detector");
    if (det == std::string("HCAL_BARREL1")) {
      const auto& recHitENorms = pset.getParameter<std::vector<double> >("recHitEnergyNorm");
      std::copy(recHitENorms.begin(), recHitENorms.end(), recHitEnergyNormInvEB_vec);
      for (auto& x : recHitEnergyNormInvEB_vec) x = std::pow(x, -1); // Invert these values 
    }
    else if (det == std::string("HCAL_ENDCAP")) {
      const auto& recHitENorms = pset.getParameter<std::vector<double> >("recHitEnergyNorm");
      std::copy(recHitENorms.begin(), recHitENorms.end(), recHitEnergyNormInvEE_vec);
      for (auto& x : recHitEnergyNormInvEE_vec) x = std::pow(x, -1); // Invert these values 
    }
    else
      std::cout<<"Unknown detector when parsing recHitEnergyNorm: "<<det<<std::endl;
  }
  //float recHitEnergyNormEB = 0.08;
  //float recHitEnergyNormEE = 0.3;
  //float minFracToKeep = 0.0000001;
  float minFracToKeep = (float)pfcConf.getParameter<double>("minFractionToKeep");
  float minFracTot = (float)pfcConf.getParameter<double>("minFracTot");

  // Max PFClustering iterations
  unsigned maxIterations = pfcConf.getParameter<unsigned>("maxIterations");

  bool excludeOtherSeeds = pfcConf.getParameter<bool>("excludeOtherSeeds");

  //float stoppingTolerance2 = (float)std::pow(pfcConf.getParameter<double>("stoppingTolerance"), 2.0);
  float stoppingTolerance = (float)pfcConf.getParameter<double>("stoppingTolerance");


  float seedEThresholdEB_vec[4], seedEThresholdEE_vec[7], seedPt2ThresholdEB = -1, seedPt2ThresholdEE = -1;
  for (const auto& pset : seedFinderConfs)
  {
    const std::string& det = pset.getParameter<std::string>("detector");
    if (det == std::string("HCAL_BARREL1")) {
      const auto& thresholds = pset.getParameter<std::vector<double> >("seedingThreshold");
      std::copy(thresholds.begin(), thresholds.end(), seedEThresholdEB_vec);
      seedPt2ThresholdEB = (float)std::pow(pset.getParameter<std::vector<double> >("seedingThresholdPt")[0], 2.0); 

    }
    else if (det == std::string("HCAL_ENDCAP")) {
      const auto& thresholds = pset.getParameter<std::vector<double> >("seedingThreshold");
      std::copy(thresholds.begin(), thresholds.end(), seedEThresholdEE_vec);
      seedPt2ThresholdEE = (float)std::pow(pset.getParameter<std::vector<double> >("seedingThresholdPt")[0], 2.0);
    }
    else
      std::cout<<"Unknown detector when parsing seedFinder: "<<det<<std::endl;
  }
  
  float topoEThresholdEB_vec[4], topoEThresholdEE_vec[7];

  const auto topoThresholdConf = initConf.getParameterSetVector("thresholdsByDetector");
  for (const auto& pset : topoThresholdConf)
  {
    const std::string& det = pset.getParameter<std::string>("detector");
    if (det == std::string("HCAL_BARREL1")) {
      const auto& thresholds = pset.getParameter<std::vector<double> >("gatheringThreshold");
      std::copy(thresholds.begin(), thresholds.end(), topoEThresholdEB_vec);
    }
    else if (det == std::string("HCAL_ENDCAP")) {
      const auto& thresholds = pset.getParameter<std::vector<double> >("gatheringThreshold");
      std::copy(thresholds.begin(), thresholds.end(), topoEThresholdEE_vec);
    }
    else
      std::cout<<"Unknown detector when parsing initClusteringStep: "<<det<<std::endl;
  }
  
  if (pfcConf.exists("timeResolutionCalcEndcap")) {
      const edm::ParameterSet& endcapTimeResConf = pfcConf.getParameterSet("timeResolutionCalcEndcap");
      endcapTimeResConsts.corrTermLowE = (float)endcapTimeResConf.getParameter<double>("corrTermLowE");
      endcapTimeResConsts.threshLowE = (float)endcapTimeResConf.getParameter<double>("threshLowE");
      endcapTimeResConsts.noiseTerm = (float)endcapTimeResConf.getParameter<double>("noiseTerm");
      endcapTimeResConsts.constantTermLowE2 = (float)std::pow(endcapTimeResConf.getParameter<double>("constantTermLowE"), 2.0);
      endcapTimeResConsts.noiseTermLowE = (float)endcapTimeResConf.getParameter<double>("noiseTermLowE");
      endcapTimeResConsts.threshHighE = (float)endcapTimeResConf.getParameter<double>("threshHighE");
      endcapTimeResConsts.constantTerm2 = (float)std::pow(endcapTimeResConf.getParameter<double>("constantTerm"), 2.0);
      endcapTimeResConsts.resHighE2 = (float)std::pow(endcapTimeResConsts.noiseTerm / endcapTimeResConsts.threshHighE, 2.0) + endcapTimeResConsts.constantTerm2;
  }

  if (pfcConf.exists("timeResolutionCalcBarrel")) {
      const edm::ParameterSet& barrelTimeResConf = pfcConf.getParameterSet("timeResolutionCalcBarrel");
      barrelTimeResConsts.corrTermLowE = (float)barrelTimeResConf.getParameter<double>("corrTermLowE");
      barrelTimeResConsts.threshLowE = (float)barrelTimeResConf.getParameter<double>("threshLowE");
      barrelTimeResConsts.noiseTerm = (float)barrelTimeResConf.getParameter<double>("noiseTerm");
      barrelTimeResConsts.constantTermLowE2 = (float)std::pow(barrelTimeResConf.getParameter<double>("constantTermLowE"), 2.0);
      barrelTimeResConsts.noiseTermLowE = (float)barrelTimeResConf.getParameter<double>("noiseTermLowE");
      barrelTimeResConsts.threshHighE = (float)barrelTimeResConf.getParameter<double>("threshHighE");
      barrelTimeResConsts.constantTerm2 = (float)std::pow(barrelTimeResConf.getParameter<double>("constantTerm"), 2.0);
      barrelTimeResConsts.resHighE2 = (float)std::pow(barrelTimeResConsts.noiseTerm / barrelTimeResConsts.threshHighE, 2.0) + barrelTimeResConsts.constantTerm2;
  }

  int nNeigh = sfConf.getParameter<int>("nNeighbours");

  if (!PFClusterCudaHCAL::initializeCudaConstants(showerSigma2,
                                             recHitEnergyNormInvEB_vec,
                                             recHitEnergyNormInvEE_vec,
                                             minFracToKeep,
                                             minFracTot,
                                             minFracInCalc,
                                             minAllowedNormalization,
                                             maxIterations,
                                             stoppingTolerance,
                                             excludeOtherSeeds,
                                             seedEThresholdEB_vec,
                                             seedEThresholdEE_vec,
                                             seedPt2ThresholdEB,
                                             seedPt2ThresholdEE,
                                             topoEThresholdEB_vec,
                                             topoEThresholdEE_vec,
                                             endcapTimeResConsts,
                                             barrelTimeResConsts,
                                             nNeigh,
                                             maxSize)) {
  
    std::cout<<"Unable to initialize Cuda constants"<<std::endl;
    return;
  }
  
  //cudaSetDeviceFlags(cudaDeviceMapHost);
  if (!PFClusterProducerCudaHCAL::initializeCudaMemory()) {
    // Problem allocating Cuda memory
    std::cout<<"Unable to allocate Cuda memory"<<std::endl;
    return;
  }
 
  cudaStream_t cudaStream = 0;  // TODO: Get from cms::cuda::ScopedContextAcquire
  //inputCPU.allocate(cudaConfig_, cudaStream);
  inputGPU.allocate(cudaConfig_, cudaStream);

  if (_prodInitClusters) {
    produces<reco::PFClusterCollection>("initialClusters");
  }
  produces<reco::PFClusterCollection>();
}

PFClusterProducerCudaHCAL::~PFClusterProducerCudaHCAL()
{
  // Free Cuda memory
  freeCudaMemory();

  MyFile->cd();
#ifdef DEBUG_HCAL_TREES  
  clusterTree->Write();
#endif
  pfcIterations->Write();
  pfcIter_vs_nRHTopo->Write();
  pfcIter_vs_nSeedsTopo->Write();
  pfcIter_vs_nFracsTopo->Write();
  topoIterations->Write();
  topoIter_vs_nRH->Write();
  nTopo_CPU->Write();
  nTopo_GPU->Write();
  topoSeeds_CPU->Write();
  topoSeeds_GPU->Write();
  sumSeed_CPU->Write();
  sumSeed_GPU->Write();
  topoEn_CPU->Write();
  topoEn_GPU->Write();
  topoEta_CPU->Write();
  topoEta_GPU->Write();
  topoPhi_CPU->Write();
  topoPhi_GPU->Write();
  nPFCluster_CPU->Write();
  nPFCluster_GPU->Write();
  enPFCluster_CPU->Write();
  enPFCluster_GPU->Write();
  pfcEta_CPU->Write();
  pfcEta_GPU->Write();
  pfcPhi_CPU->Write();
  pfcPhi_GPU->Write();
  nRH_perPFCluster_CPU->Write();
  nRH_perPFCluster_GPU->Write();
  nRH_perPFClusterTotal_CPU->Write();
  nRH_perPFClusterTotal_GPU->Write();
  matched_pfcRh_CPU->Write();
  matched_pfcRh_GPU->Write();
  matched_pfcEn_CPU->Write();
  matched_pfcEn_GPU->Write();
  matched_pfcEta_CPU->Write();
  matched_pfcEta_GPU->Write();
  matched_pfcPhi_CPU->Write();
  matched_pfcPhi_GPU->Write();
  nRh_CPUvsGPU->Write();
  enPFCluster_CPUvsGPU->Write();
  enPFCluster_CPUvsGPU_1d->Write();
  coordinate->Write();
  layer->Write();
  deltaSumSeed->Write();
  deltaRH->Write();
  deltaEn->Write();
  deltaEta->Write();
  deltaPhi->Write();
  if (numEvents > 10) {
      // Skip first 10 entries
      hTimers->Scale(1. / (numEvents - 10.));
  }
  hTimers->Write();
  // MyFile->Close();
  delete MyFile;
}

bool PFClusterProducerCudaHCAL::initializeCudaMemory(int cudaDevice) {
  h_topoIter = new int(0);
  cudaCheck(cudaMalloc(&d_topoIter, sizeof(int)));
  
  h_pcrhFracSize = new int(0);
  cudaCheck(cudaMalloc(&d_pcrhFracSize, sizeof(int)));

  h_cuda_pfc_iter = new int(0);
  cudaCheck(cudaMalloc(&d_cuda_pfc_iter, sizeof(int)));

  /*
    if (!cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&h_notDone), sizeof(bool), cudaHostAllocMapped))) {
    h_notDone = d_notDone = nullptr;
    return false;
  }
  */
  return true;
}

void PFClusterProducerCudaHCAL::freeCudaMemory(int cudaDevice) {
    cudaCheck(cudaFree(d_topoIter));
    cudaCheck(cudaFree(d_pcrhFracSize));
    cudaCheck(cudaFree(d_cuda_pfc_iter));

    delete h_topoIter;
    delete h_pcrhFracSize;
    delete h_cuda_pfc_iter;
}

void PFClusterProducerCudaHCAL::beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& es) {
  _initialClustering->update(es);
  if (_pfClusterBuilder)
    _pfClusterBuilder->update(es);
  if (_positionReCalc)
    _positionReCalc->update(es);
}

void PFClusterProducerCudaHCAL::produce(edm::Event& e, const edm::EventSetup& es) {
  _initialClustering->reset();
  if (_pfClusterBuilder)
    _pfClusterBuilder->reset();

  edm::Handle<reco::PFRecHitCollection> rechits;
  e.getByToken(_rechitsLabel, rechits);
  std::cout<<"\n===== Now on event "<<numEvents<<" with "<<rechits->size()<<" HCAL rechits ====="<<std::endl;

#ifdef DEBUG_HCAL_TREES
  GPU_timers.fill(0.0);
  __pfcIter.clear();
  __nRHTopo.clear();
  __nSeedsTopo.clear();
  __nFracsTopo.clear();
  __rechits = *rechits;
  __rh_mask.clear();
  __rh_isSeed.clear();
  __rh_x.clear();
  __rh_y.clear();
  __rh_z.clear();
  __rh_eta.clear();
  __rh_phi.clear();
  __rh_pt2.clear();
  __rh_neighbours4.clear();
  __rh_neighbours8.clear();
#endif

  _initialClustering->updateEvent(e);

  std::vector<bool> mask(rechits->size(), true);

  for (auto isMasked: mask) {
    __rh_mask.push_back(isMasked);
  }

  size_t rh_size = rechits->size();

  std::vector<float>                                    h_cuda_fracsum=std::vector<float>(rh_size,0);
  std::vector<int>                                      h_cuda_rhcount=std::vector<int>(rh_size,1);

  std::vector<float>                                    h_cuda_pcRhFrac(maxSize,-1.);
  std::vector<int>                                      h_cuda_pfNeighEightInd(rechits->size()*8,-1);
  std::vector<int>                                      h_cuda_pfNeighFourInd(rechits->size()*4,-1);
  std::vector<int>                                      h_cuda_pcRhFracInd(maxSize,-1);

  std::vector<float>                                    h_cuda_pfrh_x(rechits->size(),0);
  std::vector<float>                                    h_cuda_pfrh_y(rechits->size(),0);
  std::vector<float>                                    h_cuda_pfrh_z(rechits->size(),0);
  std::vector<float>                                    h_cuda_pfrh_energy(rechits->size(),0);
  std::vector<float>                                    h_cuda_pfrh_pt2(rechits->size(),0);
  std::vector<int>                                      h_cuda_pfrh_topoId(rechits->size(),0);//-1);
  std::vector<int>                                      h_cuda_pfrh_isSeed(rechits->size(),0);
  std::vector<int>                                      h_cuda_pfrh_layer(rechits->size(),-999);
  std::vector<int>                                      h_cuda_pfrh_depth(rechits->size(),-999);

  std::vector<int>                                      h_cuda_pfrh_edgeId(rechits->size()*8, -1);      // Rechit index for each edge 
  std::vector<int>                                      h_cuda_pfrh_edgeList(rechits->size()*8, -1);    // Sorted list of 8 neighbours for each rechit 

  int numbytes_float = rh_size*sizeof(float);
  int numbytes_int = rh_size*sizeof(int);
/*
  auto d_cuda_rhcount = cms::cuda::make_device_unique<int[]>(numbytes_int, nullptr);
  auto d_cuda_fracsum = cms::cuda::make_device_unique<float[]>(numbytes_float, nullptr);

  float*                                    d_cuda_pfrh_x;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_x, numbytes_float));
  float*                                    d_cuda_pfrh_y;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_y, numbytes_float));
  float*                                    d_cuda_pfrh_z;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_z, numbytes_float));
  float*                                    d_cuda_pfrh_energy;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_energy, numbytes_float));
  float*                                    d_cuda_pfrh_pt2;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_pt2, numbytes_float));
  int*                                      d_cuda_pfrh_topoId;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_topoId, numbytes_int));
  int*                                      d_cuda_pfrh_isSeed;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_isSeed, numbytes_int));
  int*                                      d_cuda_pfrh_layer;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_layer, numbytes_int));
  int*                                      d_cuda_pfrh_depth;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_depth, numbytes_int));
  int*                                      d_cuda_pfNeighEightInd;
  cudaCheck(cudaMalloc(&d_cuda_pfNeighEightInd, numbytes_int*8));
  int*                                      d_cuda_pfNeighFourInd;
  cudaCheck(cudaMalloc(&d_cuda_pfNeighFourInd, numbytes_int*4));

  bool*                                      d_cuda_pfrh_passTopoThresh;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_passTopoThresh, sizeof(bool)*rh_size));

  int *d_cuda_pcRhFracInd;
  cudaCheck(cudaMalloc(&d_cuda_pcRhFracInd, sizeof(int)*maxSize));
  float *d_cuda_pcRhFrac;
  cudaCheck(cudaMalloc(&d_cuda_pcRhFrac, sizeof(float)*maxSize));

  int *d_cuda_topoSeedCount;
  cudaCheck(cudaMalloc(&d_cuda_topoSeedCount, numbytes_int));
  int *d_cuda_topoRHCount;
  cudaCheck(cudaMalloc(&d_cuda_topoRHCount, numbytes_int));

  // Offsets for seeds and rechit fractions in pcRhFrac lists
  int *d_cuda_seedFracOffsets;
  cudaCheck(cudaMalloc(&d_cuda_seedFracOffsets, numbytes_int));

  // Offsets for seed lists by topo ID
  int* d_cuda_topoSeedOffsets;
  cudaCheck(cudaMalloc(&d_cuda_topoSeedOffsets, numbytes_int));

  // List of seeds indexed by topo ID
  int* d_cuda_topoSeedList;
  cudaCheck(cudaMalloc(&d_cuda_topoSeedList, numbytes_int));

  int* d_cuda_pfc_iter;
  cudaCheck(cudaMalloc(&d_cuda_pfc_iter, numbytes_int));
*/
  int p=0; 
  /*
  std::cout<<"-----------------------------------------"<<std::endl;
  std::cout<<" HCAL: Event "<<numEvents<<" has "<<rechits->size()<<" rechits"<<std::endl;
  std::cout<<"-----------------------------------------"<<std::endl;
  */
  int totalNeighbours = 0;     // Running count of 8 neighbour edges for edgeId, edgeList
  for (auto rh: *rechits){
    //std::cout<<"*** Now on rechit \t"<<p<<"\tdetId = "<<rh.detId()<<"\tneighbourInfos().size() = "<<rh.neighbourInfos().size()<<"\tneighbours4().size() = "<<rh.neighbours4().size()<<"\tneighbours8().size() = "<<rh.neighbours8().size()<<std::endl;
    //std::cout<<"*** Now on rechit \t"<<p<<"\tdetId = "<<rh.detId()<<"\tneighbours8().size() = "<<rh.neighbours8().size()<<std::endl;
    h_cuda_pfrh_x[p]=rh.position().x();
    h_cuda_pfrh_y[p]=rh.position().y();
    h_cuda_pfrh_z[p]=rh.position().z();
    h_cuda_pfrh_energy[p]=rh.energy();
    h_cuda_pfrh_pt2[p]=rh.pt2();
    h_cuda_pfrh_layer[p]=(int)rh.layer();
    h_cuda_pfrh_depth[p]=(int)rh.depth();
    h_cuda_pfrh_topoId[p]=p;

    auto theneighboursEight = rh.neighbours8();
    auto theneighboursFour = rh.neighbours4();
    __rh_x.push_back(h_cuda_pfrh_x[p]);
    __rh_y.push_back(h_cuda_pfrh_y[p]);
    __rh_z.push_back(h_cuda_pfrh_z[p]);
    __rh_eta.push_back(rh.positionREP().eta());
    __rh_phi.push_back(rh.positionREP().phi()); 
    __rh_pt2.push_back(h_cuda_pfrh_pt2[p]);
    std::vector<int> n4;
    std::vector<int> n8;
    int z = 0;
    for(auto nh: theneighboursEight)
      {
        n8.push_back((int)nh);
	  }
    std::sort(n8.begin(), n8.end());    // Sort 8 neighbour edges in ascending order for topo clustering

    for (auto nh: n8) {
    h_cuda_pfNeighEightInd[8*p+z] = (int)nh;
	h_cuda_pfrh_edgeId[totalNeighbours] = p;
    h_cuda_pfrh_edgeList[totalNeighbours] = (int)nh;
    totalNeighbours++;
    z++;
      }
    
    int y = 0;
    for(auto nh: theneighboursFour)
      {
        n4.push_back((int)nh);
        h_cuda_pfNeighFourInd[4*p+y] = (int)nh;
        y++;
      }

    p++;
    __rh_neighbours4.push_back(n4);
    __rh_neighbours8.push_back(n8);
  }//end of rechit loop  


  nEdges = totalNeighbours;
/* 
  // Allocate Cuda memory
  int*  d_cuda_pfrh_edgeId;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_edgeId, sizeof(int) * totalNeighbours));
  
  int*  d_cuda_pfrh_edgeList;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_edgeList, sizeof(int) * totalNeighbours));
 
  int*                                      d_cuda_pfrh_edgeMask;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_edgeMask, sizeof(int) * totalNeighbours));
  
  // PF cluster position vector (cartesian 3 vector with w component used for position_norm)
  float4* d_cuda_pfc_pos4;
  cudaCheck(cudaMalloc(&d_cuda_pfc_pos4, sizeof(float4) * rh_size));
  
  float4* d_cuda_pfc_prevPos4;
  cudaCheck(cudaMalloc(&d_cuda_pfc_prevPos4, sizeof(float4) * rh_size));

  float* d_cuda_pfc_energy;
  cudaCheck(cudaMalloc(&d_cuda_pfc_energy, numbytes_float)); 
*/
#ifdef DEBUG_GPU_HCAL
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaDeviceSynchronize();
  cudaEventRecord(start);
#endif

//  cudaCheck(cudaMemcpyAsync(d_cuda_fracsum.get(), h_cuda_fracsum.data(), numbytes_float, cudaMemcpyHostToDevice));
//  cudaCheck(cudaMemcpyAsync(d_cuda_rhcount.get(), h_cuda_rhcount.data(), numbytes_int, cudaMemcpyHostToDevice));
/*  
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_x, h_cuda_pfrh_x.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_y, h_cuda_pfrh_y.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_z, h_cuda_pfrh_z.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_energy, h_cuda_pfrh_energy.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_pt2, h_cuda_pfrh_pt2.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_topoId, h_cuda_pfrh_topoId.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_isSeed, h_cuda_pfrh_isSeed.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_layer, h_cuda_pfrh_layer.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_depth, h_cuda_pfrh_depth.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfNeighEightInd, h_cuda_pfNeighEightInd.data(), numbytes_int*8, cudaMemcpyHostToDevice));  
  cudaCheck(cudaMemcpyAsync(d_cuda_pfNeighFourInd, h_cuda_pfNeighFourInd.data(), numbytes_int*4, cudaMemcpyHostToDevice));  
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_edgeId, h_cuda_pfrh_edgeId.data(), sizeof(int)*totalNeighbours, cudaMemcpyHostToDevice));  
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_edgeList, h_cuda_pfrh_edgeList.data(), sizeof(int)*totalNeighbours, cudaMemcpyHostToDevice));  
*/
  
  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_x.get(), h_cuda_pfrh_x.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_y.get(), h_cuda_pfrh_y.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_z.get(), h_cuda_pfrh_z.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_energy.get(), h_cuda_pfrh_energy.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_pt2.get(), h_cuda_pfrh_pt2.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_topoId.get(), h_cuda_pfrh_topoId.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_isSeed.get(), h_cuda_pfrh_isSeed.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_layer.get(), h_cuda_pfrh_layer.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_depth.get(), h_cuda_pfrh_depth.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfNeighEightInd.get(), h_cuda_pfNeighEightInd.data(), numbytes_int*8, cudaMemcpyHostToDevice));  
  cudaCheck(cudaMemcpyAsync(inputGPU.pfNeighFourInd.get(), h_cuda_pfNeighFourInd.data(), numbytes_int*4, cudaMemcpyHostToDevice));  
  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_edgeId.get(), h_cuda_pfrh_edgeId.data(), sizeof(int)*totalNeighbours, cudaMemcpyHostToDevice));  
  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_edgeList.get(), h_cuda_pfrh_edgeList.data(), sizeof(int)*totalNeighbours, cudaMemcpyHostToDevice));  
#ifdef DEBUG_GPU_HCAL
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&GPU_timers[0], start, stop);
  //std::cout<<"(HCAL) Copy memory to device: "<<GPU_timers[0]<<" ms"<<std::endl;
//  cudaEventRecord(start);
#endif
     
     float kernelTimers[8] = {0.0};
     
     /* 
     //PFClusterCudaHCAL::PFRechitToPFCluster_HCAL_serialize(rh_size, 
     PFClusterCudaHCAL::PFRechitToPFCluster_HCALV2(rh_size, 
					      d_cuda_pfrh_x,  
					      d_cuda_pfrh_y,  
					      d_cuda_pfrh_z, 
					      d_cuda_pfrh_energy, 
					      d_cuda_pfrh_pt2, 	
					      d_cuda_pfrh_isSeed,
					      d_cuda_pfrh_passTopoThresh,
					      d_cuda_pfrh_topoId,
					      d_cuda_pfrh_layer, 
					      d_cuda_pfrh_depth, 
					      d_cuda_pfNeighEightInd, 
					      d_cuda_pfNeighFourInd, 
					      d_cuda_pcRhFracInd,
					      d_cuda_pcRhFrac,
					      d_cuda_fracsum.get(),
					      d_cuda_rhcount.get(),
					      kernelTimers
                          );
     */
     
     
     PFClusterCudaHCAL::PFRechitToPFCluster_HCAL_CCLClustering((int)rh_size, 
					      (int)totalNeighbours,
                          inputGPU.pfrh_x.get(),  
					      inputGPU.pfrh_y.get(),  
					      inputGPU.pfrh_z.get(), 
					      inputGPU.pfrh_energy.get(), 
					      inputGPU.pfrh_pt2.get(), 	
					      inputGPU.pfrh_isSeed.get(),
					      inputGPU.pfrh_topoId.get(),
					      inputGPU.pfrh_layer.get(), 
					      inputGPU.pfrh_depth.get(), 
					      inputGPU.pfNeighEightInd.get(), 
					      inputGPU.pfNeighFourInd.get(), 
					      inputGPU.pfrh_edgeId.get(), 
					      inputGPU.pfrh_edgeList.get(), 
					      inputGPU.pfrh_edgeMask.get(), 
					      inputGPU.pfrh_passTopoThresh.get(),
                          inputGPU.pcrh_fracInd.get(),
					      inputGPU.pcrh_frac.get(),
					      inputGPU.pcrh_fracSum.get(),
					      inputGPU.rhcount.get(),
					      inputGPU.topoSeedCount.get(),
                          inputGPU.topoRHCount.get(),
                          inputGPU.seedFracOffsets.get(),
                          inputGPU.topoSeedOffsets.get(),
                          inputGPU.topoSeedList.get(),
                          inputGPU.pfc_pos4.get(),
                          inputGPU.pfc_prevPos4.get(),
                          inputGPU.pfc_energy.get(),
                          kernelTimers,
                          d_topoIter,
                          inputGPU.pfc_iter.get(),
                          d_pcrhFracSize
                          );
     cudaCheck(cudaMemcpyAsync(h_topoIter, d_topoIter, sizeof(int), cudaMemcpyDeviceToHost));
     cudaCheck(cudaMemcpyAsync(h_pcrhFracSize, d_pcrhFracSize, sizeof(int), cudaMemcpyDeviceToHost));
     
     std::vector<int> h_cuda_pfc_iter(rh_size, -1);
     cudaCheck(cudaMemcpyAsync(h_cuda_pfc_iter.data(), inputGPU.pfc_iter.get(), numbytes_int, cudaMemcpyDeviceToHost));
/*
#ifdef DEBUG_GPU_HCAL
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout<<"(HCAL) GPU clustering: "<<milliseconds<<" ms"<<std::endl;
  cudaEventRecord(start);
#endif
*/

#ifdef DEBUG_GPU_HCAL
  GPU_timers[1] = kernelTimers[0];  // Seeding
  GPU_timers[2] = kernelTimers[1];  // Topo clustering
  GPU_timers[3] = kernelTimers[2];
  GPU_timers[4] = kernelTimers[3];  // PF clustering

  // Extra timers
  GPU_timers[6] = kernelTimers[4];
  GPU_timers[7] = kernelTimers[5];
  GPU_timers[8] = kernelTimers[6];

//  std::cout<<"HCAL GPU clustering (ms):\n"
//           <<"Seeding\t\t"<<GPU_timers[1]<<std::endl
//           <<"Topo clustering\t"<<GPU_timers[2]<<std::endl
//           <<"PF cluster step 1 \t"<<GPU_timers[3]<<std::endl
//           <<"PF cluster step 2 \t"<<GPU_timers[4]<<std::endl;
  cudaDeviceSynchronize();
  cudaEventRecord(start);
#endif

  std::vector<int> h_cuda_topoSeedCount(rh_size, -1);
  std::vector<int> h_cuda_topoRHCount(rh_size, -1);
  std::vector<int> h_cuda_seedFracOffsets(rh_size, -1);
  std::vector<int> h_cuda_topoSeedOffsets(rh_size, -1);
  std::vector<int> h_cuda_topoSeedList(rh_size, -1);
  cudaCheck(cudaMemcpyAsync(h_cuda_topoSeedCount.data()  , inputGPU.topoSeedCount.get()  , numbytes_int , cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpyAsync(h_cuda_topoRHCount.data()    , inputGPU.topoRHCount.get()  , numbytes_int , cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpyAsync(h_cuda_seedFracOffsets.data()    , inputGPU.seedFracOffsets.get()  , numbytes_int , cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpyAsync(h_cuda_topoSeedOffsets.data()    , inputGPU.topoSeedOffsets.get()  , numbytes_int , cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpyAsync(h_cuda_topoSeedList.data()    , inputGPU.topoSeedList.get()  , numbytes_int , cudaMemcpyDeviceToHost));


  //std::cout<<"*h_pcrhFracSize = "<<*h_pcrhFracSize<<std::endl;
  //cudaCheck(cudaMemcpyAsync(h_cuda_pcRhFracInd.data()    , inputGPU.pcRhFracInd  , numbytes_int*maxSize , cudaMemcpyDeviceToHost)); 
  //cudaCheck(cudaMemcpyAsync(h_cuda_pcRhFrac.data()       , inputGPU.pcRhFrac  , numbytes_float*maxSize , cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpyAsync(h_cuda_pcRhFracInd.data()    , inputGPU.pcrh_fracInd.get()  , sizeof(int) * *h_pcrhFracSize , cudaMemcpyDeviceToHost)); 
  cudaCheck(cudaMemcpyAsync(h_cuda_pcRhFrac.data()       , inputGPU.pcrh_frac.get()  , sizeof(int) * *h_pcrhFracSize , cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpyAsync(h_cuda_pfrh_isSeed.data()    , inputGPU.pfrh_isSeed.get()  , numbytes_int , cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpyAsync(h_cuda_pfrh_topoId.data()    , inputGPU.pfrh_topoId.get()  , numbytes_int , cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpyAsync(h_cuda_pfNeighEightInd.data()    , inputGPU.pfNeighEightInd.get()  , numbytes_int*8 , cudaMemcpyDeviceToHost));
  bool*                                                 h_cuda_pfrh_passTopoThresh = new bool[rechits->size()];
  cudaCheck(cudaMemcpyAsync(h_cuda_pfrh_passTopoThresh, inputGPU.pfrh_passTopoThresh.get(), sizeof(bool)*rechits->size(), cudaMemcpyDeviceToHost));
  

#ifdef DEBUG_GPU_HCAL
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&GPU_timers[5], start, stop);
  //std::cout<<"(HCAL) Copy results from GPU: "<<GPU_timers[5]<<" ms"<<std::endl;
#endif

  for (int topoId = 0; topoId < (int)rh_size; topoId++) {
    int nPFCIter = h_cuda_pfc_iter[topoId]; // Number of iterations for PF clustering to converge
    __pfcIter.push_back(nPFCIter);
    
    int nSeeds = h_cuda_topoSeedCount[topoId], nRHTopo = h_cuda_topoRHCount[topoId];
    int nFracsTopo = (nRHTopo-nSeeds+1) * nSeeds;
    __nRHTopo.push_back(nRHTopo);
    __nSeedsTopo.push_back(nSeeds);
    __nFracsTopo.push_back(nFracsTopo);
    
    if (nPFCIter >= 0) {
        pfcIterations->Fill(nPFCIter);
        pfcIter_vs_nRHTopo->Fill(nRHTopo, nPFCIter);
        pfcIter_vs_nSeedsTopo->Fill(nSeeds, nPFCIter);
        pfcIter_vs_nFracsTopo->Fill(nFracsTopo, nPFCIter);   // Total number of rechit fractions for a given topo cluster
    }
  }
  //std::cout<<"Freeing cuda memory"<<std::endl;
  //free up
/*
  cudaCheck(cudaFree(d_cuda_pfrh_x));
  cudaCheck(cudaFree(d_cuda_pfrh_y));
  cudaCheck(cudaFree(d_cuda_pfrh_z));
  cudaCheck(cudaFree(d_cuda_pfrh_energy));
  cudaCheck(cudaFree(d_cuda_pfrh_layer));
  cudaCheck(cudaFree(d_cuda_pfrh_depth));
  cudaCheck(cudaFree(d_cuda_pfrh_isSeed));
  cudaCheck(cudaFree(d_cuda_pfrh_topoId));
  cudaCheck(cudaFree(d_cuda_pfrh_pt2));
  cudaCheck(cudaFree(d_cuda_pfNeighEightInd));
  cudaCheck(cudaFree(d_cuda_pfNeighFourInd));
  cudaCheck(cudaFree(d_cuda_pcRhFracInd));
  cudaCheck(cudaFree(d_cuda_pcRhFrac));
  cudaCheck(cudaFree(d_cuda_pfrh_edgeId));
  cudaCheck(cudaFree(d_cuda_pfrh_edgeList));
  cudaCheck(cudaFree(d_cuda_pfrh_edgeMask));
  cudaCheck(cudaFree(d_cuda_pfrh_passTopoThresh));
  cudaCheck(cudaFree(d_cuda_topoSeedCount));
  cudaCheck(cudaFree(d_cuda_topoRHCount));
  cudaCheck(cudaFree(d_cuda_seedFracOffsets));
  cudaCheck(cudaFree(d_cuda_topoSeedOffsets));
  cudaCheck(cudaFree(d_cuda_topoSeedList));
  cudaCheck(cudaFree(d_cuda_pfc_pos4));
  cudaCheck(cudaFree(d_cuda_pfc_prevPos4));
  cudaCheck(cudaFree(d_cuda_pfc_energy));
  cudaCheck(cudaFree(d_cuda_pfc_iter));
*/

    std::unordered_map<int, std::vector<int>> nTopoRechits;
    std::unordered_map<int, int> nTopoSeeds;

    for(int rh=0; rh<(int)rechits->size();rh++){
        int topoId = h_cuda_pfrh_topoId.at(rh);
        if (topoId > -1) {
            // Valid topo id
            nTopoRechits[topoId].push_back(rh);
            if (h_cuda_pfrh_isSeed.at(rh) > 0) {
                nTopoSeeds[topoId]++;
            }
        }
    }

    int intTopoCount = 0;
    for (const auto& x: nTopoRechits) {
        int topoId = x.first;
        if (nTopoSeeds.count(topoId) > 0) {
            // This topo cluster has at least one seed
            nTopo_GPU->Fill(x.second.size());
            topoSeeds_GPU->Fill(nTopoSeeds[topoId]);
            intTopoCount++;
        }
    }

    nPFCluster_GPU->Fill(intTopoCount);


  if(doComparison){ 
      for(unsigned int i=0;i<rh_size;i++){
        int topoIda=h_cuda_pfrh_topoId[i];
        if (nTopoSeeds.count(topoIda) == 0) continue;
        for(unsigned int j=0;j<8;j++){
          if(h_cuda_pfNeighEightInd[i*8+j]>-1 && h_cuda_pfrh_topoId[h_cuda_pfNeighEightInd[i*8+j]]!=topoIda && h_cuda_pfrh_passTopoThresh[i*8+j]) std::cout<<"HCAL HAS DIFFERENT TOPOID "<<i<<"  "<<j<<"  "<<topoIda<<"  "<<h_cuda_pfrh_topoId[h_cuda_pfNeighEightInd[i*8+j]]<<std::endl; 
        }
      }
  }
  
  // Total size of allocated rechit fraction arrays (includes some extra padding for rechits that don't end up passing cuts)
  nFracs = *h_pcrhFracSize;

  topoIter = *h_topoIter;
  topoIterations->Fill(*h_topoIter);
  topoIter_vs_nRH->Fill(rh_size, *h_topoIter);

  auto pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();
  pfClustersFromCuda.reset(new reco::PFClusterCollection);


  for(int n=0; n<(int)rh_size; n++){
    if(h_cuda_pfrh_isSeed[n]==1){
      reco::PFCluster temp;
      temp.setSeed((*rechits)[n].detId());
      int offset = h_cuda_seedFracOffsets[n];
      int topoId = h_cuda_pfrh_topoId[n];
      int nSeeds = h_cuda_topoSeedCount[topoId];
      //std::cout<<"Seed "<<n<<" has topoId "<<topoId<<"\toffset "<<offset<<std::endl;
      for(int k=offset;k < (offset + h_cuda_topoRHCount[topoId] - nSeeds + 1);k++){
        //std::cout<<"\tNow on k = "<<k<<"\tindex = "<<h_cuda_pcRhFracInd[k]<<"\tfrac = "<<h_cuda_pcRhFrac[k]<<std::endl;
        //if(h_cuda_pcRhFracInd[n*maxSize+k] > -1)
        //if(h_cuda_pcRhFracInd[n*maxSize+k] > -1 && h_cuda_pcRhFrac[n*maxSize+k] > 0.0)
        if(h_cuda_pcRhFracInd[k] > -1 && h_cuda_pcRhFrac[k] > 0.0){
          const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechits,h_cuda_pcRhFracInd[k]);
          temp.addRecHitFraction( reco::PFRecHitFraction(refhit, h_cuda_pcRhFrac[k]) );
          //const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechits,h_cuda_pcRhFracInd[n*maxSize+k]);
          //temp.addRecHitFraction( reco::PFRecHitFraction(refhit, h_cuda_pcRhFrac[n*maxSize+k]) );
        }
        //if(h_cuda_pcRhFracInd[n*maxSize+k] < 0.) break;
      }
      // Check if this topoId has one only one seed
      if (nTopoSeeds.count(h_cuda_pfrh_topoId[n]) && nTopoSeeds[h_cuda_pfrh_topoId[n]] == 1 && _allCellsPositionCalc)
      {
        _allCellsPositionCalc->calculateAndSetPosition(temp);
      }
      else { 
        _positionCalc->calculateAndSetPosition(temp);
      }
      pfClustersFromCuda->insert(pfClustersFromCuda->end(), std::move(temp));
    }   
  }


  /*
  std::cout<<"topoId = \n[";
  for (int rh = 0; rh < (int)rh_size; rh++) {
    if (rh != 0) std::cout<<", ";
    std::cout<<h_cuda_pfrh_topoId[rh];
  }
  std::cout<<"]"<<std::endl<<std::endl;

  int npfc = 0;
  std::cout<<"\ntopoSeedCount = \n[";
  for (int rh = 0; rh < (int)rh_size; rh++) {
    if (rh != 0) std::cout<<", ";
    std::cout<<h_cuda_topoSeedCount[rh];
    npfc += h_cuda_topoSeedCount[rh];
  }
  std::cout<<"]"<<std::endl;
  if (npfc != (int)pfClustersFromCuda->size())
    std::cout<<"Error: Different number of seeds ("<<npfc<<") than PF clusters ("<<(int)pfClustersFromCuda->size()<<")!\n";
 
  int totSeedOffset = 0;
  std::cout<<"\ntopoRHCount = \n[";
  for (int rh = 0; rh < (int)rh_size; rh++) {
    if (rh != 0) std::cout<<", ";
    std::cout<<h_cuda_topoRHCount[rh];
    if (h_cuda_pfrh_isSeed[rh] && h_cuda_pfrh_topoId[rh] > -1) {
        totSeedOffset += h_cuda_topoRHCount[h_cuda_pfrh_topoId[rh]];
    }
  }
  std::cout<<"]"<<std::endl;

  std::cout<<"--------> totSeedOffset computed from h_cuda_topoRHCount: "<<totSeedOffset<<std::endl<<std::endl;

  std::cout<<"\nseedFracOffsets = \n[";
  for (int rh = 0; rh < (int)rh_size; rh++) {
    if (rh != 0) std::cout<<", ";
    std::cout<<h_cuda_seedFracOffsets[rh];
  }
  std::cout<<"]"<<std::endl<<std::endl;
  */
  
  //if (_energyCorrector) {
  //  _energyCorrector->correctEnergies(*pfClustersFromCuda);
  //}

  float sumEn_CPU = 0.;
  if(doComparison)
  {
    std::vector<bool> seedable(rechits->size(), false);
    _seedFinder->findSeeds(rechits, mask, seedable);
    for (auto isSeed: seedable) {
      __rh_isSeed.push_back((int)isSeed);
    }
    auto initialClusters = std::make_unique<reco::PFClusterCollection>();
    _initialClustering->buildClusters(rechits, mask, seedable, *initialClusters);
    __initialClusters = *initialClusters;  // For TTree
    
    int topoRhCount=0;
    for(auto pfc : *initialClusters)
      {
        nTopo_CPU->Fill(pfc.recHitFractions().size());
        topoEn_CPU->Fill(pfc.energy());
        topoEta_CPU->Fill(pfc.eta());
        topoPhi_CPU->Fill(pfc.phi());
        topoRhCount=topoRhCount+pfc.recHitFractions().size();
        int nSeeds = 0;
        for (const auto& rhf : pfc.recHitFractions()) {
            if (seedable[rhf.recHitRef().key()])
                nSeeds++;
        }
        topoSeeds_CPU->Fill(nSeeds);
      }
    
    nPFCluster_CPU->Fill(initialClusters->size());
    std::sort (h_cuda_pfrh_topoId.begin(), h_cuda_pfrh_topoId.end());
    
    LOGVERB("PFClusterProducer::produce()") << *_initialClustering;

    int seedSumCPU=0;
    int seedSumGPU=0;
    int maskSize = 0;
    for (int j=0;j<(int)seedable.size(); j++) seedSumCPU=seedSumCPU+seedable[j];
    for (int j=0;j<(int)h_cuda_pfrh_isSeed.size(); j++) seedSumGPU=seedSumGPU +h_cuda_pfrh_isSeed[j];
    for (int j=0;j<(int)mask.size(); j++) maskSize=maskSize +mask[j];
    
    
    
    //std::cout<<"HCAL sum CPU seeds: "<<seedSumCPU<<std::endl;
    /*
    std::cout<<"sum GPU seeds: "<<seedSumGPU<<std::endl;
    //std::cout<<"sum rechits  : "<<rh_size<<std::endl;
    std::cout<<"sum mask  : "<<maskSize<<std::endl;
    */

    sumSeed_CPU->Fill(seedSumCPU);
    sumSeed_GPU->Fill(seedSumGPU);
    deltaSumSeed->Fill(seedSumGPU - seedSumCPU);

    auto pfClusters = std::make_unique<reco::PFClusterCollection>();
    pfClusters.reset(new reco::PFClusterCollection);
    if (_pfClusterBuilder) {  // if we've defined a re-clustering step execute it
      _pfClusterBuilder->buildClusters(*initialClusters, seedable, *pfClusters);
    LOGVERB("PFClusterProducer::produce()") << *_pfClusterBuilder;
    } else {
      pfClusters->insert(pfClusters->end(), initialClusters->begin(), initialClusters->end());
    }

    int totalRHPF_CPU = 0, totalRHPF_GPU = 0;
    __pfClusters = *pfClusters;  // For TTree
    for(auto pfc : *pfClusters)
    {
      nRH_perPFCluster_CPU->Fill(pfc.recHitFractions().size());
	  totalRHPF_CPU += (int)pfc.recHitFractions().size();
      enPFCluster_CPU->Fill(pfc.energy());
      pfcEta_CPU->Fill(pfc.eta());
      pfcPhi_CPU->Fill(pfc.phi());
    sumEn_CPU += pfc.energy();
    for(auto pfcx : *pfClustersFromCuda)
	  {
	    if(pfc.seed()==pfcx.seed()){
          totalRHPF_GPU += (int)pfcx.recHitFractions().size();

          matched_pfcRh_CPU->Fill(pfc.recHitFractions().size());
          matched_pfcRh_GPU->Fill(pfcx.recHitFractions().size());
          matched_pfcEn_CPU->Fill(pfc.energy());
          matched_pfcEn_GPU->Fill(pfcx.energy());
          matched_pfcEta_CPU->Fill(pfc.eta());
          matched_pfcEta_GPU->Fill(pfcx.eta());
          matched_pfcPhi_CPU->Fill(pfc.phi());
          matched_pfcPhi_GPU->Fill(pfcx.phi());

        if (abs((int)pfcx.recHitFractions().size() - (int)pfc.recHitFractions().size()) > 0) {
            std::cout<<"HCAL mismatch nRH:\tGPU:"<<(int)pfcx.recHitFractions().size()<<"\tCPU:"<<(int)pfc.recHitFractions().size()<<std::endl;
        }
        deltaRH->Fill((int)pfcx.recHitFractions().size() - (int)pfc.recHitFractions().size());
        if (abs(pfcx.energy() - pfc.energy()) > 1e-2) {
            std::cout<<"HCAL mismatch  En:\tGPU:"<<pfcx.energy()<<"\tCPU:"<<pfc.energy()<<std::endl;
        }
        deltaEn->Fill(pfcx.energy() - pfc.energy());
        if (abs(pfcx.eta() - pfc.eta()) > 1e-4) {
            std::cout<<"HCAL mismatch Eta:\tGPU:"<<pfcx.eta()<<"\tCPU:"<<pfc.eta()<<std::endl;
	    }
        deltaEta->Fill(pfcx.eta() - pfc.eta());
        if (abs(pfcx.phi() - pfc.phi()) > 1e-4) {
            std::cout<<"HCAL mismatch Phi:\tGPU:"<<pfcx.phi()<<"\tCPU:"<<pfc.phi()<<std::endl;
	    }
        deltaPhi->Fill(pfcx.phi() - pfc.phi());


	      nRh_CPUvsGPU->Fill(pfcx.recHitFractions().size(),pfc.recHitFractions().size());
	      enPFCluster_CPUvsGPU->Fill(pfcx.energy(),pfc.energy());
	      enPFCluster_CPUvsGPU_1d->Fill((pfcx.energy()-pfc.energy())/pfc.energy());
	      if(abs((pfcx.energy()-pfc.energy())/pfc.energy())>0.05){

		coordinate->Fill(pfcx.eta(),pfcx.phi());

		for(auto rhf: pfc.recHitFractions()){
		  if(rhf.fraction()==1)layer->Fill(rhf.recHitRef()->depth());
		}
	      }
	    }
	  }
    }

    nRH_perPFClusterTotal_CPU->Fill(totalRHPF_CPU);
    nRH_perPFClusterTotal_GPU->Fill(totalRHPF_GPU);

    nRHperPFCTotal_CPU = totalRHPF_CPU;
    nRHperPFCTotal_GPU = totalRHPF_GPU;

    __pfClustersFromCuda = *pfClustersFromCuda;      // For TTree
    for(auto pfc : *pfClustersFromCuda)
    {
        nRH_perPFCluster_GPU->Fill(pfc.recHitFractions().size());
        enPFCluster_GPU->Fill(pfc.energy());
        pfcEta_GPU->Fill(pfc.eta());
        pfcPhi_GPU->Fill(pfc.phi());
    }
  }
  
  //std::cout<<"Sum En CPU = "<<sumEn_CPU<<std::endl;
  //std::cout<<"***** Filling event "<<numEvents<<std::endl;
#ifdef DEBUG_HCAL_TREES
  clusterTree->Fill();
  if (numEvents > 9) {
    for (int i = 0; i < (int)GPU_timers.size(); i++)
      hTimers->Fill(i, GPU_timers[i]);
  }
#endif
  numEvents++;
  if (_prodInitClusters)
    e.put(std::move(pfClustersFromCuda), "initialClusters");
  e.put(std::move(pfClustersFromCuda));
}



