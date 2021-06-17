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
#define DEBUG_GPU_HCAL

// Uncomment to fill TTrees
#define DEBUG_HCAL_TREES

// Uncomment to save cluster collections in TTree
#define DEBUG_SAVE_CLUSTERS


PFClusterProducerCudaHCAL::PFClusterProducerCudaHCAL(const edm::ParameterSet& conf)
  : 

  _prodInitClusters(conf.getUntrackedParameter<bool>("prodInitialClusters", false)) {
  _rechitsLabel = consumes<reco::PFRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsSource"));

#ifdef DEBUG_HCAL_TREES
  //setup TTree
  clusterTree->Branch("Event", &numEvents);
  clusterTree->Branch("nIter", &nIter, "nIter/I");
  clusterTree->Branch("nEdges", &nEdges, "nEdges/I");
  clusterTree->Branch("timers", &GPU_timers);
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
    _cleaners.emplace_back(RecHitTopologicalCleanerFactory::get()->create(cleanerName, conf));
  }

  edm::ConsumesCollector sumes = consumesCollector();

  // setup seed finding
  const edm::ParameterSet& sfConf = conf.getParameterSet("seedFinder");
  const std::string& sfName = sfConf.getParameter<std::string>("algoName");
  _seedFinder = SeedFinderFactory::get()->create(sfName, sfConf);

  const edm::VParameterSet& seedFinderConfs = sfConf.getParameterSetVector("thresholdsByDetector"); 
  

  
 
  //setup topo cluster builder
  const edm::ParameterSet& initConf = conf.getParameterSet("initialClusteringStep");
  const std::string& initName = initConf.getParameter<std::string>("algoName");
  _initialClustering = InitialClusteringStepFactory::get()->create(initName, initConf, sumes);
  //setup pf cluster builder if requested
  const edm::ParameterSet& pfcConf = conf.getParameterSet("pfClusterBuilder");
  if (!pfcConf.empty()) {
    const std::string& pfcName = pfcConf.getParameter<std::string>("algoName");
    _pfClusterBuilder = PFClusterBuilderFactory::get()->create(pfcName, pfcConf);
    /*if (pfcConf.exists("allCellsPositionCalc")) {
    const edm::ParameterSet& acConf = pfcConf.getParameterSet("allCellsPositionCalc");
    const std::string& algoac = acConf.getParameter<std::string>("algoName");
    _allCellsPosCalcCuda = PFCPositionCalculatorFactory::get()->create(algoac, acConf);*/

    if (pfcConf.exists("positionCalc")) {
        const edm::ParameterSet& acConf = pfcConf.getParameterSet("positionCalc");
        const std::string& algoac = acConf.getParameter<std::string>("algoName");
        _positionCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf);
    }

    if (pfcConf.exists("allCellsPositionCalc")) {
        const edm::ParameterSet& acConf = pfcConf.getParameterSet("allCellsPositionCalc");
        const std::string& algoac = acConf.getParameter<std::string>("algoName");
        _allCellsPositionCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf);
    }
  }
  //setup (possible) recalcuation of positions
  const edm::ParameterSet& pConf = conf.getParameterSet("positionReCalc");
  if (!pConf.empty()) {
    const std::string& pName = pConf.getParameter<std::string>("algoName");
    _positionReCalc = PFCPositionCalculatorFactory::get()->create(pName, pConf);
  }
  // see if new need to apply corrections, setup if there.
  const edm::ParameterSet& cConf = conf.getParameterSet("energyCorrector");
  if (!cConf.empty()) {
    const std::string& cName = cConf.getParameter<std::string>("algoName");
    _energyCorrector = PFClusterEnergyCorrectorFactory::get()->create(cName, cConf);
  }

  
  float showerSigma2 = (float)pfcConf.getParameter<double>("showerSigma") * (float)pfcConf.getParameter<double>("showerSigma");
  float recHitEnergyNormEB_vec[4], recHitEnergyNormEE_vec[7];
  const auto recHitEnergyNormConf = pfcConf.getParameterSetVector("recHitEnergyNorms");
  for (const auto& pset : recHitEnergyNormConf)
  {
    const std::string& det = pset.getParameter<std::string>("detector");
    if (det == std::string("HCAL_BARREL1")) {
      const auto& recHitENorms = pset.getParameter<std::vector<double> >("recHitEnergyNorm");
      std::copy(recHitENorms.begin(), recHitENorms.end(), recHitEnergyNormEB_vec);
    }
    else if (det == std::string("HCAL_ENDCAP")) {
      const auto& recHitENorms = pset.getParameter<std::vector<double> >("recHitEnergyNorm");
      std::copy(recHitENorms.begin(), recHitENorms.end(), recHitEnergyNormEE_vec);
    }
    else
      std::cout<<"Unknown detector when parsing recHitEnergyNorm: "<<det<<std::endl;
  }
  //float recHitEnergyNormEB = 0.08;
  //float recHitEnergyNormEE = 0.3;
  //float minFracToKeep = 0.0000001;
  float minFracToKeep = (float)pfcConf.getParameter<double>("minFractionToKeep");

  float seedEThresholdEB_vec[4], seedEThresholdEE_vec[7], seedPt2ThresholdEB = -1, seedPt2ThresholdEE = -1;
  //const auto seedThresholdConf = sfConf.getParameterSetVector("thresholdsByDetector");
  for (const auto& pset : seedFinderConfs)
  {
    const std::string& det = pset.getParameter<std::string>("detector");
    if (det == std::string("HCAL_BARREL1")) {
      const auto& thresholds = pset.getParameter<std::vector<double> >("seedingThreshold");
      std::copy(thresholds.begin(), thresholds.end(), seedEThresholdEB_vec);
      seedPt2ThresholdEB = (float)(pset.getParameter<std::vector<double> >("seedingThresholdPt")[0] * pset.getParameter<std::vector<double> >("seedingThresholdPt")[0]);

    }
    else if (det == std::string("HCAL_ENDCAP")) {
      const auto& thresholds = pset.getParameter<std::vector<double> >("seedingThreshold");
      std::copy(thresholds.begin(), thresholds.end(), seedEThresholdEE_vec);
      seedPt2ThresholdEE = (float)(pset.getParameter<std::vector<double> >("seedingThresholdPt")[0] * pset.getParameter<std::vector<double> >("seedingThresholdPt")[0]);
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

  int nNeigh = sfConf.getParameter<int>("nNeighbours");
  int maxSize = 100; 

  if (!PFClusterCudaHCAL::initializeCudaConstants(showerSigma2,
                                             recHitEnergyNormEB_vec,
                                             recHitEnergyNormEE_vec,
                                             minFracToKeep,
                                             seedEThresholdEB_vec,
                                             seedEThresholdEE_vec,
                                             seedPt2ThresholdEB,
                                             seedPt2ThresholdEE,
                                             topoEThresholdEB_vec,
                                             topoEThresholdEE_vec,
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
  nIterations->Write();
  nIter_vs_nRH->Write();
  nTopo_CPU->Write();
  nTopo_GPU->Write();
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
  timer->Write();
  // MyFile->Close();
  delete MyFile;
}

bool PFClusterProducerCudaHCAL::initializeCudaMemory(int cudaDevice) {
  //nIter = new int(0);
  h_nIter = new int(0);
  cudaCheck(cudaMalloc(&d_nIter, sizeof(int)));
  /*
    if (!cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&h_notDone), sizeof(bool), cudaHostAllocMapped))) {
    h_notDone = d_notDone = nullptr;
    return false;
  }
  */
  return true;
}

void PFClusterProducerCudaHCAL::freeCudaMemory(int cudaDevice) {
    cudaCheck(cudaFree(d_nIter));
}

void PFClusterProducerCudaHCAL::beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& es) {
  _initialClustering->update(es);
  if (_pfClusterBuilder)
    _pfClusterBuilder->update(es);
  if (_positionReCalc)
    _positionReCalc->update(es);
}

void PFClusterProducerCudaHCAL::produce(edm::Event& e, const edm::EventSetup& es) {
  /*
  cudaCheck(cudaSetDeviceFlags(cudaDeviceMapHost));
  if (h_notDone == nullptr) {
      if (!PFClusterProducerCudaHCAL::initializeCudaMemory()) {
        // Problem allocating Cuda memory
        std::cout<<"Unable to allocate Cuda memory"<<std::endl;
        return;
      }
  }
  */
  _initialClustering->reset();
  if (_pfClusterBuilder)
    _pfClusterBuilder->reset();

  edm::Handle<reco::PFRecHitCollection> rechits;
  e.getByToken(_rechitsLabel, rechits);
  //std::cout<<"\n===== Now on event "<<numEvents<<" with "<<rechits->size()<<" rechits ====="<<std::endl;

#ifdef DEBUG_HCAL_TREES
  GPU_timers.fill(0.0);
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
  //std::cout<<std::endl;
  //for(int l=0;l<(int)mask.size();l++) std::cout<<"mask: "<<mask[l]<<" ";
  /* for (const auto& cleaner : _cleaners) {
    cleaner->clean(rechits, mask);
    }*/

  for (auto isMasked: mask) {
    __rh_mask.push_back(isMasked);
  }

  size_t rh_size = rechits->size();
  //std::cout<<rh_size<<std::endl;

  std::vector<float>                                    h_cuda_fracsum=std::vector<float>(rh_size,0);
  std::vector<int>                                      h_cuda_rhcount=std::vector<int>(rh_size,1);

  std::vector<float>                                    h_cuda_pfRhFrac(rechits->size()*100,-1.);
  std::vector<float>                                    h_cuda_pcRhFrac(rechits->size()*100,-1.);
  std::vector<int>                                      h_cuda_pfRhFracInd(rechits->size()*100,-1);
  std::vector<int>                                    h_cuda_pfNeighEightInd(rechits->size()*8,-1);
  std::vector<int>                                    h_cuda_pfNeighFourInd(rechits->size()*4,-1);
  std::vector<int>                                      h_cuda_pcRhFracInd(rechits->size()*100,-1);

  std::vector<float>                                    h_cuda_pfrh_x(rechits->size(),0);
  std::vector<float>                                    h_cuda_pfrh_y(rechits->size(),0);
  std::vector<float>                                    h_cuda_pfrh_z(rechits->size(),0);
  std::vector<double>                                    h_cuda_pfrh_energy(rechits->size(),0);
  std::vector<double>                                    h_cuda_pfrh_pt2(rechits->size(),0);
  std::vector<int>                                      h_cuda_pfrh_topoId(rechits->size(),0);//-1);
  std::vector<int>                                      h_cuda_pfrh_isSeed(rechits->size(),0);
  std::vector<int>                                      h_cuda_pfrh_layer(rechits->size(),-999);
  std::vector<int>                                      h_cuda_pfrh_depth(rechits->size(),-999);

  std::vector<int>                                      h_cuda_pfrh_edgeId(rechits->size()*8, -1);      // Rechit index for each edge 
  std::vector<int>                                      h_cuda_pfrh_edgeList(rechits->size()*8, -1);    // Sorted list of 8 neighbours for each rechit 


//  std::vector<int> h_cuda_pfrh_edgesAll(rechits->size()*8, -1);
//  std::vector<int> h_cuda_pfrh_edgesLeft(rechits->size()*8, -1);

  int numbytes_float = rh_size*sizeof(float);
  int numbytes_double = rh_size*sizeof(double);
  int numbytes_int = rh_size*sizeof(int);
  //int numbytes_short = rh_size*sizeof(short);

  auto d_cuda_rhcount = cms::cuda::make_device_unique<int[]>(numbytes_int, nullptr);
  auto d_cuda_fracsum = cms::cuda::make_device_unique<float[]>(numbytes_float, nullptr);

  float*                                    d_cuda_pfrh_x;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_x, numbytes_float));
  float*                                    d_cuda_pfrh_y;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_y, numbytes_float));
  float*                                    d_cuda_pfrh_z;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_z, numbytes_float));
  double*                                    d_cuda_pfrh_energy;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_energy, numbytes_double));
  double*                                    d_cuda_pfrh_pt2;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_pt2, numbytes_double));
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

  int *d_cuda_pfRhFracInd;
  cudaCheck(cudaMalloc(&d_cuda_pfRhFracInd, numbytes_int*100));
  int *d_cuda_pcRhFracInd;
  cudaCheck(cudaMalloc(&d_cuda_pcRhFracInd, numbytes_int*100));
  float *d_cuda_pfRhFrac;
  cudaCheck(cudaMalloc(&d_cuda_pfRhFrac, numbytes_float*100));
  float *d_cuda_pcRhFrac;
  cudaCheck(cudaMalloc(&d_cuda_pcRhFrac, numbytes_float*100));
 
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
    // std::cout<<"depth  "<<h_cuda_pfrh_depth[p]<<std::endl;
    //std::cout<<"layer  "<<h_cuda_pfrh_layer[p]<<std::endl;

    auto theneighboursEight = rh.neighbours8();
    auto theneighboursFour = rh.neighbours4();
    // h_cuda_pfNeighEightInd[9*p] = p;
    //if (theneighboursEight.size() > 0 || theneighboursFour.size() > 0) {
    //std::cout<<"Eight neighbors:\t";
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
  //    std::cout<<nh<<" ";
      n8.push_back((int)nh);
	  }
 //   std::cout<<std::endl;
    std::sort(n8.begin(), n8.end());    // Sort 8 neighbour edges in ascending order for topo clustering
//    std::cout<<"After sorting: ";
//    for (auto x : n8) {
//        std::cout<<x<<" ";
//    }
//    std::cout<<std::endl;

    for (auto nh: n8) {
    h_cuda_pfNeighEightInd[8*p+z] = (int)nh;
	h_cuda_pfrh_edgeId[totalNeighbours] = p;
    h_cuda_pfrh_edgeList[totalNeighbours] = (int)nh;
    totalNeighbours++;
    z++;
      }
    
    //std::cout<<std::endl;
    //auto theneighboursFour = rh.neighbours4();
    int y = 0;
    //std::cout<<" Four neighbors:\t";
    for(auto nh: theneighboursFour)
      {
      //std::cout<<nh<<" ";
    n4.push_back((int)nh);
	h_cuda_pfNeighFourInd[4*p+y] = (int)nh;
	y++;
      }
    //}
    //else { std::cout<<"No neighbors found!"<<std::endl; }
    
    //std::cout<<std::endl<<std::endl;

    p++;
    __rh_neighbours4.push_back(n4);
    __rh_neighbours8.push_back(n8);
  }//end of rechit loop  

  // Resize edgeId, edgeList vectors to total 8 neighbour count
  h_cuda_pfrh_edgeId.resize(totalNeighbours);
  h_cuda_pfrh_edgeList.resize(totalNeighbours);
  nEdges = totalNeighbours;
/*
  bool* h_bool_edgeMask = new bool[totalNeighbours];
  for (int i = 0; i < totalNeighbours; i++) {
    h_cuda_pfrh_edgesAll[i] = i;
    h_cuda_pfrh_edgesLeft[i] = i;
    h_bool_edgeMask[i] = true;
  }
*/
//  bool*                                      d_bool_edgeMask;
//  cudaCheck(cudaMalloc(&d_bool_edgeMask, sizeof(bool) * totalNeighbours));
  
  
//  int*                                      d_cuda_pfrh_edgesLeft;
//  cudaCheck(cudaMalloc(&d_cuda_pfrh_edgesLeft, sizeof(int) * totalNeighbours));
//  int*                                      d_cuda_pfrh_edgesAll;
//  cudaCheck(cudaMalloc(&d_cuda_pfrh_edgesAll, sizeof(int) * totalNeighbours));
  
  // Allocate Cuda memory
  int*  d_cuda_pfrh_edgeId;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_edgeId, sizeof(int) * totalNeighbours));
  
  int*  d_cuda_pfrh_edgeList;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_edgeList, sizeof(int) * totalNeighbours));
 
  int*                                      d_cuda_pfrh_edgeMask;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_edgeMask, sizeof(int) * totalNeighbours));

  /*
  int*                                      d_cuda_pfrh_edgeLeft;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_edgeLeft, sizeof(int) * totalNeighbours));
  
  int*                                      d_cuda_pfrh_edgeRight;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_edgeRight, sizeof(int) * totalNeighbours));
  */
  

#ifdef DEBUG_GPU_HCAL
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaDeviceSynchronize();
  cudaEventRecord(start);
#endif

  cudaCheck(cudaMemcpyAsync(d_cuda_fracsum.get(), h_cuda_fracsum.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_rhcount.get(), h_cuda_rhcount.data(), numbytes_int, cudaMemcpyHostToDevice));
  
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_x, h_cuda_pfrh_x.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_y, h_cuda_pfrh_y.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_z, h_cuda_pfrh_z.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_energy, h_cuda_pfrh_energy.data(), numbytes_double, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_pt2, h_cuda_pfrh_pt2.data(), numbytes_double, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_topoId, h_cuda_pfrh_topoId.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_isSeed, h_cuda_pfrh_isSeed.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_layer, h_cuda_pfrh_layer.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_depth, h_cuda_pfrh_depth.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfNeighEightInd, h_cuda_pfNeighEightInd.data(), numbytes_int*8, cudaMemcpyHostToDevice));  
  cudaCheck(cudaMemcpyAsync(d_cuda_pfNeighFourInd, h_cuda_pfNeighFourInd.data(), numbytes_int*4, cudaMemcpyHostToDevice));  
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_edgeId, h_cuda_pfrh_edgeId.data(), sizeof(int)*totalNeighbours, cudaMemcpyHostToDevice));  
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_edgeList, h_cuda_pfrh_edgeList.data(), sizeof(int)*totalNeighbours, cudaMemcpyHostToDevice));  
  
//  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_edgesAll, h_cuda_pfrh_edgesAll.data(), sizeof(int)*totalNeighbours, cudaMemcpyHostToDevice));  
//  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_edgesLeft, h_cuda_pfrh_edgesLeft.data(), sizeof(int)*totalNeighbours, cudaMemcpyHostToDevice));  
//  cudaCheck(cudaMemcpyAsync(d_bool_edgeMask, h_bool_edgeMask, sizeof(bool)*totalNeighbours, cudaMemcpyHostToDevice));  
  
  cudaCheck(cudaMemcpyAsync(d_cuda_pfRhFrac, h_cuda_pfRhFrac.data(), numbytes_float*100, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pcRhFrac, h_cuda_pcRhFrac.data(), numbytes_float*100, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfRhFracInd, h_cuda_pfRhFracInd.data(), numbytes_int*100, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pcRhFracInd, h_cuda_pcRhFracInd.data(), numbytes_int*100, cudaMemcpyHostToDevice));

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
                          d_cuda_pfrh_x,  
					      d_cuda_pfrh_y,  
					      d_cuda_pfrh_z, 
					      d_cuda_pfrh_energy, 
					      d_cuda_pfrh_pt2, 	
					      d_cuda_pfrh_isSeed,
					      d_cuda_pfrh_topoId,
					      d_cuda_pfrh_layer, 
					      d_cuda_pfrh_depth, 
					      d_cuda_pfNeighEightInd, 
					      d_cuda_pfNeighFourInd, 
					      d_cuda_pfrh_edgeId, 
					      d_cuda_pfrh_edgeList, 
					      d_cuda_pfrh_edgeMask, 
					      d_cuda_pfrh_passTopoThresh,
                          d_cuda_pcRhFracInd,
					      d_cuda_pcRhFrac,
					      d_cuda_fracsum.get(),
					      d_cuda_rhcount.get(),
					      kernelTimers,
                          d_nIter
                          );
     cudaCheck(cudaMemcpyAsync(h_nIter, d_nIter, sizeof(int), cudaMemcpyDeviceToHost));
     nIter = *h_nIter;
     nIterations->Fill(*h_nIter);
     nIter_vs_nRH->Fill(rh_size, *h_nIter);
     
/*
#ifdef DEBUG_GPU_HCAL
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout<<"(HCAL) GPU clustering: "<<milliseconds<<" ms"<<std::endl;
  cudaEventRecord(start);
#endif
*/
  //std::cout<<"Elapsed time (ms) for HCAL topo clustering: "<<*elapsedTime<<std::endl;
  //timer->Fill(*elapsedTime);
//  delete elapsedTime;

#ifdef DEBUG_GPU_HCAL
  GPU_timers[1] = kernelTimers[0];
  GPU_timers[2] = kernelTimers[1];
  GPU_timers[3] = kernelTimers[2];
  GPU_timers[4] = kernelTimers[3];

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

  cudaCheck(cudaMemcpyAsync(h_cuda_pcRhFracInd.data()    , d_cuda_pcRhFracInd  , numbytes_int*100 , cudaMemcpyDeviceToHost)); 
  cudaCheck(cudaMemcpyAsync(h_cuda_pcRhFrac.data()       , d_cuda_pcRhFrac  , numbytes_float*100 , cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpyAsync(h_cuda_pfrh_isSeed.data()    , d_cuda_pfrh_isSeed  , numbytes_int , cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpyAsync(h_cuda_pfrh_topoId.data()    , d_cuda_pfrh_topoId  , numbytes_int , cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpyAsync(h_cuda_pfNeighEightInd.data()    , d_cuda_pfNeighEightInd  , numbytes_int*8 , cudaMemcpyDeviceToHost));
  
#ifdef DEBUG_GPU_HCAL
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&GPU_timers[5], start, stop);
  //std::cout<<"(HCAL) Copy results from GPU: "<<GPU_timers[5]<<" ms"<<std::endl;
#endif


  if(doComparison){ 
  for(unsigned int i=0;i<rh_size;i++){
    int topoIda=h_cuda_pfrh_topoId[i];
    for(unsigned int j=0;j<8;j++){
      if(h_cuda_pfNeighEightInd[i*8+j]>-1 && h_cuda_pfrh_topoId[h_cuda_pfNeighEightInd[i*8+j]]!=topoIda) std::cout<<"HCAL HAS DIFFERENT TOPOID "<<i<<"  "<<j<<"  "<<topoIda<<"  "<<h_cuda_pfrh_topoId[h_cuda_pfNeighEightInd[i*8+j]]<<std::endl; 
    }
    
  }
  
  }


  //free up
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
  cudaCheck(cudaFree(d_cuda_pfRhFracInd));
  cudaCheck(cudaFree(d_cuda_pcRhFracInd));
  cudaCheck(cudaFree(d_cuda_pfRhFrac));
  cudaCheck(cudaFree(d_cuda_pcRhFrac));
  cudaCheck(cudaFree(d_cuda_pfrh_edgeId));
  cudaCheck(cudaFree(d_cuda_pfrh_edgeList));
//  cudaCheck(cudaFree(d_cuda_pfrh_edgesAll));
//  cudaCheck(cudaFree(d_cuda_pfrh_edgesLeft));
  cudaCheck(cudaFree(d_cuda_pfrh_edgeMask));
//  cudaCheck(cudaFree(d_cuda_pfrh_edgeLeft));
//  cudaCheck(cudaFree(d_cuda_pfrh_edgeRight));
  cudaCheck(cudaFree(d_cuda_pfrh_passTopoThresh));
//  cudaCheck(cudaFree(d_bool_edgeMask));


  // Determine number of seeds per topo cluster
  std::unordered_map<int, int> topoSeedMap;
  for(int n=1; n<(int)rh_size; n++){
    if (h_cuda_pfrh_isSeed[n]) {
        topoSeedMap[h_cuda_pfrh_topoId[n]]++;
    }
  }

/*
  std::cout<<"topo id : # seeds"<<std::endl;
  for (auto &x : topoSeedMap) {
    std::cout<<x.first<<" : "<<x.second<<std::endl;
  }
*/
//  std::cout<<"****** topoID order: ******"<<std::endl<<"["<<h_cuda_pfrh_topoId[0];
//  for(int n=1; n<(int)rh_size; n++){
//    std::cout<<", "<<h_cuda_pfrh_topoId[n];
//  }
//  std::cout<<"]"<<std::endl;

  auto pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();
  pfClustersFromCuda.reset(new reco::PFClusterCollection);
  /*
  for(int n=0; n<(int)rh_size; n++){
    if(h_cuda_pfrh_isSeed[n]==1){
      reco::PFCluster temp;
      temp.setSeed((*rechits)[n].detId());
      //if((*rechits)[n]==nullptr) std::cout<<"null det seed: "<<n<<std::endl;
      for(int k=0;k<100;k++){
	if(h_cuda_pcRhFracInd[n*100+k] > -1){
	  const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechits,h_cuda_pcRhFracInd[n*100+k]);
	  temp.addRecHitFraction( reco::PFRecHitFraction(refhit, h_cuda_pcRhFrac[n*100+k]) );
	}
	if(h_cuda_pcRhFracInd[n*100+k] < 0.) break;
      }        
      pfClustersFromCuda->push_back(temp);
    }   
  }*/
  
  for(int n=0; n<(int)rh_size; n++){
    if(h_cuda_pfrh_isSeed[n]==1){
      reco::PFCluster temp;
      temp.setSeed((*rechits)[n].detId());
      //if((*rechits)[n]==nullptr) std::cout<<"null det seed: "<<n<<std::endl;
      for(int k=0;k<100;k++){
        if(h_cuda_pcRhFracInd[n*100+k] > -1){
          const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechits,h_cuda_pcRhFracInd[n*100+k]);
          temp.addRecHitFraction( reco::PFRecHitFraction(refhit, h_cuda_pcRhFrac[n*100+k]) );
        }
        if(h_cuda_pcRhFracInd[n*100+k] < 0.) break;
      }
      // Check if this topoId has one only one seed
      if (topoSeedMap[h_cuda_pfrh_topoId[n]] == 1 && _allCellsPositionCalc)
      {
        _allCellsPositionCalc->calculateAndSetPosition(temp);
      }
      else { 
        _positionCalc->calculateAndSetPosition(temp);
      }
      pfClustersFromCuda->insert(pfClustersFromCuda->end(), std::move(temp));
    }   
  }







  //_positionReCalc->calculateAndSetPositions(*pfClustersFromCuda);
  //_allCellsPosCalcCuda->calculateAndSetPositions(*pfClustersFromCuda);
  //_positionCalc->calculateAndSetPositions(*pfClustersFromCuda);

  //if (_energyCorrector) {
  //  _energyCorrector->correctEnergies(*pfClustersFromCuda);
  //}

  float sumEn_CPU = 0.f;
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
      }
    
    nPFCluster_CPU->Fill(initialClusters->size());
    std::sort (h_cuda_pfrh_topoId.begin(), h_cuda_pfrh_topoId.end());
    
    int topoCount=1;
    int intTopoCount=0;
    for(int l=1; l<(int)h_cuda_pfrh_topoId.size();l++){
      if((h_cuda_pfrh_topoId[l]==h_cuda_pfrh_topoId[l+1]) && h_cuda_pfrh_topoId[l]>-1.) topoCount++;
      else if(h_cuda_pfrh_topoId[l]>-1.){
	nTopo_GPU->Fill(topoCount);
	topoCount=1;
	intTopoCount++;
      }
    }
    /*
    std::cout<<"HCAL:"<<std::endl;
    std::cout<<"sum rechits          : "<<rh_size<<std::endl;
    std::cout<<"sum rechits in topo  : "<<topoRhCount<<std::endl;
    */
    nPFCluster_GPU->Fill(intTopoCount);
    LOGVERB("PFClusterProducer::produce()") << *_initialClustering;

    int seedSumCPU=0;
    int seedSumGPU=0;
    int maskSize = 0;
    for (int j=0;j<(int)seedable.size(); j++) seedSumCPU=seedSumCPU+seedable[j];
    for (int j=0;j<(int)h_cuda_pfrh_isSeed.size(); j++) seedSumGPU=seedSumGPU +h_cuda_pfrh_isSeed[j];
    for (int j=0;j<(int)mask.size(); j++) maskSize=maskSize +mask[j];
    
    /*for (int j=0;j<(int)seedable.size(); j++){
      if(seedable[j]!=h_cuda_pfrh_isSeed[j]){
	std::cout<<j<<" "<<seedable[j]<<"  "<<h_cuda_pfrh_isSeed[j]<<", depth:  "<<(*rechits)[j].depth()<<", layer: "<<(*rechits)[j].layer()<<std::endl;
	std::cout<<"pt2: "<<(*rechits)[j].pt2()<<std::endl;
	std::cout<<"energy: "<<(*rechits)[j].energy()<<std::endl;
	auto theneighboursFour = (*rechits)[j].neighbours4();	
	for(auto nh: theneighboursFour)
	  {
	    std::cout<<"neigh: "<<(*rechits)[nh].energy()<<std::endl;
	  }
      }

    }
    */
    
    
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


    //std::cout<<"HCAL pfClusters->size() = "<<pfClusters->size()<<std::endl; 
    __pfClusters = *pfClusters;  // For TTree
    for(auto pfc : *pfClusters)
    {
      nRH_perPFCluster_CPU->Fill(pfc.recHitFractions().size());
	  enPFCluster_CPU->Fill(pfc.energy());
      pfcEta_CPU->Fill(pfc.eta());
      pfcPhi_CPU->Fill(pfc.phi());
    sumEn_CPU += pfc.energy();
    //if (numEvents < 1) std::cout<<pfc.energy()<<std::endl;	
    for(auto pfcx : *pfClustersFromCuda)
	  {
	    if(pfc.seed()==pfcx.seed()){
          matched_pfcRh_CPU->Fill(pfc.recHitFractions().size());
          matched_pfcRh_GPU->Fill(pfcx.recHitFractions().size());
          matched_pfcEn_CPU->Fill(pfc.energy());
          matched_pfcEn_GPU->Fill(pfcx.energy());
          matched_pfcEta_CPU->Fill(pfc.eta());
          matched_pfcEta_GPU->Fill(pfcx.eta());
          matched_pfcPhi_CPU->Fill(pfc.phi());
          matched_pfcPhi_GPU->Fill(pfcx.phi());
	    
        deltaRH->Fill((int)pfcx.recHitFractions().size() - (int)pfc.recHitFractions().size());
        deltaEn->Fill(pfcx.energy() - pfc.energy());
	    deltaEta->Fill(pfcx.eta() - pfc.eta());
	    deltaPhi->Fill(pfcx.phi() - pfc.phi());


	      nRh_CPUvsGPU->Fill(pfcx.recHitFractions().size(),pfc.recHitFractions().size());
	      enPFCluster_CPUvsGPU->Fill(pfcx.energy(),pfc.energy());
	      enPFCluster_CPUvsGPU_1d->Fill((pfcx.energy()-pfc.energy())/pfc.energy());
	      if(abs((pfcx.energy()-pfc.energy())/pfc.energy())>0.05){

		coordinate->Fill(pfcx.eta(),pfcx.phi());
//	    deltaRH->Fill((int)pfcx.recHitFractions().size() - (int)pfc.recHitFractions().size());
//        deltaEn->Fill(pfcx.energy() - pfc.energy());
//	    deltaEta->Fill(pfcx.eta() - pfc.eta());
//	    deltaPhi->Fill(pfcx.phi() - pfc.phi());

		for(auto rhf: pfc.recHitFractions()){
		  if(rhf.fraction()==1)layer->Fill(rhf.recHitRef()->depth());
		}
//		std::cout<<std::endl;
//		std::cout<<"fractions"<<std::endl;
//		for(auto rhf: pfcx.recHitFractions()) std::cout<<rhf.fraction()<<", eta:"<<rhf.recHitRef()->positionREP().eta()<<", phi:"<< rhf.recHitRef()->positionREP().phi()<<"  ";
//		std::cout<<std::endl;
//		for(auto rhf: pfc.recHitFractions()) std::cout<<rhf.fraction()<<", eta:"<<rhf.recHitRef()->positionREP().eta()<<", phi:"<< rhf.recHitRef()->positionREP().phi()<<"  ";
	      }
	      /*if(abs((int)(pfcx.recHitFractions().size() - pfc.recHitFractions().size() ))>30){
		std::cout<<"fractions"<<std::endl;
		for(auto rhf: pfcx.recHitFractions()) std::cout<<rhf.fraction()<<"  ";
		std::cout<<std::endl;
		for(auto rhf: pfc.recHitFractions()) std::cout<<rhf.fraction()<<"  ";
		std::cout<<std::endl;
		}*/
	    }
	  }
    }

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
#endif
  numEvents++;
  if (_prodInitClusters)
    e.put(std::move(pfClustersFromCuda), "initialClusters");
  e.put(std::move(pfClustersFromCuda));
}



