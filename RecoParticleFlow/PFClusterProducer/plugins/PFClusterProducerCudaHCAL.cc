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
  float minFracTot = (float)pfcConf.getParameter<double>("minFracTot");

  // Max PFClustering iterations
  unsigned maxIterations = pfcConf.getParameter<unsigned>("maxIterations");

  bool excludeOtherSeeds = pfcConf.getParameter<bool>("excludeOtherSeeds");

  float stoppingTolerance = (float)pfcConf.getParameter<double>("stoppingTolerance");


  float seedEThresholdEB_vec[4], seedEThresholdEE_vec[7], seedPt2ThresholdEB = -1, seedPt2ThresholdEE = -1;
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

  if (!PFClusterCudaHCAL::initializeCudaConstants(showerSigma2,
                                             recHitEnergyNormEB_vec,
                                             recHitEnergyNormEE_vec,
                                             minFracToKeep,
                                             minFracTot,
                                             maxIterations,
                                             stoppingTolerance,
                                             excludeOtherSeeds,
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
  _initialClustering->reset();
  if (_pfClusterBuilder)
    _pfClusterBuilder->reset();

  edm::Handle<reco::PFRecHitCollection> rechits;
  e.getByToken(_rechitsLabel, rechits);
  //std::cout<<"\n===== Now on event "<<numEvents<<" with "<<rechits->size()<<" HCAL rechits ====="<<std::endl;

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

  for (auto isMasked: mask) {
    __rh_mask.push_back(isMasked);
  }

  size_t rh_size = rechits->size();

  std::vector<float>                                    h_cuda_fracsum=std::vector<float>(rh_size,0);
  std::vector<int>                                      h_cuda_rhcount=std::vector<int>(rh_size,1);

  std::vector<float>                                    h_cuda_pfRhFrac(rechits->size()*maxSize,-1.);
  std::vector<float>                                    h_cuda_pcRhFrac(rechits->size()*maxSize,-1.);
  std::vector<int>                                      h_cuda_pfRhFracInd(rechits->size()*maxSize,-1);
  std::vector<int>                                    h_cuda_pfNeighEightInd(rechits->size()*8,-1);
  std::vector<int>                                    h_cuda_pfNeighFourInd(rechits->size()*4,-1);
  std::vector<int>                                      h_cuda_pcRhFracInd(rechits->size()*maxSize,-1);

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


  int numbytes_float = rh_size*sizeof(float);
  int numbytes_double = rh_size*sizeof(double);
  int numbytes_int = rh_size*sizeof(int);

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
  cudaCheck(cudaMalloc(&d_cuda_pfRhFracInd, numbytes_int*maxSize));
  int *d_cuda_pcRhFracInd;
  cudaCheck(cudaMalloc(&d_cuda_pcRhFracInd, numbytes_int*maxSize));
  float *d_cuda_pfRhFrac;
  cudaCheck(cudaMalloc(&d_cuda_pfRhFrac, numbytes_float*maxSize));
  float *d_cuda_pcRhFrac;
  cudaCheck(cudaMalloc(&d_cuda_pcRhFrac, numbytes_float*maxSize));
 
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

  // Resize edgeId, edgeList vectors to total 8 neighbour count
  nEdges = totalNeighbours;
  
  // Allocate Cuda memory
  int*  d_cuda_pfrh_edgeId;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_edgeId, sizeof(int) * totalNeighbours));
  
  int*  d_cuda_pfrh_edgeList;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_edgeList, sizeof(int) * totalNeighbours));
 
  int*                                      d_cuda_pfrh_edgeMask;
  cudaCheck(cudaMalloc(&d_cuda_pfrh_edgeMask, sizeof(int) * totalNeighbours));
  

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
  cudaCheck(cudaMemcpyAsync(d_cuda_pfRhFrac, h_cuda_pfRhFrac.data(), numbytes_float*maxSize, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pcRhFrac, h_cuda_pcRhFrac.data(), numbytes_float*maxSize, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfRhFracInd, h_cuda_pfRhFracInd.data(), numbytes_int*maxSize, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pcRhFracInd, h_cuda_pcRhFracInd.data(), numbytes_int*maxSize, cudaMemcpyHostToDevice));

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

  cudaCheck(cudaMemcpyAsync(h_cuda_pcRhFracInd.data()    , d_cuda_pcRhFracInd  , numbytes_int*maxSize , cudaMemcpyDeviceToHost)); 
  cudaCheck(cudaMemcpyAsync(h_cuda_pcRhFrac.data()       , d_cuda_pcRhFrac  , numbytes_float*maxSize , cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpyAsync(h_cuda_pfrh_isSeed.data()    , d_cuda_pfrh_isSeed  , numbytes_int , cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpyAsync(h_cuda_pfrh_topoId.data()    , d_cuda_pfrh_topoId  , numbytes_int , cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpyAsync(h_cuda_pfNeighEightInd.data()    , d_cuda_pfNeighEightInd  , numbytes_int*8 , cudaMemcpyDeviceToHost));
  bool*                                                 h_cuda_pfrh_passTopoThresh = new bool[rechits->size()];
  cudaCheck(cudaMemcpyAsync(h_cuda_pfrh_passTopoThresh, d_cuda_pfrh_passTopoThresh, sizeof(bool)*rechits->size(), cudaMemcpyDeviceToHost));
  
#ifdef DEBUG_GPU_HCAL
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&GPU_timers[5], start, stop);
  //std::cout<<"(HCAL) Copy results from GPU: "<<GPU_timers[5]<<" ms"<<std::endl;
#endif



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
  cudaCheck(cudaFree(d_cuda_pfrh_edgeMask));
  cudaCheck(cudaFree(d_cuda_pfrh_passTopoThresh));


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

  nIter = *h_nIter;
  nIterations->Fill(*h_nIter);
  nIter_vs_nRH->Fill(rh_size, *h_nIter);

  auto pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();
  pfClustersFromCuda.reset(new reco::PFClusterCollection);
  
  for(int n=0; n<(int)rh_size; n++){
    if(h_cuda_pfrh_isSeed[n]==1){
      reco::PFCluster temp;
      temp.setSeed((*rechits)[n].detId());
      //if((*rechits)[n]==nullptr) std::cout<<"null det seed: "<<n<<std::endl;
      for(int k=0;k<maxSize;k++){
        if(h_cuda_pcRhFracInd[n*maxSize+k] > -1){
          const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechits,h_cuda_pcRhFracInd[n*maxSize+k]);
          temp.addRecHitFraction( reco::PFRecHitFraction(refhit, h_cuda_pcRhFrac[n*maxSize+k]) );
        }
        if(h_cuda_pcRhFracInd[n*maxSize+k] < 0.) break;
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


    __pfClusters = *pfClusters;  // For TTree
    for(auto pfc : *pfClusters)
    {
      nRH_perPFCluster_CPU->Fill(pfc.recHitFractions().size());
	  enPFCluster_CPU->Fill(pfc.energy());
      pfcEta_CPU->Fill(pfc.eta());
      pfcPhi_CPU->Fill(pfc.phi());
    sumEn_CPU += pfc.energy();
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

		for(auto rhf: pfc.recHitFractions()){
		  if(rhf.fraction()==1)layer->Fill(rhf.recHitRef()->depth());
		}
	      }
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



