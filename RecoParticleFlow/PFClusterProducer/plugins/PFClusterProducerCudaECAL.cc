#include "PFClusterProducerCudaECAL.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterCudaECAL.h"
#include <TFile.h>
#include <TH1F.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/ESHandle.h"

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
#define DEBUG_GPU_ECAL

// Uncomment to fill TTrees
#define DEBUG_ECAL_TREES

// Uncomment to save cluster collections in TTree
#define DEBUG_SAVE_CLUSTERS

PFClusterProducerCudaECAL::PFClusterProducerCudaECAL(const edm::ParameterSet& conf)
  : 
  _prodInitClusters(conf.getUntrackedParameter<bool>("prodInitialClusters", false)) {

  _rechitsLabel = consumes<reco::PFRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsSource"));

#ifdef DEBUG_ECAL_TREES
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
  clusterTree->Branch("rechits_neighbours8", &__rh_neighbours8);
#endif
#if defined DEBUG_ECAL_TREES && defined DEBUG_SAVE_CLUSTERS
//setup TTree
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
  

  if (conf.exists("allCellsPositionCalc")) {
    const edm::ParameterSet& acConf = conf.getParameterSet("allCellsPositionCalc");
    const std::string& algoac = acConf.getParameter<std::string>("algoName");
    _allCellsPosCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf);
  }
 
  //setup topo cluster builder
  const edm::ParameterSet& initConf = conf.getParameterSet("initialClusteringStep");
  const std::string& initName = initConf.getParameter<std::string>("algoName");
  _initialClustering = InitialClusteringStepFactory::get()->create(initName, initConf, sumes);
  //setup pf cluster builder if requested
  const edm::ParameterSet& pfcConf = conf.getParameterSet("pfClusterBuilder");
  if (!pfcConf.empty()) {
    const std::string& pfcName = pfcConf.getParameter<std::string>("algoName");
    _pfClusterBuilder = PFClusterBuilderFactory::get()->create(pfcName, pfcConf);
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
  
  //bool doComparison = false;

  
  // Initialize Cuda device constant values
  // Read values from parameter set 
  //float showerSigma = 1.5;
  float showerSigma2 = (float)pfcConf.getParameter<double>("showerSigma") * (float)pfcConf.getParameter<double>("showerSigma");
  float recHitEnergyNormEB = -1, recHitEnergyNormEE = -1;
  const auto recHitEnergyNormConf = pfcConf.getParameterSetVector("recHitEnergyNorms");
  for (const auto& pset : recHitEnergyNormConf)
  {
    const std::string& det = pset.getParameter<std::string>("detector");
    if (det == std::string("ECAL_BARREL"))
      recHitEnergyNormEB = (float)pset.getParameter<double>("recHitEnergyNorm");
    else if (det == std::string("ECAL_ENDCAP"))
      recHitEnergyNormEE = (float)pset.getParameter<double>("recHitEnergyNorm");
    else
      std::cout<<"Unknown detector when parsing recHitEnergyNorm: "<<det<<std::endl;
  }
  //float recHitEnergyNormEB = 0.08;
  //float recHitEnergyNormEE = 0.3;
  //float minFracToKeep = 0.0000001;
  float minFracToKeep = (float)pfcConf.getParameter<double>("minFractionToKeep"); 

  float seedEThresholdEB = -1, seedEThresholdEE = -1, seedPt2ThresholdEB = -1, seedPt2ThresholdEE = -1;
  const auto seedThresholdConf = sfConf.getParameterSetVector("thresholdsByDetector");
  for (const auto& pset : seedThresholdConf) 
  {
    const std::string& det = pset.getParameter<std::string>("detector");
    if (det == std::string("ECAL_BARREL")) {
      seedEThresholdEB = (float)pset.getParameter<double>("seedingThreshold");
      seedPt2ThresholdEB = (float)(pset.getParameter<double>("seedingThresholdPt") * pset.getParameter<double>("seedingThresholdPt"));

    }
    else if (det == std::string("ECAL_ENDCAP")) {
      seedEThresholdEE = (float)pset.getParameter<double>("seedingThreshold");
      seedPt2ThresholdEE = (float)(pset.getParameter<double>("seedingThresholdPt") * pset.getParameter<double>("seedingThresholdPt"));
    }
    else
      std::cout<<"Unknown detector when parsing seedFinder: "<<det<<std::endl;
  }

  //float seedEThresholdEB = 0.23;
  //float seedEThresholdEE = 0.6;
  //float seedPt2ThresholdEB = 0.0;
  //float seedPt2ThresholdEE = 0.0225;

  float topoEThresholdEB = -1, topoEThresholdEE = -1;

  const auto topoThresholdConf = initConf.getParameterSetVector("thresholdsByDetector");
  for (const auto& pset : topoThresholdConf) 
  {
    const std::string& det = pset.getParameter<std::string>("detector");
    if (det == std::string("ECAL_BARREL")) {
      topoEThresholdEB = (float)pset.getParameter<double>("gatheringThreshold");
    }
    else if (det == std::string("ECAL_ENDCAP")) {
      topoEThresholdEE = (float)pset.getParameter<double>("gatheringThreshold");
    }
    else
      std::cout<<"Unknown detector when parsing initClusteringStep: "<<det<<std::endl;
  }
  //float topoEThresholdEB = 0.08;
  //float topoEThresholdEE = 0.3;

  int nNeigh = sfConf.getParameter<int>("nNeighbours");
  //int nNeigh = 8;
  int maxSize = 50;
  
  if (!PFClusterCudaECAL::initializeCudaConstants(showerSigma2,
                                             recHitEnergyNormEB,
                                             recHitEnergyNormEE,
                                             minFracToKeep,
                                             seedEThresholdEB,
                                             seedEThresholdEE,
                                             seedPt2ThresholdEB,
                                             seedPt2ThresholdEE,
                                             topoEThresholdEB,
                                             topoEThresholdEE,
                                             nNeigh,
                                             maxSize)) {

    std::cout<<"Unable to initialize Cuda constants"<<std::endl;
    return;
  }
  
  if (!PFClusterProducerCudaECAL::initializeCudaMemory()) {
    // Problem allocating Cuda memory
    std::cout<<"Unable to allocate Cuda memory"<<std::endl;
    return;
  }

  if (_prodInitClusters) {
    produces<reco::PFClusterCollection>("initialClusters");
  }
  produces<reco::PFClusterCollection>();
 
  
}

PFClusterProducerCudaECAL::~PFClusterProducerCudaECAL()
{
  // Free Cuda memory
  freeCudaMemory();

  MyFile->cd();
#ifdef DEBUG_ECAL_TREES
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

bool PFClusterProducerCudaECAL::initializeCudaMemory(int cudaDevice) {
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

void PFClusterProducerCudaECAL::freeCudaMemory(int cudaDevice) {
    cudaCheck(cudaFree(d_nIter));
}

void PFClusterProducerCudaECAL::beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& es) {
  _initialClustering->update(es);
  if (_pfClusterBuilder)
    _pfClusterBuilder->update(es);
  if (_positionReCalc)
    _positionReCalc->update(es);
}

void PFClusterProducerCudaECAL::produce(edm::Event& e, const edm::EventSetup& es) {
  _initialClustering->reset();
  if (_pfClusterBuilder)
    _pfClusterBuilder->reset();

  edm::Handle<reco::PFRecHitCollection> rechits;
  e.getByToken(_rechitsLabel, rechits);
  //std::cout<<"\n===== Now on event "<<numEvents<<" with "<<rechits->size()<<" ECAL rechits ====="<<std::endl; 
#ifdef DEBUG_ECAL_TREES
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
  __rh_neighbours8.clear();
#endif
  
  _initialClustering->updateEvent(e);

  std::vector<bool> mask(rechits->size(), true);
  for (const auto& cleaner : _cleaners) {
    cleaner->clean(rechits, mask);
  }

  for (auto isMasked: mask) {
    __rh_mask.push_back(isMasked);
  }

  size_t rh_size = 2000;
  int numbytes_double = rh_size*sizeof(double);
  int numbytes_float = rh_size*sizeof(float);
  int numbytes_int = rh_size*sizeof(int);

  std::vector<float>                                    h_cuda_pfRhFrac=std::vector<float>(rh_size*50,-1.);
  std::vector<float>                                    h_cuda_pcRhFrac=std::vector<float>(rh_size*50,-1.);
  std::vector<int>                                      h_cuda_pfRhFracInd=std::vector<int>(rh_size*50,-1);
  std::vector<int>                                      h_cuda_pfNeighEightInd=std::vector<int>(rh_size*8,0);
  std::vector<int>                                      h_cuda_pcRhFracInd=std::vector<int>(rh_size*50,-1);
  std::vector<float>                                    h_cuda_fracsum=std::vector<float>(rh_size,0);
  std::vector<int>                                      h_cuda_rhcount=std::vector<int>(rh_size,1);
  std::vector<float>                                    h_cuda_pfrh_x=std::vector<float>(rh_size,0);
  std::vector<float>                                    h_cuda_pfrh_y=std::vector<float>(rh_size,0);
  std::vector<float>                                    h_cuda_pfrh_z=std::vector<float>(rh_size,0);
  std::vector<double>                                   h_cuda_pfrh_energy=std::vector<double>(rh_size,0);
  std::vector<double>                                   h_cuda_pfrh_pt2=std::vector<double>(rh_size,0);
  std::vector<int>                                      h_cuda_pfrh_topoId=std::vector<int>(rh_size,-1);
  std::vector<int>                                      h_cuda_pfrh_isSeed=std::vector<int>(rh_size,0);
  std::vector<int>                                      h_cuda_pfrh_layer=std::vector<int>(rh_size,-999);

  std::vector<int>                                      h_cuda_pfrh_edgeId(rechits->size()*8, -1);      // Rechit index for each edge 
  std::vector<int>                                      h_cuda_pfrh_edgeList(rechits->size()*8, -1);    // Sorted list of 8 neighbours for each rechit

auto d_cuda_pfrh_x = cms::cuda::make_device_unique<float[]>(numbytes_float, nullptr);
auto d_cuda_pfrh_y = cms::cuda::make_device_unique<float[]>(numbytes_float, nullptr);
auto d_cuda_pfrh_z = cms::cuda::make_device_unique<float[]>(numbytes_float, nullptr);
auto d_cuda_pfrh_energy = cms::cuda::make_device_unique<double[]>(numbytes_double, nullptr);
auto d_cuda_pfrh_pt2 = cms::cuda::make_device_unique<double[]>(numbytes_double, nullptr);
auto d_cuda_pfRhFrac = cms::cuda::make_device_unique<float[]>(numbytes_float*50, nullptr);
auto d_cuda_pcRhFrac = cms::cuda::make_device_unique<float[]>(numbytes_float*50, nullptr);
auto d_cuda_rhcount = cms::cuda::make_device_unique<int[]>(numbytes_int, nullptr);
auto d_cuda_fracsum = cms::cuda::make_device_unique<float[]>(numbytes_float, nullptr);
auto d_cuda_pfrh_topoId = cms::cuda::make_device_unique<int[]>(numbytes_int, nullptr);
auto d_cuda_pfrh_isSeed = cms::cuda::make_device_unique<int[]>(numbytes_int, nullptr);
auto d_cuda_pfrh_layer = cms::cuda::make_device_unique<int[]>(numbytes_int, nullptr);
auto d_cuda_pfrh_passTopoThresh = cms::cuda::make_device_unique<bool[]>(sizeof(bool)*rh_size, nullptr);
auto d_cuda_pfNeighEightInd = cms::cuda::make_device_unique<int[]>(numbytes_int*8, nullptr);
auto d_cuda_pfRhFracInd = cms::cuda::make_device_unique<int[]>(numbytes_int*50, nullptr);
auto d_cuda_pcRhFracInd = cms::cuda::make_device_unique<int[]>(numbytes_int*50, nullptr);

  int p=0;
  int totalNeighbours = 0;     // Running count of 8 neighbour edges for edgeId, edgeList
  for (auto rh: *rechits){
    //std::cout<<"*** Now on rechit \t"<<p<<"\tdetId = "<<rh.detId()<<"\tneighbourInfos().size() = "<<rh.neighbourInfos().size()<<"\tneighbours8().size() = "<<rh.neighbours8().size()<<std::endl;
    h_cuda_pfrh_x[p]=rh.position().x();
    h_cuda_pfrh_y[p]=rh.position().y();
    h_cuda_pfrh_z[p]=rh.position().z();
    h_cuda_pfrh_energy[p]=rh.energy();
    h_cuda_pfrh_pt2[p]=rh.pt2();
    h_cuda_pfrh_layer[p]=(int)rh.layer();
    h_cuda_pfrh_topoId[p]=p;
    
    __rh_x.push_back(h_cuda_pfrh_x[p]);
    __rh_y.push_back(h_cuda_pfrh_y[p]);
    __rh_z.push_back(h_cuda_pfrh_z[p]);
    __rh_eta.push_back(rh.positionREP().eta());
    __rh_phi.push_back(rh.positionREP().phi());
    __rh_pt2.push_back(h_cuda_pfrh_pt2[p]);
    std::vector<unsigned int> n8;
    //std::cout<<"Eight neighbors:\t";
    auto theneighboursEight = rh.neighbours8();
    int z = 0;
    for(auto nh: theneighboursEight)
      {
      //std::cout<<nh<<" ";
	    n8.push_back(nh);
      }
    std::sort(n8.begin(), n8.end());    // Sort 8 neighbour edges in ascending order for topo clustering
    for (auto nh: n8) {
        h_cuda_pfNeighEightInd[8*p+z] = nh;
        h_cuda_pfrh_edgeId[totalNeighbours] = p;
        h_cuda_pfrh_edgeList[totalNeighbours] = (int)nh;
        totalNeighbours++;
        z++;
      }
    //std::cout<<std::endl; 
    for(int l=z; l<8; l++)
      {
	h_cuda_pfNeighEightInd[8*p+l] = -1;
      }
    
    p++;
    __rh_neighbours8.push_back(n8);
  }//end of rechit loop  
  //std::cout<<"p: "<<p<<std::endl;

  // Resize edgeId, edgeList vectors to total 8 neighbour count
  h_cuda_pfrh_edgeId.resize(totalNeighbours);
  h_cuda_pfrh_edgeList.resize(totalNeighbours);
  nEdges = totalNeighbours;

  auto d_cuda_pfrh_edgeId = cms::cuda::make_device_unique<int[]>(sizeof(int)*totalNeighbours, nullptr);
  auto d_cuda_pfrh_edgeList = cms::cuda::make_device_unique<int[]>(sizeof(int)*totalNeighbours, nullptr);
  auto d_cuda_pfrh_edgeMask = cms::cuda::make_device_unique<int[]>(sizeof(int)*totalNeighbours, nullptr);

#ifdef DEBUG_GPU_ECAL
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaDeviceSynchronize();
  cudaEventRecord(start);
#endif

  cudaCheck(cudaMemcpyAsync(d_cuda_fracsum.get(), h_cuda_fracsum.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_rhcount.get(), h_cuda_rhcount.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_x.get(), h_cuda_pfrh_x.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_y.get(), h_cuda_pfrh_y.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_z.get(), h_cuda_pfrh_z.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_energy.get(), h_cuda_pfrh_energy.data(), numbytes_double, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_pt2.get(), h_cuda_pfrh_pt2.data(), numbytes_double, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_topoId.get(), h_cuda_pfrh_topoId.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_isSeed.get(), h_cuda_pfrh_isSeed.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_layer.get(), h_cuda_pfrh_layer.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfNeighEightInd.get(), h_cuda_pfNeighEightInd.data(), numbytes_int*8, cudaMemcpyHostToDevice));  
  cudaCheck(cudaMemcpyAsync(d_cuda_pfRhFrac.get(), h_cuda_pfRhFrac.data(), numbytes_float*50, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pcRhFrac.get(), h_cuda_pcRhFrac.data(), numbytes_float*50, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfRhFracInd.get(), h_cuda_pfRhFracInd.data(), numbytes_int*50, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pcRhFracInd.get(), h_cuda_pcRhFracInd.data(), numbytes_int*50, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_edgeId.get(), h_cuda_pfrh_edgeId.data(), sizeof(int)*totalNeighbours, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(d_cuda_pfrh_edgeList.get(), h_cuda_pfrh_edgeList.data(), sizeof(int)*totalNeighbours, cudaMemcpyHostToDevice)); 

#ifdef DEBUG_GPU_ECAL
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&GPU_timers[0], start, stop);
  //std::cout<<"(ECAL) Copy memory to device: "<<GPU_timers[0]<<" ms"<<std::endl;
#endif
  
  float kernelTimers[8] = {0.0};

  /* 
  //PFClusterCudaECAL::PFRechitToPFCluster_ECAL_serialize(rechits->size(), 
  PFClusterCudaECAL::PFRechitToPFCluster_ECALV2(rechits->size(), 
					      d_cuda_pfrh_x.get(),  
					      d_cuda_pfrh_y.get(),  
					      d_cuda_pfrh_z.get(), 
					      d_cuda_pfrh_energy.get(), 
					      d_cuda_pfrh_pt2.get(), 	
					      d_cuda_pfrh_isSeed.get(),
					      d_cuda_pfrh_topoId.get(),
					      d_cuda_pfrh_layer.get(), 
					      d_cuda_pfNeighEightInd.get(), 
					      d_cuda_pfRhFrac.get(), 
					      d_cuda_pfRhFracInd.get(), 
					      d_cuda_pcRhFracInd.get(),
					      d_cuda_pcRhFrac.get(),
					      d_cuda_fracsum.get(),
					      d_cuda_rhcount.get(),
                          kernelTimers 
					      );
  */
    
  PFClusterCudaECAL::PFRechitToPFCluster_ECAL_CCLClustering((int)rechits->size(),
                          (int)totalNeighbours,
                          d_cuda_pfrh_x.get(),
                          d_cuda_pfrh_y.get(),
                          d_cuda_pfrh_z.get(),
                          d_cuda_pfrh_energy.get(),
                          d_cuda_pfrh_pt2.get(),
                          d_cuda_pfrh_isSeed.get(),
                          d_cuda_pfrh_topoId.get(),
                          d_cuda_pfrh_layer.get(),
                          d_cuda_pfNeighEightInd.get(),
                          d_cuda_pfrh_edgeId.get(),
                          d_cuda_pfrh_edgeList.get(),
                          d_cuda_pfrh_edgeMask.get(),
                          d_cuda_pfrh_passTopoThresh.get(),
                          d_cuda_pcRhFracInd.get(),
                          d_cuda_pcRhFrac.get(),
                          d_cuda_fracsum.get(),
                          d_cuda_rhcount.get(),
                          kernelTimers,
                          d_nIter
                          );
     cudaCheck(cudaMemcpy(h_nIter, d_nIter, sizeof(int), cudaMemcpyDeviceToHost));
     nIter = *h_nIter;
     nIterations->Fill(*h_nIter);
     nIter_vs_nRH->Fill(rechits->size(), *h_nIter);
  

#ifdef DEBUG_GPU_ECAL
  GPU_timers[1] = kernelTimers[0];
  GPU_timers[2] = kernelTimers[1];
  GPU_timers[3] = kernelTimers[2];
  GPU_timers[4] = kernelTimers[3];
//  std::cout<<"ECAL GPU clustering (ms):\n"
//           <<"Seeding\t\t"<<GPU_timers[1]<<std::endl
//           <<"Topo clustering\t"<<GPU_timers[2]<<std::endl
//           <<"PF cluster step 1 \t"<<GPU_timers[3]<<std::endl
//           <<"PF cluster step 2 \t"<<GPU_timers[4]<<std::endl;
  cudaDeviceSynchronize();
  cudaEventRecord(start);
#endif
/*
  std::cout<<"Elapsed time (ms) for ECAL topo clustering: "<<*elapsedTime<<std::endl;
  timer->Fill(*elapsedTime); 
*/
  //delete elapsedTime;


  cudaMemcpyAsync(h_cuda_pcRhFracInd.data()    , d_cuda_pcRhFracInd.get()  , numbytes_int*50 , cudaMemcpyDeviceToHost);  
  cudaMemcpyAsync(h_cuda_pcRhFrac.data()       , d_cuda_pcRhFrac.get()  , numbytes_float*50 , cudaMemcpyDeviceToHost);  
  cudaMemcpyAsync(h_cuda_pfrh_isSeed.data()    , d_cuda_pfrh_isSeed.get()  , numbytes_int , cudaMemcpyDeviceToHost);  
  cudaMemcpyAsync(h_cuda_pfrh_topoId.data()    , d_cuda_pfrh_topoId.get()  , numbytes_int , cudaMemcpyDeviceToHost);  
  cudaMemcpyAsync(h_cuda_pfNeighEightInd.data()    , d_cuda_pfNeighEightInd.get()  , numbytes_int*8 , cudaMemcpyDeviceToHost); 

#ifdef DEBUG_GPU_ECAL
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&GPU_timers[5], start, stop);
//  std::cout<<"(ECAL) Copy results from GPU: "<<GPU_timers[5]<<" ms"<<std::endl;
#endif
  /*
  std::vector<int> negTopoId;  
  std::cout<<"ECAL topoId"<<std::endl<<"[";
 
  for (int i = 0; i < (int)rechits->size(); i++) {
    int topoId = h_cuda_pfrh_topoId.at(i);
    std::cout<<topoId;
    if (topoId == -1)
        negTopoId.push_back(i);
    if (i != ((int)rechits->size()-1))
        std::cout<<",";  
  }
  std::cout<<"]"<<std::endl;

  if ((int)negTopoId.size() > 0) {
    std::cout<<"\nFound rechits with negative topoId: [";
    for (int i = 0; i < (int)negTopoId.size(); i++) {
        std::cout<<negTopoId.at(i);
        if (i != ((int)negTopoId.size()-1))
            std::cout<<",";
    }
    std::cout<<"]"<<std::endl<<std::endl;
  }
  */
  /* 
  if(doComparison){
      for(unsigned int i=0;i<rh_size;i++){
        int topoIda=h_cuda_pfrh_topoId[i];
        for(unsigned int j=0;j<8;j++){
          if(h_cuda_pfNeighEightInd[i*8+j]>-1 && h_cuda_pfrh_topoId[h_cuda_pfNeighEightInd[i*8+j]]!=topoIda) std::cout<<"ECAL HAS DIFFERENT TOPOID "<<i<<"  "<<j<<"  "<<topoIda<<"  "<<h_cuda_pfrh_topoId[h_cuda_pfNeighEightInd[i*8+j]]<<std::endl;
        }
      }
  }
  */

  auto pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();
  pfClustersFromCuda.reset(new reco::PFClusterCollection);
  for(int n=0; n<(int)rh_size; n++){
    if(h_cuda_pfrh_isSeed[n]==1){
      reco::PFCluster temp;
      temp.setSeed((*rechits)[n].detId()); 
      for(int k=0;k<50;k++){
	if(h_cuda_pcRhFracInd[n*50+k] > -1){
	  const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechits,h_cuda_pcRhFracInd[n*50+k]);
	  temp.addRecHitFraction( reco::PFRecHitFraction(refhit, h_cuda_pcRhFrac[n*50+k]) );
	}
	if(h_cuda_pcRhFracInd[n*50+k] < 0.) break;
      }    
      pfClustersFromCuda->push_back(temp);
    }   
  }
  _positionReCalc->calculateAndSetPositions(*pfClustersFromCuda);



  if(doComparison)
  {
    std::vector<bool> seedable(rechits->size(), false);
    _seedFinder->findSeeds(rechits, mask, seedable);
    for (auto isSeed: seedable) {
      __rh_isSeed.push_back((int)isSeed);
    }
    auto initialClusters = std::make_unique<reco::PFClusterCollection>();
    _initialClustering->buildClusters(rechits, mask, seedable, *initialClusters);
    LOGVERB("PFClusterProducer::produce()") << *_initialClustering;
    __initialClusters = *initialClusters;  // For TTree

    int topoRhCount=0;
    int clusterCount = 0;
    for(auto pfc : *initialClusters)
      {
        nTopo_CPU->Fill(pfc.recHitFractions().size());
        /*
        std::cout<<"Cluster "<<clusterCount<<" has "<<pfc.recHitFractions().size()<<" rechits"<<std::endl;
        for (auto rhf : pfc.recHitFractions()) {
            std::cout<<"rhf.recHitRef().index() = "<<rhf.recHitRef().index()<<"\trhf.recHitRef()->detId() = "<<rhf.recHitRef()->detId()<<"\trhf.recHitRef().get() = "<<rhf.recHitRef().get()<<std::endl;
        }
        std::cout<<std::endl;
        */
        topoEn_CPU->Fill(pfc.energy());
        topoEta_CPU->Fill(pfc.eta());
        topoPhi_CPU->Fill(pfc.phi());
        
        topoRhCount=topoRhCount+pfc.recHitFractions().size();
        //std::cout<<"Cluster "<<clusterCount<<" has "<<pfc.recHitFractions().size()<<" rechit fractions"<<std::endl;
        for (auto rhf : pfc.recHitFractions())
        {
            auto rh = *rhf.recHitRef().get();
            //std::cout<<"detId = "<<rh.detId()<<"\teta = "<<rh.position().eta()<<"\tphi = "<<rh.position().phi()<<std::endl;

        }
        //std::cout<<std::endl<<std::endl;
        clusterCount++;
      }

    nPFCluster_CPU->Fill(initialClusters->size());
    
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
    std::sort (h_cuda_pfrh_topoId.begin(), h_cuda_pfrh_topoId.end());
    /* 
    int topoCount=1;
    int intTopoCount=0;
    for(int l=0; l<(int)h_cuda_pfrh_topoId.size();l++){
      if((h_cuda_pfrh_topoId[l]==h_cuda_pfrh_topoId[l+1]) && h_cuda_pfrh_topoId[l]>-1.) topoCount++;
      else if(h_cuda_pfrh_topoId[l]>-1.){
    nTopo_GPU->Fill(topoCount);
    topoCount=1;
    intTopoCount++;
      }
    }
    
    nPFCluster_GPU->Fill(intTopoCount);
    */
    int seedSumCPU=0;
    int seedSumGPU=0;
    int maskSize = 0;
    for (int j=0;j<(int)seedable.size(); j++) seedSumCPU=seedSumCPU+seedable[j];
    for (int j=0;j<(int)h_cuda_pfrh_isSeed.size(); j++) seedSumGPU=seedSumGPU +h_cuda_pfrh_isSeed[j];
    for (int j=0;j<(int)mask.size(); j++) maskSize=maskSize +mask[j];

    //std::cout<<"ECAL sum CPU seeds: "<<seedSumCPU<<std::endl; 

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
          
          nRh_CPUvsGPU->Fill(pfcx.recHitFractions().size(),pfc.recHitFractions().size());
	      enPFCluster_CPUvsGPU->Fill(pfcx.energy(),pfc.energy());
            deltaRH->Fill((int)pfcx.recHitFractions().size() - (int)pfc.recHitFractions().size());
            deltaEn->Fill(pfcx.energy() - pfc.energy());
            deltaEta->Fill(pfcx.eta() - pfc.eta());
            deltaPhi->Fill(pfcx.phi() - pfc.phi());

          if(abs((pfcx.energy()-pfc.energy())/pfc.energy())>0.05){

            coordinate->Fill(pfcx.eta(),pfcx.phi());
//            deltaRH->Fill((int)pfcx.recHitFractions().size() - (int)pfc.recHitFractions().size());
//            deltaEn->Fill(pfcx.energy() - pfc.energy());
//            deltaEta->Fill(pfcx.eta() - pfc.eta());
//            deltaPhi->Fill(pfcx.phi() - pfc.phi());

            for(auto rhf: pfc.recHitFractions()){
              if(rhf.fraction()==1)layer->Fill(rhf.recHitRef()->depth());
             }
            }
	      /*if(pfcx.recHitFractions().size()>30){
		std::cout<<"fractions"<<std::endl;
		for(auto rhf: pfcx.recHitFractions()) std::cout<<rhf.fraction()<<"  ";
		std::cout<<std::endl;
		for(auto rhf: pfc.recHitFractions()) std::cout<<rhf.fraction()<<"  ";
		std::cout<<std::endl;*/
	      //}
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

#ifdef DEBUG_ECAL_TREES
  clusterTree->Fill();
#endif
  numEvents++;
  if (_prodInitClusters)
    e.put(std::move(pfClustersFromCuda), "initialClusters");
  e.put(std::move(pfClustersFromCuda));
}



