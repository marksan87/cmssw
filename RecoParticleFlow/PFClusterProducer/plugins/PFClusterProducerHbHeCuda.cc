#include "PFClusterProducerHbHeCuda.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/ClusterSeedingCuda.h"

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

PFClusterProducerHbHeCuda::PFClusterProducerHbHeCuda(const edm::ParameterSet& conf)
  : 
  HE_depths(conf.getParameter<std::vector<int>>("HE_depths")),
  HE_seedingThreshold(conf.getParameter<std::vector<double>>("HE_seedingThreshold")),
  HE_seedingThresholdPt(conf.getParameter<std::vector<double>>("HE_seedingThresholdPt")),
  HB_depths(conf.getParameter<std::vector<int>>("HB_depths")),
  HB_seedingThreshold(conf.getParameter<std::vector<double>>("HB_seedingThreshold")),
  HB_seedingThresholdPt(conf.getParameter<std::vector<double>>("HB_seedingThresholdPt")),

  _prodInitClusters(conf.getUntrackedParameter<bool>("prodInitialClusters", false)) {
  _rechitsLabel = consumes<reco::PFRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsSource"));
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

  if (_prodInitClusters) {
    produces<reco::PFClusterCollection>("initialClusters");
  }
  produces<reco::PFClusterCollection>();
}

void PFClusterProducerHbHeCuda::beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& es) {
  _initialClustering->update(es);
  if (_pfClusterBuilder)
    _pfClusterBuilder->update(es);
  if (_positionReCalc)
    _positionReCalc->update(es);
}

void PFClusterProducerHbHeCuda::produce(edm::Event& e, const edm::EventSetup& es) {
  _initialClustering->reset();
  if (_pfClusterBuilder)
    _pfClusterBuilder->reset();

  edm::Handle<reco::PFRecHitCollection> rechits;
  e.getByToken(_rechitsLabel, rechits);
  

  _initialClustering->updateEvent(e);

  std::vector<bool> mask(rechits->size(), true);
  for (const auto& cleaner : _cleaners) {
    cleaner->clean(rechits, mask);
  }

  std::vector<bool> seedable(rechits->size(), false);
  _seedFinder->findSeeds(rechits, mask, seedable);

  size_t rh_size = rechits->size();

  std::vector<int>   					h_cuda_seedable(rechits->size(),0);
  std::vector<ClusterSeedingCuda::pfRhForSeeding>   	h_cuda_pfrh;

  int   					*d_cuda_seedable;
  int numbytes_bools = rh_size*sizeof(int);
  cudaCheck(cudaMalloc(&d_cuda_seedable, numbytes_bools));

  ClusterSeedingCuda::pfRhForSeeding  	*d_cuda_pfrh;
  int numbytes_pfrh  = rh_size*sizeof(ClusterSeedingCuda::pfRhForSeeding);
  cudaCheck(cudaMalloc(&d_cuda_pfrh, numbytes_pfrh));
    
  for (auto rh: *rechits){
    ClusterSeedingCuda::pfRhForSeeding temp;
    temp.rho = rh.positionREP().rho();
    temp.eta = rh.positionREP().eta();
    temp.phi = rh.positionREP().phi();
    temp.energy = rh.energy();
    temp.layer =(int) rh.layer();

    auto theneighbours = rh.neighbours4();
    int k = 0;
    for(auto nh: theneighbours)
	{
      	temp.neigh_Ens[k]=(*rechits)[nh].energy();
	k++;
     	}
    for(int l=k; l<4; l++){temp.neigh_Ens[l]=0.0;}
    h_cuda_pfrh.push_back(temp);
    //std::cout<<h_cuda_pfrh[-1].neigh_Ens[1]<<std::endl;
    }  
  
  
  cudaCheck(cudaMemcpy(d_cuda_pfrh, h_cuda_pfrh.data(), numbytes_pfrh, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_seedable, h_cuda_seedable.data(), numbytes_bools, cudaMemcpyHostToDevice));
 
  ClusterSeedingCuda::seedingWrapperXYZ(d_cuda_pfrh, d_cuda_seedable, rh_size/*, HE_depths, HE_seedingThreshold, HB_depths, HB_seedingThreshold */);
  //ClusterSeedingCuda::seedingWrapperXYZ(5,9);

 cudaMemcpy(h_cuda_seedable.data(), d_cuda_seedable, numbytes_bools, cudaMemcpyDeviceToHost);
  cudaFree(d_cuda_seedable);
  cudaFree(d_cuda_pfrh);

  /*
  for(auto rh: *rechits){
    auto theneighbours = rh.neighbours8();
    std::cout<<"neighbours type: "<<typeid(theneighbours).name()<<std::endl;
    int rhlayer=(int)rh.layer();
    std::cout<<"size of neighbours: "<<rh.neighbourInfos().size()<<std::endl;
    std::cout<<"layer of rechit (normal): "<<std::dec<<rh.layer()<<std::endl;
    std::cout<<"layer of rechit (int): "<<rhlayer<<std::endl;
    std::cout<<"detid of rechit: "<<std::dec<<rh.detId()<<std::endl;
    std::cout<<"(int)PFLayer::HCAL_BARREL1: "<<(int)PFLayer::HCAL_BARREL1<<std::endl;
    std::cout<<"(int)PFLayer::HCAL_BARREL2_RING0: "<<(int)PFLayer::HCAL_BARREL2<<std::endl;
    std::cout<<"(int)PFLayer::HCCAL_ENDCAP: "<<(int)PFLayer::HCAL_ENDCAP<<std::endl; 
    
    for(auto nh: theneighbours){
      std::cout<<(*rechits)[nh].energy()<<std::endl;
      //std::cout<<typeid(nh).name()<<std::endl;
    }
    }*/


  /*std::cout<<std::endl<<"seedable size: "<<seedable.size()<<std::endl;
  for(unsigned int i=0;i<seedable.size();i++){
    std::cout<<seedable[i];
    }*/

  //std::cout<<std::endl<<"seedable cuda size: "<<h_cuda_seedable.size()<<std::endl;


  if(h_cuda_seedable.size()==seedable.size()){
 	for(unsigned int i=0;i<h_cuda_seedable.size();i++){
	  if(h_cuda_seedable[i]!=seedable[i]) std::cout<<"they are not the same [cuda,normal]: "<<h_cuda_seedable[i]<<" "<<seedable[i]<<std::endl;
    }
  }
  

  

  auto initialClusters = std::make_unique<reco::PFClusterCollection>();
  _initialClustering->buildClusters(rechits, mask, seedable, *initialClusters);
  LOGVERB("PFClusterProducerHbHeCuda::produce()") << *_initialClustering;

  auto pfClusters = std::make_unique<reco::PFClusterCollection>();
  pfClusters.reset(new reco::PFClusterCollection);
  if (_pfClusterBuilder) {  // if we've defined a re-clustering step execute it
    _pfClusterBuilder->buildClusters(*initialClusters, seedable, *pfClusters);
    LOGVERB("PFClusterProducerHbHeCuda::produce()") << *_pfClusterBuilder;
  } else {
    pfClusters->insert(pfClusters->end(), initialClusters->begin(), initialClusters->end());
  }

  if (_positionReCalc) {
    _positionReCalc->calculateAndSetPositions(*pfClusters);
  }

  if (_energyCorrector) {
    _energyCorrector->correctEnergies(*pfClusters);
  }

  if (_prodInitClusters)
    e.put(std::move(initialClusters), "initialClusters");
  e.put(std::move(pfClusters));
}
