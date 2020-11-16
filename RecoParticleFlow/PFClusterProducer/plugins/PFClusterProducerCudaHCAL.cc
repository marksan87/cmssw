#include "PFClusterProducerCudaHCAL.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterCudaHCAL.h"
#include <TFile.h>
#include <TH1F.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"


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

PFClusterProducerCudaHCAL::PFClusterProducerCudaHCAL(const edm::ParameterSet& conf)
  : 

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
    if (pfcConf.exists("allCellsPositionCalc")) {
    const edm::ParameterSet& acConf = pfcConf.getParameterSet("allCellsPositionCalc");
    const std::string& algoac = acConf.getParameter<std::string>("algoName");
    _allCellsPosCalcCuda = PFCPositionCalculatorFactory::get()->create(algoac, acConf);
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

  if (_prodInitClusters) {
    produces<reco::PFClusterCollection>("initialClusters");
  }
  produces<reco::PFClusterCollection>();

  
}

PFClusterProducerCudaHCAL::~PFClusterProducerCudaHCAL()
{
  
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
  
  _initialClustering->updateEvent(e);

  std::vector<bool> mask(rechits->size(), true);
  for (const auto& cleaner : _cleaners) {
    cleaner->clean(rechits, mask);
  }

  size_t rh_size = rechits->size();
  //std::cout<<rh_size<<std::endl;

  std::vector<float>                                    h_cuda_pfRhFrac(rechits->size()*50,-1.);
  std::vector<float>                                    h_cuda_pcRhFrac(rechits->size()*50,-1.);
  std::vector<int>                                      h_cuda_pfRhFracInd(rechits->size()*50,-1);
  std::vector<int>                                      h_cuda_pfNeighEightInd(rechits->size()*8,0);
  std::vector<int>                                      h_cuda_pfNeighFourInd(rechits->size()*4,0);
  std::vector<int>                                      h_cuda_pcRhFracInd(rechits->size()*50,-1);

  std::vector<float>                                    h_cuda_pfrh_x(rechits->size(),0);
  std::vector<float>                                    h_cuda_pfrh_y(rechits->size(),0);
  std::vector<float>                                    h_cuda_pfrh_z(rechits->size(),0);
  std::vector<float>                                    h_cuda_pfrh_energy(rechits->size(),0);
  std::vector<float>                                    h_cuda_pfrh_pt2(rechits->size(),0);
  std::vector<int>                                      h_cuda_pfrh_topoId(rechits->size(),-1);
  std::vector<int>                                      h_cuda_pfrh_isSeed(rechits->size(),0);
  std::vector<int>                                      h_cuda_pfrh_layer(rechits->size(),-999);
  std::vector<int>                                      h_cuda_pfrh_depth(rechits->size(),-999);

  

  int numbytes_float = rh_size*sizeof(float);
  int numbytes_int = rh_size*sizeof(int);

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

  int *d_cuda_pfRhFracInd;
  cudaCheck(cudaMalloc(&d_cuda_pfRhFracInd, numbytes_int*50));
  int *d_cuda_pcRhFracInd;
  cudaCheck(cudaMalloc(&d_cuda_pcRhFracInd, numbytes_int*50));
  float *d_cuda_pfRhFrac;
  cudaCheck(cudaMalloc(&d_cuda_pfRhFrac, numbytes_float*50));
  float *d_cuda_pcRhFrac;
  cudaCheck(cudaMalloc(&d_cuda_pcRhFrac, numbytes_float*50));
 
  int p=0; 
  for (auto rh: *rechits){

    h_cuda_pfrh_x[p]=rh.position().x();
    h_cuda_pfrh_y[p]=rh.position().y();
    h_cuda_pfrh_z[p]=rh.position().z();
    h_cuda_pfrh_energy[p]=rh.energy();
    h_cuda_pfrh_pt2[p]=rh.pt2();
    h_cuda_pfrh_layer[p]=(int)rh.layer();
    h_cuda_pfrh_depth[p]=(int)rh.depth();
    h_cuda_pfrh_topoId[p]=p;

    auto theneighboursEight = rh.neighbours8();
    int z = 0;
    for(auto nh: theneighboursEight)
      {
	h_cuda_pfNeighEightInd[8*p+z] = nh;
	z++;
      }
    for(int l=z; l<8; l++)
      {
	h_cuda_pfNeighEightInd[8*p+l] = -1;
      }

    auto theneighboursFour = rh.neighbours4();
    int y = 0;
    for(auto nh: theneighboursFour)
      {
	h_cuda_pfNeighEightInd[4*p+y] = nh;
	y++;
      }
    for(int l=y; l<4; l++)
      {
	h_cuda_pfNeighEightInd[4*p+l] = -1;
      }

    p++;
  }//end of rechit loop  
  
  
  cudaCheck(cudaMemcpy(d_cuda_pfrh_x, h_cuda_pfrh_x.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_pfrh_y, h_cuda_pfrh_y.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_pfrh_z, h_cuda_pfrh_z.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_pfrh_energy, h_cuda_pfrh_energy.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_pfrh_pt2, h_cuda_pfrh_pt2.data(), numbytes_float, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_pfrh_topoId, h_cuda_pfrh_topoId.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_pfrh_isSeed, h_cuda_pfrh_isSeed.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_pfrh_layer, h_cuda_pfrh_layer.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_pfrh_depth, h_cuda_pfrh_depth.data(), numbytes_int, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_pfNeighEightInd, h_cuda_pfNeighEightInd.data(), numbytes_int*8, cudaMemcpyHostToDevice));  
   cudaCheck(cudaMemcpy(d_cuda_pfNeighFourInd, h_cuda_pfNeighFourInd.data(), numbytes_int*4, cudaMemcpyHostToDevice));  

  
  cudaCheck(cudaMemcpy(d_cuda_pfRhFrac, h_cuda_pfRhFrac.data(), numbytes_float*50, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_pcRhFrac, h_cuda_pcRhFrac.data(), numbytes_float*50, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_pfRhFracInd, h_cuda_pfRhFracInd.data(), numbytes_int*50, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_pcRhFracInd, h_cuda_pcRhFracInd.data(), numbytes_int*50, cudaMemcpyHostToDevice));

  PFClusterCudaHCAL::PFRechitToPFCluster_HCAL(rh_size, 
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
					      d_cuda_pfRhFrac, 
					      d_cuda_pfRhFracInd, 
					      d_cuda_pcRhFracInd,
					      d_cuda_pcRhFrac
					      );
     
    
  cudaMemcpy(h_cuda_pcRhFracInd.data()    , d_cuda_pcRhFracInd  , numbytes_int*50 , cudaMemcpyDeviceToHost);  
  cudaMemcpy(h_cuda_pcRhFrac.data()       , d_cuda_pcRhFrac  , numbytes_float*50 , cudaMemcpyDeviceToHost);  
  cudaMemcpy(h_cuda_pfrh_isSeed.data()    , d_cuda_pfrh_isSeed  , numbytes_int , cudaMemcpyDeviceToHost);  
  
  //free up
  cudaFree(d_cuda_pfrh_x);
  cudaFree(d_cuda_pfrh_y);
  cudaFree(d_cuda_pfrh_z);
  cudaFree(d_cuda_pfrh_energy);
  cudaFree(d_cuda_pfrh_layer);
  cudaFree(d_cuda_pfrh_depth);
  cudaFree(d_cuda_pfrh_isSeed);
  cudaFree(d_cuda_pfrh_topoId);
  cudaFree(d_cuda_pfrh_pt2);  
  cudaFree(d_cuda_pfNeighEightInd);
  cudaFree(d_cuda_pfNeighFourInd);
  cudaFree(d_cuda_pfRhFracInd);
  cudaFree(d_cuda_pcRhFracInd);
  cudaFree(d_cuda_pfRhFrac);
  cudaFree(d_cuda_pcRhFrac);

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
  //_positionReCalc->calculateAndSetPositions(*pfClustersFromCuda);
  _allCellsPosCalcCuda->calculateAndSetPositions(*pfClustersFromCuda);
  
  if (_energyCorrector) {
    _energyCorrector->correctEnergies(*pfClustersFromCuda);
  }

  if (_prodInitClusters)
    e.put(std::move(pfClustersFromCuda), "initialClusters");
  e.put(std::move(pfClustersFromCuda));
}



