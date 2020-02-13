#include "PFClusterProducerHbHeCuda.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/ClusterSeedingCuda.h"
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

PFClusterProducerHbHeCuda::PFClusterProducerHbHeCuda(const edm::ParameterSet& conf)
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
  for(const auto& pset : seedFinderConfs){
    if(pset.getParameter< std::string >("detector")==std::string("HCAL_ENDCAP") || pset.getParameter< std::string >("detector")==std::string("HCAL_BARREL1")){
    theThresh.push_back(pset.getParameter< std::vector<double> >("seedingThreshold"));
    }
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

  if (_prodInitClusters) {
    produces<reco::PFClusterCollection>("initialClusters");
  }
  produces<reco::PFClusterCollection>();

  
}

PFClusterProducerHbHeCuda::~PFClusterProducerHbHeCuda()
{
  //MyFile->Open();
  MyFile->cd();
  nRecHits->Write();
  nTopoCluster->Write();
  nSeeds->Write();
  nRecHitsPerTopoCluster->Write();
  nRecHitsPerPfCluster->Write();
  nSeedsPerTopoCluster->Write();
  PFvsTopo->Write();
  theMap->Write();
  nRhDiff->Write();
  // MyFile->Close();
  delete MyFile;
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

  std::vector<int>   				h_cuda_seedable(rechits->size(),0);
  std::vector<ClusterSeedingCuda::pfRhForSeeding>   	h_cuda_pfrh;
  std::vector<float>  	h_hbthresh;
  std::vector<float>  	h_hethresh;

  int   					*d_cuda_seedable;
  int numbytes_bools = rh_size*sizeof(int);
  cudaCheck(cudaMalloc(&d_cuda_seedable, numbytes_bools));

  ClusterSeedingCuda::pfRhForSeeding  	*d_cuda_pfrh;
  int numbytes_pfrh  = rh_size*sizeof(ClusterSeedingCuda::pfRhForSeeding);
  cudaCheck(cudaMalloc(&d_cuda_pfrh, numbytes_pfrh));

  float*  	d_hbthresh;
  float*  	d_hethresh;
  
  cudaCheck(cudaMalloc(&d_hbthresh, 4*sizeof(float)));
  cudaCheck(cudaMalloc(&d_hethresh, 7*sizeof(float)));


  /*std::cout<<"The RecHit size: "<<rh_size<<", The mask size: "<<mask.size()<<std::endl;
  for(int k=0; k<(int)rh_size; k++){
    std::cout<<mask[k];
    }*/

  
  int l=0;
  for (auto rh: *rechits){
    ClusterSeedingCuda::pfRhForSeeding temp;
    temp.rho = rh.positionREP().rho();
    temp.eta = rh.positionREP().eta();
    temp.phi = rh.positionREP().phi();
    temp.energy = rh.energy();
    temp.layer =(int) rh.layer();
    temp.depth =(int) rh.depth();
    temp.mask = mask[l];

    int etabin = theMap->GetXaxis()->FindBin(rh.positionREP().eta());
    int phibin = theMap->GetYaxis()->FindBin(rh.positionREP().phi());
    theMap->SetBinContent(etabin, phibin,rh.energy());
    
    //std::cout<<"eta: "<<rh.positionREP().eta()<<", phi: "<<rh.positionREP().phi()<<", layer: "<<(int) rh.layer()<<std::endl;

    auto theneighbours = rh.neighbours4();
    int k = 0;
    for(auto nh: theneighbours)
	{
      		temp.neigh_Ens[k]=(*rechits)[nh].energy();
		k++;
     	}
    for(int l=k; l<4; l++)
	{
		temp.neigh_Ens[l]=0.0;
		}
    h_cuda_pfrh.push_back(temp);
    l++;
    }  
  
  
  cudaCheck(cudaMemcpy(d_cuda_pfrh, h_cuda_pfrh.data(), numbytes_pfrh, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_seedable, h_cuda_seedable.data(), numbytes_bools, cudaMemcpyHostToDevice));

  for(int i=0;i<4;i++){h_hbthresh.push_back(theThresh[0][i]);}
  for(int i=0;i<7;i++){h_hethresh.push_back(theThresh[1][i]);}

  
  cudaCheck(cudaMemcpy(d_hbthresh, h_hbthresh.data(), 4*sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_hethresh, h_hethresh.data(), 7*sizeof(float), cudaMemcpyHostToDevice));

  ClusterSeedingCuda::seedingWrapperXYZ(d_cuda_pfrh, d_cuda_seedable, rh_size, d_hbthresh, d_hethresh);

  cudaMemcpy(h_cuda_seedable.data(), d_cuda_seedable, numbytes_bools, cudaMemcpyDeviceToHost);
  cudaFree(d_cuda_seedable);
  cudaFree(d_cuda_pfrh);
  cudaFree(d_hbthresh);
  cudaFree(d_hethresh);

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

  int count=0;
  int countdiff = 0;
  if(h_cuda_seedable.size()==seedable.size()){
 	for(unsigned int i=0;i<h_cuda_seedable.size();i++){
	  //std::cout<<h_cuda_seedable[i]<<" "<<seedable[i]<<std::endl;
	  //std::cout<<h_cuda_seedable[i];
	  if(seedable[i]==1.)  count++;
	  if(h_cuda_seedable[i]!=seedable[i]) countdiff++;
	    /*{
	      std::cout<<"they are not the same [cuda,normal]: ["<<h_cuda_seedable[i]<<" "<<seedable[i]<<"], index: "<<i<<std::endl;
	      std::cout<<"layer "<<h_cuda_pfrh[i].layer<<" "<<(*rechits)[i].layer()<<std::endl;
	      std::cout<<"depth "<<h_cuda_pfrh[i].depth<<" "<<(*rechits)[i].depth()<<std::endl;
	      std::cout<<"energy "<<h_cuda_pfrh[i].energy<<" "<<(*rechits)[i].energy()<<std::endl;
	      int k = 0;
	      auto theneighbours = (*rechits)[i].neighbours4();
	      for(auto nh: theneighbours)
		{
		  std::cout<<(*rechits)[nh].energy()<<std::endl;
		  std::cout<<h_cuda_pfrh[i].neigh_Ens[k]<<std::endl;
		  std::cout<<mask[i]<<"  "<<h_cuda_pfrh[i].mask<<std::endl;
		  k++;
		  }
		}*/
	}
    }

  nRhDiff->Fill(countdiff);

  if(h_cuda_seedable.size()!=seedable.size()) std::cout<<"The two seed vectors ahve different size."<<std::endl;
  //std::cout<<"number of RecHits: "<<rh_size<<std::endl;
  nRecHits->Fill(rh_size);
   //std::cout<<"number of Seeds: "<<count<<std::endl;
  nSeeds->Fill(count);

  ///////for (unsigned int i = 0;i<h_cuda_seedable.size(); i++){ seedable[i] = h_cuda_seedable[i];}


  auto initialClusters = std::make_unique<reco::PFClusterCollection>();
  _initialClustering->buildClusters(rechits, mask, seedable, *initialClusters);
  LOGVERB("PFClusterProducerHbHeCuda::produce()") << *_initialClustering;

  //std::cout<<"number of Topo Clusters: "<<initialClusters->size()<<std::endl;
  nTopoCluster->Fill(initialClusters->size());
  int seedPerTopo=0;
  for(auto tc : *initialClusters)
    {
    
  	nRecHitsPerTopoCluster->Fill(tc.recHitFractions().size());
	seedPerTopo=0;
	for(auto rf : tc.recHitFractions()){
	  
	  for(unsigned int i=0;i<h_cuda_seedable.size();i++){
	    if(rf.recHitRef()->positionREP().rho() == (*rechits)[i].positionREP().rho() && rf.recHitRef()->positionREP().eta() == (*rechits)[i].positionREP().eta() && rf.recHitRef()->positionREP().phi() == (*rechits)[i].positionREP().phi() && h_cuda_seedable[i]==1 ) seedPerTopo++;
	  }
	  
	  
	}
	nSeedsPerTopoCluster->Fill(seedPerTopo);
    }

  nSeedsPerTopoCluster->Fill(seedPerTopo);

  auto pfClusters = std::make_unique<reco::PFClusterCollection>();
  pfClusters.reset(new reco::PFClusterCollection);
  if (_pfClusterBuilder) {  // if we've defined a re-clustering step execute it
    _pfClusterBuilder->buildClusters(*initialClusters, seedable, *pfClusters);
    LOGVERB("PFClusterProducerHbHeCuda::produce()") << *_pfClusterBuilder;
  } else {
    pfClusters->insert(pfClusters->end(), initialClusters->begin(), initialClusters->end());
  }

  //std::cout<<"number of PF Clusters: "<<pfClusters->size()<<std::endl;
  for(auto pfc : *pfClusters)
    {
  	nRecHitsPerPfCluster->Fill(pfc.recHitFractions().size());	
    }

  PFvsTopo->Fill(initialClusters->size(),pfClusters->size());

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



