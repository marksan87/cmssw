#include "PFClusterProducerECALCuda.h"
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

PFClusterProducerECALCuda::PFClusterProducerECALCuda(const edm::ParameterSet& conf)
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
    if(pset.getParameter< std::string >("detector")==std::string("ECAL_ENDCAP") || pset.getParameter< std::string >("detector")==std::string("ECAL_BARREL")){
      std::vector<double> temp;
      double tempd=pset.getParameter< double >("seedingThreshold");
      temp.push_back(tempd);
      theThresh.push_back(temp);
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

PFClusterProducerECALCuda::~PFClusterProducerECALCuda()
{
  //MyFile->Open();
  MyFile->cd();
  nRecHits->Write();
  nTopoCluster->Write();
  nTopoClusterCuda->Write();
  nSeeds->Write();
  nRecHitsPerTopoCluster->Write();
  nRecHitsPerPfCluster->Write();
  nSeedsPerTopoCluster->Write();
  PFvsTopo->Write();
  theMap->Write();
  theMap1->Write();
  theMap2->Write();
  theMap3->Write();
  theMap4->Write();
  nRhDiff->Write();
  topoVS->Write();
  // MyFile->Close();
  delete MyFile;
}



void PFClusterProducerECALCuda::beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& es) {
  _initialClustering->update(es);
  if (_pfClusterBuilder)
    _pfClusterBuilder->update(es);
  if (_positionReCalc)
    _positionReCalc->update(es);
}

void PFClusterProducerECALCuda::produce(edm::Event& e, const edm::EventSetup& es) {
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
  std::cout<<rh_size<<std::endl;
  std::vector<int> h_cuda_seedable(rechits->size(),0);
  std::vector<ClusterSeedingCuda::pfRhForSeeding>   	h_cuda_pfrh;
  std::vector<float>  	h_hbthresh;
  std::vector<float>  	h_hethresh;

  int *d_cuda_seedable;
  int numbytes_bools = rh_size*sizeof(int);
  cudaCheck(cudaMalloc(&d_cuda_seedable, numbytes_bools));

  ClusterSeedingCuda::pfRhForSeeding  	*d_cuda_pfrh;
  int numbytes_pfrh  = rh_size*sizeof(ClusterSeedingCuda::pfRhForSeeding);
  cudaCheck(cudaMalloc(&d_cuda_pfrh, numbytes_pfrh));

  float*  	d_hbthresh;
  float*  	d_hethresh;
  
  cudaCheck(cudaMalloc(&d_hbthresh, theThresh[0].size()*sizeof(float)));
  cudaCheck(cudaMalloc(&d_hethresh, theThresh[1].size()*sizeof(float)));
  
  int p=0;
  int countHe=0;
  int countHb=0;
  for (auto rh: *rechits){
    ClusterSeedingCuda::pfRhForSeeding temp;
    //math::XYZPoint  pos_rh= rh.position();
    //std::cout<<"rh position (x,y,z): "<<pos_rh.x()<<" "<<pos_rh.y()<<" "<<pos_rh.z()<<std::endl;
    temp.rho = rh.positionREP().rho();
    temp.eta = rh.positionREP().eta();
    temp.phi = rh.positionREP().phi();
    temp.energy = rh.energy();
    temp.pt2 = rh.pt2();
    temp.layer =(int) rh.layer();
    temp.depth =(int) rh.depth();
    temp.mask = mask[p];
    temp.isSeed = 0;
    temp.topoId = p;
    
    if(temp.layer==-1) countHb++;
    if(temp.layer==-2) countHe++;

    int etabin = theMap->GetXaxis()->FindBin(rh.positionREP().eta());
    int phibin = theMap->GetYaxis()->FindBin(rh.positionREP().phi());
    if(temp.layer==1 && temp.depth==1)theMap1->SetBinContent(etabin, phibin,rh.energy());
    if(temp.layer==1 && temp.depth==2)theMap2->SetBinContent(etabin, phibin,rh.energy());
    if(temp.layer==1 && temp.depth==3)theMap3->SetBinContent(etabin, phibin,rh.energy());
    if(temp.layer==1 && temp.depth==4)theMap4->SetBinContent(etabin, phibin,rh.energy());

    auto theneighbours = rh.neighbours4();
    int k = 0;
    for(auto nh: theneighbours)
	{
      		temp.neigh_Ens[k]=(*rechits)[nh].energy();
		double dist = (rh.position() - (*rechits)[nh].position()).mag();
		std::cout<<"ECAL distance to neighbour "<<k<<": "<<dist<<std::endl;
		temp.neigh_Index[k]=nh;
		k++;
     	}
    for(int l=k; l<4; l++)
	{
		temp.neigh_Ens[l]=0.0;
		temp.neigh_Index[l]=-1;
		}
    h_cuda_pfrh.push_back(temp);
    p++;
    }  
  //std::cout<<"nHb RHL "<<countHb<<std::endl;
  //std::cout<<"nHe RHL "<<countHe<<std::endl;
  
  cudaCheck(cudaMemcpy(d_cuda_pfrh, h_cuda_pfrh.data(), numbytes_pfrh, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_cuda_seedable, h_cuda_seedable.data(), numbytes_bools, cudaMemcpyHostToDevice));

  for(int i=0;i<(int)theThresh[0].size();i++){h_hbthresh.push_back(theThresh[0][i]);}
  for(int i=0;i<(int)theThresh[1].size();i++){h_hethresh.push_back(theThresh[1][i]);}

  
  cudaCheck(cudaMemcpy(d_hbthresh, h_hbthresh.data(), theThresh[0].size()*sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_hethresh, h_hethresh.data(), theThresh[1].size()*sizeof(float), cudaMemcpyHostToDevice));

  ClusterSeedingCuda::seedingWrapperXYZ_2ECAL(d_cuda_pfrh, d_cuda_seedable, rh_size, d_hbthresh, d_hethresh);
  cudaMemcpy(h_cuda_pfrh.data(), d_cuda_pfrh, numbytes_pfrh, cudaMemcpyDeviceToHost);

  
  cudaMemcpy(h_cuda_seedable.data(), d_cuda_seedable, numbytes_bools, cudaMemcpyDeviceToHost);
  

  cudaFree(d_cuda_seedable);
  cudaFree(d_cuda_pfrh);
  cudaFree(d_hbthresh);
  cudaFree(d_hethresh);

  
  /* for(auto rh: *rechits){
    auto theneighbours = rh.neighbours4();
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
  std::vector<int> nTopo;
  for(int o=0; o<(int)h_cuda_pfrh.size();o++){
    bool isIn=false;
    for (int a=0; a<(int)nTopo.size();a++){
      if(h_cuda_pfrh[o].topoId==nTopo[a]) isIn=true;
    }
    if(!isIn) {

      nTopo.push_back(h_cuda_pfrh[o].topoId);
      //std::cout<<h_cuda_pfrh[o].topoId<<std::endl;
    }
    }
  nTopoClusterCuda->Fill(nTopo.size());
  
  if(h_cuda_seedable.size()==seedable.size()){
 	for(unsigned int i=0;i<h_cuda_seedable.size();i++){
	  std::vector<int> nRecHitPerTopoCuda;
	  if(seedable[i]==1.)  count++;
	  if(h_cuda_pfrh[i].isSeed!=seedable[i]) 
	    {
	      countdiff++;
	      /*std::cout<<"they are not the same [cuda,normal]: ["<<h_cuda_pfrh[i].isSeed<<" "<<seedable[i]<<"], index: "<<i<<std::endl;

	      std::cout<<"layer "<<h_cuda_pfrh[i].layer<<" "<<(*rechits)[i].layer()<<std::endl;
	      std::cout<<"depth "<<h_cuda_pfrh[i].depth<<" "<<(*rechits)[i].depth()<<std::endl;
	      std::cout<<"energy "<<h_cuda_pfrh[i].energy<<" "<<(*rechits)[i].energy()<<std::endl;
	      std::cout<<"pt2 "<<h_cuda_pfrh[i].pt2<<" "<<(*rechits)[i].pt2()<<std::endl;
	      int k = 0;
	      auto theneighbours = (*rechits)[i].neighbours4();
	      for(auto nh: theneighbours)
		{
		  std::cout<<(*rechits)[nh].energy()<<std::endl;
		  std::cout<<h_cuda_pfrh[i].neigh_Ens[k]<<std::endl;
		  std::cout<<mask[i]<<"  "<<h_cuda_pfrh[i].mask<<std::endl;
		  k++;
		  }*/
	      }

	  /*int etabin = theMap->GetXaxis()->FindBin(h_cuda_pfrh[i].eta);
	  int phibin = theMap->GetYaxis()->FindBin(h_cuda_pfrh[i].phi);
	  if(h_cuda_pfrh[i].layer==1 && h_cuda_pfrh[i].depth==2) {theMap->SetBinContent(etabin, phibin,h_cuda_pfrh[i].topoId);
	    std::cout<<" "<<h_cuda_pfrh[i].topoId<<"("<<h_cuda_pfrh[i].eta<<","<<h_cuda_pfrh[i].phi<<")";
	  for(int h=0;h<4;h++)  std::cout<<" "<<h_cuda_pfrh[h_cuda_pfrh[i].neigh_Index[h]].topoId;
	  std::cout<<std::endl;}*/
	  //std::cout<<h_cuda_pfrh[i].topoId<<std::endl;
	  
	  
	  


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
  LOGVERB("PFClusterProducerECALCuda::produce()") << *_initialClustering;

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
    LOGVERB("PFClusterProducerECALCuda::produce()") << *_pfClusterBuilder;
  } else {
    pfClusters->insert(pfClusters->end(), initialClusters->begin(), initialClusters->end());
  }

  //std::cout<<"number of PF Clusters: "<<pfClusters->size()<<std::endl;
  for(auto pfc : *pfClusters)
    {
  	nRecHitsPerPfCluster->Fill(pfc.recHitFractions().size());	
    }

  PFvsTopo->Fill(initialClusters->size(),pfClusters->size());
  topoVS->Fill(nTopo.size(),initialClusters->size());
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



