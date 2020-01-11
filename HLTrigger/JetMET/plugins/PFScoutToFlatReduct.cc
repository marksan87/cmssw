// Standard C++ includes
#include <memory>
#include <vector>
#include <iostream>

// ROOT includes
#include <TTree.h>
#include <TLorentzVector.h>
#include <TPRegexp.h>

// CMSSW framework includes
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// CMSSW data formats
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Scouting/interface/ScoutingMuon.h"
#include "DataFormats/Scouting/interface/ScoutingParticle.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"

// Other relevant CMSSW includes
#include "CommonTools/UtilAlgos/interface/TFileService.h" 
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"


#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Scouting/interface/ScoutingElectron.h"
#include "DataFormats/Scouting/interface/ScoutingPhoton.h"
#include "DataFormats/Scouting/interface/ScoutingPFJet.h"

#include "DataFormats/Scouting/interface/ScoutingVertex.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/PatCandidates/interface/PackedTriggerPrescales.h"
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionParser.h"

#include <DataFormats/TrackReco/interface/TrackBase.h>

#include "DataFormats/Math/interface/libminifloat.h"

class PFScoutToFlatReduct : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  explicit PFScoutToFlatReduct(const edm::ParameterSet&);
  ~PFScoutToFlatReduct();
		
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
	
	
private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  const edm::InputTag triggerResultsTag;
  const edm::EDGetTokenT<edm::TriggerResults>             	triggerResultsToken;

  const edm::EDGetTokenT<std::vector<ScoutingMuon> >      muonsToken;
  const edm::EDGetTokenT<std::vector<ScoutingElectron> >  	electronsToken;
  const edm::EDGetTokenT<std::vector<ScoutingPhoton> >  	photonsToken;
  const edm::EDGetTokenT<std::vector<ScoutingParticle> >  	pfcandsToken;
  const edm::EDGetTokenT<std::vector<ScoutingPFJet> >  		pfjetsToken;
  

  //const edm::EDGetTokenT<GenEventInfoProduct>             genEvtInfoToken;

  std::vector<std::string> triggerPathsVector;
  std::map<std::string, int> triggerPathsMap;

        
	
  bool doL1;       
  triggerExpression::Data triggerCache_;
      
	
  // Generator-level information
  // Flags for the different types of triggers used in the analysis
  // For now we are interested in events passing either the single or double lepton triggers
  unsigned char                trig;
       
  edm::InputTag                algInputTag_;       
  edm::EDGetToken              algToken_;
  l1t::L1TGlobalUtil          *l1GtUtils_;
  std::vector<std::string>     l1Seeds_;
  std::vector<bool>            l1Result_;
       
        


  //Photon
  const static int 	max_pho = 1000;
  UInt_t n_pho;
  Float16_t 	    	photonpt[max_pho];
  Float16_t        	photoneta[max_pho];
  Float16_t        	photonphi[max_pho];
  Float16_t	    	photonm[max_pho];
  Float16_t	    	photonsigmaietaieta[max_pho];
  Float16_t	    	photonHoE[max_pho];
  Float16_t        	photonecaliso[max_pho];
  Float16_t	    	photonhcaliso[max_pho];

  //Electron
  const static int 	max_ele = 1000;
  UInt_t n_ele;
  Float16_t 	    electronpt[max_ele];
  Float16_t        electroneta[max_ele];
  Float16_t        electronphi[max_ele];
  Float16_t	    electronm[max_ele];
  Float16_t        electrond0[max_ele];
  Float16_t	    electrondz[max_ele];
  Float16_t	    electrondetain[max_ele];
  Float16_t	    electrondphiin[max_ele];
  Float16_t	    electronsigmaietaieta[max_ele];
  Float16_t	    electronHoE[max_ele];
  Float16_t	    electronooEMOop[max_ele];
  Float16_t	    electronmHits[max_ele];
  Float16_t        electroncharge[max_ele];
  Float16_t        electronecaliso[max_ele];
  Float16_t	    electronhcaliso[max_ele];
  Float16_t        electrontkiso[max_ele];

  //Muon
  const static int 	max_mu = 1000;
  UInt_t n_mu;
  Float16_t 	muonpt[max_mu];
  Float16_t 	muoneta[max_mu];
  Float16_t 	muonphi[max_mu];
  Float16_t 	muonm[max_mu];
  Float16_t 	muonecaliso[max_mu];
  Float16_t 	muonhcaliso[max_mu];
  Float16_t 	muontrkiso[max_mu];
  Float16_t 	muonchi2[max_mu];
  Float16_t 	muonndof[max_mu];
  Float16_t 	muoncharge[max_mu];
  Float16_t 	muondxy[max_mu];
  Float16_t 	muondz[max_mu];
  Float16_t 	muonnvalidmuonhits[max_mu];
  Float16_t 	muonnvalidpixelhits[max_mu];
  
  Float16_t 	muonnmatchedstations[max_mu];
  Float16_t       muontype[max_mu];
  Float16_t       muonnvalidstriphits[max_mu];
  Float16_t       muontrkqoverp[max_mu];
  Float16_t       muontrklambda[max_mu];
  Float16_t       muontrkpt[max_mu];
  Float16_t       muontrkphi[max_mu];
  Float16_t       muontrketa[max_mu];
  Float16_t       muontrkqoverperror[max_mu];
  Float16_t       muontrklambdaerror[max_mu];
  Float16_t       muontrkpterror[max_mu];
  Float16_t       muontrkphierror[max_mu];
  Float16_t       muontrketaerror[max_mu];
  Float16_t       muontrkdszerror[max_mu];
  Float16_t       muontrkdsz[max_mu];
  int muontvtxind[max_mu];

  //PFJets
  const static int 	max_jet = 1000;
  UInt_t n_jet;
  Float16_t 	    jetpt[max_jet];
  Float16_t         jeteta[max_jet];
  Float16_t         jetphi[max_jet];
  Float16_t	    jetm[max_jet];
  Float16_t	    jetarea[max_jet];
  Float16_t	    jetchargedHadronEnergy[max_jet];
  Float16_t         jetneutralHadronEnergy[max_jet];
  Float16_t	    jetphotonEnergy[max_jet];
  Float16_t	    jetelectronEnergy[max_jet];
  Float16_t	    jetmuonEnergy[max_jet];
  Float16_t	    jetHFHadronEnergy[max_jet];
  Float16_t	    jetHFEMEnergy[max_jet];
  Float16_t	    jetHOEnergy[max_jet];
  Float16_t	    jetchargedHadronMultiplicity[max_jet];
  Float16_t         jetneutralHadronMultiplicity[max_jet];
  Float16_t	    jetphotonMultiplicity[max_jet];
  Float16_t	    jetelectronMultiplicity[max_jet];
  Float16_t	    jetmuonMultiplicity[max_jet];
  Float16_t	    jetHFHadronMultiplicity[max_jet];
  Float16_t	    jetHFEMMultiplicity[max_jet];
  Float16_t 	    jetcsv[max_jet];
  Float16_t 	    jetmvaDiscriminator[max_jet];
  std::vector< std::vector<int16_t> >  	    jetconstituents;

  //PFCand
  const static int 	max_pfcand = 10000;
  UInt_t n_pfcand;
  Float16_t 	    pfcandpt[max_pfcand];
  Float16_t         pfcandeta[max_pfcand];
  Float16_t         pfcandphi[max_pfcand];
  Float16_t	    pdcandm[max_pfcand];
  Float16_t	    pfcandpdgid[max_pfcand];
  Float16_t	    pfcandvertex[max_pfcand];

        
  // TTree carrying the event weight information
  TTree* tree;

  //Run and lumisection
  int run;
  int lumSec;

};

PFScoutToFlatReduct::PFScoutToFlatReduct(const edm::ParameterSet& iConfig): 
  triggerResultsTag        (iConfig.getParameter<edm::InputTag>("triggerresults")),
  triggerResultsToken      (consumes<edm::TriggerResults>                    (triggerResultsTag)),


  muonsToken               (consumes<std::vector<ScoutingMuon> >             (iConfig.getParameter<edm::InputTag>("muons"))), 
  electronsToken           (consumes<std::vector<ScoutingElectron> >         (iConfig.getParameter<edm::InputTag>("electrons"))), 
  photonsToken           (consumes<std::vector<ScoutingPhoton> >         (iConfig.getParameter<edm::InputTag>("photons"))), 
  pfcandsToken             (consumes<std::vector<ScoutingParticle> >         (iConfig.getParameter<edm::InputTag>("pfcands"))), 
  pfjetsToken              (consumes<std::vector<ScoutingPFJet> >            (iConfig.getParameter<edm::InputTag>("pfjets"))), 
//  pileupInfoToken          (consumes<std::vector<PileupSummaryInfo> >        (iConfig.getParameter<edm::InputTag>("pileupinfo"))),
//  gensToken                (consumes<std::vector<reco::GenParticle> >        (iConfig.getParameter<edm::InputTag>("gens"))),
  //genEvtInfoToken          (consumes<GenEventInfoProduct>                    (iConfig.getParameter<edm::InputTag>("geneventinfo"))),    
  doL1                     (iConfig.existsAs<bool>("doL1")               ?    iConfig.getParameter<bool>  ("doL1")            : false)
{
  usesResource("TFileService");
  if (doL1) {
    algInputTag_ = iConfig.getParameter<edm::InputTag>("AlgInputTag");
    algToken_ = consumes<BXVector<GlobalAlgBlk>>(algInputTag_);
    l1Seeds_ = iConfig.getParameter<std::vector<std::string> >("l1Seeds");
    l1GtUtils_ = new l1t::L1TGlobalUtil(iConfig,consumesCollector());	
  }
  else {
    l1Seeds_ = std::vector<std::string>();
    l1GtUtils_ = 0;
  }

 // Access the TFileService
  edm::Service<TFileService> fs;

  // Create the TTree
  tree = fs->make<TTree>("tree"       , "tree");

  // Event weights
    
  tree->Branch("lumSec"		, &lumSec			 , "lumSec/i" );
  tree->Branch("run"			, &run				 , "run/i" );
  //tree->Branch("nvtx"			, &nvtx				 , "nvtx/i" );
    
  // Triggers
  tree->Branch("trig"                 , &trig                          , "trig/b");
  tree->Branch("l1Result"		, "std::vector<bool>"             ,&l1Result_	, 32000, 0);		
  // Pileup info
  //tree->Branch("nvtx"                 , &nvtx                          , "nvtx/i"       );

  //Electrons
  tree->Branch("n_ele"            	   ,&n_ele 			, "n_ele/i"		);
  tree->Branch("electronpt"         ,electronpt 		, "electronpt[n_ele]/f"		);
  tree->Branch("electroneta"               ,electroneta 		, "electroneta[n_ele]/f" 	);
  tree->Branch("electronphi"               ,electronphi 		, "electronphi[n_ele]/f"	);
  tree->Branch("electroncharge"            ,electroncharge 		, "electroncharge[n_ele]/f"	);
  tree->Branch("electronm"            	   ,electronm 			,"electronm[n_ele]/f" );
tree->Branch("electrontkiso"               ,electrontkiso 		,"electrontkiso[n_ele]/f" );
tree->Branch("electronHoE"            	   ,electronHoE 		,"electronHoE[n_ele]/f" );
tree->Branch("electronsigmaietaieta"       ,electronsigmaietaieta 	,"electronsigmaietaieta[n_ele]/f" );
 tree->Branch("electrondphiin"              ,electrondphiin 		,"electrondphiin[n_ele]/f" );
 tree->Branch("electrondetain"              ,electrondetain 		,"electrondetain[n_ele]/f" );
 tree->Branch("electronmHits"               ,electronmHits 		,"electronmHits[n_ele]/f" );
 tree->Branch("electronooEMOop"             ,electronooEMOop  		,"electronooEMOop[n_ele]/f" );

  //Photons
  tree->Branch("n_pho"            	   ,&n_pho 			, "n_pho/i"		);
  tree->Branch("photonpt"            	   ,photonpt 			,"photonpt[n_pho]/f");
  tree->Branch("photoneta"            	   ,photoneta 			,"photoneta[n_pho]/f");
  tree->Branch("photonphi"            	   ,photonphi 			,"photonphi[n_pho]/f");	
  tree->Branch("photonm"            	   ,photonm 			,"photonm[n_pho]/f");
  tree->Branch("photonhcaliso"             ,photonhcaliso 		,"photonhcaliso[n_pho]/f");
  tree->Branch("photonecaliso"             ,photonecaliso 		,"photonecaliso[n_pho]/f");
  tree->Branch("photonHoE"            	   ,photonHoE 			,"photonHoE[n_pho]/f");
  tree->Branch("photonsigmaietaieta"       ,photonsigmaietaieta		,"photonsigmaietaieta[n_pho]/f" );

  tree->Branch("n_pfcand"            	   ,&n_pfcand 		,"n_pfcand/i"		);	
  tree->Branch("pfcandpt"        	   ,pfcandpt 		,"pfcandpt[n_pfcand]/f" );
  tree->Branch("pfcandeta"            	   ,pfcandeta 		,"pfcandeta[n_pfcand]/f" );
  tree->Branch("pfcandphi"            	   ,pfcandphi		,"pfcandphi[n_pfcand]/f" );
  tree->Branch("pdcandm"            	   ,pdcandm 		,"pfcandm[n_pfcand]/f" );
  tree->Branch("pfcandpdgid"               ,pfcandpdgid		,"pfcandpdgid[n_pfcand]/f" );
  tree->Branch("pfcandvertex"              ,pfcandvertex 	,"pfcandvertex[n_pfcand]/f" );

  tree->Branch("n_mu"            	   ,&n_mu 			, "n_mu/i"		);
  tree->Branch("muonpt", muonpt	,"muonpt[n_mu]/f");
  tree->Branch("muoneta", muoneta	,"muoneta[n_mu]/f");
  tree->Branch("muonphi", muonphi	,"muonphi[n_mu]/f");
  tree->Branch("muonm", muonm	,"muonm[n_mu]/f");
  tree->Branch("muonecaliso", muonecaliso	,"muonecaliso[n_mu]/f");
  tree->Branch("muonhcaliso", muonhcaliso	,"muonhcaliso[n_mu]/f");
  tree->Branch("muontrkiso", muontrkiso	,"muontrkiso[n_mu]/f");
  tree->Branch("muonchi2", muonchi2	,"muonchi2[n_mu]/f");
  tree->Branch("muonndof", muonndof	,"muonndof[n_mu]/f");
  tree->Branch("muoncharge", muoncharge	,"muoncharge[n_mu]/f");
  tree->Branch("muondxy", muondxy	,"muondxy[n_mu]/f");
  tree->Branch("muondz", muondz	,"muondz[n_mu]/f");
  tree->Branch("muonnvalidmuonhits", muonnvalidmuonhits	,"muonnvalidmuonhits[n_mu]/f");
  tree->Branch("muonvalidpixelhits", muonnvalidpixelhits  ,"muonnvalidpixelhits[n_mu]/f");
  
  tree->Branch("muonnmatchedstations", muonnmatchedstations	,"muonnmatchedstations[n_mu]/f");
  tree->Branch("muontype",   muontype    ,"muontype[n_mu]/f");
  tree->Branch("muonnvalidstriphits",    muonnvalidstriphits   ,"muonnvalidstriphits[n_mu]/f");
  tree->Branch("muontrkqoverp",    muontrkqoverp   ,"muontrkqoverp[n_mu]/f");
  tree->Branch("muontrklambda",   muontrklambda    ,"muontrklambda[n_mu]/f");
  tree->Branch("muontrkpt",   muontrkpt    ,"muontrkpt[n_mu]/f");
  tree->Branch("muontrkphi",  muontrkphi     ,"muontrkphi[n_mu]/f");
  tree->Branch("muontrketa",   muontrketa    ,"muontrketa[n_mu]/f");
  tree->Branch("muontrkqoverperror",   muontrkqoverperror    ,"muontrkqoverperror[n_mu]/f");
  tree->Branch("muontrklambdaerror",   muontrklambdaerror    ,"muontrklambdaerror[n_mu]/f");
  tree->Branch("muontrkpterror",   muontrkpterror    ,"muontrkpterror[n_mu]/f");
  tree->Branch("muontrkphierror",   muontrkphierror    ,"muontrkphierror[n_mu]/f");
  tree->Branch("muontrketaerror",   muontrketaerror    ,"muontrketaerror[n_mu]/f");
  tree->Branch("muontrkdszerror",   muontrkdszerror    ,"muontrkdszerror[n_mu]/f");
  tree->Branch("muontrkdsz",    muontrkdsz   ,"muontrkdsz[n_mu]/f");


  tree->Branch("n_jet"            	   	,&n_jet 			, "n_jet/i"		);
  tree->Branch("jetpt"            	   	,jetpt 				,"jetpt[n_jet]/f" );
  tree->Branch("jeteta"            	   	,jeteta 			,"jeteta[n_jet]/f" );
  tree->Branch("jetphi"            	   	,jetphi 			,"jetphi[n_jet]/f" );
  tree->Branch("jetm"            	   	,jetm 				,"jetm[n_jet]/f" );
  tree->Branch("jetarea"            	   	,jetarea			,"jetarea[n_jet]/f" );
  tree->Branch("jetchargedHadronEnergy"         ,jetchargedHadronEnergy 	,"jetchargedHadronEnergy[n_jet]/f" );
  tree->Branch("jetneutralHadronEnergy"         ,jetneutralHadronEnergy 	,"jetneutralHadronEnergy[n_jet]/f" );
  tree->Branch("jetphotonEnergy"            	,jetphotonEnergy 		,"jetphotonEnergy[n_jet]/f" );
  tree->Branch("jetelectronEnergy"              ,jetelectronEnergy 		,"jetelectronEnergy[n_jet]/f" );
  tree->Branch("jetmuonEnergy"    		   ,jetmuonEnergy 		,"jetmuonEnergy[n_jet]/f" );
  tree->Branch("jetHFHadronEnergy"            	   ,jetHFHadronEnergy 		,"jetHFHadronEnergy[n_jet]/f" );
  tree->Branch("jetHFEMEnergy"            	   ,jetHFEMEnergy 		,"jetHFEMEnergy[n_jet]/f" );
  tree->Branch("jetHOEnergy"            	   ,jetHOEnergy 		,"jeHOENergyt[n_jet]/f" );
  tree->Branch("jetchargedHadronMultiplicity"      ,jetchargedHadronMultiplicity 		,"jetchargedHadronMultiplicity[n_jet]/f" );
  tree->Branch("jetneutralHadronMultiplicity"      ,jetneutralHadronMultiplicity 		,"jetneutralHadronMultiplicity[n_jet]/f" );
  tree->Branch("jetphotonMultiplicity"            	   ,jetphotonMultiplicity 		,"jetphotonMultiplicity[n_jet]/f" );
  tree->Branch("jetelectronMultiplicity"            	   ,jetelectronMultiplicity 		,"jetelectronMultiplicity[n_jet]/f" );
  tree->Branch("jetmuonMultiplicity"            	   ,jetmuonMultiplicity 		,"jetmuonMultiplicity[n_jet]/f" );
  tree->Branch("jetHFHadronMultiplicity"            	   ,jetHFHadronMultiplicity 		,"jetHFHadronMultiplicity[n_jet]/f" );
  tree->Branch("jetHFEMMultiplicity"            	   ,jetHFEMMultiplicity 		,"jetHFEMMultiplicity[n_jet]/f" );
  tree->Branch("jetcsv"            	   	,jetcsv 		,"jetcsv[n_jet]/f" );
  tree->Branch("jetmvaDiscriminator"            	   ,jetmvaDiscriminator 		,"jetmvaDiscriminator[n_jet]/f" );
  tree->Branch("jetconstituents"            	, "std::vector< vector<int16_t> >"   , &jetconstituents 		, 32000, 0);
  

 
}


PFScoutToFlatReduct::~PFScoutToFlatReduct() {
}

void PFScoutToFlatReduct::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  using namespace reco;
    
  // Handles to the EDM content
  edm::Handle<edm::TriggerResults> triggerResultsH;
  iEvent.getByToken(triggerResultsToken, triggerResultsH);
    
  Handle<vector<ScoutingElectron> > electronsH;
  iEvent.getByToken(electronsToken, electronsH);

  Handle<vector<ScoutingMuon> > muonsH;
  iEvent.getByToken(muonsToken, muonsH);

  Handle<vector<ScoutingPhoton> > photonsH;
  iEvent.getByToken(photonsToken, photonsH);

  Handle<vector<ScoutingPFJet> > pfjetsH;
  iEvent.getByToken(pfjetsToken, pfjetsH);
    
  Handle<vector<ScoutingParticle> > pfcandsH;
  iEvent.getByToken(pfcandsToken, pfcandsH);

  run = iEvent.eventAuxiliary().run();
  lumSec = iEvent.eventAuxiliary().luminosityBlock();


  // Which triggers fired
  for (size_t i = 0; i < triggerPathsVector.size(); i++) {
    if (triggerPathsMap[triggerPathsVector[i]] == -1) continue;
    if (i == 0  && triggerResultsH->accept(triggerPathsMap[triggerPathsVector[i]])) trig +=   1; // DST_L1DoubleMu_CaloScouting_PFScouting
    if (i == 1  && triggerResultsH->accept(triggerPathsMap[triggerPathsVector[i]])) trig +=   2; // DST_DoubleMu3_Mass10_CaloScouting_PFScouting
    if (i == 2  && triggerResultsH->accept(triggerPathsMap[triggerPathsVector[i]])) trig +=   4; // DST_ZeroBias_CaloScouting_PFScouting
    if (i == 3  && triggerResultsH->accept(triggerPathsMap[triggerPathsVector[i]])) trig +=   8; // DST_L1HTT_CaloScouting_PFScouting
    if (i == 4  && triggerResultsH->accept(triggerPathsMap[triggerPathsVector[i]])) trig +=  16; // DST_CaloJet40_CaloScouting_PFScouting
    if (i == 5  && triggerResultsH->accept(triggerPathsMap[triggerPathsVector[i]])) trig +=  32; // DST_HT250_CaloScouting
    if (i == 6  && triggerResultsH->accept(triggerPathsMap[triggerPathsVector[i]])) trig +=  64; // DST_HT410_PFScouting
    if (i == 7  && triggerResultsH->accept(triggerPathsMap[triggerPathsVector[i]])) trig += 128; // DST_HT450_PFScouting
  }
  
  jetconstituents.clear();

  n_ele = 0;
  for (auto electrons_iter = electronsH->begin(); electrons_iter != electronsH->end(); ++electrons_iter) 
    {
      electronpt[n_ele]	=electrons_iter->pt();
      electroneta[n_ele]	=electrons_iter->eta();
      electronphi[n_ele]	=electrons_iter->phi();		
      electronm[n_ele]		=electrons_iter->m();
      electrondetain[n_ele]	=electrons_iter->dEtaIn();
      electrondphiin[n_ele]	=electrons_iter->dPhiIn();
      electronsigmaietaieta[n_ele]=electrons_iter->sigmaIetaIeta();
      electronHoE[n_ele]	=electrons_iter->hOverE();		
      electronooEMOop[n_ele]	=electrons_iter->ooEMOop();
      electronmHits[n_ele]	=electrons_iter->missingHits();
      electroncharge[n_ele]	=electrons_iter->charge();
      electrontkiso[n_ele]	=electrons_iter->trackIso();
      electronecaliso[n_ele]	=electrons_iter->ecalIso();
      electronhcaliso[n_ele]	=electrons_iter->hcalIso();
      n_ele++;
    }

  n_pho = 0;

  for (auto photons_iter = photonsH->begin(); photons_iter != photonsH->end(); ++photons_iter) {
    photonpt[n_pho]=photons_iter->pt();
    photoneta[n_pho]=photons_iter->eta();
    photonphi[n_pho]=photons_iter->phi();
    photonm[n_pho]=photons_iter->m();
    photonsigmaietaieta[n_pho]=photons_iter->sigmaIetaIeta();
    photonHoE[n_pho]=photons_iter->hOverE();
    photonecaliso[n_pho]=photons_iter->ecalIso();
    photonhcaliso[n_pho]=photons_iter->hcalIso();
    
    n_pho++;
  }

  n_pfcand = 0;
    for (auto pfcands_iter = pfcandsH->begin(); pfcands_iter != pfcandsH->end(); ++pfcands_iter) {
      pfcandpt[n_pfcand]=MiniFloatConverter::float16to32(MiniFloatConverter::float32to16(pfcands_iter->pt()));
      pfcandeta[n_pfcand]=MiniFloatConverter::float16to32(MiniFloatConverter::float32to16(pfcands_iter->eta()));
      pfcandphi[n_pfcand]=MiniFloatConverter::float16to32(MiniFloatConverter::float32to16(pfcands_iter->phi()));
    pdcandm[n_pfcand]=pfcands_iter->m();
    pfcandpdgid[n_pfcand]=pfcands_iter->pdgId();
    pfcandvertex[n_pfcand]=pfcands_iter->vertex();

    n_pfcand++;
  } 

     n_mu=0;
for (auto muons_iter = muonsH->begin(); muons_iter != muonsH->end(); ++muons_iter) {
 	muonpt[n_mu]=muons_iter->pt();
   	muoneta[n_mu]=muons_iter->eta();
   	muonphi[n_mu]=muons_iter->phi();
   	muonm[n_mu]=muons_iter->m();
   	muonecaliso[n_mu]=muons_iter->ecalIso();
   	muonhcaliso[n_mu]=muons_iter->hcalIso();
   	muontrkiso[n_mu]=muons_iter->chi2();
   	muonchi2[n_mu]=muons_iter->ndof();
   	muonndof[n_mu]=muons_iter->charge();
   	muoncharge[n_mu]=muons_iter->dxy();
   	muondxy[n_mu]=muons_iter->dz();
   	muondz[n_mu]=muons_iter->nValidMuonHits();
   	muonnvalidmuonhits[n_mu]=muons_iter->nValidPixelHits();
   	muonnvalidpixelhits[n_mu]=muons_iter->nMatchedStations();
   	muonnmatchedstations[n_mu]=muons_iter->nTrackerLayersWithMeasurement();
         muontype[n_mu]=muons_iter->type();
         muonnvalidstriphits[n_mu]=muons_iter->nValidStripHits();
         muontrkqoverp[n_mu]=muons_iter->trk_qoverp();
         muontrklambda[n_mu]=muons_iter->trk_lambda();
         muontrkpt[n_mu]=muons_iter->trk_pt();
         muontrkphi[n_mu]=muons_iter->trk_phi();
         muontrketa[n_mu]=muons_iter->trk_eta();
         muontrkqoverperror[n_mu]=muons_iter->dxyError();
         muontrklambdaerror[n_mu]=muons_iter->dzError();
         muontrkpterror[n_mu]=muons_iter->trk_qoverpError();
         muontrkphierror[n_mu]=muons_iter->trk_lambdaError();
         muontrketaerror[n_mu]=muons_iter->trk_phiError();
         muontrkdszerror[n_mu]=muons_iter->trk_dsz();
         muontrkdsz[n_mu]=muons_iter->trk_dszError();
         n_mu++;
 }


  n_jet = 0;
   for (auto pfjets_iter = pfjetsH->begin(); pfjets_iter != pfjetsH->end(); ++pfjets_iter) {
    jetpt[n_jet]			=pfjets_iter->pt();
    jeteta[n_jet]			=pfjets_iter->eta();
    jetphi[n_jet]			=pfjets_iter->phi();
    jetm[n_jet]				=pfjets_iter->m();
    jetarea[n_jet]			=pfjets_iter->jetArea();
    jetchargedHadronEnergy[n_jet]	=pfjets_iter->chargedHadronEnergy();
    jetneutralHadronEnergy[n_jet]	=pfjets_iter->neutralHadronEnergy();
    jetphotonEnergy[n_jet]		=pfjets_iter->photonEnergy();
    jetelectronEnergy[n_jet]		=pfjets_iter->electronEnergy();
    jetmuonEnergy[n_jet]		=pfjets_iter->muonEnergy();
    jetHFHadronEnergy[n_jet]		=pfjets_iter->HFHadronEnergy();
    jetHFEMEnergy[n_jet]		=pfjets_iter->HFEMEnergy();
    jetHOEnergy[n_jet]			=pfjets_iter->HOEnergy();
    
    jetchargedHadronMultiplicity[n_jet]	=pfjets_iter->chargedHadronMultiplicity();
    jetneutralHadronMultiplicity[n_jet]	=pfjets_iter->neutralHadronMultiplicity();
    jetphotonMultiplicity[n_jet]	=pfjets_iter->photonMultiplicity();
    jetelectronMultiplicity[n_jet]	=pfjets_iter->electronMultiplicity();
    jetmuonMultiplicity[n_jet]		=pfjets_iter->muonMultiplicity();
    jetHFHadronMultiplicity[n_jet]	=pfjets_iter->HFHadronMultiplicity();
    jetHFEMMultiplicity[n_jet]		=pfjets_iter->HFEMMultiplicity();
    jetcsv[n_jet]			=pfjets_iter->csv();
    jetmvaDiscriminator[n_jet]		=pfjets_iter->mvaDiscriminator();
    jetconstituents.push_back(vector<int16_t>(pfjets_iter->constituents()));
    n_jet++;
  }
  
 if (doL1) {
    l1GtUtils_->retrieveL1(iEvent,iSetup,algToken_);
    /*	for( int r = 99; r<280; r++){
	string name ("empty");
	bool algoName_ = false;
	algoName_ = l1GtUtils_->getAlgNameFromBit(r,name);
	cout << "getAlgNameFromBit = " << algoName_  << endl;
	cout << "L1 bit number = " << r << " ; L1 bit name = " << name << endl;
	}*/
    for( unsigned int iseed = 0; iseed < l1Seeds_.size(); iseed++ ) {
      bool l1htbit = 0;	
			
      l1GtUtils_->getFinalDecisionByName(string(l1Seeds_[iseed]), l1htbit);
      //cout<<string(l1Seeds_[iseed])<<"  "<<l1htbit<<endl;
      l1Result_.push_back( l1htbit );
      }
 }


 tree->Fill();	
	
}


void PFScoutToFlatReduct::beginJob() {
  
}

void PFScoutToFlatReduct::endJob() {
}

void PFScoutToFlatReduct::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  // HLT paths

  triggerPathsVector.push_back("DST_DoubleMu1_noVtx_CaloScouting_v*");
  triggerPathsVector.push_back("DST_DoubleMu3_noVtx_CaloScouting_v*");
  triggerPathsVector.push_back("DST_DoubleMu3_noVtx_Mass10_PFScouting_v*");
  triggerPathsVector.push_back("DST_L1HTT_CaloScouting_PFScouting_v*");
  triggerPathsVector.push_back("DST_CaloJet40_CaloScouting_PFScouting_v*");
  triggerPathsVector.push_back("DST_HT250_CaloScouting_v*");
  triggerPathsVector.push_back("DST_HT410_PFScouting_v*");
  triggerPathsVector.push_back("DST_HT450_PFScouting_v*");

  HLTConfigProvider hltConfig;
  bool changedConfig = false;
  hltConfig.init(iRun, iSetup, triggerResultsTag.process(), changedConfig);

  for (size_t i = 0; i < triggerPathsVector.size(); i++) {
    triggerPathsMap[triggerPathsVector[i]] = -1;
  }

  for(size_t i = 0; i < triggerPathsVector.size(); i++){
    TPRegexp pattern(triggerPathsVector[i]);
    for(size_t j = 0; j < hltConfig.triggerNames().size(); j++){
      std::string pathName = hltConfig.triggerNames()[j];
      if(TString(pathName).Contains(pattern)){
	triggerPathsMap[triggerPathsVector[i]] = j;
      }
    }
  }
}

void PFScoutToFlatReduct::endRun(edm::Run const&, edm::EventSetup const&) {
}

void PFScoutToFlatReduct::beginLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) {
}

void PFScoutToFlatReduct::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
}

void PFScoutToFlatReduct::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(PFScoutToFlatReduct);
