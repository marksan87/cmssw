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
#include "DataFormats/JetReco/interface/GenJet.h"
//#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Scouting/interface/ScoutingMuon.h"
#include "DataFormats/Scouting/interface/ScoutingParticle.h"
#include "DataFormats/Scouting/interface/ScoutingVertex.h"
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
#include "DataFormats/Scouting/interface/ScoutingPFJet.h"
#include "DataFormats/Scouting/interface/ScoutingMuon.h"
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


class pfTreeProducer2 : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
	public:
		explicit pfTreeProducer2(const edm::ParameterSet&);
		~pfTreeProducer2();
		
		static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
	
	
	private:
        virtual void beginJob() override;
        virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
        virtual void endJob() override;

        virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
        virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
        virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
        virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
        
        const edm::EDGetTokenT<std::vector<ScoutingPFJet> >     jetsToken;
  const edm::EDGetTokenT<std::vector<reco::GenJet> >      genjetToken;
        const edm::EDGetTokenT<std::vector<ScoutingMuon> >      muonsToken;
        
	edm::EDGetTokenT<double> metpttoken, metphitoken;
  
        const edm::EDGetTokenT<GenEventInfoProduct>             genEvtInfoToken;

       
        TTree* tree;

  	//Run and lumisection
  	int run;
  	int lumSec;
  	std::vector<float>           jpt;
double recx;
double recy;
  double my;
  double mx;
  double u1;
  double u2;

  double chf;
  double nhf;
  double phf;
  double elf;
  double muf;
  double hf_hf;
  double hf_emf;
  double hof;
  int chm;
  int chMult;
  int neMult;
  int npr; 
  TH1F* chn;
TH1F* nhn;
TH1F* phn;
TH1F* ehn;
TH1F* mhn;

TH1F* chn_sim;
TH1F* nhn_sim;
TH1F* phn_sim;
TH1F* ehn_sim;
TH1F* mhn_sim;

  TH1F* jresponse;
  TH1F* upara;
  TH1F* uperp;
  




  int nch_;
  int nnh_;
  int nph_;
  

};

pfTreeProducer2::pfTreeProducer2(const edm::ParameterSet& iConfig): 
  jetsToken            (consumes<std::vector<ScoutingPFJet> >           (iConfig.getParameter<edm::InputTag>("jetsAK4"))),
  genjetToken            (consumes<std::vector<reco::GenJet> >           (iConfig.getParameter<edm::InputTag>("genJet"))),
  muonsToken            (consumes<std::vector<ScoutingMuon> >           (iConfig.getParameter<edm::InputTag>("muons"))),
  metpttoken            (consumes<double>           (iConfig.getParameter<edm::InputTag>("metpt"))),
  metphitoken            (consumes<double>           (iConfig.getParameter<edm::InputTag>("metphi")))
   
{
usesResource("TFileService");	
}


pfTreeProducer2::~pfTreeProducer2() {
}

void pfTreeProducer2::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    using namespace edm;
    using namespace std;
    using namespace reco;
    


    jpt.clear();
    
    
    //recx.clear();
    //recy.clear();
    // Handles to the EDM content
    
    Handle<vector<ScoutingPFJet>> jetsH;
    iEvent.getByToken(jetsToken, jetsH);

    Handle<vector<GenJet>> genjetsH;
    iEvent.getByToken(genjetToken, genjetsH);

    Handle<vector<ScoutingMuon>> muonsH;
    iEvent.getByToken(muonsToken, muonsH);
    Handle<double> metptH;
    iEvent.getByToken(metpttoken, metptH);
    Handle<double> metphiH;
    iEvent.getByToken(metphitoken, metphiH);
    
    for (auto ijet = jetsH->begin(); ijet != jetsH->end(); ++ijet) 
      {
	if(ijet->pt()<20) continue;
	TLorentzVector rtemp;
	rtemp.SetPtEtaPhiM(ijet->pt(),ijet->eta(),ijet->phi(),ijet->m());
	
	bool ismuon = false;
	for (auto imuon = muonsH->begin(); imuon != muonsH->end(); ++imuon) 
	  {
	    TLorentzVector mtemp;
	    mtemp.SetPtEtaPhiM(imuon->pt(),imuon->eta(),imuon->phi(),0.105);
	    if(rtemp.DeltaR(mtemp)<0.1) 
	      {
		ismuon = true;
		continue;
	      }
	  }
	if(ismuon) continue;
	float recopt = ijet->pt();
	float genpt  = -999.;
	bool isgenjet=false;
	for (auto gjet = genjetsH->begin(); gjet != genjetsH->end(); ++gjet) 
	  {	
	    TLorentzVector gtemp;
	    gtemp.SetPtEtaPhiM(gjet->pt(),gjet->eta(),gjet->phi(),gjet->mass());	  
	    if(rtemp.DeltaR(gtemp)<0.1) 
	      {
		isgenjet = true;
		chn_sim->Fill(gjet->chargedHadronMultiplicity());
		nhn_sim->Fill(gjet->neutralHadronMultiplicity());
		ehn_sim->Fill(gjet->chargedEmMultiplicity());
		phn_sim->Fill(gjet->neutralEmMultiplicity());
		mhn_sim->Fill(gjet->muonMultiplicity());
		genpt = gjet->pt();
		continue;
	      }
	  }		
	if(!ismuon && isgenjet)
	  {
	    chn->Fill(ijet->chargedHadronMultiplicity());
	    nhn->Fill(ijet->neutralHadronMultiplicity());
	    ehn->Fill(ijet->electronMultiplicity());
	    phn->Fill(ijet->photonMultiplicity());
	    mhn->Fill(ijet->muonMultiplicity());
	    jresponse->Fill((recopt-genpt)/genpt);
	  }

    }
    

    // jet information
    int a=0;
    vector<TLorentzVector> jetsV;
    /*for (auto ijet = jetsH->begin(); ijet != jetsH->end(); ++ijet) {
      //if(a>0) continue;
	double jet_energy = ijet->photonEnergy() + ijet->chargedHadronEnergy()
                          + ijet->neutralHadronEnergy() + ijet->electronEnergy()
                          + ijet->muonEnergy();

        chf = ijet->chargedHadronEnergy()/jet_energy;
        nhf = ijet->neutralHadronEnergy()/jet_energy;
        phf = ijet->photonEnergy()/jet_energy;
        elf = ijet->electronEnergy()/jet_energy;
        muf = ijet->muonEnergy()/jet_energy;

        hf_hf = ijet->HFHadronEnergy()/jet_energy;
        hf_emf= ijet->HFEMEnergy()/jet_energy;
        hof   = ijet->HOEnergy()/jet_energy;

        chm    = ijet->chargedHadronMultiplicity();

        chMult = ijet->chargedHadronMultiplicity()
                   + ijet->electronMultiplicity() + ijet->muonMultiplicity();
        neMult = ijet->neutralHadronMultiplicity()
                   + ijet->photonMultiplicity();
        npr    = chMult + neMult;

        // Juska's added fractions for identical JetID with recommendations
        double nemf = ijet->photonEnergy()/jet_energy;
        double cemf = ijet->electronEnergy()/jet_energy;
        int NumConst = npr;

        float eta  = ijet->eta();
        float pt   = ijet->pt();

        // https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID
        int idL = (nhf < 0.99 && nemf < 0.99 && NumConst > 1 && muf < 0.8)
            && ((fabs(eta) <= 2.4 && chf > 0 && chMult > 0 && cemf < 0.99)
                || fabs(eta) > 2.4);
        int idT = (nhf < 0.90 && nemf < 0.90 && NumConst > 1 && muf < 0.8)
            && ((fabs(eta) <= 2.4 && chf > 0 && chMult > 0 && cemf < 0.90)
                || fabs(eta) > 2.4);
      
	cout<<ijet->pt()<<"  "<<ijet->eta()<<"  "<<ijet->phi()<<"  "<<idL<<"  "<<idT<<"   "<<ijet->chargedHadronMultiplicity()<<endl;
       jpt.push_back(ijet->pt());
       if(idL==1 ){
       TLorentzVector tempj;
       tempj.SetPtEtaPhiM(pt,eta,ijet->phi(),ijet->m());
       jetsV.push_back(tempj);}
       
       
       a++;
       }*/

    //cout<"made it to after muons"<<endl;
    vector<TLorentzVector> muonsV;
    int c=0;
    for (auto imuon = muonsH->begin(); imuon != muonsH->end(); ++imuon) {
      if(imuon->pt()>20. && abs(imuon->eta())<2.4){
	if(c>1) continue;
      double pt=imuon->pt();
      double eta=imuon->eta();
      double phi=imuon->phi();
      double m=0.104;
      TLorentzVector temp;
      cout<<" muon: "<<pt<<"  "<<pt<<"  "<<eta<<"   "<<phi<<endl;
      temp.SetPtEtaPhiM(pt,eta,phi,m);
      muonsV.push_back(temp);
      c++;
      }
      
    }

    if (muonsV.size()>=2 /*&& (muonsV[0]+muonsV[1]).Pt()>7.*/) {
    //cout<"made it to Z"<<endl;
    
    //cout<"made it to after Z"<<endl;
    mx=TMath::Sin((*metphiH))*(*metptH);
    my=TMath::Cos((*metphiH))*(*metptH);
    

    TVector2 vMet(my, mx);
    TVector2 vZPt((muonsV[0]+muonsV[1]).Px(),(muonsV[0]+muonsV[1]).Py());
    TVector2 vU = -1.0*(vMet+vZPt);
    
    u1 = (((muonsV[0]+muonsV[1]).Px())*(vU.Px()) + ((muonsV[0]+muonsV[1]).Py())*(vU.Py()))/((muonsV[0]+muonsV[1]).Pt());  // u1 = (pT . u)/|pT|
    u2 = (((muonsV[0]+muonsV[1]).Px())*(vU.Py()) - ((muonsV[0]+muonsV[1]).Py())*(vU.Px()))/((muonsV[0]+muonsV[1]).Pt());  // u2 = (pT x u)/|pT|
    uperp->Fill(u1);
    upara->Fill(u2);
    tree->Fill();	
    }	
}


void pfTreeProducer2::beginJob() {
    // Access the TFileService
    edm::Service<TFileService> fs;


    //book Histos
    chn = fs->make<TH1F>("chn", "chn", 25, 0, 25);
    nhn = fs->make<TH1F>("nhn", "nhn", 25, 0, 25);
    phn = fs->make<TH1F>("phn", "phn", 25, 0, 25);
    ehn = fs->make<TH1F>("ehn", "ehn", 25, 0, 25);
    mhn = fs->make<TH1F>("mhn", "mhn", 25, 0, 25);
    chn_sim = fs->make<TH1F>("chn_sim", "chn_sim", 25, 0, 25);
    nhn_sim = fs->make<TH1F>("nhn_sim", "nhn_sim", 25, 0, 25);
    phn_sim = fs->make<TH1F>("phn_sim", "phn_sim", 25, 0, 25);
    ehn_sim = fs->make<TH1F>("ehn_sim", "ehn_sim", 25, 0, 25);
    mhn_sim = fs->make<TH1F>("mhn_sim", "mhn_sim", 25, 0, 25);
    jresponse = fs->make<TH1F>("jresponse", "jresponse", 20, -2, 2);
    uperp = fs->make<TH1F>("uperp", "uperp", 100, -200, 200);
    upara = fs->make<TH1F>("upara", "upara", 100, -200, 200);
    


    // Create the TTree
    tree = fs->make<TTree>("tree"      , "tree");
    tree->Branch("jpt"                 , "std::vector<float>"          , &jpt      , 32000, 0);
    tree->Branch("recx"                , &recx                         , "recx/D"        );
    tree->Branch("recy"                , &recy                         , "recy/D"        );
    tree->Branch("mx"                  , &mx                           , "mx/D"        );
    tree->Branch("my"                  , &my                           , "my/D"        );
    tree->Branch("u1"                  , &u1                           , "u1/D"        );
    tree->Branch("u2"                  , &u2                           , "u2/D"        );

    tree->Branch("chf"                  , &chf                           , "chf/D"        );
    tree->Branch("nhf"                  , &nhf                           , "nhf/D"        );
    tree->Branch("phf"                  , &phf                           , "phf/D"        );

    tree->Branch("elf"                  , &elf                           , "elf/D"        );
    tree->Branch("muf"                  , &muf                           , "muf/D"        );
    tree->Branch("hf_hf"                , &hf_hf                         , "hf_hf/D"        );
    tree->Branch("hf_emf"               , &hf_emf                        , "hf_emf/D"        );
    tree->Branch("hof"                  , &hof                           , "hof/D"        );

    tree->Branch("chm"                  , &chm                           , "chm/D"        );
    tree->Branch("chMult"               , &chMult                        , "chMult/D"        );
    tree->Branch("neMult"               , &neMult                        , "neMult/D"        );
    tree->Branch("npr"                  , &npr                           , "npr/D"        );

    // Event weights
    
}

void pfTreeProducer2::endJob() {
}

void pfTreeProducer2::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
    
}

void pfTreeProducer2::endRun(edm::Run const&, edm::EventSetup const&) {
}

void pfTreeProducer2::beginLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) {
}

void pfTreeProducer2::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
}

void pfTreeProducer2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
	edm::ParameterSetDescription desc;
	desc.setUnknown();
	descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(pfTreeProducer2);
