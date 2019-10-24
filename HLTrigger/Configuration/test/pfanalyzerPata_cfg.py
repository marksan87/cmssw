import FWCore.ParameterSet.Config as cms

process = cms.Process('LL')

process.load('PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')



## ----------------- Global Tag -----------------
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')


#--------------------- Report and output ---------------------------   

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.TFileService=cms.Service("TFileService",
                                 fileName=cms.string("recoilgpuretry.root")
                                 )

process.options = cms.untracked.PSet(
        allowUnscheduled = cms.untracked.bool(True),
        wantSummary = cms.untracked.bool(False),
)



##-------------------- Define the source  ----------------------------

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        #'file:outputScoutingPF_fullTrackingDevMCFull.root'
        'file:outputScoutingPF_pixelTracking_gpu_DevMC.root'
        #'file:outputScoutingPF_pixelTrackingDevMC.root'
    )
)

##--- l1 stage2 digis ---
process.load("EventFilter.L1TRawToDigi.gtStage2Digis_cfi")
process.gtStage2Digis.InputLabel = cms.InputTag( "hltFEDSelectorL1" )


##-------------------- User analyzer  --------------------------------
#import trigger conf


process.dijetscouting = cms.EDAnalyzer(
    'pfTreeProducer2',
    ## JETS/MET ########################################
    jetsAK4    = cms.InputTag('hltScoutingPFPacker'),
    muons      = cms.InputTag("hltScoutingMuonPacker"),
    metpt      = cms.InputTag('hltScoutingPFPacker:pfMetPt'),
    metphi     = cms.InputTag('hltScoutingPFPacker:pfMetPhi'),
    genJet     = cms.InputTag('ak4GenJets')
    
)


# ------------------ path --------------------------

process.p = cms.Path(process.dijetscouting)
