import ROOT
import os,sys
from ROOT import TGraph
from array import array
from ROOT import *
from math import *

ROOT.gStyle.SetOptStat(0)
files=[]
files.append(ROOT.TFile().Open("recoilfull.root"))
files.append(ROOT.TFile().Open("recoilgpuretry.root"))
files.append(ROOT.TFile().Open("recoilcpu.root"))

quant=["chn","nhn","ehn","phn","mhn","jresponse","uperp","upara"]
label=["chargedHadronMultiplicity","neutralHadronMultiplicity","electronMultiplicity","photonMultiplicity","muonMultiplicity","jet response","Z u-parallel [GeV]", "Z u-perpendicular [GeV]"]
#col=[kBlue,kGreen,kRed,kMagenta]
col=[kBlue-6,kGreen-6,kYellow-5,kRed-6]
t=["Full Tracking","PataTrack Tracks","Pixel Tracks","Gen Level"]
qn=0
for q in quant:
    canvas=ROOT.TCanvas(q,q,400,400)
    legend=ROOT.TLegend(0.65,0.65,0.9,0.9)
    for i in range(1,2):
        hist=files[i].Get("dijetscouting/"+q)
        hist.SetLineWidth(2)
        hist.SetLineColor(col[i])
        hist.SetXTitle(label[qn])
        hist.Draw("same")
        legend.AddEntry(hist,t[i],"l")
        if i is 1 and qn<5:
            hist=files[i].Get("dijetscouting/"+q+"_sim")
            hist.SetLineWidth(2)
            hist.SetLineColor(col[3])
            hist.Draw("same")
            legend.AddEntry(hist,t[3],"l")
            

    qn=qn+1
    legend.Draw()
    canvas.SaveAs(q+"Pata.png")
