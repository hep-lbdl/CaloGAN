from ROOT import *
import numpy as np

nbins1x = 3;
nbins2x = 12;
nbins3x = 12;
nbins1y = 96;
nbins2y = 12;
nbins3y = 6;
lvl1 = nbins1x * nbins1y;
lvl2 = nbins2x * nbins2y;
lvl3 = nbins3x * nbins3y;

def get_z(myindex):
    if myindex == 504:
        return 0
    elif myindex == 505:
        return 1
    elif myindex == 506:
        return 2
    elif (myindex >= lvl1 + lvl2):
        return 2
    elif (myindex >= lvl1):
        return 1
    else:
        return 0
    pass

def get_y(myindex,zbin):
    if (myindex >=504):
        return -1
    elif (zbin==0):
        return myindex % nbins1y
    elif (zbin==1):
        return myindex % nbins2y
    else:
        return myindex % nbins3y
    pass

def get_x(myindex,ybin,zbin):
    if (myindex >=504):
        return -1
    elif (zbin==0):
        return (myindex - ybin)/nbins1y
    elif (zbin==1):
        return (myindex - lvl1 - ybin)/nbins2y
    else:
        return (myindex - lvl1 - lvl2 - ybin)/nbins3y
    pass

myfile = TFile("cpion-10GeV-1k.root")
#pion-10GeV-1k.root
#electrons-10GeV-5k.root 
#photons-10GeV-1k.root
mytree = myfile.Get("fancy_tree")

c = TCanvas("a","a",500,500)
zsegmentation = TH1F("","",3,np.array([-240.,-150.,197.,240.]))
sampling1_eta = TH2F("","",3,-240.,240.,480/5,-240.,240.)
sampling2_eta = TH2F("","",480/40,-240.,240.,480/40,-240.,240.)
sampling3_eta = TH2F("","",480/40,-240.,240.,480/80,-240.,240.)

Fraction_in_thirdlayer = TH1F("","",100,0,0.01)
Fraction_not_in = TH1F("","",100,0,0.01)
Middle_lateral_width = TH1F("","",100,0,100)
Front_lateral_width = TH1F("","",100,0,100)
Shower_Depth = TH1F("","",100,0,1.5)
Shower_Depth_width = TH1F("","",100,0,1.0)

for i in range(min(1000,mytree.GetEntries())):
    mytree.GetEntry(i)
    if (i%100==0):
        print i,mytree.GetEntries()
        pass
    y = "energy"
    exec("%s = %s" % (y,"mytree.cell_0"))
    total_energy = 0.
    third_layer = 0.
    not_in = 0.
    lateral_depth = 0.
    lateral_depth2 = 0.
    second_layer_X = 0.
    second_layer_X2 = 0.
    first_layer_X = 0.
    first_layer_X2 = 0.
    front_energy = 0.
    middle_energy = 0.
    for j in range(507):
        exec("%s = %s" % (y,"mytree.cell_"+str(j)))
        xbin = get_x(j,get_y(j,get_z(j)),get_z(j))
        ybin = get_y(j,get_z(j))
        zbin = get_z(j)
        zsegmentation.Fill(zsegmentation.GetXaxis().GetBinCenter(zbin+1),energy)
        zvalue = zsegmentation.GetXaxis().GetBinCenter(zbin+1)
        yvalue = 0.;
        xvalue = 0.;
        total_energy+=energy
        lateral_depth+=zbin*energy
        lateral_depth2+=zbin*zbin*energy
        if (zbin==2):
            third_layer+=energy
            pass
        if (xbin < 0 or ybin < 0):
            not_in+=energy
            pass
        if (zbin==0):
            sampling1_eta.Fill(sampling1_eta.GetXaxis().GetBinCenter(xbin+1),sampling1_eta.GetYaxis().GetBinCenter(ybin+1),energy)
            xvalue = sampling1_eta.GetXaxis().GetBinCenter(xbin+1)
            yvalue = sampling1_eta.GetYaxis().GetBinCenter(ybin+1)
        elif (zbin==1):
            sampling2_eta.Fill(sampling2_eta.GetXaxis().GetBinCenter(xbin+1),sampling2_eta.GetYaxis().GetBinCenter(ybin+1),energy)
            xvalue = sampling2_eta.GetXaxis().GetBinCenter(xbin+1)
            yvalue = sampling2_eta.GetYaxis().GetBinCenter(ybin+1)
        else:
            sampling3_eta.Fill(sampling3_eta.GetXaxis().GetBinCenter(xbin+1),sampling3_eta.GetYaxis().GetBinCenter(ybin+1),energy)
            xvalue = sampling3_eta.GetXaxis().GetBinCenter(xbin+1)
            yvalue = sampling3_eta.GetYaxis().GetBinCenter(ybin+1)
            pass
        if (zbin==0):
            first_layer_X += xvalue*energy
            first_layer_X2 += xvalue*xvalue*energy
            front_energy+=energy
        elif (zbin==1):
            second_layer_X += xvalue*energy
            second_layer_X2 += xvalue*xvalue*energy
            middle_energy+=energy
            pass
        pass
    Fraction_in_thirdlayer.Fill(third_layer/total_energy)
    Fraction_not_in.Fill(not_in/total_energy)
    Shower_Depth.Fill(lateral_depth/total_energy)
    if (middle_energy > 0):
        Middle_lateral_width.Fill(((second_layer_X2/middle_energy) - (second_layer_X/middle_energy)**2)**0.5)
        pass
    if (front_energy > 0):
        Front_lateral_width.Fill((first_layer_X2/front_energy - (first_layer_X/front_energy)**2)**0.5)
        pass
    Shower_Depth_width.Fill((lateral_depth2/total_energy - (lateral_depth/total_energy)**2)**0.5)
    pass

zsegmentation.Draw()
c.Print("plots/tot_zprofil.pdf")
sampling1_eta.Draw("colz")
c.Print("plots/tot_xy_layer1.pdf")
sampling2_eta.Draw("colz")
c.Print("plots/tot_xy_layer2.pdf")
gPad.SetLogz()
sampling3_eta.Draw("colz")
for i in range(1,sampling3_eta.GetNbinsX()+1):
    for j in range(1,sampling3_eta.GetNbinsY()+1):
        print i,j,sampling3_eta.GetBinContent(i,j)
        pass
    pass
c.Print("plots/tot_xy_layer3.pdf")
gPad.SetLogz(0)
Fraction_in_thirdlayer.Draw()
c.Print("plots/Fraction_in_thirdlayer.pdf")
Fraction_not_in.Draw()
c.Print("plots/Fraction_not_in.pdf")
Shower_Depth.Draw()
c.Print("plots/Shower_Depth.pdf")
Middle_lateral_width.Draw()
c.Print("plots/Middle_lateral_width.pdf")
Front_lateral_width.Draw()
c.Print("plots/Front_lateral_width.pdf")
Shower_Depth_width.Draw()
c.Print("plots/Shower_Depth_width.pdf")
