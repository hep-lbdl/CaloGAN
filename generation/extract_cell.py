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

myfile = TFile("plz_work_kthxbai.root")
mytree = myfile.Get("fancy_tree")

c = TCanvas("a","a",500,500)
zsegmentation_tot = TH1F("","",3,np.array([-240.,-150.,197.,240.]))
sampling1_eta_tot = TH2F("","",3,-240.,240.,480/5,-240.,240.)
sampling2_eta_tot = TH2F("","",480/40,-240.,240.,480/40,-240.,240.)
sampling3_eta_tot = TH2F("","",480/40,-240.,240.,480/80,-240.,240.)
for i in range(mytree.GetEntries()):
    mytree.GetEntry(i)
    if (i%100==0):
        print i,mytree.GetEntries()
        pass
    zsegmentation = TH1F("","",3,np.array([-240.,-150.,197.,240.]))
    sampling1_eta = TH2F("","",3,-240.,240.,480/5,-240.,240.)
    sampling2_eta = TH2F("","",480/40,-240.,240.,480/40,-240.,240.)
    sampling3_eta = TH2F("","",480/40,-240.,240.,480/80,-240.,240.)

    y = "energy"
    exec("%s = %s" % (y,"mytree.cell_0"))
    for j in range(507):
        exec("%s = %s" % (y,"mytree.cell_"+str(j)))
        xbin = get_x(j,get_y(j,get_z(j)),get_z(j))
        ybin = get_y(j,get_z(j))
        zbin = get_z(j)
        zsegmentation.Fill(zsegmentation.GetXaxis().GetBinCenter(zbin+1),energy)
        zsegmentation_tot.Fill(zsegmentation.GetXaxis().GetBinCenter(zbin+1),energy)
        if (zbin==0):
            sampling1_eta.Fill(sampling1_eta.GetXaxis().GetBinCenter(xbin+1),sampling1_eta.GetYaxis().GetBinCenter(ybin+1),energy)
            sampling1_eta_tot.Fill(sampling1_eta.GetXaxis().GetBinCenter(xbin+1),sampling1_eta.GetYaxis().GetBinCenter(ybin+1),energy)
        elif (zbin==1):
            sampling2_eta.Fill(sampling2_eta.GetXaxis().GetBinCenter(xbin+1),sampling2_eta.GetYaxis().GetBinCenter(ybin+1),energy)
            sampling2_eta_tot.Fill(sampling2_eta.GetXaxis().GetBinCenter(xbin+1),sampling2_eta.GetYaxis().GetBinCenter(ybin+1),energy)
        else:
            sampling3_eta.Fill(sampling3_eta.GetXaxis().GetBinCenter(xbin+1),sampling3_eta.GetYaxis().GetBinCenter(ybin+1),energy)
            sampling3_eta_tot.Fill(sampling3_eta.GetXaxis().GetBinCenter(xbin+1),sampling3_eta.GetYaxis().GetBinCenter(ybin+1),energy)
            pass
        pass
    if (i < 10):
        zsegmentation.Draw()
        c.Print("plots/zprofile_"+str(i)+".pdf")
        sampling1_eta.Draw("colz")
        c.Print("plots/xy_layer1_"+str(i)+".pdf")
        sampling2_eta.Draw("colz")
        c.Print("plots/xy_layer2_"+str(i)+".pdf")
        sampling3_eta.Draw("colz")
        c.Print("plots/xy_layer3_"+str(i)+".pdf")
        pass
    pass

zsegmentation_tot.Draw()
c.Print("plots/tot_zprofile_"+str(i)+".pdf")
sampling1_eta_tot.Draw("colz")
c.Print("plots/tot_xy_layer1_"+str(i)+".pdf")
sampling2_eta_tot.Draw("colz")
c.Print("plots/tot_xy_layer2_"+str(i)+".pdf")
sampling3_eta_tot.Draw("colz")
c.Print("plots/tot_xy_layer3_"+str(i)+".pdf")
