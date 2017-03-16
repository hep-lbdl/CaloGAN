from ROOT import *
import numpy as np

myfile = open("Geant4_10GeVeminus_LaROnly_10k.txt")

myimage = TH2F("","",500,-240,240,500,-240,240)
zsegmentation = TH1F("","",3,np.array([-240.,-150.,197.,240.]))
sampling1_eta = TH2F("","",3,-240.,240.,480/5,-240.,240.)
sampling2_eta = TH2F("","",480/40,-240.,240.,480/40,-240.,240.)
sampling3_eta = TH2F("","",480/40,-240.,240.,480/80,-240.,240.)

mycounter = 0

outfile = open("Geant4_10GeVeminus_LaROnly_10k_cells_v2.txt","write")
outfile.write(str(sampling1_eta.GetNbinsX())+","+str(sampling1_eta.GetNbinsY())+"|"+str(sampling2_eta.GetNbinsX())+","+str(sampling2_eta.GetNbinsY())+"|"+str(sampling3_eta.GetNbinsX())+","+str(sampling3_eta.GetNbinsY())+"\n")

for line in myfile:
    #print line.split()
    if ('sqr' not in line):
        continue
    x = float(line.split()[1])
    y = float(line.split()[2])
    z = float(line.split()[3])
    E = float(line.split()[4])
    zbin = zsegmentation.GetXaxis().FindBin(z)

    if (z==-288 ):
        if (mycounter > 20000):
            outfile.close()
            exit(1)
        print mycounter,E,sampling1_eta.Integral()+sampling2_eta.Integral()+sampling3_eta.Integral()
        for i in range(1,sampling1_eta.GetNbinsX()+1):
            for j in range(1,sampling1_eta.GetNbinsY()+1):
                outfile.write(str(sampling1_eta.GetBinContent(i,j)))
                if (i == sampling1_eta.GetNbinsX() and j == sampling1_eta.GetNbinsY()):
                    outfile.write("|")
                else:
                    outfile.write(",")
                    pass
                sampling1_eta.SetBinContent(i,j,0.)
                pass
            pass
        for i in range(1,sampling2_eta.GetNbinsX()+1):
            for j in range(1,sampling2_eta.GetNbinsY()+1):
                outfile.write(str(sampling2_eta.GetBinContent(i,j)))
                if (i == sampling2_eta.GetNbinsX() and j == sampling2_eta.GetNbinsY()):
                    outfile.write("|")
                else:
                    outfile.write(",")
                    pass
                sampling2_eta.SetBinContent(i,j,0.)
                pass
            pass
        for i in range(1,sampling3_eta.GetNbinsX()+1):
            for j in range(1,sampling3_eta.GetNbinsY()+1):
                outfile.write(str(sampling3_eta.GetBinContent(i,j)))
                if (i == sampling3_eta.GetNbinsX() and j == sampling3_eta.GetNbinsY()):
                    pass
                else:
                    outfile.write(",")
                    pass
                sampling3_eta.SetBinContent(i,j,0.)
                pass
            pass
        outfile.write("\n")
        mycounter+=1

    if (zbin == 1):
        sampling1_eta.Fill(x,y,E)
    elif (zbin == 2):
        sampling2_eta.Fill(x,y,E)
        pass
    elif (zbin == 3):
        sampling3_eta.Fill(x,y,E)
        pass
    pass

