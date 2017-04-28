//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// $Id: RunData.cc 69223 2013-04-23 12:36:10Z gcosmo $
//
/// \file RunData.cc
/// \brief Implementation of the RunData class

#include "RunData.hh"
#include "Analysis.hh"

#include "G4RunManager.hh"
#include "G4UnitsTable.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunData::RunData() : G4Run()//, fNumCells(4815)
{
  // fVolumeNames[0] = "Absorber";
  // fVolumeNames[1] = "Gap";
 
  for ( G4int i=0; i < kNumCells; i++) {
    fEdep[i] = 0.;
    // fTrackLength[i] = 0.; 
  }  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunData::~RunData()
{;}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunData::FillPerEvent()
{
  // get analysis manager
  G4AnalysisManager* analysisManager = G4AnalysisManager::Instance();
  //accumulate statistic
  //

  for (int i = 0; i < kNumCells; ++i) {
    // analysisManager->CreateNtupleDColumn("cell_" + std::to_string(i));
    analysisManager->FillNtupleDColumn(i, fEdep[i]);
  }
  analysisManager->FillNtupleDColumn(kNumCells, GetTotalEnergy());

  // for ( G4int i=0; i<kDim; i++) {
  //   // fill histograms
  //   // analysisManager->FillH1(i+1, fEdep[i]);
  //   // analysisManager->FillH1(kDim+i+1, fTrackLength[i]);

  //   // fill ntuple
  //   analysisManager->FillNtupleDColumn(i, fEdep[i]);
  //   analysisManager->FillNtupleDColumn(kDim+i, fTrackLength[i]);
  // }  

  analysisManager->AddNtupleRow();  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunData::Reset()
{ 
  for ( G4int i=0; i<kNumCells; i++) {
    fEdep[i] = 0.;
    // fTrackLength[i] = 0.; 
  }  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
