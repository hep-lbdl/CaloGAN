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
// $Id: RunAction.cc 68835 2013-04-06 07:12:41Z asaim $
//
/// \file RunAction.cc
/// \brief Implementation of the RunAction class

#include "RunAction.hh"
#include "RunData.hh"
#include "Analysis.hh"

#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include <sstream>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::RunAction()
 : G4UserRunAction()
{ 
  // set printing event number per each event
  G4RunManager::GetRunManager()->SetPrintProgress(1);     

  // Create analysis manager
  // The choice of analysis technology is done via selectin of a namespace
  // in Analysis.hh
  G4AnalysisManager* analysisManager = G4AnalysisManager::Instance();
  G4cout << "Using " << analysisManager->GetType() << G4endl;

  // Create directories 
  //analysisManager->SetHistoDirectoryName("histograms");
  //analysisManager->SetNtupleDirectoryName("ntuple");
  analysisManager->SetVerboseLevel(1);
  analysisManager->SetFirstHistoId(1);

  // Book histograms, ntuple
  //
  
  // Creating histograms
  // analysisManager->CreateH1("1","Edep in absorber", 100, 0., 800*MeV);
  // analysisManager->CreateH1("2","Edep in gap", 100, 0., 100*MeV);
  // analysisManager->CreateH1("3","trackL in absorber", 100, 0., 1*m);
  // analysisManager->CreateH1("4","trackL in gap", 100, 0., 50*cm);

  // Creating ntuple
  //

  char const* val = getenv("GAN_TREENAME"); 
  std::string fname = (val == NULL ? std::string("fancy_tree") : std::string(val));


  analysisManager->CreateNtuple(fname.c_str(), "Edep and TrackL");

  int total_bins = 504 + 3;  // 3 overflow bins for the three calo layers

  for (int i = 0; i < total_bins; ++i) {

    std::stringstream out;
    out << i;
    analysisManager->CreateNtupleDColumn("cell_" + out.str());
  }
  analysisManager->CreateNtupleDColumn("TotalEnergy");
  
  // analysisManager->CreateNtupleDColumn("Eabs");
  // analysisManager->CreateNtupleDColumn("Egap");
  // analysisManager->CreateNtupleDColumn("Labs");
  // analysisManager->CreateNtupleDColumn("Lgap");



  analysisManager->FinishNtuple();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::~RunAction()
{
  delete G4AnalysisManager::Instance();  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4Run* RunAction::GenerateRun()
{
  return (new RunData);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::BeginOfRunAction(const G4Run* run)
{ 
  G4cout << "### Run " << run->GetRunID() << " start." << G4endl;

  //inform the runManager to save random number seed
  //G4RunManager::GetRunManager()->SetRandomNumberStore(true);
  
  // Get analysis manager
  G4AnalysisManager* analysisManager = G4AnalysisManager::Instance();

  // Open an output file
  //

  char const* val = getenv("GAN_FNAME"); 
  std::string fname = (val == NULL ? std::string("plz_work_kthxbai") : std::string(val));


  G4String fileName = fname.c_str();
  analysisManager->OpenFile(fileName);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::EndOfRunAction(const G4Run* /*aRun*/)
{
  // print histogram statistics
  //
  G4AnalysisManager* analysisManager = G4AnalysisManager::Instance();
  // if ( analysisManager->GetH1(1) ) {
  //   G4cout << G4endl << " ----> print histograms statistic ";
  //   if(isMaster) {
  //     G4cout << "for the entire run " << G4endl << G4endl; 
  //   }
  //   else {
  //     G4cout << "for the local thread " << G4endl << G4endl; 
  //   }
    
  //   G4cout << " EAbs : mean = " 
  //      << G4BestUnit(analysisManager->GetH1(1)->mean(), "Energy") 
  //      << " rms = " 
  //      << G4BestUnit(analysisManager->GetH1(1)->rms(),  "Energy") << G4endl;
    
  //   G4cout << " EGap : mean = " 
  //      << G4BestUnit(analysisManager->GetH1(2)->mean(), "Energy") 
  //      << " rms = " 
  //      << G4BestUnit(analysisManager->GetH1(2)->rms(),  "Energy") << G4endl;
    
  //   G4cout << " LAbs : mean = " 
  //     << G4BestUnit(analysisManager->GetH1(3)->mean(), "Length") 
  //     << " rms = " 
  //     << G4BestUnit(analysisManager->GetH1(3)->rms(),  "Length") << G4endl;

  //   G4cout << " LGap : mean = " 
  //     << G4BestUnit(analysisManager->GetH1(4)->mean(), "Length") 
  //     << " rms = " 
  //     << G4BestUnit(analysisManager->GetH1(4)->rms(),  "Length") << G4endl;
  // }

  // save histograms & ntuple
  //
  analysisManager->Write();
  analysisManager->CloseFile();

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
