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
// $Id: SteppingAction.cc 69223 2013-04-23 12:36:10Z gcosmo $
// 
/// \file SteppingAction.cc
/// \brief Implementation of the SteppingAction class

#include "SteppingAction.hh"
#include "RunData.hh"
#include "DetectorConstruction.hh"

#include "G4Step.hh"
#include "G4RunManager.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::SteppingAction(
                      const DetectorConstruction* detectorConstruction)
  : G4UserSteppingAction(),
    fDetConstruction(detectorConstruction)
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::~SteppingAction()
{ 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SteppingAction::UserSteppingAction(const G4Step* step)
{
// Collect energy and track length step by step

  // get volume of the current step
  G4VPhysicalVolume* volume 
    = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume();
  
  // energy deposit
  G4double edep = step->GetTotalEnergyDeposit();
  
  // step length
  G4double stepLength = 0.;
  if ( step->GetTrack()->GetDefinition()->GetPDGCharge() != 0. ) {
    stepLength = step->GetStepLength();
  }

  G4StepPoint* point1 = step->GetPreStepPoint();
  G4StepPoint* point2 = step->GetPostStepPoint();
  G4ThreeVector pos1 = point1->GetPosition();
  G4ThreeVector pos2 = point2->GetPosition();

  //G4cout << "sqr " << pos1.z() << " " << pos2.z() << " " << pos1.x() << " " << pos2.x() << " " << edep << " " << step->GetTrack()->GetDefinition()->GetParticleName() << " " << step->GetTrack()->GetKineticEnergy() << G4endl;
      
  G4cout << "sqr " << pos1.x() << " " << pos1.y() << " " << pos1.z() << " " << edep << G4endl;
  
  RunData* runData = static_cast<RunData*>
    (G4RunManager::GetRunManager()->GetNonConstCurrentRun());

  if ( volume == fDetConstruction->GetAbsorberPV() ) {
    runData->Add(kAbs, edep, stepLength);
  }
  else if ( volume == fDetConstruction->GetGapPV() ) {
    runData->Add(kGap, edep, stepLength);
  }
  else{
    runData->Add(kAbs, edep, stepLength);
    G4cout << "where am i ??? " << G4endl;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
