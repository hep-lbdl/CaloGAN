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
// $Id: RunData.hh 69223 2013-04-23 12:36:10Z gcosmo $
// 
/// \file RunData.hh
/// \brief Definition of the RunData class

#ifndef RunData_h
#define RunData_h 1

#include "G4Run.hh"
#include "globals.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

enum {
  kAbs = 0,
  kGap = 1,
  kDim = 2, 
  kNumCells = 504 + 3 // 3 overflow bins for the three calo layers
};  

///  Run data class
///
/// It defines data members to hold the energy deposit and track lengths
/// of charged particles in Absober and Gap layers.
/// 
/// In order to reduce the number of data members a 2-dimensions array 
/// is introduced for each quantity:
/// - fEdep[], fTrackLength[].
///
/// The data are collected step by step in SteppingAction, and
/// the accumulated values are filled in histograms and entuple
/// event by event in EventAction.

class RunData : public G4Run
{
public:
  RunData();
  virtual ~RunData();

  // void Add(G4int id, G4double de, G4double dl);
  void Add(G4int id, G4double de);
  void FillPerEvent();
  
  void Reset();

  // Get methods
  // G4String  GetVolumeName(G4int id) const;
  G4double  GetEdep(G4int id) const;
  G4double GetTotalEnergy(){return TotalEnergy;};
  void SetTotalEnergy(G4double e){TotalEnergy = e;};
  // G4double  GetTrackLength(G4int id) const; 

private:
  // G4String  fVolumeNames[kDim];
  G4double  fEdep[kNumCells];
  G4double TotalEnergy;
  // G4double  fTrackLength[kDim];
};

// inline functions

// inline void RunData::Add(G4int id, G4double de, G4double dl) {
inline void RunData::Add(G4int id, G4double de) {
  fEdep[id] += de; 
  // fTrackLength[id] += dl;
}

// inline G4String  RunData::GetVolumeName(G4int id) const {
//   return fVolumeNames[id];
// }

inline G4double  RunData::GetEdep(G4int id) const {
  return fEdep[id];
}   

// inline G4double  RunData::GetTrackLength(G4int id) const {
//   return fTrackLength[id];
// }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif

