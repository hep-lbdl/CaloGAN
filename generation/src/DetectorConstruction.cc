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
// $Id: DetectorConstruction.cc 87359 2014-12-01 16:04:27Z gcosmo $
// 
/// \file DetectorConstruction.cc
/// \brief Implementation of the DetectorConstruction class

#include "DetectorConstruction.hh"

#include "G4Material.hh"
#include "G4NistManager.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4PVReplica.hh"
#include "G4GlobalMagFieldMessenger.hh"
#include "G4AutoDelete.hh"

#include "G4GeometryManager.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4SolidStore.hh"

#include "G4VisAttributes.hh"
#include "G4Colour.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4ThreadLocal 
G4GlobalMagFieldMessenger* DetectorConstruction::fMagFieldMessenger = 0; 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::DetectorConstruction()
 : G4VUserDetectorConstruction(),
   fAbsorberPV(0),
   fGapPV(0),
   fAbsorberPV2(0),
   fGapPV2(0),
   fCheckOverlaps(true)
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::~DetectorConstruction()
{ 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  // Define materials 
  DefineMaterials();
  
  // Define volumes
  return DefineVolumes();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::DefineMaterials()
{ 
  // Lead material defined using NIST Manager
  G4NistManager* nistManager = G4NistManager::Instance();
  nistManager->FindOrBuildMaterial("G4_Pb");

 // Lead material defined using NIST Manager
  //G4NistManager* nistManager = G4NistManager::Instance();
  nistManager->FindOrBuildMaterial("G4_W");
 
  // Liquid argon material
  G4double a;  // mass of a mole;
  G4double z;  // z=mean number of protons;  
  G4double density; 
  new G4Material("liquidArgon", z=18., a= 39.95*g/mole, density= 1.390*g/cm3);
         // The argon by NIST Manager is a gas with a different density

  // Vacuum
  new G4Material("Galactic", z=1., a=1.01*g/mole,density= universe_mean_density,
                  kStateGas, 2.73*kelvin, 3.e-18*pascal);

  // Print materials
  G4cout << *(G4Material::GetMaterialTable()) << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* DetectorConstruction::DefineVolumes()
{
  // Geometry parameters
  G4int nofLayers = 80;
  G4double absoThickness = 2.*mm;
  G4double gapThickness =  4.*mm;

  G4int nofLayers2 = 80;//80;
  G4double absoThickness2 = 8.*mm; //2.*mm;
  G4double gapThickness2 =  16.*mm; //4.*mm;
  G4double calorSizeXY  = 100.*cm;

  G4double layerThickness = absoThickness + gapThickness;
  G4double calorThickness = nofLayers * layerThickness;
  G4double layerThickness2 = absoThickness2 + gapThickness2;
  G4double calorThickness2 = nofLayers2 * layerThickness2;
  G4double worldSizeXY = 1.2 * calorSizeXY;
  G4double worldSizeZ  = 1.2 * (calorThickness+calorThickness2); 
  
  // Get materials
  G4Material* defaultMaterial = G4Material::GetMaterial("Galactic");
  G4Material* absorberMaterial = G4Material::GetMaterial("G4_Pb");
  G4Material* gapMaterial = G4Material::GetMaterial("liquidArgon");

  G4Material* absorberMaterial2 = G4Material::GetMaterial("G4_W");
  G4Material* gapMaterial2 = G4Material::GetMaterial("liquidArgon");
  
  if ( ! defaultMaterial || ! absorberMaterial || ! gapMaterial | ! absorberMaterial2 || ! gapMaterial2) {
    G4ExceptionDescription msg;
    msg << "Cannot retrieve materials already defined."; 
    G4Exception("DetectorConstruction::DefineVolumes()",
      "MyCode0001", FatalException, msg);
  }  
   
  //     
  // World
  //
  G4VSolid* worldS 
    = new G4Box("World",           // its name
                 worldSizeXY/2, worldSizeXY/2, worldSizeZ/2); // its size
                         
  G4LogicalVolume* worldLV
    = new G4LogicalVolume(
                 worldS,           // its solid
                 defaultMaterial,  // its material
                 "World");         // its name
                                   
  G4VPhysicalVolume* worldPV
    = new G4PVPlacement(
                 0,                // no rotation
                 G4ThreeVector(),  // at (0,0,0)
                 worldLV,          // its logical volume                         
                 "World",          // its name
                 0,                // its mother  volume
                 false,            // no boolean operation
                 0,                // copy number
                 fCheckOverlaps);  // checking overlaps 
  
  //                               
  // Calorimeter
  //  
  G4VSolid* calorimeterS
    = new G4Box("Calorimeter",     // its name
                 calorSizeXY/2, calorSizeXY/2, calorThickness/2); // its size
                         
  G4LogicalVolume* calorLV
    = new G4LogicalVolume(
                 calorimeterS,     // its solid
                 defaultMaterial,  // its material
                 "Calorimeter");   // its name
                                   
  new G4PVPlacement(
                 0,                // no rotation
                 G4ThreeVector(0.,0.,-calorThickness2/2),  // at (0,0,0)
                 calorLV,          // its logical volume                         
                 "Calorimeter",    // its name
                 worldLV,          // its mother  volume
                 false,            // no boolean operation
                 0,                // copy number
                 fCheckOverlaps);  // checking overlaps 
  
  //                                 
  // Layer
  //
  G4VSolid* layerS 
    = new G4Box("Layer",           // its name
                 calorSizeXY/2, calorSizeXY/2, layerThickness/2); // its size
                         
  G4LogicalVolume* layerLV
    = new G4LogicalVolume(
                 layerS,           // its solid
                 defaultMaterial,  // its material
                 "Layer");         // its name

  new G4PVReplica(
                 "Layer",          // its name
                 layerLV,          // its logical volume
                 calorLV,          // its mother
                 kZAxis,           // axis of replication
                 nofLayers,        // number of replica
                 layerThickness);  // witdth of replica
  
  //                               
  // Absorber
  //
  G4VSolid* absorberS 
    = new G4Box("Abso",            // its name
                 calorSizeXY/2, calorSizeXY/2, absoThickness/2); // its size
                         
  G4LogicalVolume* absorberLV
    = new G4LogicalVolume(
                 absorberS,        // its solid
                 absorberMaterial, // its material
                 "Abso");          // its name
                                   
  fAbsorberPV
    = new G4PVPlacement(
                 0,                // no rotation
                 G4ThreeVector(0., 0., -gapThickness/2), // its position
                 absorberLV,       // its logical volume                         
                 "Abso",           // its name
                 layerLV,          // its mother  volume
                 false,            // no boolean operation
                 0,                // copy number
                 fCheckOverlaps);  // checking overlaps 

  //                               
  // Gap
  //
  G4VSolid* gapS 
    = new G4Box("Gap",             // its name
                 calorSizeXY/2, calorSizeXY/2, gapThickness/2); // its size
                         
  G4LogicalVolume* gapLV
    = new G4LogicalVolume(
                 gapS,             // its solid
                 gapMaterial,      // its material
                 "Gap");           // its name
                                   
  fGapPV
    = new G4PVPlacement(
                 0,                // no rotation
                 G4ThreeVector(0., 0., absoThickness/2), // its position
                 gapLV,            // its logical volume                         
                 "Gap",            // its name
                 layerLV,          // its mother  volume
                 false,            // no boolean operation
                 0,                // copy number
                 fCheckOverlaps);  // checking overlaps 
  
  //
  // Calorimeter 2
  //
  G4VSolid* calorimeterS2
    = new G4Box("Calorimeter 2",
		calorSizeXY/2, calorSizeXY/2, calorThickness2/2);

  G4LogicalVolume* calorLV2
    = new G4LogicalVolume(
			  calorimeterS2,
			  defaultMaterial,
			  "Calorimeter2");

  new G4PVPlacement(
		    0,                // no rotation
		    G4ThreeVector(0,0,calorThickness/2),  // at (0,0,0)
		    calorLV2,          // its logical volume
		    "Calorimeter2",    // its name
		    worldLV,          // its mother  volume
		    false,            // no boolean operation
		    0,                // copy number
		    fCheckOverlaps);  // checking overlaps
  

  G4VSolid* layerS2
    = new G4Box("Layer2",
		calorSizeXY/2, calorSizeXY/2, layerThickness2/2);
  
  G4LogicalVolume* layerLV2
    = new G4LogicalVolume(layerS2,
			  defaultMaterial,
			  "Layer2");
  
  new G4PVReplica(
		  "Layer2",
		  layerLV2,
		  calorLV2,
		  kZAxis,
		  nofLayers2,
		  layerThickness2);

  G4VSolid* absorberS2
    = new G4Box("Abso2",
		calorSizeXY/2, calorSizeXY/2, absoThickness2/2);
  
  G4LogicalVolume* absorberLV2
    = new G4LogicalVolume(
			  absorberS2,
			  absorberMaterial2,
			  "Abso2");

  fAbsorberPV2
    = new G4PVPlacement(
			0,
			G4ThreeVector(0., 0., -gapThickness2/2),
			absorberLV2,
			"Abso2",
			layerLV2,
			false,
			0,
			fCheckOverlaps);

  G4VSolid* gapS2
    = new G4Box("Gap2",
		calorSizeXY/2, calorSizeXY/2, gapThickness2/2);
  
  G4LogicalVolume* gapLV2
    = new G4LogicalVolume(
			  gapS2,
			  gapMaterial2,
			  "Gap2");

  fGapPV2
    = new G4PVPlacement(
			0,
			G4ThreeVector(0., 0., absoThickness2/2),
			gapLV2, 
			"Gap2",
			layerLV2,
			false,
			0,
			fCheckOverlaps);

  //
  // print parameters
  //
  G4cout
    << G4endl 
    << "------------------------------------------------------------" << G4endl
    << "---> The ECAL is " << nofLayers << " layers of: [ "
    << absoThickness/mm << "mm of " << absorberMaterial->GetName() 
    << " + "
    << gapThickness/mm << "mm of " << gapMaterial->GetName() << " ] " << G4endl
    << "---> The HCAL is " << nofLayers2 << " layers of: [ "
    << absoThickness2/mm << "mm of " << absorberMaterial2->GetName()
    << " + "
    << gapThickness2/mm << "mm of " << gapMaterial2->GetName() << " ] " << G4endl
    << "------------------------------------------------------------" << G4endl;
  
  //                                        
  // Visualization attributes
  //
  worldLV->SetVisAttributes (G4VisAttributes::Invisible);

  G4VisAttributes* simpleBoxVisAtt= new G4VisAttributes(G4Colour(1.0,1.0,1.0));
  simpleBoxVisAtt->SetVisibility(true);
  calorLV->SetVisAttributes(simpleBoxVisAtt);

  //
  // Always return the physical World
  //
  return worldPV;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::ConstructSDandField()
{ 
  // Create global magnetic field messenger.
  // Uniform magnetic field is then created automatically if
  // the field value is not zero.
  G4ThreeVector fieldValue = G4ThreeVector();
  fMagFieldMessenger = new G4GlobalMagFieldMessenger(fieldValue);
  fMagFieldMessenger->SetVerboseLevel(1);
  
  // Register the field messenger for deleting
  G4AutoDelete::Register(fMagFieldMessenger);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
