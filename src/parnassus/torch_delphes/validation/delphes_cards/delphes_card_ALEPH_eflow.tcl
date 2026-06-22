#Author: Lorenzo Marafatto
#date: 23/02/2025
#version: 1.0
#lmarafat@cern.ch
#
set ExecutionPath {
  ParticlePropagator

  ChargedHadronTrackingEfficiency
  ElectronTrackingEfficiency
  MuonTrackingEfficiency

  ChargedHadronMomentumSmearing
  ElectronMomentumSmearing
  MuonMomentumSmearing

  TrackMerger

  ECal
  HCal

  Calorimeter
  EFlowMerger

  TreeWriter
}

#################################
# Propagate particles in cylinder
#################################

module ParticlePropagator ParticlePropagator {
  set InputArray Delphes/stableParticles

  set OutputArray stableParticles
  set ChargedHadronOutputArray chargedHadrons
  set ElectronOutputArray electrons
  set MuonOutputArray muons
  set NeutralOutputArray neutrals

  # radius of the magnetic field coverage, in m
  set Radius 1.5
  # half-length of the magnetic field coverage, in m
  set HalfLength 2.5
  # magnetic field
  set Bz 0.435

}

####################################
# Charged hadron tracking efficiency
####################################

module Efficiency ChargedHadronTrackingEfficiency {
  set InputArray ParticlePropagator/chargedHadrons
  set OutputArray chargedHadrons

  # add EfficiencyFormula {efficiency formula as a function of eta and pt}

  # tracking efficiency formula for charged hadrons
  set EfficiencyFormula {
  (pt <= 0.1) * (0.0) +
  (abs(eta) <= 1.5) * (pt > 0.1) * (0.98) +
  (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 0.1) * (0.95) +
  (abs(eta) > 2.5) * (0.0)
}

}

##############################
# Electron tracking efficiency
##############################

module Efficiency ElectronTrackingEfficiency {
  set InputArray ParticlePropagator/electrons
  set OutputArray electrons

  # set EfficiencyFormula {efficiency formula as a function of eta and pt}

  # tracking efficiency formula for electrons
  set EfficiencyFormula {
  (pt <= 0.1) * (0.0) +
  (abs(eta) <= 1.5) * (pt > 0.1) * (0.98) +
  (abs(eta) > 1.5) * (0.0)
}
}

##########################
# Muon tracking efficiency
##########################

module Efficiency MuonTrackingEfficiency {
  set InputArray ParticlePropagator/muons
  set OutputArray muons

  # set EfficiencyFormula {efficiency formula as a function of eta and pt}

  # tracking efficiency formula for muons
  set EfficiencyFormula {
  (pt <= 0.1) * (0.0) +
  (abs(eta) <= 1.5) * (pt > 0.1) * (0.99) +
  (abs(eta) > 1.5) * (0.0)
}
}

########################################
# Momentum resolution for charged tracks
########################################

module MomentumSmearing ChargedHadronMomentumSmearing {
  set InputArray ChargedHadronTrackingEfficiency/chargedHadrons
  set OutputArray chargedHadrons

  # set ResolutionFormula {resolution formula as a function of eta and pt}

  # resolution formula for charged hadrons
  set ResolutionFormula {
  (abs(eta) <= 2.5) * sqrt( (0.0003 * pt)^2 + (0.015)^2 )
}


}

###################################
# Momentum resolution for electrons
###################################

module MomentumSmearing ElectronMomentumSmearing {
  set InputArray ElectronTrackingEfficiency/electrons
  set OutputArray electrons

  # set ResolutionFormula {resolution formula as a function of eta and energy}

   # resolution formula for electrons
  set ResolutionFormula {
  (abs(eta) <= 2.5) * sqrt( (0.00012 * pt)^2 + (0.005)^2 )
}
}

###############################
# Momentum resolution for muons
###############################

module MomentumSmearing MuonMomentumSmearing {
  set InputArray MuonTrackingEfficiency/muons
  set OutputArray muons

  # set ResolutionFormula {resolution formula as a function of eta and pt}

   # resolution formula for muons
  set ResolutionFormula {
  (abs(eta) <= 2.5) * sqrt( (0.00015 * pt)^2 + (0.010)^2 )
}

}

##############
# Track merger
##############

module Merger TrackMerger {
# add InputArray InputArray
  add InputArray ChargedHadronMomentumSmearing/chargedHadrons
  add InputArray ElectronMomentumSmearing/electrons
  add InputArray MuonMomentumSmearing/muons
  set OutputArray tracks
}

#############
#   ECAL
#############

module SimpleCalorimeter ECal {
  set ParticleInputArray ParticlePropagator/stableParticles
  set TrackInputArray TrackMerger/tracks

  set TowerOutputArray ecalTowers
  set EFlowTrackOutputArray eflowTracks
  set EFlowTowerOutputArray eflowPhotons

  set IsEcal true

  set EnergyMin 0.5
  set EnergySignificanceMin 1.0

  set SmearTowerCenter true

  set pi [expr {acos(-1)}]

  # lists of the edges of each tower in eta and phi
  # each list starts with the lower edge of the first tower
  # the list ends with the higher edged of the last tower

  # 1.0 degree towers (3 cm x 3 cm)
  set PhiBins {}
  for {set i -180} {$i <= 180} {incr i} {
    add PhiBins [expr {$i * $pi/180.0}]
  }

  # 0.02 unit in eta up to eta = 3.0
  for {set i -150} {$i <= 150} {incr i} {
    set eta [expr {$i * 0.02}]
    add EtaPhiBins $eta $PhiBins
  }

  # default energy fractions {abs(PDG code)} {fraction of energy deposited in ECAL}

  add EnergyFraction {0} {0.0}
  # energy fractions for e, gamma and pi0
  add EnergyFraction {11} {1.0}
  add EnergyFraction {22} {1.0}
  add EnergyFraction {111} {1.0}
  # energy fractions for muon, neutrinos and neutralinos
  add EnergyFraction {12} {0.0}
  add EnergyFraction {13} {0.0}
  add EnergyFraction {14} {0.0}
  add EnergyFraction {16} {0.0}
  add EnergyFraction {1000022} {0.0}
  add EnergyFraction {1000023} {0.0}
  add EnergyFraction {1000025} {0.0}
  add EnergyFraction {1000035} {0.0}
  add EnergyFraction {1000045} {0.0}
  # energy fractions for K0short and Lambda
  add EnergyFraction {310} {0.3}
  add EnergyFraction {3122} {0.3}

  # set ECalResolutionFormula {resolution formula as a function of eta and energy}

  set ResolutionFormula {
  (abs(eta) <= 2.5) * sqrt( (0.07 / sqrt(energy))^2 + (0.02)^2 )
}

}

#############
#   HCAL
#############

module SimpleCalorimeter HCal {
  set ParticleInputArray ParticlePropagator/stableParticles
  set TrackInputArray ECal/eflowTracks

  set TowerOutputArray hcalTowers
  set EFlowTrackOutputArray eflowTracks
  set EFlowTowerOutputArray eflowNeutralHadrons

  set IsEcal false

  set EnergyMin 1.0
  set EnergySignificanceMin 1.0

  set SmearTowerCenter true

  set pi [expr {acos(-1)}]

  # lists of the edges of each tower in eta and phi
  # each list starts with the lower edge of the first tower
  # the list ends with the higher edged of the last tower


  # 2.0 degree towers (6 cm x 6 cm)
  set PhiBins {}
  for {set i -180} {$i <= 180} {incr i} {
    add PhiBins [expr {$i * $pi/180.0}]
  }

  # 0.04 unit in eta up to eta = 3.0
  for {set i -75} {$i <= 75} {incr i} {
    set eta [expr {$i * 0.04}]
    add EtaPhiBins $eta $PhiBins
  }


  # default energy fractions {abs(PDG code)} {Fecal Fhcal}
  add EnergyFraction {0} {1.0}
  # energy fractions for e, gamma and pi0
  add EnergyFraction {11} {0.0}
  add EnergyFraction {22} {0.0}
  add EnergyFraction {111} {0.0}
  # energy fractions for muon, neutrinos and neutralinos
  add EnergyFraction {12} {0.0}
  add EnergyFraction {13} {0.0}
  add EnergyFraction {14} {0.0}
  add EnergyFraction {16} {0.0}
  add EnergyFraction {1000022} {0.0}
  add EnergyFraction {1000023} {0.0}
  add EnergyFraction {1000025} {0.0}
  add EnergyFraction {1000035} {0.0}
  add EnergyFraction {1000045} {0.0}
  # energy fractions for K0short and Lambda
  add EnergyFraction {310} {0.7}
  add EnergyFraction {3122} {0.7}

  # set HCalResolutionFormula {resolution formula as a function of eta and energy}

  set ResolutionFormula {
  (abs(eta) <= 2.5) * sqrt( (0.6 / sqrt(energy))^2 + (0.06)^2 )
}

}

#################
# Electron filter
#################

module PdgCodeFilter ElectronFilter {
  set InputArray HCal/eflowTracks
  set OutputArray electrons
  set Invert true
  add PdgCode {11}
  add PdgCode {-11}
}

######################
# ChargedHadronFilter
######################

module PdgCodeFilter ChargedHadronFilter {
  set InputArray HCal/eflowTracks
  set OutputArray chargedHadrons

  add PdgCode {11}
  add PdgCode {-11}
  add PdgCode {13}
  add PdgCode {-13}
}



###################################################
# Tower Merger (in case not using e-flow algorithm)
###################################################

module Merger Calorimeter {
# add InputArray InputArray
  add InputArray ECal/ecalTowers
  add InputArray HCal/hcalTowers
  set OutputArray towers
}


#################
# Neutrino Filter
#################

module PdgCodeFilter NeutrinoFilter {

  set InputArray Delphes/stableParticles
  set OutputArray filteredParticles

  set PTMin 0.0

  add PdgCode {12}
  add PdgCode {14}
  add PdgCode {16}
  add PdgCode {-12}
  add PdgCode {-14}
  add PdgCode {-16}

}

###################
# Photon efficiency
###################

module Efficiency PhotonEfficiency {
  set InputArray ECal/eflowPhotons
  set OutputArray photons

  # set EfficiencyFormula {efficiency formula as a function of eta and pt}

  # efficiency formula for photons
  set EfficiencyFormula {
  (energy > 0.5) * (0.95) +
  (energy <= 0.5) * (0.0)
}
}

#####################
# Electron efficiency
#####################

module Efficiency ElectronEfficiency {
  set InputArray ElectronFilter/electrons
  set OutputArray electrons

  # set EfficiencyFormula {efficiency formula as a function of eta and pt}

  # efficiency formula for electrons
  set EfficiencyFormula {
  (energy <= 2.0) * (0.00) +
  (abs(eta) <= 1.5) * (energy > 2.0) * (0.99) +
  (abs(eta) > 1.5 && abs(eta) <= 3.0) * (energy > 2.0) * (0.98) +
  (abs(eta) > 3.0) * (0.0)
}
}


#################
# Muon efficiency
#################

module Efficiency MuonEfficiency {
  set InputArray MuonMomentumSmearing/muons
  set OutputArray muons

  # set EfficiencyFormula {efficiency as a function of eta and pt}
  set EfficiencyFormula {                                      (energy <= 2.0) * (0.00) +
                                           (abs(eta) <= 1.5) * (energy > 2.0)  * (0.99) +
                         (abs(eta) > 1.5 && abs(eta) <= 3.0) * (energy > 2.0)  * (0.99) +
                         (abs(eta) > 3.0)                                      * (0.00)}
}


module Merger EFlowMerger {
# add InputArray InputArray
  add InputArray HCal/eflowTracks
  add InputArray ECal/eflowPhotons
  add InputArray HCal/eflowNeutralHadrons
  set OutputArray eflow
}


##################
# ROOT tree writer
##################

module TreeWriter TreeWriter {

 # All particles before propagation
  add Branch Delphes/allParticles Particle GenParticle
  add Branch Delphes/stableParticles ParticleBeforeProp GenParticle

  # All particles after propagation
  add Branch ParticlePropagator/stableParticles ParticleAfterProp GenParticle

  # Separated outputs from ParticlePropagator
  add Branch ParticlePropagator/chargedHadrons ChargedHadron Track
  add Branch ParticlePropagator/electrons Electron Track
  add Branch ParticlePropagator/muons Muon Track
  add Branch ParticlePropagator/neutrals NeutralParticle Track

  # Charged hadrons after tracking efficiency filter
  add Branch ChargedHadronTrackingEfficiency/chargedHadrons ChargedHadronEfficiency Track

  # Electrons after tracking efficiency filter
  add Branch ElectronTrackingEfficiency/electrons ElectronEfficiency Track

  # Muons after tracking efficiency filter
  add Branch MuonTrackingEfficiency/muons MuonEfficiency Track

  # Charged hadrons after momentum smearing
  add Branch ChargedHadronMomentumSmearing/chargedHadrons ChargedHadronSmeared Track

  # Electrons after momentum smearing
  add Branch ElectronMomentumSmearing/electrons ElectronSmeared Track

  # Muons after momentum smearing
  add Branch MuonMomentumSmearing/muons MuonSmeared Track

  # Merged tracks
  add Branch TrackMerger/tracks Track Track

  # ECAL
  add Branch ECal/ecalTowers ECalTower Tower
  add Branch ECal/eflowTracks ECal_EFlowTrack Track
  add Branch ECal/eflowPhotons EFlowPhoton Tower

  # HCAL
  add Branch HCal/eflowTracks EFlowTrack Track
  add Branch HCal/hcalTowers HCalTower Tower
  add Branch HCal/eflowNeutralHadrons EFlowNeutralHadron Tower

  # Calorimeter merged towers
  add Branch Calorimeter/towers Tower Tower

  # EFlow merged objects
  add Branch EFlowMerger/eflow EFlowObject ParticleFlowCandidate
}
