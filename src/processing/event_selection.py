import numpy as np
from awkward.array.jagged import JaggedArray

import processing.kinematics as kinematics


def select_jet(events) -> JaggedArray:
    """Create boolean mask to apply jet selection criteria from ATLAS

    :param events: Delphes event TTree containing
    :type events: TTree
    :return: boolean mask to select jets in events
    :rtype: JaggedArray
    """
    jet_pt = events["Jet.PT"].array()
    jet_phi = events["Jet.Phi"].array()
    jet_eta = events["Jet.Eta"].array()
    electron_phi = events["Electron.Phi"].array()
    electron_eta = events["Electron.Eta"].array()
    muon_phi = events["Muon.Phi"].array()
    muon_eta = events["Muon.Eta"].array()

    pt_mask = jet_pt > 25
    eta_mask = np.abs(jet_eta) < 2.5
    electron_dR_mask = []
    muon_dR_mask = []
    for event_idx in range(len(events)):
        jet_phi_idx = jet_phi[event_idx]
        jet_eta_idx = jet_eta[event_idx]
        electron_dR_event_mask = np.ones_like(jet_phi_idx, dtype=int)
        muon_dR_event_mask = np.ones_like(jet_phi_idx, dtype=int)
        for elec_idx in range(len(electron_phi[event_idx])):
            dPhi = kinematics.normalize_dPhi(jet_phi_idx - electron_phi[event_idx][elec_idx])
            dEta = jet_eta_idx - electron_eta[event_idx][elec_idx]
            dR = np.sqrt(dPhi**2 + dEta**2)
            electron_dR_event_mask *= (dR > 0.2)
        for muon_idx in range(len(muon_phi[event_idx])):
            dPhi = kinematics.normalize_dPhi(jet_phi_idx - muon_phi[event_idx][muon_idx])
            dEta = jet_eta_idx - muon_eta[event_idx][muon_idx]
            dR = np.sqrt(dPhi**2 + dEta**2)
            muon_dR_event_mask *= (dR > 0.4)
        electron_dR_mask.append(electron_dR_event_mask.astype(bool))
        muon_dR_mask.append(muon_dR_event_mask.astype(bool))
    electron_dR_mask = JaggedArray.fromiter(electron_dR_mask)
    muon_dR_mask = JaggedArray.fromiter(muon_dR_mask)
    mask = pt_mask * eta_mask * electron_dR_mask * muon_dR_mask
    return mask


def select_electron(events) -> JaggedArray:
    """Create boolean mask to apply electron selection criteria from ATLAS

    :param events: Delphes event TTree containing
    :type events: TTree
    :return: boolean mask to select electrons in events
    :rtype: JaggedArray
    """
    jet_phi = events["Jet.Phi"].array()
    jet_eta = events["Jet.Eta"].array()
    jet_mass = events["Jet.Mass"].array()
    jet_pt = events["Jet.PT"].array()
    electron_pt = events["Electron.PT"].array()
    electron_phi = events["Electron.Phi"].array()
    electron_eta = events["Electron.Eta"].array()

    pt_mask = electron_pt > 25
    eta_mask1 = (np.abs(electron_eta) < 2.5) * (np.abs(electron_eta) > 1.52)
    eta_mask2 = np.abs(electron_eta) < 1.37
    eta_mask = eta_mask1 + eta_mask2
    jet_dR_mask = []
    for event_idx in range(len(events)):
        electron_phi_idx = electron_phi[event_idx]
        electron_eta_idx = electron_eta[event_idx]
        jet_dR_event_mask = np.ones_like(electron_phi_idx, dtype=int)
        for jet_idx in range(len(jet_phi[event_idx])):
            jet_eta_idx = jet_eta[event_idx][jet_idx]
            jet_mass_idx = jet_mass[event_idx][jet_idx]
            jet_pt_idx = jet_pt[event_idx][jet_idx]
            jet_rapidity = jet_eta_idx - (np.tanh(jet_eta_idx)/2)*(jet_mass_idx/jet_pt_idx)**2
            dPhi = kinematics.normalize_dPhi(electron_phi_idx - jet_phi[event_idx][jet_idx])
            dEta = electron_eta_idx - jet_rapidity
            dR = np.sqrt(dPhi**2 + dEta**2)
            jet_dR_event_mask *= (dR > 0.4)
        jet_dR_mask.append(jet_dR_event_mask.astype(bool))
    jet_dR_mask = JaggedArray.fromiter(jet_dR_mask)
    mask = pt_mask * eta_mask * jet_dR_mask
    return mask


def select_muon(events) -> JaggedArray:
    """Create boolean mask to apply muon selection criteria from ATLAS

    :param events: Delphes event TTree containing
    :type events: TTree
    :return: boolean mask to select muon in events
    :rtype: JaggedArray
    """
    jet_phi = events["Jet.Phi"].array()
    jet_eta = events["Jet.Eta"].array()
    muon_pt = events["Muon.PT"].array()
    muon_phi = events["Muon.Phi"].array()
    muon_eta = events["Muon.Eta"].array()

    pt_mask = muon_pt > 25
    eta_mask = np.abs(muon_eta) < 2.5
    jet_dR_mask = []
    for event_idx in range(len(events)):
        muon_phi_idx = muon_phi[event_idx]
        muon_eta_idx = muon_eta[event_idx]
        jet_dR_event_mask = np.ones_like(muon_phi_idx, dtype=int)
        for jet_idx in range(len(jet_phi[event_idx])):
            dPhi = kinematics.normalize_dPhi(muon_phi_idx - jet_phi[event_idx][jet_idx])
            dEta = muon_eta_idx - jet_eta[event_idx][jet_idx]
            dR = np.sqrt(dPhi**2 + dEta**2)
            jet_dR_event_mask *= (dR > 0.4)
        jet_dR_mask.append(jet_dR_event_mask.astype(bool))
    jet_dR_mask = JaggedArray.fromiter(jet_dR_mask)
    mask = pt_mask * eta_mask * jet_dR_mask
    return mask
