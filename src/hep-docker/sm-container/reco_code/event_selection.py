from typing import Literal

import awkward as ak
import numpy as np
from tqdm import tqdm

import kinematics


def select_jet(events) -> ak.Array:
    """Create boolean mask to apply jet selection criteria from ATLAS

    :param events: Delphes event TTree containing
    :type events: TTree
    :return: boolean mask to select jets in events
    :rtype: Array
    """

    pt_mask = get_jets_pt_mask(events=events)
    eta_mask = get_jets_eta_mask(events=events)
    electron_dR_mask = get_jet_to_lepton_mask(
        events=events, lepton="Electron", min_dR=0.2
    )
    muon_dR_mask = get_jet_to_lepton_mask(events=events, lepton="Muon", min_dR=0.4)
    mask = pt_mask * eta_mask * electron_dR_mask * muon_dR_mask
    return mask


def get_jets_pt_mask(events) -> ak.Array:
    jet_pt = events["Jet.PT"].array()
    pt_mask = jet_pt > 25
    return pt_mask


def get_jets_eta_mask(events) -> ak.Array:
    jet_eta = events["Jet.Eta"].array()
    eta_mask = np.abs(jet_eta) < 2.5
    return eta_mask


def get_jet_to_lepton_mask(
    events, lepton: Literal["Electron", "Muon"], min_dR: float
) -> ak.Array:
    jet_phi = events["Jet.Phi"].array()
    jet_eta = events["Jet.Eta"].array()
    lepton_phi = events[f"{lepton}.Phi"].array()
    lepton_eta = events[f"{lepton}.Eta"].array()

    lepton_dR_mask = []
    for event_idx in tqdm(range(len(lepton_phi)), desc=f"Jet-{lepton} Separation"):
        jet_phi_idx = np.array(jet_phi[event_idx])
        jet_eta_idx = np.array(jet_eta[event_idx])
        lepton_dR_event_mask = np.ones_like(jet_phi_idx, dtype=int)
        for lep_idx in range(len(lepton_phi[event_idx])):
            dPhi = kinematics.normalize_dPhi(
                jet_phi_idx - lepton_phi[event_idx][lep_idx]
            )
            dEta = jet_eta_idx - lepton_eta[event_idx][lep_idx]
            dR = np.sqrt(dPhi**2 + dEta**2)
            lepton_dR_event_mask *= dR > min_dR
        lepton_dR_mask.append(lepton_dR_event_mask.astype(bool))
    lepton_dR_mask = ak.from_iter(lepton_dR_mask)
    return lepton_dR_mask


def select_electron(events) -> ak.Array:
    """Create boolean mask to apply electron selection criteria from ATLAS

    :param events: Delphes event TTree containing
    :type events: TTree
    :return: boolean mask to select electrons in events
    :rtype: Array
    """

    pt_mask = get_electron_pt_mask(events=events)
    eta_mask = get_electron_eta_mask(events=events)
    jet_dR_mask = get_electron_to_jet_mask(events=events)
    mask = pt_mask * eta_mask * jet_dR_mask
    return mask


def get_electron_pt_mask(events) -> ak.Array:
    electron_pt = events["Electron.PT"].array()
    pt_mask = electron_pt > 25
    return pt_mask


def get_electron_eta_mask(events) -> ak.Array:
    electron_eta = events["Electron.Eta"].array()
    eta_mask1 = (np.abs(electron_eta) < 2.5) * (np.abs(electron_eta) > 1.52)
    eta_mask2 = np.abs(electron_eta) < 1.37
    eta_mask = eta_mask1 + eta_mask2
    return eta_mask


def get_electron_to_jet_mask(events) -> ak.Array:
    jet_phi = events["Jet.Phi"].array()
    jet_eta = events["Jet.Eta"].array()
    jet_mass = events["Jet.Mass"].array()
    jet_pt = events["Jet.PT"].array()
    electron_phi = events["Electron.Phi"].array()
    electron_eta = events["Electron.Eta"].array()

    jet_dR_mask = []
    for event_idx in tqdm(range(len(electron_phi)), desc="Electron-Jet Separation"):
        electron_phi_idx = np.array(electron_phi[event_idx])
        electron_eta_idx = np.array(electron_eta[event_idx])
        jet_dR_event_mask = np.ones_like(electron_phi_idx, dtype=int)
        for jet_idx in range(len(jet_phi[event_idx])):
            jet_eta_idx = jet_eta[event_idx][jet_idx]
            jet_mass_idx = jet_mass[event_idx][jet_idx]
            jet_pt_idx = jet_pt[event_idx][jet_idx]
            jet_rapidity = (
                jet_eta_idx
                - (np.tanh(jet_eta_idx) / 2) * (jet_mass_idx / jet_pt_idx) ** 2
            )
            dPhi = kinematics.normalize_dPhi(
                electron_phi_idx - jet_phi[event_idx][jet_idx]
            )
            dEta = electron_eta_idx - jet_rapidity
            dR = np.sqrt(dPhi**2 + dEta**2)
            jet_dR_event_mask *= dR > 0.4
        jet_dR_mask.append(jet_dR_event_mask.astype(bool))
    jet_dR_mask = ak.from_iter(jet_dR_mask)
    return jet_dR_mask


def select_muon(events) -> ak.Array:
    """Create boolean mask to apply muon selection criteria from ATLAS

    :param events: Delphes event TTree containing
    :type events: TTree
    :return: boolean mask to select muon in events
    :rtype: Array
    """
    pt_mask = get_muon_pt_mask(events=events)
    eta_mask = get_muon_eta_mask(events=events)
    jet_dR_mask = get_muon_to_jet_mask(events=events)
    mask = pt_mask * eta_mask * jet_dR_mask
    return mask


def get_muon_pt_mask(events) -> ak.Array:
    muon_pt = events["Muon.PT"].array()
    pt_mask = muon_pt > 25
    return pt_mask


def get_muon_eta_mask(events) -> ak.Array:
    muon_eta = events["Muon.Eta"].array()
    eta_mask = np.abs(muon_eta) < 2.5
    return eta_mask


def get_muon_to_jet_mask(events) -> ak.Array:
    jet_phi = events["Jet.Phi"].array()
    jet_eta = events["Jet.Eta"].array()
    muon_phi = events["Muon.Phi"].array()
    muon_eta = events["Muon.Eta"].array()

    jet_dR_mask = []
    for event_idx in tqdm(range(len(muon_phi)), desc="Muon-Jet Separation"):
        muon_phi_idx = np.array(muon_phi[event_idx])
        muon_eta_idx = np.array(muon_eta[event_idx])
        jet_dR_event_mask = np.ones_like(muon_phi_idx, dtype=int)
        for jet_idx in range(len(jet_phi[event_idx])):
            dPhi = kinematics.normalize_dPhi(muon_phi_idx - jet_phi[event_idx][jet_idx])
            dEta = muon_eta_idx - jet_eta[event_idx][jet_idx]
            dR = np.sqrt(dPhi**2 + dEta**2)
            jet_dR_event_mask *= dR > 0.4
        jet_dR_mask.append(jet_dR_event_mask.astype(bool))
    jet_dR_mask = ak.from_iter(jet_dR_mask)
    return jet_dR_mask
