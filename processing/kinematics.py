import numpy as np
from typing import List
from processing import event_selection


def normalize_dPhi(dphi: float) -> float:
    """Normalize delta phi to values between 0 and pi

    :param dphi: Unnormalized delta phi.
    :type dphi: float
    :return: normalized delta phi.
    :rtype: float
    """
    if dphi < -np.pi:
        return dphi + 2*np.pi
    if dphi > np.pi:
        return 2*np.pi - dphi
    return abs(dphi)


def dphi_dilepton(events) -> List[float]:
    """Calculate delta phi between two leptons in the event. We require
    that the two leptons have opposite charge. The function assumes that
    there is maximum one muon per event.

    :param events: Delphes TTree
    :type events: TTree
    :return: Delta phi between the two leptons in the event.
    :rtype: List[float]
    """
    muon_mask = event_selection.select_muon(events)
    elec_mask = event_selection.select_electron(events)
    muon_phi = events["Muon.Phi"].array()[muon_mask]
    elec_phi = events["Electron.Phi"].array()[elec_mask]
    muon_charge = events["Muon.Charge"].array()[muon_mask]
    elec_charge = events["Electron.Charge"].array()[elec_mask]

    dphi_vals = []
    for idx, (phi, charge) in enumerate(zip(elec_phi, elec_charge)):
        if len(phi) == 0:
            if len(muon_charge[idx]) == 2:
                dphi_vals.append(normalize_dPhi(muon_phi[idx][0] - muon_phi[idx][1]))
            else:
                continue
        elif len(phi) == 1:
            if len(muon_phi[idx]) == 0:
                continue
            if (charge[0] + muon_charge[idx][0]) != 0:
                continue
            dphi_vals.append(normalize_dPhi(phi[0] - muon_phi[idx][0]))
        elif len(phi) == 2:
            if (charge[0] + charge[1]) != 0:
                continue
            dphi_vals.append(normalize_dPhi(phi[0] - phi[1]))
    return dphi_vals


def invariant_mass_ttbar(events) -> np.array:
    """Calculate invariant mass for ttbar system

    :param events: Delphes TTree
    :type events: TTree
    :return: Mass values for ttbar system
    :rtype: np.array
    """
    status_mask = events["Particle.Status"].array() == 22
    t_mask = (events["Particle.PID"].array() == 6) * status_mask
    tbar_mask = (events["Particle.PID"].array() == -6) * status_mask
    mass = events["Particle.Mass"].array()[t_mask] + events["Particle.Mass"].array()[tbar_mask]
    mass_vals = mass.flatten()
    return mass_vals


def pt_ttbar(events) -> np.array:
    """Calculate total pT for ttbar system

    :param events: Delphes TTree
    :type events: TTree
    :return: pT values for ttbar system
    :rtype: np.array
    """
    status_mask = events["Particle.Status"].array() == 22
    t_mask = (events["Particle.PID"].array() == 6) * status_mask
    tbar_mask = (events["Particle.PID"].array() == -6) * status_mask
    pT = events["Particle.PT"].array()[t_mask] + events["Particle.PT"].array()[tbar_mask]
    pT_vals = pT.flatten()
    return pT_vals
