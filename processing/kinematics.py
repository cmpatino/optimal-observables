import numpy as np
from typing import List


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
    muon_phi = events["Muon.Phi"].array()
    elec_phi = events["Electron.Phi"].array()
    muon_charge = events["Muon.Charge"].array()
    elec_charge = events["Electron.Charge"].array()

    dphi_vals = []
    for idx, (phi, charge) in enumerate(zip(elec_phi, elec_charge)):
        if len(phi) == 0:
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
