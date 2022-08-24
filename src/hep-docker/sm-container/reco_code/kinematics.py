from typing import List

import numpy as np

import event_selection


def four_momentum(pt: np.ndarray, phi: np.ndarray, eta: np.ndarray,
                  mass: np.ndarray) -> np.ndarray:
    """Set four momentum from pt, phi, eta and mass.

    :param pt: Transverse momentum.
    :type pt: np.ndarray
    :param phi: Azimuth angle.
    :type phi: np.ndarray
    :param eta: Pseudorapidity.
    :type eta: np.ndarray
    :param mass: Particle's mass.
    :type mass: np.ndarray
    :return: Four momentum in (x, y, z, E) coordinates.
    :rtype: np.ndarray
    """
    pt = np.abs(pt)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2).reshape(-1, 1)
    return np.concatenate([px, py, pz, E], axis=1)


def neutrino_four_momentum(px: float, py: float, eta: float) -> np.ndarray:
    """Generate neutrino's four momentum.

    :param px: momentum's x component.
    :type px: float
    :param py: momentum's y component.
    :type py: float
    :param eta: Pseudorapidity.
    :type eta: float
    :return: Four momentum in (x, y, z, E) coordinates.
    :rtype: np.ndarray
    """
    pt = np.sqrt(px**2 + py**2)
    pz = pt * np.sinh(eta)
    E = np.sqrt(pt ** 2 + pz ** 2)
    return np.array([px, py, pz, E])


def normalize_dPhi(dphi: np.ndarray) -> np.ndarray:
    """Normalize delta phi to values between 0 and pi

    :param dphi: Unnormalized delta phi.
    :type dphi: float
    :return: normalized delta phi.
    :rtype: float
    """
    mask1 = dphi < -np.pi
    mask2 = dphi >= np.pi
    normed_dphi = dphi
    normed_dphi[mask1] += (2 * np.pi)
    normed_dphi[mask2] = (2 * np.pi) - normed_dphi[mask2]
    return np.abs(normed_dphi)


def eta(p: np.ndarray) -> np.ndarray:
    p_norm = np.linalg.norm(p[:, :3], axis=1)
    cos_theta = p[:, 2] / p_norm
    unstable_mask = (cos_theta * cos_theta) >= 1
    np.putmask(cos_theta, unstable_mask, 0)
    eta_result = -0.5 * np.log((1 - cos_theta) / (1 + cos_theta))
    np.putmask(eta_result, unstable_mask, 10e10)
    return eta_result


def phi(p: np.ndarray) -> np.ndarray:
    p_x = p[:, 0]
    p_y = p[:, 1]
    return np.arctan2(p_y, p_x)


def mass(p: np.ndarray) -> np.ndarray:
    x2 = np.sum(p[:, :3] ** 2, axis=1, keepdims=True)
    m2 = p[:, 3:] - x2
    m2 *= ((np.sign(m2) < 0) * -1)
    return np.sqrt(m2)


def dR(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    dEta = eta(p1) - eta(p2)
    dPhi = normalize_dPhi(phi(p1) - phi(p2))
    return np.sqrt((dPhi ** 2) + (dEta ** 2))


def dphi_dilepton(events) -> List[np.ndarray]:
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
                dphi_vals.append(normalize_dPhi(np.array(muon_phi[idx][0] - muon_phi[idx][1])))
            else:
                continue
        elif len(phi) == 1:
            if len(muon_phi[idx]) == 0:
                continue
            if (charge[0] + muon_charge[idx][0]) != 0:
                continue
            dphi_vals.append(normalize_dPhi(np.array(phi[0] - muon_phi[idx][0])))
        elif len(phi) == 2:
            if (charge[0] + charge[1]) != 0:
                continue
            dphi_vals.append(normalize_dPhi(np.array(phi[0] - phi[1])))
    return dphi_vals


def invariant_mass_ttbar(events) -> np.ndarray:
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


def pt_ttbar(events) -> np.ndarray:
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
