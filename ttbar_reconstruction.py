import uproot
import numpy as np

from typing import List, Tuple
from itertools import combinations
from tqdm import tqdm

from processing import event_selection


# m_t = 172.5
# m_w = 80.4
# m_e = 0.000510998902
# m_m = 0.105658389


def four_momentum(pt: float, phi: float, eta: float, mass: float) -> Tuple[float]:
    """Calculate the four-momentum for a particle.

    :param pt: Transverse momentum
    :type pt: float
    :param phi: Phi coordinate
    :type phi: float
    :param eta: Pseudorapidity
    :type eta: float
    :param mass: Particle mass
    :type mass: float
    :return: Particle's four-momentum
    :rtype: Tuple[float]
    """
    pt = np.abs(pt)
    px = pt*np.cos(phi)
    py = pt*np.sin(phi)
    pz = pt*np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return px, py, pz, E


def ttbar_bjets_kinematics(event_bjets_pt: np.ndarray, event_bjets_phi: np.ndarray,
                           event_bjets_eta: np.ndarray, event_bjets_mass: np.ndarray, idx_t: int,
                           idx_tbar: int) -> Tuple[Tuple[float], Tuple[float], float, float]:
    """Calculate four-momentum for two b-jets. One jet is assumed to come from the
    top quark and the other from the anti-top quark.

    :param event_bjets_pt: Transverse momentum of two b-jets
    :type event_bjets_pt: np.ndarray
    :param event_bjets_phi: Phi angles for two b-jets
    :type event_bjets_phi: np.ndarray
    :param event_bjets_eta: Pseudorapidity of two b-jets
    :type event_bjets_eta: np.ndarray
    :param event_bjets_mass: Mass of two b-jets
    :type event_bjets_mass: np.ndarray
    :param idx_t: Index of b-jet assumed to be from the top quark
    :type idx_t: int
    :param idx_tbar: Index of b-jet assumed to be from the anti-top quark
    :type idx_tbar: int
    :return: Four-momenta and masses for the two b-jets assigned to each quark.
    :rtype: Tuple[Tuple[float], Tuple[float], float, float]
    """
    pt_b_t = event_bjets_pt[idx_t]
    pt_b_tbar = event_bjets_pt[idx_tbar]
    phi_b_t = event_bjets_phi[idx_t]
    phi_b_tbar = event_bjets_phi[idx_tbar]
    eta_b_t = event_bjets_eta[idx_t]
    eta_b_tbar = event_bjets_eta[idx_tbar]
    m_b_t = event_bjets_mass[idx_t]
    m_b_tbar = event_bjets_mass[idx_tbar]
    p_b_t = four_momentum(pt_b_t, phi_b_t, eta_b_t, m_b_t)
    p_b_tbar = four_momentum(pt_b_tbar, phi_b_tbar, eta_b_tbar, m_b_tbar)
    return p_b_t, p_b_tbar, m_b_t, m_b_tbar


def ttbar_leptons_kinematics(event_ls_pt: List[float], event_ls_phi: List[float],
                             event_ls_eta: List[float], event_ls_charge: List[float],
                             m_ls: List[float]) -> Tuple[Tuple[float], Tuple[float], float, float]:
    """Calculate four-momentum for two leptons. The leptons are assigned to a
    specific top quark using their charge.

    :param event_ls_pt: Leptons' transverse momenta
    :type event_ls_pt: List[float]
    :param event_ls_phi: Lepton's phi angle
    :type event_ls_phi: List[float]
    :param event_ls_eta: Leptons' Pseudorapidity
    :type event_ls_eta: List[float]
    :param event_ls_charge: Leptons' charge
    :type event_ls_charge: List[float]
    :param m_ls: Leptons' masses
    :type m_ls: List[float]
    :return: Four-momenta and masses for leptons assigned to each quark.
    :rtype: Tuple[Tuple[float], Tuple[float], float, float]
    """
    if event_ls_charge[0] == 1:
        l_idx_t = 0
        l_idx_tbar = 1
    else:
        l_idx_t = 1
        l_idx_tbar = 0
    pt_l_t = event_ls_pt[l_idx_t]
    phi_l_t = event_ls_phi[l_idx_t]
    eta_l_t = event_ls_pt[l_idx_t]
    m_l_t = m_ls[l_idx_t]
    p_l_t = four_momentum(pt_l_t, phi_l_t, eta_l_t, m_l_t)

    pt_l_tbar = event_ls_pt[l_idx_tbar]
    phi_l_tbar = event_ls_phi[l_idx_tbar]
    eta_l_tbar = event_ls_pt[l_idx_tbar]
    m_l_tbar = m_ls[l_idx_tbar]
    p_l_tbar = four_momentum(pt_l_tbar, phi_l_tbar, eta_l_tbar, m_l_tbar)

    return p_l_t, p_l_tbar, m_l_t, m_l_tbar


def calculate_neutrino_py(eta: float, m_b: float, p_b: Tuple[float],
                          m_l: float, p_l: Tuple[float], m_t: float,
                          m_w=80.4) -> Tuple[np.ndarray, float, float]:
    """Calculate possible solutions for the py momentum of a neutrino in the event.
    The solutions are calculated solving a second order polynomial. The b-jet and
    lepton used in the calculation are the ones that result of the same top quark
    decay as the neutrino.

    :param eta: Assummed pseudorapidity for the neutrino.
    :type eta: float
    :param m_b: Mass of the b-jet.
    :type m_b: float
    :param p_b: Four-momentum of the b-jet.
    :type p_b: Tuple[float]
    :param m_l: Lepton's mass.
    :type m_l: float
    :param p_l: Lepton's four-momentum.
    :type p_l: Tuple[float]
    :param m_t: Assumed top quark mass.
    :type m_t: float
    :param m_w: Assumed W boson mass., defaults to 80.4
    :type m_w: float, optional
    :return: Possible solutions for py. eps and kappa are quantities required
             for calculating the neutrino's px.
    :rtype: Tuple[np.ndarray, float, float]
    """
    alpha_1 = (m_t**2 - m_b**2 - m_w**2)
    alpha_2 = (m_w**2 - m_l**2)/2

    beta_b = p_b[3]*np.sinh(eta) - p_b[2]*np.cosh(eta)
    A_b = p_b[0]/beta_b
    B_b = p_b[1]/beta_b
    C_b = alpha_1/beta_b

    beta_l = p_l[3]*np.sinh(eta) - p_l[2]*np.cosh(eta)
    A_l = p_l[0]/beta_l
    B_l = p_l[1]/beta_l
    C_l = alpha_2/beta_l

    kappa = (B_l - B_b)/(A_b - A_l)
    eps = (C_l - C_b)/(A_b - A_l)

    coeff_2 = (kappa**2)*(A_b**2 - 1) + B_b**2 - 1
    coeff_1 = 2*eps*kappa*(A_b**2 - 1) + 2*A_b*C_b*kappa + 2*B_b*C_b
    coeff_0 = (A_b**2 - 1)*eps**2 + 2*eps*A_b*C_b + C_b**2
    if not np.isfinite(coeff_2 + coeff_1 + coeff_0):
        return None, eps, kappa
    roots = np.roots([coeff_2, coeff_1, coeff_0])
    return roots, eps, kappa


def calculate_neutrino_px(neutrino_py: np.ndarray, eps: float, kappa: float) -> np.ndarray:
    """Calculate neutrino's px.""

    :param neutrino_py: Potential solutions for neutrino's py
    :type neutrino_py: np.ndarray
    :param eps: Variable encapsulating lepton and b-jet kinematics used for py solutions.
    :type eps: float
    :param kappa: Variable encapsulating lepton and b-jet kinematics used for py solutions.
    :type kappa: float
    :return: Potential solutions for neutrino's px
    :rtype: np.ndarray
    """
    return kappa*neutrino_py + eps


def solution_weight(met_x: float, met_y: float, neutrino_px: float, neutrino_py: float,
                    met_resolution: float) -> float:
    """Calculate the weight of the solution using potential neutrino's momentum solution
    and observed missing ET.

    :param met_x: x component of Missing ET.
    :type met_x: float
    :param met_y: x component of Missing ET.
    :type met_y: float
    :param neutrino_px: Potential solution of neutrino's px.
    :type neutrino_px: float
    :param neutrino_py: Potential solution of neutrino's py.'
    :type neutrino_py: float
    :param met_resolution: Resolution of MET measurement.
    :type met_resolution: float
    :return: Solution's weights.
    :rtype: float
    """
    weight_x = np.exp(-(met_x - neutrino_px)**2/(2*met_resolution**2))
    weight_y = np.exp(-(met_y - neutrino_py)**2/(2*met_resolution**2))
    return weight_x*weight_y


def total_neutrino_momentum(nu_eta_t: float, m_b_t: float, p_b_t: Tuple[float], m_l_t: float,
                            p_l_t: Tuple[float], nu_eta_tbar: float, m_b_tbar: float,
                            p_b_tbar: Tuple[float], m_l_tbar: float, p_l_tbar: Tuple[float],
                            m_t_val: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate total momentum of the two neutrinos in the x and y components.

    :param nu_eta_t: Pseudorapidity of neutrino assigned to top quark.
    :type nu_eta_t: float
    :param m_b_t: Mass of b-jet assigned to top quark.
    :type m_b_t: float
    :param p_b_t: Four-momentum of b-jet assigned to top quark.
    :type p_b_t: Tuple[float]
    :param m_l_t: Mass of lepton assigned to top quark.
    :type m_l_t: float
    :param p_l_t: Four-momentum of lepton assigned to top quark.
    :type p_l_t: Tuple[float]
    :param nu_eta_tbar: Pseudorapidity of neutrino assigned to anti-top quark.
    :type nu_eta_tbar: float
    :param m_b_tbar: Mass of b-jet assigned to anti-top quark.
    :type m_b_tbar: float
    :param p_b_tbar: Four-momentum of b-jet assigned to anti-top quark.
    :type p_b_tbar: Tuple[float]
    :param m_l_tbar: Mass of lepton assigned to anti-top quark.
    :type m_l_tbar: float
    :param p_l_tbar: Four-momentum of lepton assigned to anti-top quark.
    :type p_l_tbar: Tuple[float]
    :param m_t_val: Assumed mass for top quark.
    :type m_t_val: float
    :return: x and y components of total neutrino momentum in the event.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    nu_t_py, eps, kappa = calculate_neutrino_py(
        nu_eta_t,
        m_b_t,
        p_b_t,
        m_l_t,
        p_l_t,
        m_t_val
    )
    if nu_t_py is None:
        return None, None
    nu_t_px = calculate_neutrino_px(nu_t_py, eps, kappa)

    nu_tbar_py, eps, kappa = calculate_neutrino_py(
        nu_eta_tbar,
        m_b_tbar,
        p_b_tbar,
        m_l_tbar,
        p_l_tbar,
        m_t_val
    )
    if nu_tbar_py is None:
        return None, None
    nu_tbar_px = calculate_neutrino_px(nu_tbar_py, eps, kappa)

    total_nu_px = nu_t_px + nu_tbar_px
    total_nu_py = nu_t_py + nu_tbar_py
    return total_nu_px, total_nu_py


def lepton_kinematics(electron_pt: np.ndarray, electron_phi: np.ndarray, electron_eta: np.ndarray,
                      electron_charge: np.ndarray, muon_pt: np.ndarray, muon_phi: np.ndarray,
                      muon_eta: np.ndarray, muon_charge: np.ndarray
                      ) -> Tuple[Tuple[float], Tuple[float], float, float]:
    """Calculate lepton kinematics according to the types of leptons present in the event.

    :param electron_pt: Transverse momenta of electrons in the event.
    :type electron_pt: np.ndarray
    :param electron_phi: Phi of electrons in the event.
    :type electron_phi: np.ndarray
    :param electron_eta: Pseudorapidities of electrons in the event.
    :type electron_eta: np.ndarray
    :param electron_charge: Charges of electrons in the event.
    :type electron_charge: np.ndarray
    :param muon_pt: Transverse momenta of muons in the event.
    :type muon_pt: np.ndarray
    :param muon_phi: Phi of muons in the event.
    :type muon_phi: np.ndarray
    :param muon_eta: Pseudorapidities of muons in the event.
    :type muon_eta: np.ndarray
    :param muon_charge: Charges of muons in the event.
    :type muon_charge: np.ndarray
    :raises ValueError: The number of leptons in the event is greater than two.
    :return: Four-momenta and masses for leptons assigned to top and anti-top quarks.
    :rtype: Tuple[Tuple[float], Tuple[float], float, float]
    """
    if len(electron_pt) + len(muon_pt) < 2:
        return None, None, None, None
    n_electrons = len(electron_pt)
    n_muons = len(muon_pt)
    if n_electrons == 2:
        if np.sum(electron_charge) != 0:
            return None, None, None, None

        m_ls = [0.000510998902] * 2
        p_l_t, p_l_tbar, m_l_t, m_l_tbar = ttbar_leptons_kinematics(
            electron_pt,
            electron_phi,
            electron_eta,
            electron_charge,
            m_ls
        )
        return p_l_t, p_l_tbar, m_l_t, m_l_tbar

    elif n_muons == 2:
        if np.sum(muon_charge) != 0:
            return None, None, None, None

        m_ls = [0.105658389] * 2
        p_l_t, p_l_tbar, m_l_t, m_l_tbar = ttbar_leptons_kinematics(
            muon_pt,
            muon_phi,
            muon_eta,
            muon_charge,
            m_ls
        )
        return p_l_t, p_l_tbar, m_l_t, m_l_tbar

    elif (n_electrons == 1) and (n_muons == 1):
        if (electron_charge[0] + muon_charge[0]) != 0:
            return None, None, None, None

        m_ls = [0.000510998902, 0.105658389]
        event_ls_pt = [electron_pt[0], muon_pt[0]]
        event_ls_phi = [electron_phi[0], muon_phi[0]]
        event_ls_eta = [electron_eta[0], muon_eta[0]]
        event_ls_charge = [electron_charge[0], muon_charge[0]]
        p_l_t, p_l_tbar, m_l_t, m_l_tbar = ttbar_leptons_kinematics(
            event_ls_pt,
            event_ls_phi,
            event_ls_eta,
            event_ls_charge,
            m_ls
        )
        return p_l_t, p_l_tbar, m_l_t, m_l_tbar

    else:
        raise ValueError(
            "Event does not have a valid combination of leptons: "
            f"{n_electrons} electrons and {n_muons} muons in the event."
        )


if __name__ == "__main__":
    sm_path = "./mg5_data/SM-process_spin-ON/Events/run_01_decayed_1/tag_1_delphes_events.root"
    sm_events = uproot.open(sm_path)["Delphes"]

    # Apply ATLAS selection criteria
    electron_mask = event_selection.select_electron(sm_events)
    muon_mask = event_selection.select_muon(sm_events)
    jets_mask = event_selection.select_jet(sm_events)

    # Get mask for b-jets
    bjets_mask = sm_events["Jet.BTag"].array()[jets_mask].astype(bool)

    # Select b-jets that pass selection criteria from Jet TTree
    bjets_mass = sm_events["Jet.Mass"].array()[jets_mask][bjets_mask]
    bjets_pt = sm_events["Jet.PT"].array()[jets_mask][bjets_mask]
    bjets_phi = sm_events["Jet.Phi"].array()[jets_mask][bjets_mask]
    bjets_eta = sm_events["Jet.Eta"].array()[jets_mask][bjets_mask]

    # Select electrons that pass selection criteria
    electron_pt = sm_events["Electron.PT"].array()[electron_mask]
    electron_phi = sm_events["Electron.Phi"].array()[electron_mask]
    electron_eta = sm_events["Electron.Eta"].array()[electron_mask]
    electron_charge = sm_events["Electron.Charge"].array()[electron_mask]

    # Select muons that pass selection criteria
    muon_pt = sm_events["Muon.PT"].array()[muon_mask]
    muon_phi = sm_events["Muon.Phi"].array()[muon_mask]
    muon_eta = sm_events["Muon.Eta"].array()[muon_mask]
    muon_charge = sm_events["Muon.Charge"].array()[muon_mask]

    best_weights = []
    for idx in tqdm(range(len(bjets_mass))):
        best_weight = -1
        for m_t_val in np.linspace(171, 174, 7):
            p_l_t, p_l_tbar, m_l_t, m_l_tbar = lepton_kinematics(
                electron_pt[idx], electron_phi[idx], electron_eta[idx], electron_charge[idx],
                muon_pt[idx], muon_phi[idx], muon_eta[idx], muon_charge[idx]
            )
            if p_l_t is None:
                continue

            if len(bjets_mass[idx]) < 2:
                continue
            bjets_combinations = list(combinations(range(len(bjets_mass[idx])), 2))
            for idx_t, idx_tbar in bjets_combinations:
                smeared_jets_pt = np.random.normal(
                    bjets_pt[idx],
                    bjets_pt[idx]*0.08,
                    (5, len(bjets_pt[idx]))
                )
                for bjets_pt_idx in smeared_jets_pt:
                    p_b_t, p_b_tbar, m_b_t, m_b_tbar = ttbar_bjets_kinematics(
                        bjets_pt_idx,
                        bjets_phi[idx],
                        bjets_eta[idx],
                        bjets_mass[idx],
                        idx_t,
                        idx_tbar
                    )

                    met = sm_events["MissingET.MET"].array()[idx]
                    met_phi = sm_events["MissingET.Phi"].array()[idx]
                    met_resolution = 20 + met/20
                    met_x = (met * np.cos(met_phi))[0]
                    met_y = (met * np.sin(met_phi))[0]

                    eta_range = np.linspace(-5, 5, 51)
                    eta_grid = np.array(np.meshgrid(eta_range, eta_range)).T.reshape(-1, 2)
                    for nu_eta_t, nu_eta_tbar in eta_grid:
                        total_nu_px, total_nu_py = total_neutrino_momentum(
                            nu_eta_t, m_b_t, p_b_t, m_l_t, p_l_t,
                            nu_eta_tbar, m_b_tbar, p_b_tbar, m_l_tbar,  p_l_tbar, m_t_val
                        )
                        if total_nu_px is None:
                            continue

                        for nu_px, nu_py in zip(total_nu_px, total_nu_py):
                            if np.iscomplex(nu_px) or np.iscomplex(nu_py):
                                continue
                            weight = solution_weight(met_x, met_y, nu_px, nu_py, met_resolution)
                            if weight > best_weight:
                                best_weight = weight
        best_weights.append(best_weight)
