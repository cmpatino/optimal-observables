import uproot
import numpy as np
import jax.numpy as jnp

from jax import vmap
from typing import List, Tuple
from itertools import permutations

from processing import event_selection


# m_t = 172.5
# m_w = 80.4
# m_e = 0.000510998902
# m_m = 0.105658389


def four_momentum(pt: np.ndarray, phi: np.ndarray, eta: np.ndarray,
                  mass: np.ndarray) -> np.ndarray:
    pt = np.abs(pt)
    px = pt*np.cos(phi)
    py = pt*np.sin(phi)
    pz = pt*np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2).reshape(-1, 1)
    return np.concatenate([px, py, pz, E], axis=1)


def neutrino_four_momentum(px, py, eta):
    pt = np.sqrt(px**2 + py**2)
    pz = pt * np.sinh(eta)
    E = pt * np.cosh(eta)
    return np.array([px, py, pz, E])


def ttbar_bjets_kinematics(smeared_bjets_pt, bjets_phi, bjets_eta,
                           bjets_mass, bjets_combinations_idxs):
    n_smears = smeared_bjets_pt.shape[0]
    pt_combinations = smeared_bjets_pt[:, bjets_combinations_idxs].reshape(-1, 2)
    phi_combinations = np.tile(bjets_phi[bjets_combinations_idxs], (n_smears, 1))
    eta_combinations = np.tile(bjets_eta[bjets_combinations_idxs], (n_smears, 1))
    mass_combinations = np.tile(bjets_mass[bjets_combinations_idxs], (n_smears, 1))
    p_b_t = four_momentum(
        pt_combinations[:, 0:1],
        phi_combinations[:, 0:1],
        eta_combinations[:, 0:1],
        mass_combinations[:, 0:1]
    )
    p_b_tbar = four_momentum(
        pt_combinations[:, 1:],
        phi_combinations[:, 1:],
        eta_combinations[:, 1:],
        mass_combinations[:, 1:]
    )
    return p_b_t, p_b_tbar, mass_combinations[:, 0:1], mass_combinations[:, 1:]


def ttbar_leptons_kinematics(event_ls_pt: List[float], event_ls_phi: List[float],
                             event_ls_eta: List[float], event_ls_charge: List[float],
                             m_ls: List[float]) -> Tuple[Tuple[float], Tuple[float], float, float]:
    if event_ls_charge[0] == 1:
        l_idx_t = 0
        l_idx_tbar = 1
    else:
        l_idx_t = 1
        l_idx_tbar = 0
    pt_l_t = np.array(event_ls_pt[l_idx_t]).reshape(-1, 1)
    phi_l_t = np.array(event_ls_phi[l_idx_t]).reshape(-1, 1)
    eta_l_t = np.array(event_ls_eta[l_idx_t]).reshape(-1, 1)
    m_l_t = np.array(m_ls[l_idx_t]).reshape(-1, 1)
    p_l_t = four_momentum(pt_l_t, phi_l_t, eta_l_t, m_l_t)

    pt_l_tbar = np.array(event_ls_pt[l_idx_tbar]).reshape(-1, 1)
    phi_l_tbar = np.array(event_ls_phi[l_idx_tbar]).reshape(-1, 1)
    eta_l_tbar = np.array(event_ls_eta[l_idx_tbar]).reshape(-1, 1)
    m_l_tbar = np.array(m_ls[l_idx_tbar]).reshape(-1, 1)
    p_l_tbar = four_momentum(pt_l_tbar, phi_l_tbar, eta_l_tbar, m_l_tbar)

    return p_l_t, p_l_tbar, m_l_t, m_l_tbar


def find_roots(coeffs):
    return jnp.roots(coeffs, strip_zeros=False)


def calculate_neutrino_py(eta: float, m_b: float, p_b: Tuple[float],
                          m_l: float, p_l: Tuple[float], m_t: float,
                          m_w=80.4) -> Tuple[np.ndarray, float, float]:

    alpha_1 = (m_t**2 - m_b**2 - m_w**2)/2
    alpha_2 = (m_w**2 - m_l**2)/2

    beta_b = p_b[:, 3:]*np.sinh(eta) - p_b[:, 2:3]*np.cosh(eta)
    A_b = p_b[:, 0:1]/beta_b
    B_b = p_b[:, 1:2]/beta_b
    C_b = alpha_1/beta_b

    beta_l = p_l[:, 3:]*np.sinh(eta) - p_l[:, 2:3]*np.cosh(eta)
    A_l = p_l[:, 0:1]/beta_l
    B_l = p_l[:, 1:2]/beta_l
    C_l = alpha_2/beta_l

    kappa = (B_l - B_b)/(A_b - A_l)
    eps = (C_l - C_b)/(A_b - A_l)

    coeff_2 = (kappa**2)*(A_b**2 - 1) + B_b**2 - 1
    coeff_1 = 2*eps*kappa*(A_b**2 - 1) + 2*A_b*C_b*kappa + 2*B_b*C_b
    coeff_0 = (A_b**2 - 1)*eps**2 + 2*eps*A_b*C_b + C_b**2

    coeffs = np.concatenate([coeff_2, coeff_1, coeff_0], axis=1)
    jcoeffs = jnp.array(coeffs)
    jsols = vmap(find_roots)(jcoeffs)
    sols = np.array(jsols)

    nu_py = np.concatenate([sols[:, 0:1], sols[:, 1:]], axis=0)
    eps = np.tile(eps, (2, 1))
    kappa = np.tile(kappa, (2, 1))
    return nu_py, eps, kappa


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


def total_neutrino_momentum(nu_eta_t, m_b_t, p_b_t, m_l_t,
                            p_l_t, nu_eta_tbar, m_b_tbar,
                            p_b_tbar, m_l_tbar, p_l_tbar,
                            m_t_val) -> Tuple[np.ndarray, np.ndarray]:
    nu_t_py, eps, kappa = calculate_neutrino_py(
        nu_eta_t,
        m_b_t,
        p_b_t,
        m_l_t,
        p_l_t,
        m_t_val
    )
    nu_t_px = calculate_neutrino_px(nu_t_py, eps, kappa)

    nu_tbar_py, eps, kappa = calculate_neutrino_py(
        nu_eta_tbar,
        m_b_tbar,
        p_b_tbar,
        m_l_tbar,
        p_l_tbar,
        m_t_val
    )
    nu_tbar_px = calculate_neutrino_px(nu_tbar_py, eps, kappa)

    return nu_t_px, nu_t_py, nu_tbar_px, nu_tbar_py


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


def reconstruct_event(bjets_mass, bjets_pt, bjets_phi, bjets_eta,
                      electron_pt, electron_phi, electron_eta, electron_charge,
                      muon_pt, muon_phi, muon_eta, muon_charge,
                      met, met_phi, idx):
    if (idx % 100) == 0:
        print(f"Event {idx}")

    p_l_t, p_l_tbar, m_l_t, m_l_tbar = lepton_kinematics(
        electron_pt, electron_phi, electron_eta, electron_charge,
        muon_pt, muon_phi, muon_eta, muon_charge
    )
    if p_l_t is None:
        return None

    if len(bjets_mass) < 2:
        return None

    bjets_combinations_idxs = np.array(list(permutations(range(len(bjets_mass)), 2)))
    smeared_bjets_pt = np.random.normal(
        bjets_pt,
        bjets_pt * 0.14,
        (5, len(bjets_pt))
    )
    p_b_t, p_b_tbar, m_b_t, m_b_tbar = ttbar_bjets_kinematics(
        smeared_bjets_pt,
        bjets_phi,
        bjets_eta,
        bjets_mass,
        bjets_combinations_idxs
    )

    met_resolution = 20 + met / 20
    met_x = (met * np.cos(met_phi))[0]
    met_y = (met * np.sin(met_phi))[0]

    # Vectorize Eta grid for loop
    eta_range = np.linspace(-5, 5, 51)
    eta_grid = np.array(np.meshgrid(eta_range, eta_range)).T.reshape(-1, 2)

    eta_vectorized_mask = [i for i in range(eta_grid.shape[0])
                           for j in range(p_b_t.shape[0])]
    nu_etas = eta_grid[eta_vectorized_mask]

    p_l_t = np.tile(p_l_t, (eta_grid.shape[0] * p_b_t.shape[0], 1))
    p_l_tbar = np.tile(p_l_tbar, (eta_grid.shape[0] * p_b_t.shape[0], 1))
    m_l_t = np.tile(m_l_t, (eta_grid.shape[0] * p_b_t.shape[0], 1))
    m_l_tbar = np.tile(m_l_tbar, (eta_grid.shape[0] * p_b_t.shape[0], 1))

    p_b_t = np.tile(p_b_t, (eta_grid.shape[0], 1))
    p_b_tbar = np.tile(p_b_tbar, (eta_grid.shape[0], 1))
    m_b_t = np.tile(m_b_t, (eta_grid.shape[0], 1))
    m_b_tbar = np.tile(m_b_tbar, (eta_grid.shape[0], 1))

    # Vectorize top mass for loop
    m_t_search = np.linspace(171, 174, 7).reshape(-1, 1)
    mass_vectorized_mask = [i for i in range(m_t_search.shape[0])
                            for j in range(p_b_t.shape[0])]
    m_t_val = m_t_search[mass_vectorized_mask]

    p_l_t = np.tile(p_l_t, (m_t_search.shape[0], 1))
    p_l_tbar = np.tile(p_l_tbar, (m_t_search.shape[0], 1))
    m_l_t = np.tile(m_l_t, (m_t_search.shape[0], 1))
    m_l_tbar = np.tile(m_l_tbar, (m_t_search.shape[0], 1))

    p_b_t = np.tile(p_b_t, (m_t_search.shape[0], 1))
    p_b_tbar = np.tile(p_b_tbar, (m_t_search.shape[0], 1))
    m_b_t = np.tile(m_b_t, (m_t_search.shape[0], 1))
    m_b_tbar = np.tile(m_b_tbar, (m_t_search.shape[0], 1))

    nu_etas = np.tile(nu_etas, (m_t_search.shape[0], 1))

    nu_eta_t = nu_etas[:, 0:1]
    nu_eta_tbar = nu_etas[:, 1:]

    nu_t_px, nu_t_py, nu_tbar_px, nu_tbar_py = total_neutrino_momentum(
        nu_eta_t, m_b_t, p_b_t, m_l_t, p_l_t,
        nu_eta_tbar, m_b_tbar, p_b_tbar, m_l_tbar, p_l_tbar, m_t_val
    )
    total_nu_px = nu_t_px + nu_tbar_px
    total_nu_py = nu_t_py + nu_tbar_py

    real_mask = np.isreal(total_nu_px) * np.isreal(total_nu_py)
    real_mask_momentum = np.tile(real_mask, (1, 4))

    p_b_t = np.tile(p_b_t, (2, 1))[real_mask_momentum].reshape(-1, 4)
    p_l_t = np.tile(p_l_t, (2, 1))[real_mask_momentum].reshape(-1, 4)
    nu_eta_t = np.tile(nu_eta_t, (2, 1))[real_mask]
    nu_t_px = nu_t_px[real_mask]
    nu_t_py = nu_t_py[real_mask]

    p_b_tbar = np.tile(p_b_tbar, (2, 1))[real_mask_momentum].reshape(-1, 4)
    p_l_tbar = np.tile(p_l_tbar, (2, 1))[real_mask_momentum].reshape(-1, 4)
    nu_eta_tbar = np.tile(nu_eta_tbar, (2, 1))[real_mask]
    nu_tbar_px = nu_tbar_px[real_mask]
    nu_tbar_py = nu_tbar_py[real_mask]

    total_nu_px = total_nu_px[real_mask]
    total_nu_py = total_nu_py[real_mask]

    weights = solution_weight(met_x, met_y, total_nu_px, total_nu_py, met_resolution)
    if len(weights) == 0:
        return None
    best_weight_idx = np.argmax(weights)

    best_weight = weights[best_weight_idx]
    best_b_t = p_b_t[best_weight_idx]
    best_l_t = p_l_t[best_weight_idx]
    best_nu_t = neutrino_four_momentum(
        np.real(nu_t_px[best_weight_idx]),
        np.real(nu_t_py[best_weight_idx]),
        nu_eta_t[best_weight_idx]
    )
    best_b_tbar = p_b_tbar[best_weight_idx]
    best_l_tbar = p_l_tbar[best_weight_idx]
    best_nu_tbar = neutrino_four_momentum(
        np.real(nu_tbar_px[best_weight_idx]),
        np.real(nu_tbar_py[best_weight_idx]),
        nu_eta_tbar[best_weight_idx]
    )
    print(f"Best weight: {best_weight}")
    p_top = best_b_t + best_l_t + best_nu_t
    p_tbar = best_b_tbar + best_l_tbar + best_nu_tbar
    return p_top, best_l_t, p_tbar, best_l_tbar


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

    # MET for all events
    met = sm_events["MissingET.MET"].array()
    met_phi = sm_events["MissingET.Phi"].array()

    reconstructed_event = [
        reconstruct_event(
            bjets_mass[idx], bjets_pt[idx], bjets_phi[idx], bjets_eta[idx],
            electron_pt[idx], electron_phi[idx], electron_eta[idx], electron_charge[idx],
            muon_pt[idx], muon_phi[idx], muon_eta[idx], muon_charge[idx],
            met[idx], met_phi[idx], idx
        )
        for idx in range(len(bjets_mass))
    ]
