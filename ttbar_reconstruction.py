import os
import uproot
import numpy as np
import jax.numpy as jnp

from jax import vmap
from typing import List, Tuple
from itertools import permutations
from tqdm import tqdm

from processing import event_selection


# m_t = 172.5
# m_w = 80.4
# m_e = 0.000510998902
# m_m = 0.105658389


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
    E = pt * np.cosh(eta)
    return np.array([px, py, pz, E])


def ttbar_bjets_kinematics(smeared_bjets_pt: np.ndarray, bjets_phi: np.ndarray,
                           bjets_eta: np.ndarray, bjets_mass: np.ndarray,
                           bjets_combinations_idxs: np.ndarray) -> Tuple[np.ndarray]:
    """Create four momenta for b-jets from possible jet permutations. pt comes with
    extra entries due to smearing for the reconstruction.

    :param smeared_bjets_pt: pt for two b-jets smeared n times.
    :type smeared_bjets_pt: np.ndarray
    :param bjets_phi: phi for two b-jets.
    :type bjets_phi: np.ndarray
    :param bjets_eta: eta for two b-jets.
    :type bjets_eta: np.ndarray
    :param bjets_mass: mass for two b-jets.
    :type bjets_mass: np.ndarray
    :param bjets_combinations_idxs: Indexes for possible permutations of b-jets.
    :type bjets_combinations_idxs: np.ndarray
    :return: Four-momenta for two b-jets and their masses assigned to top quarks.
    :rtype: Tuple[np.ndarray]
    """
    n_smears = smeared_bjets_pt.shape[0]
    pt_combinations = smeared_bjets_pt[:, bjets_combinations_idxs].reshape(-1, 2)
    phi_combinations = np.tile(bjets_phi[bjets_combinations_idxs], (n_smears, 1))
    eta_combinations = np.tile(bjets_eta[bjets_combinations_idxs], (n_smears, 1))
    mass_combinations = np.tile(bjets_mass[bjets_combinations_idxs], (n_smears, 1))
    p_b_t = four_momentum(
        pt=pt_combinations[:, 0:1],
        phi=phi_combinations[:, 0:1],
        eta=eta_combinations[:, 0:1],
        mass=mass_combinations[:, 0:1]
    )
    p_b_tbar = four_momentum(
        pt=pt_combinations[:, 1:],
        phi=phi_combinations[:, 1:],
        eta=eta_combinations[:, 1:],
        mass=mass_combinations[:, 1:]
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


def find_roots(coeffs: jnp.DeviceArray):
    """Wrapper of Jax's implementation of np.roots.

    :param coeffs: Coefficients of polynomial to solve.
    :type coeffs: jnp.DeviceArray
    :return: Roots of polynomial
    :rtype: jnp.DeviceArray
    """
    return jnp.roots(coeffs, strip_zeros=False)


def calculate_neutrino_py(eta: np.ndarray, m_b: np.ndarray, p_b: np.ndarray,
                          m_l: np.ndarray, p_l: np.ndarray, m_t: np.ndarray,
                          m_w=80.4) -> Tuple[np.ndarray]:
    """Calculate solutions for neutrino's py by solving the roots of the polynomial from the Neutrino
    Weighting method (https://lss.fnal.gov/archive/thesis/2000/fermilab-thesis-2006-53.pdf
    Appendix B)

    :param eta: Neutrino eta.
    :type eta: np.ndarray
    :param m_b: b-jet mass.
    :type m_b: np.ndarray
    :param p_b: b-jet four momentum.
    :type p_b: np.ndarray
    :param m_l: Lepton's mass.
    :type m_l: np.ndarray
    :param p_l: Lepton's four momentum.
    :type p_l: np.ndarray
    :param m_t: Top's mass.
    :type m_t: np.ndarray
    :param m_w: W boson's mass., defaults to 80.4
    :type m_w: float, optional
    :return: Neutrino's py and quantities for calculate px.
    :rtype: Tuple[np.ndarray]
    """
    alpha_1 = (m_t**2 - m_b**2 - m_w**2) / 2
    alpha_2 = (m_w**2 - m_l**2) / 2

    beta_b = p_b[:, 3:] * np.cosh(eta) - p_b[:, 2:3] * np.sinh(eta)
    A_b = p_b[:, 0:1] / beta_b
    B_b = p_b[:, 1:2] / beta_b
    C_b = alpha_1 / beta_b

    beta_l = p_l[:, 3:] * np.cosh(eta) - p_l[:, 2:3] * np.sinh(eta)
    A_l = p_l[:, 0:1] / beta_l
    B_l = p_l[:, 1:2] / beta_l
    C_l = alpha_2 / beta_l

    kappa = (B_l - B_b) / (A_b - A_l)
    eps = (C_l - C_b) / (A_b - A_l)

    coeff_2 = ((kappa**2) * (A_b**2 - 1)) + B_b**2 - 1
    coeff_1 = (2 * eps * kappa * (A_b**2 - 1)) + (2 * A_b * C_b * kappa) + (2 * B_b * C_b)
    coeff_0 = ((A_b**2 - 1) * eps**2) + (2 * eps * A_b * C_b) + C_b**2

    coeffs = np.concatenate([coeff_2, coeff_1, coeff_0], axis=1)
    jcoeffs = jnp.array(coeffs)
    jsols = vmap(find_roots)(jcoeffs)
    sols = np.array(jsols)

    nu_py = np.concatenate([sols[:, 0:1], sols[:, 1:]], axis=0)
    eps = np.tile(eps, (2, 1))
    kappa = np.tile(kappa, (2, 1))
    return nu_py, eps, kappa


def calculate_neutrino_px(nu_py: np.ndarray, eps: np.ndarray,
                          kappa: np.ndarray) -> np.ndarray:
    return (kappa * nu_py) + eps


def solution_weight(met_x: np.ndarray, met_y: np.ndarray, neutrino_px: np.ndarray,
                    neutrino_py: np.ndarray, met_resolution: np.ndarray) -> np.ndarray:
    weight_x = np.exp(-(met_x - neutrino_px)**2/(2*met_resolution**2))
    weight_y = np.exp(-(met_y - neutrino_py)**2/(2*met_resolution**2))
    return weight_x * weight_y


def get_neutrino_momentum(nu_eta_t, m_b_t, p_b_t, m_l_t,
                          p_l_t, nu_eta_tbar, m_b_tbar,
                          p_b_tbar, m_l_tbar, p_l_tbar,
                          m_t_val) -> Tuple[np.ndarray]:
    nu_t_py, eps, kappa = calculate_neutrino_py(
        eta=nu_eta_t,
        m_b=m_b_t,
        p_b=p_b_t,
        m_l=m_l_t,
        p_l=p_l_t,
        m_t=m_t_val
    )
    nu_t_px = calculate_neutrino_px(nu_py=nu_t_py, eps=eps, kappa=kappa)

    nu_tbar_py, eps, kappa = calculate_neutrino_py(
        eta=nu_eta_tbar,
        m_b=m_b_tbar,
        p_b=p_b_tbar,
        m_l=m_l_tbar,
        p_l=p_l_tbar,
        m_t=m_t_val
    )
    nu_tbar_px = calculate_neutrino_px(nu_py=nu_tbar_py, eps=eps, kappa=kappa)

    return nu_t_px, nu_t_py, nu_tbar_px, nu_tbar_py


def lepton_kinematics(electron_pt: np.ndarray, electron_phi: np.ndarray, electron_eta: np.ndarray,
                      electron_charge: np.ndarray, muon_pt: np.ndarray, muon_phi: np.ndarray,
                      muon_eta: np.ndarray, muon_charge: np.ndarray
                      ) -> Tuple[Tuple[float], Tuple[float], float, float]:
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
        return None, None, None, None
        raise ValueError(
            "Event does not have a valid combination of leptons: "
            f"{n_electrons} electrons and {n_muons} muons in the event."
        )


def reconstruct_event(bjets_mass, bjets_pt, bjets_phi, bjets_eta,
                      electron_pt, electron_phi, electron_eta, electron_charge,
                      muon_pt, muon_phi, muon_eta, muon_charge,
                      met, met_phi, idx):

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

    nu_t_px, nu_t_py, nu_tbar_px, nu_tbar_py = get_neutrino_momentum(
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
    if weights[best_weight_idx] < 0.4:
        return None

    best_weight = np.real(weights[best_weight_idx])
    best_b_t = p_b_t[best_weight_idx]
    best_l_t = p_l_t[best_weight_idx]
    best_nu_t = neutrino_four_momentum(
        px=np.real(nu_t_px[best_weight_idx]),
        py=np.real(nu_t_py[best_weight_idx]),
        eta=nu_eta_t[best_weight_idx]
    )
    best_b_tbar = p_b_tbar[best_weight_idx]
    best_l_tbar = p_l_tbar[best_weight_idx]
    best_nu_tbar = neutrino_four_momentum(
        px=np.real(nu_tbar_px[best_weight_idx]),
        py=np.real(nu_tbar_py[best_weight_idx]),
        eta=nu_eta_tbar[best_weight_idx]
    )

    p_top = best_b_t + best_l_t + best_nu_t
    p_tbar = best_b_tbar + best_l_tbar + best_nu_tbar
    idx_arr = np.array([idx])

    return (p_top, best_l_t, best_b_t, best_nu_t,
            p_tbar, best_l_tbar, best_b_tbar, best_nu_tbar, idx_arr, best_weight)


if __name__ == "__main__":
    sm_path = "mg5_data/SM-process_spin-ON_100k/Events/run_01_decayed_1/tag_1_delphes_events.root"
    output_dir = "reconstructions/SM_spin-ON_100k"
    n_batches = 10

    print("Loading events...", end="\r")
    sm_events = uproot.open(sm_path)["Delphes"]
    print("Loading events...Done")

    print("Applying selection criteria...", end="\r")
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

    # Get original quarks
    status_mask = sm_events["Particle.Status"].array() == 22
    t_mask = (sm_events["Particle.PID"].array() == 6) * status_mask
    tbar_mask = (sm_events["Particle.PID"].array() == -6) * status_mask

    # MET for all events
    met = sm_events["MissingET.MET"].array()
    met_phi = sm_events["MissingET.Phi"].array()
    print("Applying selection criteria...Done")

    reco_names = [
        "p_top", "p_l_t", "p_b_t", "p_nu_t",
        "p_tbar", "p_l_tbar", "p_b_tbar", "p_nu_tbar", "idx", "weight"
    ]
    step_size = len(muon_phi) // n_batches
    for batch_idx in range(n_batches):
        init_idx = batch_idx * step_size
        end_idx = init_idx + step_size
        print(f"Reconstructing events from {init_idx} to {end_idx}")
        reconstructed_events = [
            reconstruct_event(
                bjets_mass[idx], bjets_pt[idx], bjets_phi[idx], bjets_eta[idx],
                electron_pt[idx], electron_phi[idx], electron_eta[idx], electron_charge[idx],
                muon_pt[idx], muon_phi[idx], muon_eta[idx], muon_charge[idx],
                met[idx], met_phi[idx], idx
            )
            for idx in tqdm(range(init_idx, end_idx))
        ]

        recos = {name: [] for name in reco_names}

        for event in reconstructed_events:
            if event is None:
                continue
            for name, reco_p in zip(reco_names, event):
                recos[name].append(reco_p.reshape(1, -1))

        reco_arrays = {name: np.concatenate(reco_list, axis=0) for name, reco_list in recos.items()}

        for name, p_array in reco_arrays.items():
            with open(os.path.join(output_dir, f"{name}_batch_{batch_idx}.npy"), "wb") as f:
                np.save(f, p_array)
        del recos, reco_arrays, reconstructed_events
