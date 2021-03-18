import os
import uproot
import numpy as np

from typing import List, Tuple

from processing import event_selection


M_T = 172.5
M_W = 80.4
M_ELECTRON = 0.000510998902
M_MUON = 0.105658389
SIGMA_X = 10.
SIGMA_Y = 10.


def four_momentum(pt: float, phi: float, eta: float, mass: float) -> np.ndarray:
    pt = np.abs(pt)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return np.array([px, py, pz, E])


def neutrino_four_momentum(px: float, py: float, eta: float) -> np.ndarray:
    pt = np.sqrt(px**2 + py**2)
    pz = pt * np.sinh(eta)
    E = np.sqrt(pt ** 2 + pz ** 2)
    return np.array([px, py, pz, E])


def ttbar_leptons_kinematics(event_ls_pt: List[float], event_ls_phi: List[float],
                             event_ls_eta: List[float], event_ls_charge: List[float],
                             m_ls: List[float]) -> Tuple[Tuple[float], Tuple[float], float, float]:
    if event_ls_charge[0] == 1:
        l_idx_t = 0
        l_idx_tbar = 1
    else:
        l_idx_t = 1
        l_idx_tbar = 0
    pt_l_t = event_ls_pt[l_idx_t]
    phi_l_t = event_ls_phi[l_idx_t]
    eta_l_t = event_ls_eta[l_idx_t]
    m_l_t = m_ls[l_idx_t]
    p_l_t = four_momentum(
        pt=pt_l_t,
        phi=phi_l_t,
        eta=eta_l_t,
        mass=m_l_t
    )

    pt_l_tbar = event_ls_pt[l_idx_tbar]
    phi_l_tbar = event_ls_phi[l_idx_tbar]
    eta_l_tbar = event_ls_eta[l_idx_tbar]
    m_l_tbar = m_ls[l_idx_tbar]
    p_l_tbar = four_momentum(
        pt=pt_l_tbar,
        phi=phi_l_tbar,
        eta=eta_l_tbar,
        mass=m_l_tbar
    )

    return p_l_t, p_l_tbar, m_l_t, m_l_tbar


def solve_quadratic_equation(a: float, b: float, c: float) -> List[float]:
    a_c = complex(a)
    b_c = complex(b)
    c_c = complex(c)

    det = np.sqrt(b_c ** 2 - (4 * a_c * c_c))
    sol1 = ((-b_c) + det) / (2 * a_c)
    sol2 = ((-b_c) - det) / (2 * a_c)
    return np.array([sol1, sol2])


def scalar_product(p1: np.ndarray, p2: np.ndarray) -> float:
    return p1[3] * p2[3] - ((p1[0] * p2[0]) + (p1[1] * p2[1]) + (p1[2] * p2[2]))


def solve_p_nu(eta, p_l, p_b, m_t, m_b, m_w=M_W):

    E_l_prime = (p_l[3] * np.cosh(eta)) - (p_l[2] * np.sinh(eta))
    E_b_prime = (p_b[3] * np.cosh(eta)) - (p_b[2] * np.sinh(eta))

    den = p_b[0] * E_l_prime - p_l[0] * E_b_prime
    A = (p_l[1] * E_b_prime - p_b[1] * E_l_prime) / den

    l_b_prod = scalar_product(p1=p_l, p2=p_b)
    alpha = m_t ** 2 - m_w ** 2 - m_b ** 2 - 2 * l_b_prod
    B = (E_l_prime * alpha - E_b_prime * m_w ** 2) / (-2 * den)

    par1 = (p_l[0] * A + p_l[1]) / E_l_prime
    C = A ** 2 + 1 - par1 ** 2

    par2 = ((m_w ** 2) / 2 + p_l[0] * B) / E_l_prime
    D = 2 * (A * B - par2 * par1)
    F = B ** 2 - par2 ** 2

    sols = solve_quadratic_equation(a=C, b=D, c=F)

    py1 = sols[0]
    py2 = sols[1]
    px1 = A * py1 + B
    px2 = A * py2 + B
    return px1, px2, py1, py2


def solution_weight(met_x: np.ndarray, met_y: np.ndarray,
                    neutrino_px: np.ndarray, neutrino_py: np.ndarray) -> np.ndarray:
    dx = met_x - neutrino_px
    dy = met_y - neutrino_py
    weight_x = np.exp(-(dx ** 2) / (2 * SIGMA_X ** 2))
    weight_y = np.exp(-(dy ** 2) / (2 * SIGMA_Y ** 2))
    return weight_x * weight_y


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

        m_ls = [M_ELECTRON] * 2
        p_l_t, p_l_tbar, m_l_t, m_l_tbar = ttbar_leptons_kinematics(
            event_ls_pt=electron_pt,
            event_ls_phi=electron_phi,
            event_ls_eta=electron_eta,
            event_ls_charge=electron_charge,
            m_ls=m_ls
        )
        return p_l_t, p_l_tbar, m_l_t, m_l_tbar

    elif n_muons == 2:
        if np.sum(muon_charge) != 0:
            return None, None, None, None

        m_ls = [M_MUON] * 2
        p_l_t, p_l_tbar, m_l_t, m_l_tbar = ttbar_leptons_kinematics(
            event_ls_pt=muon_pt,
            event_ls_phi=muon_phi,
            event_ls_eta=muon_eta,
            event_ls_charge=muon_charge,
            m_ls=m_ls
        )
        return p_l_t, p_l_tbar, m_l_t, m_l_tbar

    elif (n_electrons == 1) and (n_muons == 1):
        if (electron_charge[0] + muon_charge[0]) != 0:
            return None, None, None, None

        m_ls = [M_ELECTRON, M_MUON]
        event_ls_pt = [electron_pt[0], muon_pt[0]]
        event_ls_phi = [electron_phi[0], muon_phi[0]]
        event_ls_eta = [electron_eta[0], muon_eta[0]]
        event_ls_charge = [electron_charge[0], muon_charge[0]]
        p_l_t, p_l_tbar, m_l_t, m_l_tbar = ttbar_leptons_kinematics(
            event_ls_pt=event_ls_pt,
            event_ls_phi=event_ls_phi,
            event_ls_eta=event_ls_eta,
            event_ls_charge=event_ls_charge,
            m_ls=m_ls
        )
        return p_l_t, p_l_tbar, m_l_t, m_l_tbar

    else:
        return None, None, None, None


def neutrino_weight(total_px: float, total_py: float, met_ex: float, met_ey: float) -> float:
    dx = met_ex - total_px
    dy = met_ey - total_py
    return np.exp((-(dx ** 2) / (2. * SIGMA_X ** 2)) - ((dy ** 2) / (2. * SIGMA_Y ** 2)))


def reconstruct_event(bjets_mass, bjets_pt, bjets_phi, bjets_eta,
                      electron_pt, electron_phi, electron_eta, electron_charge,
                      muon_pt, muon_phi, muon_eta, muon_charge,
                      met, met_phi, idx, rng):

    p_l_t, p_l_tbar, m_l_t, m_l_tbar = lepton_kinematics(
        electron_pt=electron_pt,
        electron_phi=electron_phi,
        electron_eta=electron_eta,
        electron_charge=electron_charge,
        muon_pt=muon_pt,
        muon_phi=muon_phi,
        muon_eta=muon_eta,
        muon_charge=muon_charge
    )
    if p_l_t is None:
        return None

    if len(bjets_mass) < 2:
        return None

    smeared_bjets_pt = rng.normal(
        bjets_pt,
        bjets_pt * 0.14,
        (5, len(bjets_pt))
    )
    met_x = (met * np.cos(met_phi))[0]
    met_y = (met * np.sin(met_phi))[0]

    best_weight = -1
    best_totalx = None
    best_totaly = None
    nu_t = None
    nu_tbar = None

    for nu_tbar_eta in np.linspace(-5, 5, 51):
        for nu_t_eta in np.linspace(-5, 5, 51):
            for m_t_val in np.linspace(171, 174, 7):
                for smeared_pt in smeared_bjets_pt:
                    for bjet_t_idx, bjet_tbar_idx in [(0, 1), (1, 0)]:
                        p_b_t = four_momentum(
                            pt=smeared_pt[bjet_t_idx],
                            phi=bjets_phi[bjet_t_idx],
                            eta=bjets_eta[bjet_t_idx],
                            mass=bjets_mass[bjet_t_idx]
                        )
                        px1_t, px2_t, py1_t, py2_t = solve_p_nu(
                            eta=nu_t_eta,
                            p_l=p_l_t,
                            p_b=p_b_t,
                            m_t=m_t_val,
                            m_b=bjets_mass[bjet_t_idx]
                        )

                        p_b_tbar = four_momentum(
                            pt=smeared_pt[bjet_tbar_idx],
                            phi=bjets_phi[bjet_tbar_idx],
                            eta=bjets_eta[bjet_tbar_idx],
                            mass=bjets_mass[bjet_tbar_idx]
                        )
                        px1_tbar, px2_tbar, py1_tbar, py2_tbar = solve_p_nu(
                            eta=nu_tbar_eta,
                            p_l=p_l_tbar,
                            p_b=p_b_tbar,
                            m_t=m_t_val,
                            m_b=bjets_mass[bjet_tbar_idx]
                        )

                        sols_combinations = [
                            (px1_t, py1_t, px1_tbar, py1_tbar),
                            (px1_t, py1_t, px2_tbar, py2_tbar),
                            (px2_t, py2_t, px1_tbar, py1_tbar),
                            (px2_t, py2_t, px2_tbar, py2_tbar),
                        ]

                        for px_t, py_t, px_tbar, py_tbar in sols_combinations:
                            total_px = px_t + px_tbar
                            total_py = py_t + py_tbar
                            if (not np.isreal(total_px)) or (not np.isreal(total_py)):
                                continue
                            weight = neutrino_weight(
                                total_px=total_px,
                                total_py=total_py,
                                met_ex=met_x,
                                met_ey=met_y
                            )

                            if np.isreal(weight):
                                weight = np.real(weight)
                                if weight > best_weight:
                                    best_weight = weight
                                    best_totalx = np.real(total_px)
                                    best_totaly = np.real(total_py)
                                    nu_t = neutrino_four_momentum(
                                        np.real(px_t),
                                        np.real(py_t),
                                        nu_t_eta
                                    )
                                    nu_tbar = neutrino_four_momentum(
                                        np.real(px_tbar),
                                        np.real(py_tbar),
                                        nu_tbar_eta
                                    )

    print('-----------', idx)
    print("Best weight: ", best_weight)
    print("metx: ", met_x, "total_px: ", best_totalx)
    print("mety: ", met_y, "total_py: ", best_totaly)
    print("nu_t: ", nu_t)
    print("nu_tbar: ", nu_tbar)
    print('-----------\n')
    return None


if __name__ == "__main__":
    sm_path = "mg5_data/SM-process_spin-ON_10k/Events/run_01_decayed_1/tag_1_delphes_events.root"
    output_dir = "reconstructions/Naive-Reco"
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

    # MET for all events
    met = sm_events["MissingET.MET"].array()
    met_phi = sm_events["MissingET.Phi"].array()
    print("Applying selection criteria...Done")

    reco_names = [
        "p_top", "p_l_t", "p_b_t", "p_nu_t",
        "p_tbar", "p_l_tbar", "p_b_tbar", "p_nu_tbar", "idx", "weight"
    ]
    step_size = len(muon_phi) // n_batches
    rng = np.random.default_rng(940202)
    for batch_idx in range(n_batches):
        init_idx = batch_idx * step_size
        end_idx = init_idx + step_size
        reconstructed_events = [
            reconstruct_event(
                bjets_mass=bjets_mass[idx],
                bjets_pt=bjets_pt[idx],
                bjets_phi=bjets_phi[idx],
                bjets_eta=bjets_eta[idx],
                electron_pt=electron_pt[idx],
                electron_phi=electron_phi[idx],
                electron_eta=electron_eta[idx],
                electron_charge=electron_charge[idx],
                muon_pt=muon_pt[idx],
                muon_phi=muon_phi[idx],
                muon_eta=muon_eta[idx],
                muon_charge=muon_charge[idx],
                met=met[idx],
                met_phi=met_phi[idx],
                idx=idx,
                rng=rng
            )
            for idx in range(init_idx, end_idx)
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
