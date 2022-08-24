from itertools import permutations
from typing import Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import jit

import kinematics
from objects import MET, Particle, ReconstructedEvent

M_W = 80.4
M_ELECTRON = 0.000510998902
M_MUON = 0.105658389
SIGMA_X = 10.0
SIGMA_Y = 10.0


def ttbar_bjets_kinematics(
    smeared_bjets_pt: np.ndarray,
    bjets_phi: np.ndarray,
    bjets_eta: np.ndarray,
    bjets_mass: np.ndarray,
    bjets_combinations_idxs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    p_b_t = kinematics.four_momentum(
        pt=pt_combinations[:, 0:1],
        phi=phi_combinations[:, 0:1],
        eta=eta_combinations[:, 0:1],
        mass=mass_combinations[:, 0:1],
    )
    p_b_tbar = kinematics.four_momentum(
        pt=pt_combinations[:, 1:],
        phi=phi_combinations[:, 1:],
        eta=eta_combinations[:, 1:],
        mass=mass_combinations[:, 1:],
    )
    return p_b_t, p_b_tbar, mass_combinations[:, 0:1], mass_combinations[:, 1:]


def ttbar_leptons_kinematics(
    event_ls_pt: np.ndarray,
    event_ls_phi: np.ndarray,
    event_ls_eta: np.ndarray,
    event_ls_charge: np.ndarray,
    m_ls: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Assign lepton to parent quark using leptons' charge.

    :param event_ls_pt: PT of leptons in the event.
    :type event_ls_pt: np.ndarray
    :param event_ls_phi: Phi of leptons in the event.
    :type event_ls_phi: np.ndarray
    :param event_ls_eta: Eta of leptons in the event.
    :type event_ls_eta: np.ndarray
    :param event_ls_charge: Charge of lepton in the event.
    :type event_ls_charge: np.ndarray
    :param m_ls: Mass of leptons in the event.
    :type m_ls: np.ndarray
    :return: Leptons' four-momenta and masses assigned to parent quark.
    :rtype: Tuple[Tuple[float], Tuple[float], float, float]
    """
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
    p_l_t = kinematics.four_momentum(pt=pt_l_t, phi=phi_l_t, eta=eta_l_t, mass=m_l_t)

    pt_l_tbar = np.array(event_ls_pt[l_idx_tbar]).reshape(-1, 1)
    phi_l_tbar = np.array(event_ls_phi[l_idx_tbar]).reshape(-1, 1)
    eta_l_tbar = np.array(event_ls_eta[l_idx_tbar]).reshape(-1, 1)
    m_l_tbar = np.array(m_ls[l_idx_tbar]).reshape(-1, 1)
    p_l_tbar = kinematics.four_momentum(
        pt=pt_l_tbar, phi=phi_l_tbar, eta=eta_l_tbar, mass=m_l_tbar
    )

    return p_l_t, p_l_tbar, m_l_t, m_l_tbar


def scalar_product(p1: jnp.DeviceArray, p2: jnp.DeviceArray) -> jnp.DeviceArray:
    """Calculate four-vector scalar product with (-,-,-,+) metric.

    :param p1: First four-vector.
    :type p1: jnp.DeviceArray
    :param p2: Second four-vector.
    :type p2: jnp.DeviceArray
    :return: Scalar product of two four-vectors.
    :rtype: jnp.DeviceArray
    """
    return p1[:, 3:] * p2[:, 3:] - jnp.sum(p1[:, :3] * p2[:, :3], axis=1, keepdims=True)


def solve_quadratic_equation(
    a: jnp.DeviceArray, b: jnp.DeviceArray, c: jnp.DeviceArray
) -> jnp.DeviceArray:
    """Solve quadratic equation.

    :param a: Coefficient of x**2.
    :type a: jnp.DeviceArray
    :param b: Coefficient of x**1.
    :type b: jnp.DeviceArray
    :param c: Coeffiecient of x**0.
    :type c: jnp.DeviceArray
    :return: Solutions for quadratic equation.
    :rtype: jnp.DeviceArray
    """
    a_c = a.astype(jnp.complex64)
    b_c = b.astype(jnp.complex64)
    c_c = c.astype(jnp.complex64)

    det = jnp.sqrt(b_c**2 - (4 * a_c * c_c))
    sol1 = ((-b_c) + det) / (2 * a_c)
    sol2 = ((-b_c) - det) / (2 * a_c)
    return jnp.concatenate([sol1, sol2], axis=1)


def solve_p_nu(
    eta: jnp.DeviceArray,
    p_l: jnp.DeviceArray,
    p_b: jnp.DeviceArray,
    m_t: jnp.DeviceArray,
    m_b: jnp.DeviceArray,
    m_w=M_W,
) -> Tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]:
    """Get possible solutions for neutrino's px and py.

    :param eta: Neutrino's eta.
    :type eta: jnp.DeviceArray
    :param p_l: Lepton's four-momentum.
    :type p_l: jnp.DeviceArray
    :param p_b: b-jet's four-momentum.
    :type p_b: jnp.DeviceArray
    :param m_t: Top's mass value.
    :type m_t: jnp.DeviceArray
    :param m_b: Bottom's mass value.
    :type m_b: jnp.DeviceArray
    :param m_w: W's mass value, defaults to M_W
    :type m_w: float, optional
    :return: Solutions for neutrino's px and py.
    :rtype: Tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]
    """
    E_l_prime = (p_l[:, 3:] * jnp.cosh(eta)) - (p_l[:, 2:3] * jnp.sinh(eta))
    E_b_prime = (p_b[:, 3:] * jnp.cosh(eta)) - (p_b[:, 2:3] * jnp.sinh(eta))

    den = p_b[:, 0:1] * E_l_prime - p_l[:, 0:1] * E_b_prime
    A = (p_l[:, 1:2] * E_b_prime - p_b[:, 1:2] * E_l_prime) / den

    l_b_prod = scalar_product(p1=p_l, p2=p_b)
    alpha = m_t**2 - m_w**2 - m_b**2 - 2 * l_b_prod
    B = (E_l_prime * alpha - E_b_prime * m_w**2) / (-2 * den)

    par1 = (p_l[:, 0:1] * A + p_l[:, 1:2]) / E_l_prime
    C = A**2 + 1 - par1**2

    par2 = ((m_w**2) / 2 + p_l[:, 0:1] * B) / E_l_prime
    D = 2 * (A * B - par2 * par1)
    F = B**2 - par2**2

    sols = solve_quadratic_equation(a=C, b=D, c=F)

    py1 = sols[:, 0:1]
    py2 = sols[:, 1:]
    px1 = A * py1 + B
    px2 = A * py2 + B
    return px1, px2, py1, py2


def solution_weight(
    met_x: np.ndarray,
    met_y: np.ndarray,
    neutrino_px: np.ndarray,
    neutrino_py: np.ndarray,
) -> np.ndarray:
    """Calculate weight for neutrino's solution.

    :param met_x: Missing ET in x direction in event.
    :type met_x: np.ndarray
    :param met_y: Missing ET in y direction in event.
    :type met_y: np.ndarray
    :param neutrino_px: Total neutrino px in the event.
    :type neutrino_px: np.ndarray
    :param neutrino_py: Total neutrino py in the event.
    :type neutrino_py: np.ndarray
    :return: Solution's weight.
    :rtype: np.ndarray
    """
    dx = met_x - neutrino_px
    dy = met_y - neutrino_py
    weight_x = np.exp(-(dx**2) / (2 * SIGMA_X**2))
    weight_y = np.exp(-(dy**2) / (2 * SIGMA_Y**2))
    return weight_x * weight_y


@jit
def get_neutrino_momentum(
    nu_eta_t: jnp.DeviceArray,
    p_l_t: jnp.DeviceArray,
    p_b_t: jnp.DeviceArray,
    m_b_t: jnp.DeviceArray,
    nu_eta_tbar: jnp.DeviceArray,
    p_l_tbar: jnp.DeviceArray,
    p_b_tbar: jnp.DeviceArray,
    m_b_tbar: jnp.DeviceArray,
    m_t_val: jnp.DeviceArray,
) -> Tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]:
    """Get neutrino px and py solutions for neutrinos assigned to top and
    anti-top quarks.

    :param nu_eta_t: Eta for neutrino assigned to top quark.
    :type nu_eta_t: jnp.DeviceArray
    :param p_l_t: Four-momentum of lepton assigned to top quark.
    :type p_l_t: jnp.DeviceArray
    :param p_b_t: Four-momentum of b-jet assigned to top quark.
    :type p_b_t: jnp.DeviceArray
    :param m_b_t: Mass of b-jet assigned to top quark.
    :type m_b_t: jnp.DeviceArray
    :param nu_eta_tbar: Eta for neutrino assigned to anti-top quark.
    :type nu_eta_tbar: jnp.DeviceArray
    :param p_l_tbar: Four-momentum of lepton assigned to anti-top quark.
    :type p_l_tbar: jnp.DeviceArray
    :param p_b_tbar: Four-momentum of b-jet assigned to anti-top quark.
    :type p_b_tbar: jnp.DeviceArray
    :param m_b_tbar: Mass of b-jet assigned to anti-top quark.
    :type m_b_tbar: jnp.DeviceArray
    :param m_t_val: Top quark's mass.
    :type m_t_val: jnp.DeviceArray
    :return: Solutions for neutrino's px and py assigned to top and anti-top quarks.
    :rtype: Tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]
    """
    nu_t_px1, nu_t_px2, nu_t_py1, nu_t_py2 = solve_p_nu(
        eta=nu_eta_t, p_l=p_l_t, p_b=p_b_t, m_t=m_t_val, m_b=m_b_t
    )

    nu_tbar_px1, nu_tbar_px2, nu_tbar_py1, nu_tbar_py2 = solve_p_nu(
        eta=nu_eta_tbar,
        p_l=p_l_tbar,
        p_b=p_b_tbar,
        m_t=m_t_val,
        m_b=m_b_tbar,
    )

    nu_t_px = jnp.concatenate([nu_t_px1, nu_t_px1, nu_t_px2, nu_t_px2], axis=0)
    nu_t_py = jnp.concatenate([nu_t_py1, nu_t_py1, nu_t_py2, nu_t_py2], axis=0)

    nu_tbar_px = jnp.concatenate(
        [nu_tbar_px1, nu_tbar_px2, nu_tbar_px1, nu_tbar_px2], axis=0
    )
    nu_tbar_py = jnp.concatenate(
        [nu_tbar_py1, nu_tbar_py2, nu_tbar_py1, nu_tbar_py2], axis=0
    )
    return nu_t_px, nu_t_py, nu_tbar_px, nu_tbar_py


def lepton_kinematics(
    electron_pt: np.ndarray,
    electron_phi: np.ndarray,
    electron_eta: np.ndarray,
    electron_charge: np.ndarray,
    muon_pt: np.ndarray,
    muon_phi: np.ndarray,
    muon_eta: np.ndarray,
    muon_charge: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Assign lepton four-momenta to top and anti-top quarks using combination
    of muons and electrons in the event.

    :param electron_pt: PT of electrons in the event.
    :type electron_pt: np.ndarray
    :param electron_phi: Phi of electrons in the event.
    :type electron_phi: np.ndarray
    :param electron_eta: Eta of electrons in the event.
    :type electron_eta: np.ndarray
    :param electron_charge: Charges of electrons in the event.
    :type electron_charge: np.ndarray
    :param muon_pt: PT of muons in the event.
    :type muon_pt: np.ndarray
    :param muon_phi: Phi of muons in the event.
    :type muon_phi: np.ndarray
    :param muon_eta: Eta of muons in the event.
    :type muon_eta: np.ndarray
    :param muon_charge: Charges of muons in the event.
    :type muon_charge: np.ndarray
    :return: Leptons' four-momenta and masses assigned to parent top and anti-top quarks.
    :rtype: Tuple[np.ndarray, np.ndarray, float, float]
    """
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
            m_ls=m_ls,
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
            m_ls=m_ls,
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
            m_ls=m_ls,
        )
        return p_l_t, p_l_tbar, m_l_t, m_l_tbar

    else:
        return None, None, None, None


def reconstruct_event(
    bjet: Particle,
    electron: Particle,
    muon: Particle,
    met: MET,
    idx: int,
    rng: np.random.Generator,
) -> Union[ReconstructedEvent, None]:
    """Reconstruct neutrinos in the event.

    :param bjet: Kinematics of b-jets in event.
    :type bjet: Particle
    :param electron: Kinematics of electrons in event.
    :type electron: Particle
    :param muon: Kinematics of muons in event.
    :type muon: Particle
    :param met: MET in event.
    :type met: MET
    :param idx: Event index.
    :type idx: int
    :param rng: Numpy's random number generator.
    :type rng: np.random.Generator
    :return: Particles in the event with idx and reconstruction weight. Return
             None if event doesn't meet selection criteria.
    :rtype: Union[ReconstructedEvent, None]
    """
    p_l_t, p_l_tbar, m_l_t, m_l_tbar = lepton_kinematics(
        electron_pt=electron.pt[idx],
        electron_phi=electron.phi[idx],
        electron_eta=electron.eta[idx],
        electron_charge=electron.charge[idx],
        muon_pt=muon.pt[idx],
        muon_phi=muon.phi[idx],
        muon_eta=muon.eta[idx],
        muon_charge=muon.charge[idx],
    )
    if p_l_t is None:
        return None

    if len(bjet.mass[idx]) < 2:
        return None

    bjets_combinations_idxs = np.array(
        list(permutations(range(len(bjet.mass[idx])), 2))
    )
    smeared_bjets_pt = rng.normal(
        bjet.pt[idx], bjet.pt[idx] * 0.14, (5, len(bjet.pt[idx]))
    )
    p_b_t, p_b_tbar, m_b_t, m_b_tbar = ttbar_bjets_kinematics(
        smeared_bjets_pt=smeared_bjets_pt,
        bjets_phi=bjet.phi[idx],
        bjets_eta=bjet.eta[idx],
        bjets_mass=bjet.mass[idx],
        bjets_combinations_idxs=bjets_combinations_idxs,
    )

    met_x = (met.magnitude[idx] * np.cos(met.phi[idx]))[0]
    met_y = (met.magnitude[idx] * np.sin(met.phi[idx]))[0]

    # Vectorize Eta grid for loop
    eta_range = np.linspace(-5, 5, 51)
    eta_grid = np.array(np.meshgrid(eta_range, eta_range)).T.reshape(-1, 2)

    eta_vectorized_mask = [
        i for i in range(eta_grid.shape[0]) for j in range(p_b_t.shape[0])
    ]
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
    mass_vectorized_mask = [
        i for i in range(m_t_search.shape[0]) for j in range(p_b_t.shape[0])
    ]
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
        nu_eta_t=jnp.array(nu_eta_t),
        p_l_t=jnp.array(p_l_t),
        p_b_t=jnp.array(p_b_t),
        m_b_t=jnp.array(m_b_t),
        nu_eta_tbar=jnp.array(nu_eta_tbar),
        p_l_tbar=jnp.array(p_l_tbar),
        p_b_tbar=jnp.array(p_b_tbar),
        m_b_tbar=jnp.array(m_b_tbar),
        m_t_val=jnp.array(m_t_val),
    )
    nu_t_px = np.array(nu_t_px)
    nu_t_py = np.array(nu_t_py)
    nu_tbar_px = np.array(nu_tbar_px)
    nu_tbar_py = np.array(nu_tbar_py)

    total_nu_px = nu_t_px + nu_tbar_px
    total_nu_py = nu_t_py + nu_tbar_py

    real_mask = np.isreal(total_nu_px) * np.isreal(total_nu_py)
    real_mask_momentum = np.tile(real_mask, (1, 4))

    p_b_t = np.tile(p_b_t, (4, 1))[real_mask_momentum].reshape(-1, 4)
    p_l_t = np.tile(p_l_t, (4, 1))[real_mask_momentum].reshape(-1, 4)
    nu_eta_t = np.tile(nu_eta_t, (4, 1))[real_mask]
    nu_t_px = nu_t_px[real_mask]
    nu_t_py = nu_t_py[real_mask]

    p_b_tbar = np.tile(p_b_tbar, (4, 1))[real_mask_momentum].reshape(-1, 4)
    p_l_tbar = np.tile(p_l_tbar, (4, 1))[real_mask_momentum].reshape(-1, 4)
    nu_eta_tbar = np.tile(nu_eta_tbar, (4, 1))[real_mask]
    nu_tbar_px = nu_tbar_px[real_mask]
    nu_tbar_py = nu_tbar_py[real_mask]

    total_nu_px = total_nu_px[real_mask]
    total_nu_py = total_nu_py[real_mask]

    weights = solution_weight(
        met_x=met_x, met_y=met_y, neutrino_px=total_nu_px, neutrino_py=total_nu_py
    )
    if len(weights) == 0:
        return None
    best_weight_idx = np.argmax(weights)
    if weights[best_weight_idx] < 0.4:
        return None

    best_weight = np.real(weights[best_weight_idx])
    best_b_t = p_b_t[best_weight_idx]
    best_l_t = p_l_t[best_weight_idx]
    best_nu_t = kinematics.neutrino_four_momentum(
        px=np.real(nu_t_px[best_weight_idx]),
        py=np.real(nu_t_py[best_weight_idx]),
        eta=nu_eta_t[best_weight_idx],
    )
    best_b_tbar = p_b_tbar[best_weight_idx]
    best_l_tbar = p_l_tbar[best_weight_idx]
    best_nu_tbar = kinematics.neutrino_four_momentum(
        px=np.real(nu_tbar_px[best_weight_idx]),
        py=np.real(nu_tbar_py[best_weight_idx]),
        eta=nu_eta_tbar[best_weight_idx],
    )

    p_top = best_b_t + best_l_t + best_nu_t
    p_tbar = best_b_tbar + best_l_tbar + best_nu_tbar
    idx_arr = np.array([idx])

    reconstructed_event = ReconstructedEvent(
        p_top=p_top,
        p_l_t=best_l_t,
        p_b_t=best_b_t,
        p_nu_t=best_nu_t,
        p_tbar=p_tbar,
        p_l_tbar=best_l_tbar,
        p_b_tbar=best_b_tbar,
        p_nu_tbar=best_nu_tbar,
        idx=idx_arr,
        weight=best_weight,
    )
    return reconstructed_event
