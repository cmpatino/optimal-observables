import numpy as np

from optimal_observables.reconstruction import kinematics


def boost_to_frame(p_particle, p_frame):
    b = -p_frame[:, :3] / p_frame[:, 3:]
    b2 = np.sum(b**2, axis=1, keepdims=True)
    gamma = 1 / np.sqrt(1 - b2)
    bp = np.sum(p_particle[:, :3] * b, axis=1, keepdims=True)
    gamma2 = np.clip((gamma - 1) / b2, a_min=0, a_max=None)

    space_coords = (
        p_particle[:, :3] + (gamma2 * bp * b) + (gamma * b * p_particle[:, 3:])
    )
    time_coord = gamma * (p_particle[:, 3:] + bp)
    p_particle_boosted = np.concatenate([space_coords, time_coord], axis=1)
    return p_particle_boosted


def boost_to_com(p1, p2):
    p_com = p1 + p2
    return boost_to_frame(p1, p_com), boost_to_frame(p2, p_com), p_com


def create_basis(top_p_com):
    k_hat = top_p_com[:, :3] / np.linalg.norm(top_p_com[:, :3], axis=-1, keepdims=True)
    p_hat = np.array([0, 0, 1]).reshape(1, -1)
    theta = np.arccos(np.sum(k_hat * p_hat, axis=-1, keepdims=True))

    n_hat = np.cross(p_hat, k_hat) / np.sin(theta)
    r_hat = (p_hat - (k_hat * np.cos(theta))) / np.sin(theta)
    sign_mask = np.sign(np.cos(theta))
    n_hat *= sign_mask
    r_hat *= sign_mask
    return k_hat, r_hat, n_hat


def calculate_cosine_obs(p_particle, k_hat, r_hat, n_hat):
    basis_change = np.linalg.inv(np.stack([k_hat, r_hat, n_hat], axis=-1))
    p_new_basis = np.matmul(basis_change, np.expand_dims(p_particle[:, :3], axis=-1))
    obs = (
        p_new_basis[:, :3] / np.linalg.norm(p_new_basis, axis=1, keepdims=True)
    ).squeeze(-1)
    cos_k = obs[:, 0:1]
    cos_r = obs[:, 1:2]
    cos_n = obs[:, 2:3]
    return cos_k, cos_r, cos_n


def get_angles_matrix(p_l_t, p_l_tbar, p_top, p_tbar, include_cosine_prods=False):
    p_top_com, p_tbar_com, p_com = boost_to_com(p_top, p_tbar)
    k_hat, r_hat, n_hat = create_basis(p_top_com)

    p_l_t_frame = boost_to_frame(p_l_t, p_top)
    p_l_tbar_frame = boost_to_frame(p_l_tbar, p_tbar)

    cos_k1, cos_r1, cos_n1 = calculate_cosine_obs(p_l_t_frame, k_hat, r_hat, n_hat)
    cos_k2, cos_r2, cos_n2 = calculate_cosine_obs(p_l_tbar_frame, k_hat, r_hat, n_hat)

    obs_matrix = np.concatenate(
        [cos_k1, cos_k2, cos_r1, cos_r2, cos_n1, cos_n2], axis=1
    )
    if include_cosine_prods:
        obs_matrix = np.concatenate(
            [
                obs_matrix,
                cos_k1 * cos_k2,
                cos_r1 * cos_r2,
                cos_n1 * cos_n2,
                (cos_r1 * cos_k2) + (cos_k1 * cos_r2),
                (cos_r1 * cos_k2) - (cos_k1 * cos_r2),
                (cos_n1 * cos_r2) + (cos_r1 * cos_n2),
                (cos_n1 * cos_r2) - (cos_r1 * cos_n2),
                (cos_n1 * cos_k2) + (cos_k1 * cos_n2),
                (cos_n1 * cos_k2) - (cos_k1 * cos_n2),
            ],
            axis=1,
        )
    return obs_matrix


def get_mtt(p_top: np.ndarray, p_tbar: np.ndarray) -> np.ndarray:
    """Calculate the invariant mass of the top quark pair.

    :param p_top: Four momentum of the top quark.
    :type p_top: np.ndarray
    :param p_tbar: Four momentum of the anti-top quark.
    :type p_tbar: np.ndarray
    :return: Invariant mass of the top quark pair.
    :rtype: np.ndarray
    """
    p_ttbar = p_top + p_tbar
    space_component = np.sum(p_ttbar[:, :3] ** 2, axis=1, keepdims=True)
    energy_component = p_ttbar[:, 3:] ** 2
    return np.sqrt(energy_component - space_component)


def get_pt_ttbar(p_top: np.ndarray, p_tbar: np.ndarray) -> np.ndarray:
    """Calculate the transverse momentum of the top quark pair.

    :param p_top: Four momentum of the top quark.
    :type p_top: np.ndarray
    :param p_tbar: Four momentum of the anti-top quark.
    :type p_tbar: np.ndarray
    :return: Transverse momentum of the top quark pair.
    :rtype: np.ndarray
    """
    p_ttbar = p_top + p_tbar
    return np.linalg.norm(p_ttbar[:, :2], axis=1, keepdims=True)


def get_dPhi_ll(p_l_t: np.ndarray, p_l_tbar: np.ndarray):
    phi_l_t = kinematics.phi(p_l_t)
    phi_l_tbar = kinematics.phi(p_l_tbar)
    dPhi_ll = kinematics.normalize_dPhi(phi_l_tbar - phi_l_t).reshape(-1, 1)
    return dPhi_ll
