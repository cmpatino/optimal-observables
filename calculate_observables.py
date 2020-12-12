import numpy as np


def boost_to_frame(p_particle, p_frame):
    b = - p_frame[:, :3] / p_frame[:, 3:]
    b2 = np.sum(b**2, axis=1, keepdims=True)
    gamma = 1 / np.sqrt(1 - b2)
    bp = np.sum(p_particle[:, :3] * b, axis=1, keepdims=True)
    gamma2 = np.clip((gamma - 1) / b2, a_min=0, a_max=None)

    space_coords = p_particle[:, :3] + (gamma2 * bp * b) + (gamma * b * p_particle[:, 3:])
    time_coord = gamma * (p_particle[:, 3:] + bp)
    p_particle_boosted = np.concatenate([space_coords, time_coord], axis=1)
    return p_particle_boosted


def boost_to_com(p1, p2):
    p_com = p1 + p2
    return boost_to_frame(p1, p_com), boost_to_frame(p2, p_com), p_com


def create_basis(top_p_com):
    k_hat = top_p_com[:, :3] / np.linalg.norm(top_p_com[:, :3])
    p_hat = np.array([0, 0, 1]).reshape(1, -1)
    theta = np.arccos(np.sum(k_hat * p_hat))

    n_hat = np.cross(p_hat, k_hat) / np.sin(theta)
    r_hat = (p_hat - (k_hat * np.cos(theta))) / np.sin(theta)
    if theta > (np.pi / 2):
        n_hat *= -1
        r_hat *= -1
    return k_hat, r_hat, n_hat
