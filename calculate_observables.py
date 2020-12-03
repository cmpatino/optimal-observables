import numpy as np


def boost_to_frame(p_particle, p_frame):
    b = -p_frame[:3] / p_frame[3]
    b2 = np.sum(b**2)
    gamma = 1 / np.sqrt(1 - b2)
    bp = np.sum(p_particle[:3] * b)
    gamma2 = max((gamma - 1) / b2, 0)

    p_particle_boosted = np.zeros_like(p_particle)
    p_particle_boosted[:3] = p_particle[:3] + (gamma2 * bp * b) + (gamma * b * p_particle[3])
    p_particle_boosted[3] = gamma * (p_particle[3] + bp)
    return p_particle_boosted
