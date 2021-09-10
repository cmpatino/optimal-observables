from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np


@dataclass
class Particle:
    pt: np.ndarray
    phi: np.ndarray
    eta: np.ndarray
    mass: Union[float, np.ndarray]
    charge: Optional[Union[float, np.ndarray]] = None


@dataclass
class MET:
    magnitude: Optional[np.ndarray] = None
    phi: Optional[np.ndarray] = None
    eta: Optional[np.ndarray] = None


@dataclass
class ReconstructedEvent:
    p_top: np.ndarray
    p_l_t: np.ndarray
    p_b_t: np.ndarray
    p_nu_t: np.ndarray
    p_tbar: np.ndarray
    p_l_tbar: np.ndarray
    p_b_tbar: np.ndarray
    p_nu_tbar: np.ndarray
    idx: np.ndarray
    weight: np.ndarray

    def return_values(self) -> Dict[str, np.ndarray]:
        return {
            "p_top": self.p_top,
            "p_l_t": self.p_l_t,
            "p_b_t": self.p_b_t,
            "p_nu_t": self.p_nu_t,
            "p_tbar": self.p_tbar,
            "p_l_tbar": self.p_l_tbar,
            "p_b_tbar": self.p_b_tbar,
            "p_nu_tbar": self.p_nu_tbar,
            "idx": self.idx,
            "weight": self.weight,
        }
