from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pysr

from scipy.optimize import curve_fit


class Explainer(ABC):
    """Abstract class for creating a simplified symbolic
    expression for an arbitrary function."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        pass


class SymbolicRegressionExplainer(Explainer):
    def __init__(
        self,
        binary_operators: List[str],
        niterations: int = 5,
        variable_names: Optional[List[str]] = None,
    ):
        self.niterations = niterations
        self.binary_operators = binary_operators
        self.variable_names = variable_names
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = pysr.PySRRegressor(
            niterations=self.niterations,
            binary_operators=self.binary_operators,
        )
        self.model.fit(X, y, variable_names=self.variable_names)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fit yet.")
        return self.model.predict(X)


class ParametricExplainer(Explainer):
    def __init__(self, func, p0: Optional[List[float]] = None):
        self.func = func
        self.p0 = p0
        self.popt = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        popt, _ = curve_fit(self.func, X, y, p0=self.p0)
        self.popt = popt

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.popt is None:
            raise ValueError("Model has not been fit yet.")
        return self.func(X, *self.popt)
