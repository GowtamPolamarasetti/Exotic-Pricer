from abc import ABC, abstractmethod
import numpy as np

class BaseInstrument(ABC):
    """Abstract base class for all financial instruments."""
    def __init__(self, K: float, T: float):
        self.K = K  # Strike price
        self.T = T  # Time to maturity

    @abstractmethod
    def payoff(self, asset_paths: np.ndarray) -> np.ndarray:
        """Calculates the payoff for each path."""
        pass