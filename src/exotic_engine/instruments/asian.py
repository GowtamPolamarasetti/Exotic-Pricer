import numpy as np
from .base_instrument import BaseInstrument

class AsianOption(BaseInstrument):
    """An Asian option whose payoff depends on the average asset price."""
    def __init__(self, K: float, T: float, option_type: str = "call"):
        super().__init__(K, T)
        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        self.option_type = option_type

    def payoff(self, asset_paths: np.ndarray) -> np.ndarray:
        # Calculate the arithmetic average of prices along each path (excluding S0)
        average_prices = np.mean(asset_paths[:, 1:], axis=1)
        
        if self.option_type == "call":
            payoffs = np.maximum(average_prices - self.K, 0)
        else: # put
            payoffs = np.maximum(self.K - average_prices, 0)
            
        return payoffs