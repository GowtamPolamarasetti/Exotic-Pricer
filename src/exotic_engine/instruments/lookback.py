import numpy as np
from .base_instrument import BaseInstrument

class LookbackOption(BaseInstrument):
    """A Lookback option with either a fixed or floating strike."""
    def __init__(self, K: float, T: float, option_type: str = "call", strike_type: str = "fixed"):
        super().__init__(K, T)
        self.option_type = option_type
        self.strike_type = strike_type

    def payoff(self, asset_paths: np.ndarray) -> np.ndarray:
        final_prices = asset_paths[:, -1]
        
        if self.option_type == "call":
            if self.strike_type == "fixed":
                max_prices = np.max(asset_paths, axis=1)
                payoffs = np.maximum(max_prices - self.K, 0)
            else: # floating strike
                min_prices = np.min(asset_paths, axis=1)
                payoffs = final_prices - min_prices
        else: # put
            if self.strike_type == "fixed":
                min_prices = np.min(asset_paths, axis=1)
                payoffs = np.maximum(self.K - min_prices, 0)
            else: # floating strike
                max_prices = np.max(asset_paths, axis=1)
                payoffs = max_prices - final_prices
                
        return np.maximum(payoffs, 0)