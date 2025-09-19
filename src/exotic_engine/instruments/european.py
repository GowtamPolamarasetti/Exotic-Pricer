import numpy as np
from .base_instrument import BaseInstrument

class EuropeanOption(BaseInstrument):
    """
    A standard European vanilla option.
    """
    def __init__(self, K: float, T: float, option_type: str = "call"):
        """
        Initializes the European option.

        :param K: Strike price.
        :param T: Time to maturity.
        :param option_type: Type of the option, either 'call' or 'put'.
        """
        super().__init__(K, T)
        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        self.option_type = option_type

    def payoff(self, asset_prices: np.ndarray) -> np.ndarray:
        """
        Calculates the terminal payoff. Handles both 1D (FDM) and 2D (MC) inputs.

        :param asset_prices: A NumPy array of asset prices. Can be 1D for FDM
                             or 2D (paths) for Monte Carlo.
        :return: A NumPy array of corresponding payoffs.
        """
        # For MC, asset_prices is a 2D array of paths. Use the final price.
        if asset_prices.ndim == 2:
            final_prices = asset_prices[:, -1]
        # For FDM, asset_prices is a 1D vector of prices at maturity.
        else:
            final_prices = asset_prices

        if self.option_type == "call":
            return np.maximum(final_prices - self.K, 0)
        else: # put
            return np.maximum(self.K - final_prices, 0)

