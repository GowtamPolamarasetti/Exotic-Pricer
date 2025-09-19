import numpy as np
from typing import List
from .base_instrument import BaseInstrument

class BermudanOption(BaseInstrument):
    """
    A Bermudan option, allowing early exercise on a set of discrete dates.
    """
    def __init__(self, K: float, T: float, exercise_dates: List[float], option_type: str = "call"):
        """
        Initializes the Bermudan option.

        :param K: Strike price.
        :param T: Time to maturity.
        :param exercise_dates: A list of times (in years) when the option can be exercised.
        :param option_type: Type of the option, either 'call' or 'put'.
        """
        super().__init__(K, T)
        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        self.option_type = option_type
        self.exercise_dates = sorted(exercise_dates)

    def payoff(self, asset_prices: np.ndarray) -> np.ndarray:
        """
        Calculates the intrinsic value if exercised. This is the same as a European payoff.
        The decision *when* to exercise is handled by the pricing engine.
        
        :param asset_prices: A NumPy array of asset prices.
        :return: A NumPy array of corresponding intrinsic values (payoffs if exercised).
        """
        if self.option_type == "call":
            return np.maximum(asset_prices - self.K, 0)
        else: # put
            return np.maximum(self.K - asset_prices, 0)
