import numpy as np
from .base_instrument import BaseInstrument

class BarrierOption(BaseInstrument):
    """A barrier option that is activated or extinguished by a barrier."""
    def __init__(self, K: float, T: float, B: float, barrier_type: str, option_type: str = "call"):
        super().__init__(K, T)
        self.B = B # Barrier level
        self.barrier_type = barrier_type
        self.option_type = option_type

    def payoff(self, asset_paths: np.ndarray) -> np.ndarray:
        """
        Calculates the payoff for the barrier option.
        This method is robust to handle 0D (scalar), 1D, or 2D (paths) inputs.
        """
        # --- FDM case: Input is a 0D or 1D array of prices at a single time ---
        # This handles both the terminal condition (1D) and boundary conditions (0D)
        if asset_paths.ndim <= 1:
            if self.option_type == "call":
                return np.maximum(asset_paths - self.K, 0)
            else: # put
                return np.maximum(self.K - asset_paths, 0)

        # --- Monte Carlo case: Input is a 2D array of paths ---
        final_prices = asset_paths[:, -1]
        num_sims = asset_paths.shape[0]
        is_active = np.ones(num_sims, dtype=bool)

        # Determine which paths hit the barrier
        if self.barrier_type == "up-and-out":
            knocked_out = np.any(asset_paths >= self.B, axis=1)
            is_active = ~knocked_out
        elif self.barrier_type == "down-and-out":
            knocked_out = np.any(asset_paths <= self.B, axis=1)
            is_active = ~knocked_out
        elif self.barrier_type == "up-and-in":
            knocked_in = np.any(asset_paths >= self.B, axis=1)
            is_active = knocked_in
        elif self.barrier_type == "down-and-in":
            knocked_in = np.any(asset_paths <= self.B, axis=1)
            is_active = knocked_in
        else:
            raise ValueError("Invalid barrier type specified.")

        # Calculate standard call/put payoff
        if self.option_type == "call":
            standard_payoff = np.maximum(final_prices - self.K, 0)
        else: # put
            standard_payoff = np.maximum(self.K - final_prices, 0)
            
        # Apply the barrier condition
        return standard_payoff * is_active


# import numpy as np
# from .base_instrument import BaseInstrument

# class BarrierOption(BaseInstrument):
#     """A barrier option that is activated or extinguished by a barrier."""
#     def __init__(self, K: float, T: float, B: float, barrier_type: str, option_type: str = "call"):
#         super().__init__(K, T)
#         self.B = B # Barrier level
#         self.barrier_type = barrier_type
#         self.option_type = option_type

#     def payoff(self, asset_paths: np.ndarray) -> np.ndarray:
#         """
#         Calculates the payoff, handling both 1D (FDM) and 2D (MC) inputs.
#         """
#         # --- FDM case: Input is a 1D array of asset prices at maturity ---
#         # The barrier logic is handled by the FDM solver's backward pass,
#         # so here we only need the standard terminal payoff.
#         if asset_paths.ndim == 1:
#             if self.option_type == "call":
#                 return np.maximum(asset_paths - self.K, 0)
#             else: # put
#                 return np.maximum(self.K - asset_paths, 0)

#         # --- Monte Carlo case: Input is a 2D array of paths ---
#         final_prices = asset_paths[:, -1]
#         num_sims = asset_paths.shape[0]
#         is_active = np.ones(num_sims, dtype=bool)

#         # Determine which paths hit the barrier
#         if self.barrier_type == "up-and-out":
#             knocked_out = np.any(asset_paths >= self.B, axis=1)
#             is_active = ~knocked_out
#         elif self.barrier_type == "down-and-out":
#             knocked_out = np.any(asset_paths <= self.B, axis=1)
#             is_active = ~knocked_out
#         elif self.barrier_type == "up-and-in":
#             knocked_in = np.any(asset_paths >= self.B, axis=1)
#             is_active = knocked_in
#         elif self.barrier_type == "down-and-in":
#             knocked_in = np.any(asset_paths <= self.B, axis=1)
#             is_active = knocked_in
#         else:
#             raise ValueError("Invalid barrier type specified.")

#         # Calculate standard call/put payoff
#         if self.option_type == "call":
#             standard_payoff = np.maximum(final_prices - self.K, 0)
#         else: # put
#             standard_payoff = np.maximum(self.K - final_prices, 0)
            
#         # Apply the barrier condition
#         return standard_payoff * is_active
