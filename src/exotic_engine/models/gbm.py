import numpy as np
from typing import Optional

class GeometricBrownianMotionProcess:
    """
    Models asset price evolution via Geometric Brownian Motion (GBM):
      dS_t = r S_t dt + sigma S_t dW_t
    """
    def __init__(self, s0: float, r: float, sigma: float):
        self.s0 = float(s0)
        self.r = float(r)
        self.sigma = float(sigma)

    def simulate_paths(
        self,
        T: float,
        num_steps: int,
        num_sims: int,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Vectorized path simulation. Returns array shape (num_sims, num_steps+1).
        """
        rng = np.random.default_rng(random_seed)
        dt = float(T) / num_steps

        # generate standard normals once
        Z = rng.standard_normal((num_sims, num_steps))

        # increments per step (shape: num_sims x num_steps)
        increments = np.exp((self.r - 0.5 * self.sigma**2) * dt
                            + self.sigma * np.sqrt(dt) * Z)

        # cumulative product across time then prepend S0 column
        price_paths = self.s0 * np.cumprod(increments, axis=1)
        paths = np.column_stack((np.full((num_sims, 1), self.s0), price_paths))
        return paths
