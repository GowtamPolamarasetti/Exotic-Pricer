import numpy as np
from .base_engine import BaseEngine
from ..instruments.base_instrument import BaseInstrument
from ..models.gbm import GeometricBrownianMotionProcess
from typing import Tuple, Optional

class MonteCarloEngine(BaseEngine):
    def __init__(self, num_sims: int, num_steps: int, use_antithetic: bool = False, random_seed: Optional[int] = None):
        self.num_sims = int(num_sims)
        self.num_steps = int(num_steps)
        self.use_antithetic = bool(use_antithetic)
        self.random_seed = random_seed

    def calculate(self, instrument: BaseInstrument, process: GeometricBrownianMotionProcess, return_stderr: bool = False):
        rng = np.random.default_rng(self.random_seed)

        if self.use_antithetic:
            # handle odd num_sims by generating ceil(num_sims/2) pairs then slicing
            half = (self.num_sims + 1) // 2
            Z1 = rng.standard_normal((half, self.num_steps))
            Z = np.vstack((Z1, -Z1))
            Z = Z[: self.num_sims, :]  # ensure exact num_sims rows

            dt = instrument.T / self.num_steps
            # simulate steps and cumulative product
            steps = np.exp((process.r - 0.5 * process.sigma ** 2) * dt + process.sigma * np.sqrt(dt) * Z)
            paths_no_s0 = process.s0 * np.cumprod(steps, axis=1)
            paths = np.column_stack((np.full((self.num_sims, 1), process.s0), paths_no_s0))
        else:
            paths = process.simulate_paths(instrument.T, self.num_steps, self.num_sims, self.random_seed)

        payoffs = instrument.payoff(paths)
        discount_factor = np.exp(-process.r * instrument.T)
        discounted = payoffs * discount_factor

        price = float(np.mean(discounted))
        if return_stderr:
            stderr = float(np.std(discounted, ddof=1) / np.sqrt(self.num_sims))
            return price, stderr
        return price
