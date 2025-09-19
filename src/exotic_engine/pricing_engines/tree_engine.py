import numpy as np
from .base_engine import BaseEngine
from ..instruments.base_instrument import BaseInstrument
from ..instruments.european import EuropeanOption
from ..instruments.barrier import BarrierOption
from ..instruments.bermudan import BermudanOption
from ..models.gbm import GeometricBrownianMotionProcess

class TreeEngine(BaseEngine):
    """
    Prices options using the Binomial Tree method (Cox-Ross-Rubinstein).
    """
    def __init__(self, num_steps: int):
        """
        Initializes the Binomial Tree engine.
        
        :param num_steps: Number of time steps in the binomial tree (N).
        """
        self.N = num_steps

    def calculate(self, instrument: BaseInstrument, process: GeometricBrownianMotionProcess) -> float:
        """
        Calculates the option price using the binomial tree backward induction algorithm.
        
        :param instrument: The financial instrument to be priced.
        :param process: The stochastic process governing the underlying.
        :return: The estimated fair value of the instrument.
        """
        # 1. --- Tree and CRR Parameter Setup ---
        dt = instrument.T / self.N
        u = np.exp(process.sigma * np.sqrt(dt)) # Up-factor
        d = 1 / u # Down-factor
        p = (np.exp(process.r * dt) - d) / (u - d) # Risk-neutral probability
        discount_factor = np.exp(-process.r * dt)

        # 2. --- Build Asset Price Tree (Forward) ---
        # We only need the prices at the terminal nodes
        terminal_prices = np.zeros(self.N + 1)
        for i in range(self.N + 1):
            # Price at terminal node i is S0 * u^i * d^(N-i)
            terminal_prices[i] = process.s0 * (u ** i) * (d ** (self.N - i))

        # 3. --- Backward Induction ---
        # Initialize option values at maturity
        option_values = instrument.payoff(terminal_prices)

        # Step backward through the tree
        for j in range(self.N - 1, -1, -1):
            # Calculate the option values at the current time step j
            # The number of nodes at step j is j+1
            continuation_values = np.zeros(j + 1)
            for i in range(j + 1):
                # V_ij = discount * [p * V_{i+1, j+1} + (1-p) * V_{i, j+1}]
                v_up = option_values[i + 1]
                v_down = option_values[i]
                continuation_values[i] = discount_factor * (p * v_up + (1 - p) * v_down)
            
            option_values = continuation_values

            # --- Apply Exotic Feature Logic for the Current Time Step ---
            current_prices = process.s0 * (u ** np.arange(j + 1)) * (d ** (j - np.arange(j + 1)))

            if isinstance(instrument, BarrierOption):
                if "out" in instrument.barrier_type:
                    if "down" in instrument.barrier_type:
                        # Set value to 0 if price is below the barrier
                        option_values[current_prices <= instrument.B] = 0
                    elif "up" in instrument.barrier_type:
                         # Set value to 0 if price is above the barrier
                        option_values[current_prices >= instrument.B] = 0

            elif isinstance(instrument, BermudanOption):
                current_time = j * dt
                # Check if current time is close to an exercise date
                is_exercise_time = any(np.isclose(current_time, ex_date) for ex_date in instrument.exercise_dates)
                
                if is_exercise_time:
                    exercise_values = instrument.payoff(current_prices)
                    option_values = np.maximum(exercise_values, continuation_values)

        return option_values[0]
