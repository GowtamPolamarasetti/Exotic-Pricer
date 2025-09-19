from ..models.gbm import GeometricBrownianMotionProcess
from ..instruments.base_instrument import BaseInstrument
from ..pricing_engines.base_engine import BaseEngine
# Import the FDEngine to check its type
from ..pricing_engines.fd_engine import FDEngine
import copy

class GreeksCalculator:
    """
    Calculates option risk sensitivities (Greeks).
    - For FDEngine, it retrieves pre-calculated greeks.
    - For other engines, it uses numerical bumping.
    """
    def __init__(self, engine: BaseEngine, ds: float = 0.01):
        self.engine = engine
        self.ds = ds

    def calculate_gamma(self, instrument: BaseInstrument, process: GeometricBrownianMotionProcess) -> float:
        """Calculates Gamma."""
        # Use the efficient, stable method if the engine supports it
        if isinstance(self.engine, FDEngine):
            _, _, gamma = self.engine.calculate(instrument, process)
            return gamma
        else:
            # Fallback to the original bumping method for other engines
            price_mid = self.engine.calculate(instrument, process)

            process_up = copy.deepcopy(process)
            process_up.s0 += self.ds
            price_up = self.engine.calculate(instrument, process_up)

            process_down = copy.deepcopy(process)
            process_down.s0 -= self.ds
            price_down = self.engine.calculate(instrument, process_down)
            
            return (price_up - 2 * price_mid + price_down) / (self.ds**2)

