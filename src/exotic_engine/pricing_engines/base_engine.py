from abc import ABC, abstractmethod
from ..instruments.base_instrument import BaseInstrument
from ..models.gbm import GeometricBrownianMotionProcess

class BaseEngine(ABC):
    """Abstract base class for all pricing engines."""
    @abstractmethod
    def calculate(self, instrument: BaseInstrument, process: GeometricBrownianMotionProcess) -> float:
        """Calculates the price of the instrument."""
        pass