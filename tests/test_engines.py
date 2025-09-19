import pytest
from src.exotic_engine.models.gbm import GeometricBrownianMotionProcess
from src.exotic_engine.instruments.european import EuropeanOption
from src.exotic_engine.instruments.barrier import BarrierOption
from src.exotic_engine.pricing_engines.mc_engine import MonteCarloEngine

def test_barrier_in_out_parity():
    """
    Validates the In-Out Parity relationship for barrier options:
    Vanilla Option = Knock-In Option + Knock-Out Option
    """
    # 1. Setup common parameters
    s0 = 100.0
    K = 105.0
    T = 1.0
    r = 0.05
    sigma = 0.25
    B = 120.0 # Barrier

    process = GeometricBrownianMotionProcess(s0, r, sigma)
    
    # Use a high number of simulations for accuracy
    engine = MonteCarloEngine(num_sims=100000, num_steps=252, use_antithetic=True, random_seed=42)

    # 2. Define the three instruments
    vanilla_call = EuropeanOption(K=K, T=T, option_type="call")
    up_in_call = BarrierOption(K=K, T=T, B=B, barrier_type="up-and-in", option_type="call")
    up_out_call = BarrierOption(K=K, T=T, B=B, barrier_type="up-and-out", option_type="call")

    # 3. Price the instruments
    price_vanilla = engine.calculate(vanilla_call, process)
    price_in = engine.calculate(up_in_call, process)
    price_out = engine.calculate(up_out_call, process)

    # 4. Assert that the parity holds within a reasonable tolerance
    assert price_vanilla == pytest.approx(price_in + price_out, abs=0.1)
