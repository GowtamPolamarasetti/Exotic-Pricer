import numpy as np
import pytest
from src.exotic_engine.instruments.european import EuropeanOption
from src.exotic_engine.instruments.asian import AsianOption
from src.exotic_engine.instruments.barrier import BarrierOption

def test_european_call_payoff():
    """Tests the payoff of a European call option."""
    call = EuropeanOption(K=100, T=1, option_type="call")
    # Test in-the-money, at-the-money, and out-of-the-money
    asset_prices = np.array([110, 100, 90])
    payoffs = call.payoff(asset_prices)
    expected_payoffs = np.array([10, 0, 0])
    assert np.array_equal(payoffs, expected_payoffs)

def test_european_put_payoff():
    """Tests the payoff of a European put option."""
    put = EuropeanOption(K=100, T=1, option_type="put")
    asset_prices = np.array([110, 100, 90])
    payoffs = put.payoff(asset_prices)
    expected_payoffs = np.array([0, 0, 10])
    assert np.array_equal(payoffs, expected_payoffs)

def test_asian_call_payoff():
    """Tests the payoff of an Asian call option on an average price."""
    asian_call = AsianOption(K=100, T=1, option_type="call")
    # Path with average price > K
    path1 = np.array([[100, 105, 110, 115]]) # Avg = 110
    # Path with average price < K
    path2 = np.array([[100, 95, 90, 85]]) # Avg = 90
    paths = np.vstack([path1, path2])
    payoffs = asian_call.payoff(paths)
    expected_payoffs = np.array([10, 0]) # max(110-100, 0) and max(90-100, 0)
    assert np.array_equal(payoffs, expected_payoffs)
    
def test_barrier_knock_out_payoff():
    """Tests the knock-out logic for a barrier option."""
    barrier_option = BarrierOption(K=100, T=1, B=120, barrier_type="up-and-out", option_type="call")
    # Path that is in-the-money and does NOT breach the barrier
    path_alive = np.array([[100, 105, 110, 115]])
    # Path that is in-the-money but DOES breach the barrier
    path_knocked_out = np.array([[100, 110, 125, 115]])
    paths = np.vstack([path_alive, path_knocked_out])
    payoffs = barrier_option.payoff(paths)
    expected_payoffs = np.array([15, 0]) # 115-100 for first, 0 for second
    assert np.array_equal(payoffs, expected_payoffs)
