import numpy as np
import pytest
from src.exotic_engine.models.gbm import GeometricBrownianMotionProcess

def test_gbm_statistical_properties():
    """
    Tests if the simulated GBM paths have the correct log-normal properties.
    Expected log(S_T) ~ N(log(S0) + (r - 0.5*sigma^2)*T, sigma^2*T)
    """
    s0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    num_sims = 200000  # Use a large number for statistical significance
    num_steps = 2 # Only need the endpoint for this test

    process = GeometricBrownianMotionProcess(s0, r, sigma)
    paths = process.simulate_paths(T, num_steps, num_sims, random_seed=42)
    
    final_prices = paths[:, -1]
    log_returns = np.log(final_prices / s0)

    # Theoretical mean and variance of log returns
    expected_mean = (r - 0.5 * sigma**2) * T
    expected_variance = (sigma**2) * T

    # Calculated mean and variance from simulation
    actual_mean = np.mean(log_returns)
    actual_variance = np.var(log_returns)

    # Assert that the actual values are close to the theoretical ones
    # Use a tolerance (atol) appropriate for statistical tests
    assert actual_mean == pytest.approx(expected_mean, abs=1e-3)
    assert actual_variance == pytest.approx(expected_variance, abs=1e-3)
