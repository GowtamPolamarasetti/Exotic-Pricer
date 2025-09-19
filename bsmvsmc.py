import numpy as np
from scipy.stats import norm
import time
import matplotlib.pyplot as plt

# --- Instruments ---
class BaseInstrument:
    def __init__(self, K, T):
        self.K = K
        self.T = T

class EuropeanCall(BaseInstrument):
    def payoff(self, asset_paths: np.ndarray) -> np.ndarray:
        return np.maximum(asset_paths[:, -1] - self.K, 0)

class EuropeanPut(BaseInstrument):
    def payoff(self, asset_paths: np.ndarray) -> np.ndarray:
        return np.maximum(self.K - asset_paths[:, -1], 0)

# --- Process ---
class GeometricBrownianMotionProcess:
    def __init__(self, S0, r, sigma):
        self.S0 = S0
        self.r = r
        self.sigma = sigma

    def simulate(self, num_sims, num_steps, T, seed=None):
        if seed is not None:
            np.random.seed(seed)
        dt = T / num_steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        vol = self.sigma * np.sqrt(dt)
        Z = np.random.normal(0, 1, (num_sims, num_steps))
        increments = drift + vol * Z
        log_paths = np.cumsum(increments, axis=1)
        log_paths = np.hstack([np.zeros((num_sims, 1)), log_paths])
        return self.S0 * np.exp(log_paths)

# --- Monte Carlo Engine ---
class MonteCarloEngine:
    def __init__(self, num_sims=100_000, num_steps=252, random_seed=None):
        self.num_sims = num_sims
        self.num_steps = num_steps
        self.random_seed = random_seed

    def calculate(self, instrument, process, return_stderr=False):
        paths = process.simulate(self.num_sims, self.num_steps, instrument.T, self.random_seed)
        payoffs = instrument.payoff(paths) * np.exp(-process.r * instrument.T)
        price = np.mean(payoffs)
        if return_stderr:
            stderr = np.std(payoffs) / np.sqrt(self.num_sims)
            return price, stderr
        return price

# --- Black-Scholes formulas ---
def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- Test Suite ---
if __name__ == "__main__":
    # Parameters
    S0, K, T, r, sigma = 100., 100., 1.0, 0.05, 0.2

    process = GeometricBrownianMotionProcess(S0, r, sigma)
    engine = MonteCarloEngine(num_sims=200_000, num_steps=252, random_seed=42)

    # European Call
    call = EuropeanCall(K, T)
    mc_call, stderr_call = engine.calculate(call, process, return_stderr=True)
    bs_call = bs_call_price(S0, K, T, r, sigma)

    print("\n--- European Call Option ---")
    print(f"Monte Carlo : {mc_call:.4f} ± {stderr_call:.4f} (95% CI ≈ ±{1.96*stderr_call:.4f})")
    print(f"Black–Scholes: {bs_call:.4f}")

    # European Put
    put = EuropeanPut(K, T)
    mc_put, stderr_put = engine.calculate(put, process, return_stderr=True)
    bs_put = bs_put_price(S0, K, T, r, sigma)

    print("\n--- European Put Option ---")
    print(f"Monte Carlo : {mc_put:.4f} ± {stderr_put:.4f} (95% CI ≈ ±{1.96*stderr_put:.4f})")
    print(f"Black–Scholes: {bs_put:.4f}")

    # --- Convergence Test ---
    print("\n--- Convergence Test (European Call) ---")
    sim_sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000]
    errors, prices = [], []
    for sims in sim_sizes:
        engine = MonteCarloEngine(num_sims=sims, num_steps=252, random_seed=42)
        start = time.time()
        mc_price, _ = engine.calculate(call, process, return_stderr=True)
        elapsed = time.time() - start
        err = abs(mc_price - bs_call)
        errors.append(err)
        prices.append(mc_price)
        print(f"Sims={sims:<8d} MC={mc_price:.4f} Error={err:.4e} Time={elapsed:.3f}s")

    # Plot convergence
    plt.figure(figsize=(8,5))
    plt.plot(sim_sizes, errors, marker="o", label="Error vs. BS")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of simulations (log scale)")
    plt.ylabel("Absolute error (log scale)")
    plt.title("Monte Carlo Convergence for European Call")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
