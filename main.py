from src.exotic_engine.models.gbm import GeometricBrownianMotionProcess
from src.exotic_engine.instruments.european import EuropeanOption
from src.exotic_engine.instruments.barrier import BarrierOption
from src.exotic_engine.instruments.bermudan import BermudanOption
from src.exotic_engine.pricing_engines.mc_engine import MonteCarloEngine
from src.exotic_engine.pricing_engines.fd_engine import FDEngine
from src.exotic_engine.pricing_engines.tree_engine import TreeEngine

# 1. Define Market and Model Parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2

# 2. Instantiate the Stochastic Process
gbm_process = GeometricBrownianMotionProcess(s0=S0, r=r, sigma=sigma)

# 3. Instantiate the Financial Instruments
vanilla_call = EuropeanOption(K=K, T=T, option_type="call")
barrier_option = BarrierOption(K=K, T=T, B=120.0, barrier_type="up-and-out", option_type="call")
bermudan_put = BermudanOption(K=K, T=T, exercise_dates=[0.25, 0.5, 0.75], option_type="put")

# 4. Instantiate all three Pricing Engines
# Note: Tree and FDM are better for fewer steps, MC needs more sims
mc_engine = MonteCarloEngine(num_sims=50000, num_steps=252, use_antithetic=True, random_seed=42)
fd_engine = FDEngine(num_asset_steps=500, num_time_steps=500)
tree_engine = TreeEngine(num_steps=500)

# 5. Price all instruments with all applicable engines
vanilla_price_mc = mc_engine.calculate(vanilla_call, gbm_process)
vanilla_price_fd = fd_engine.calculate(vanilla_call, gbm_process)
vanilla_price_tree = tree_engine.calculate(vanilla_call, gbm_process)

barrier_price_mc = mc_engine.calculate(barrier_option, gbm_process)
barrier_price_fd = fd_engine.calculate(barrier_option, gbm_process)
barrier_price_tree = tree_engine.calculate(barrier_option, gbm_process)

# Note: Monte Carlo is not ideal for Bermudan options without complex algorithms (e.g., Longstaff-Schwartz)
bermudan_price_tree = tree_engine.calculate(bermudan_put, gbm_process)

# 6. Print the results in a comparison table
print("="*60)
print(f"{'Exotic Option Pricing Engine Results':^60}")
print("="*60)
print(f"{'Instrument':<25} | {'Monte Carlo':<12} | {'Finite Diff':<12} | {'Binomial Tree':<12}")
print("-"*60)
print(f"{'Vanilla Call':<25} | ${vanilla_price_mc: <11.4f} | ${vanilla_price_fd: <11.4f} | ${vanilla_price_tree: <11.4f}")
print(f"{'Up-and-Out Barrier Call':<25} | ${barrier_price_mc: <11.4f} | ${barrier_price_fd: <11.4f} | ${barrier_price_tree: <11.4f}")
print(f"{'Bermudan Put':<25} | {'N/A':<12} | {'N/A':<12} | ${bermudan_price_tree: <11.4f}")
print("-"*60)

