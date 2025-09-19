import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import all components from our engine
from src.exotic_engine.models.gbm import GeometricBrownianMotionProcess
from src.exotic_engine.instruments.european import EuropeanOption
from src.exotic_engine.instruments.asian import AsianOption
from src.exotic_engine.instruments.lookback import LookbackOption
from src.exotic_engine.instruments.barrier import BarrierOption
from src.exotic_engine.instruments.bermudan import BermudanOption
from src.exotic_engine.pricing_engines.mc_engine import MonteCarloEngine
from src.exotic_engine.pricing_engines.fd_engine import FDEngine
from src.exotic_engine.pricing_engines.tree_engine import TreeEngine
from src.exotic_engine.analysis.greeks import GreeksCalculator

# --- App Configuration ---
st.set_page_config(
    page_title="Exotic Derivatives Pricing Engine",
    layout="wide"
)

sns.set_style("darkgrid")

# --- App Title and Description ---
st.title("Exotic Derivatives Pricing Engine")
st.markdown("""
This application showcases the capabilities of a Python-based pricing engine for exotic financial derivatives.
Select an option type and adjust the parameters on the left to see how the prices change in real-time,
calculated by three different numerical methods.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Model and Contract Parameters")

# Option Selection
option_selection = st.sidebar.selectbox(
    "Select Option Class",
    ("European Option", "Asian Option", "Lookback Option", "Barrier Option", "Bermudan Option")
)

# Dynamic UI for Barrier Options
option_type_choice = "call" # Default
barrier_type_choice = None
if option_selection == "Barrier Option":
    st.sidebar.subheader("Barrier Specifics")
    option_type_choice = st.sidebar.selectbox("Option Type", ("call", "put"))
    barrier_type_choice = st.sidebar.selectbox(
        "Barrier Type",
        ("up-and-out", "down-and-out", "up-and-in", "down-and-in")
    )

# Market Parameters
st.sidebar.subheader("Market Parameters")
S0 = st.sidebar.slider("Spot Price ($)", min_value=50.0, max_value=150.0, value=100.0, step=0.5)
r = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.1) / 100
sigma = st.sidebar.slider("Volatility (%)", min_value=5.0, max_value=50.0, value=20.0, step=0.5) / 100
T = st.sidebar.slider("Time to Maturity (Years)", min_value=0.1, max_value=2.0, value=1.0, step=0.05)

# Contract Parameters
st.sidebar.subheader("Contract Parameters")
K = st.sidebar.number_input("Strike Price ($)", min_value=50.0, max_value=150.0, value=100.0, step=1.0)

# Instrument-specific parameters
B = None
if option_selection == "Barrier Option":
    if 'up' in barrier_type_choice:
        B = st.sidebar.number_input("Barrier Level ($) (Up)", min_value=S0, max_value=S0 + 50.0, value=S0 + 20.0, step=1.0)
    elif 'down' in barrier_type_choice:
        B = st.sidebar.number_input("Barrier Level ($) (Down)", min_value=S0 - 50.0, max_value=S0, value=S0 - 20.0, step=1.0)
elif option_selection == "Bermudan Option":
    st.sidebar.info("Exercise dates are fixed at [0.25, 0.5, 0.75] years.")


# --- Main Panel for Results ---

# 1. Instantiate the Process
process = GeometricBrownianMotionProcess(s0=S0, r=r, sigma=sigma)

# 2. Instantiate the Selected Instrument
instrument = None
display_name = ""
if option_selection == "European Option":
    display_name = "European Call"
    instrument = EuropeanOption(K=K, T=T, option_type="call")
elif option_selection == "Asian Option":
    display_name = "Asian Call"
    instrument = AsianOption(K=K, T=T, option_type="call")
elif option_selection == "Lookback Option":
    display_name = "Lookback Call"
    instrument = LookbackOption(K=K, T=T, option_type="call", strike_type="fixed")
elif option_selection == "Barrier Option":
    display_name = f"Barrier {option_type_choice.title()} ({barrier_type_choice})"
    instrument = BarrierOption(K=K, T=T, B=B, barrier_type=barrier_type_choice, option_type=option_type_choice)
elif option_selection == "Bermudan Option":
    display_name = "Bermudan Put"
    instrument = BermudanOption(K=95.0, T=T, option_type="put", exercise_dates=[0.25, 0.5, 0.75])

# 3. Instantiate Engines
mc_engine = MonteCarloEngine(num_sims=10000, num_steps=100, use_antithetic=True, random_seed=42)
fd_engine = FDEngine(num_asset_steps=150, num_time_steps=150)
tree_engine = TreeEngine(num_steps=300)

engines = {
    "Monte Carlo": mc_engine,
    "Finite Diff": fd_engine,
    "Binomial Tree": tree_engine
}

# 4. Calculate Prices
prices = {}
for name, engine in engines.items():
    is_path_dependent = option_selection in ["Asian Option", "Lookback Option"]
    is_early_exercise = option_selection == "Bermudan Option"

    if is_path_dependent and name in ["Finite Diff", "Binomial Tree"]:
        prices[name] = "N/A (Path-Dep.)"
        continue
    if is_early_exercise and name in ["Monte Carlo", "Finite Diff"]:
        prices[name] = "N/A (Early Ex.)"
        continue

    try:
        if isinstance(engine, FDEngine):
            price, _, _ = engine.calculate(instrument, process)
        else:
            price = engine.calculate(instrument, process)
        prices[name] = f"${price:.4f}"
    except (TypeError, ValueError, NotImplementedError):
        prices[name] = "N/A"

# 5. Display Prices
st.header("Pricing Results")
st.markdown("Prices calculated by the different numerical engines.")

df = pd.DataFrame([prices], index=[display_name])
st.dataframe(df)
st.info("Note: 'N/A' indicates an engine is not suited for an option type. (Path-Dep.) requires path history (only MC). (Early Ex.) requires early exercise logic (only Tree).")


# 6. Display Gamma Plot for Barrier Option
if option_selection == "Barrier Option":
    st.header("Sensitivity Analysis: Gamma Profile")
    st.markdown("This plot shows the option's Gamma across a range of spot prices. Note the extreme instability near the barrier.")

    with st.spinner("Calculating Gamma profile..."):
        if 'up' in barrier_type_choice:
             spot_prices_plot = np.linspace(S0 * 0.8, B * 1.1, 100)
        else: # down barrier
             spot_prices_plot = np.linspace(B * 0.9, S0 * 1.2, 100)
        
        gammas = []
        fd_engine_greeks = FDEngine(num_asset_steps=300, num_time_steps=300, s_max_factor=2.0)
        
        for s0_val in spot_prices_plot:
            proc = GeometricBrownianMotionProcess(s0=s0_val, r=r, sigma=sigma)
            _, _, gamma = fd_engine_greeks.calculate(instrument, proc)
            gammas.append(gamma)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(spot_prices_plot, gammas, color='crimson', lw=2.5)
        ax.axvline(x=B, color='black', linestyle='--', label=f'Barrier Level = ${B:.2f}')
        ax.set_title(f'Gamma of a {display_name}', fontsize=16)
        ax.set_xlabel('Spot Price ($)', fontsize=12)
        ax.set_ylabel('Gamma', fontsize=12)
        ax.legend()
        st.pyplot(fig)

