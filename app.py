import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from models import black_scholes, monte_carlo_option_price
from utils import get_oil_data
from utils import calculate_annualized_volatility
from utils import calculate_rolling_volatility


# Get latest oil price
try:
    df = get_oil_data()
    latest_price = round(df['Oil Price'].iloc[-1], 2)
except Exception as e:
    st.warning("Couldn't fetch live oil data. Using default spot price.")
    latest_price = 80.0  # fallback

st.title("Oil Option Fair Value Estimator") 
st.write("Estimate the fair value of oil options using Black-Scholes and Monte Carlo simulation.")

st.header(" Input Parameters")

col1, col2 = st.columns(2)

with col1:
    S = st.number_input("Spot Price of Oil ($)", min_value=1.0, value=latest_price, step=0.1)
    K = st.number_input("Strike Price ($)", min_value=1.0, value=85.0)
    option_type = st.selectbox("Option Type", ['call', 'put'])

with col2:
    T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=0.5)
    r = st.number_input("Risk-Free Rate (e.g. 0.05 = 5%)", min_value=0.0, max_value=1.0, value=0.05)
    use_estimated_vol = st.checkbox("Estimate volatility from historical oil prices")

    if use_estimated_vol:
        est_vol = calculate_annualized_volatility(df)
        sigma = st.number_input("Estimated Volatility (1Y Annualized)", value=est_vol, step=0.01)
    else:
        sigma = st.number_input("Volatility (e.g. 0.3 = 30%)", min_value=0.01, max_value=1.0, value=0.3)


simulations = st.slider("Number of Monte Carlo Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

if st.button("Estimate Fair Value"):
    bs_price = black_scholes(S, K, T, r, sigma, option_type)
    mc_price, prob_itm = monte_carlo_option_price(S, K, T, r, sigma, simulations, option_type)

    st.subheader("Model Outputs")
    st.markdown(f"**Black-Scholes Price:** ${bs_price:.2f}")
    st.markdown(f"**Monte Carlo Estimated Price:** ${mc_price:.2f}")
    st.markdown(f"**Probability In-The-Money:** {prob_itm*100:.2f}%")

#Downloadable CSV File
result_data = {
        "Spot Price": [S],
        "Strike Price": [K],
        "Time to Maturity": [T],
        "Risk-Free Rate": [r],
        "Volatility": [sigma],
        "Option Type": [option_type],
        "Black-Scholes Price": [bs_price],
        "Monte Carlo Price": [mc_price],
        "Probability ITM": [prob_itm]
    }
result_df = pd.DataFrame(result_data)

csv = result_df.to_csv(index=False)
st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="option_fair_value_results.csv",
        mime="text/csv"
    )

#plotting
def plot_simulated_prices(S, T, r, sigma, simulations):
    Z = np.random.standard_normal(simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    fig, ax = plt.subplots()
    ax.hist(ST, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(S, color='red', linestyle='--', label="Spot Price")
    ax.axvline(K, color='green', linestyle='--', label="Strike Price")
    ax.set_title("Simulated Final Prices at Maturity")
    ax.set_xlabel("Final Price")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig

if st.button("Show Price Distribution"):
    fig = plot_simulated_prices(S, T, r, sigma, simulations)
    st.pyplot(fig)

st.header("Historical Oil Prices")

if st.checkbox("Show historical data and rolling volatility"):
    df = get_oil_data()
    df = calculate_rolling_volatility(df)

    st.line_chart(df[['Oil Price', 'Rolling Volatility']])

    st.write(f"Latest Oil Price: ${df['Oil Price'].iloc[-1]:.2f}")
    st.write(f"Latest Annualized Volatility: {df['Rolling Volatility'].iloc[-1]*100:.2f}%")

