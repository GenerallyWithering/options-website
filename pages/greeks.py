import streamlit as st
import numpy as np
from scipy.stats import norm


if not st.session_state.get("logged_in", False):
    st.warning("You must log in to access this page.")
    st.rerun()
# Black Scholes Greeks function
def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return {'Delta': 0, 'Theta': 0, 'Rho': 0, 'Gamma': 0, 'Vega': 0}
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    greeks = {}

    if option_type == 'call':
        greeks['Delta'] = norm.cdf(d1)
        greeks['Theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r*K*np.exp(-r*T)*norm.cdf(d2)
        greeks['Rho'] = K * T * np.exp(-r*T) * norm.cdf(d2)
    else:
        greeks['Delta'] = -norm.cdf(-d1)
        greeks['Theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) + r*K*np.exp(-r*T)*norm.cdf(-d2)
        greeks['Rho'] = -K * T * np.exp(-r*T) * norm.cdf(-d2)

    greeks['Gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    greeks['Vega'] = S * norm.pdf(d1) * np.sqrt(T)

    return greeks

st.title("Greek Calculator")

with st.form("greek_calc"):
    col1, col2, col3 = st.columns(3)
    spot = col1.number_input("Spot Price (S)", value=100.0)
    strike = col2.number_input("Strike Price (K)", value=100.0)
    T = col3.number_input("Time to Expiry (in years, T)", value=0.5)

    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    sigma = st.number_input("Volatility (sigma)", value=0.2)
    option_type = st.selectbox("Option Type", ["call", "put"])

    calc = st.form_submit_button("Calculate Greeks")
    if calc:
        greeks = black_scholes_greeks(spot, strike, T, r, sigma, option_type)
        st.write("### Greeks")
        for greek, value in greeks.items():
            st.write(f"{greek}: {value:.4f}")
