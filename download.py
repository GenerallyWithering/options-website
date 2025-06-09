import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import datetime 
import plotly.graph_objects as go
import math
from collections import defaultdict


# Greeks helper functions
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def black_scholes_delta(S, K, T, r, sigma, option_type):
    D1 = d1(S, K, T, r, sigma)
    if option_type == "Call":
        return norm.cdf(D1)
    else:
        return norm.cdf(D1) - 1

def black_scholes_gamma(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    return norm.pdf(D1) / (S * sigma * np.sqrt(T))

def black_scholes_theta(S, K, T, r, sigma, option_type):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    first_term = -(S * norm.pdf(D1) * sigma) / (2 * np.sqrt(T))
    if option_type == "Call":
        second_term = r * K * np.exp(-r * T) * norm.cdf(D2)
        return first_term - second_term
    else:
        second_term = r * K * np.exp(-r * T) * norm.cdf(-D2)
        return first_term + second_term

def black_scholes_vega(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(D1) * np.sqrt(T)

def black_scholes_rho(S, K, T, r, sigma, option_type):
    D2 = d2(S, K, T, r, sigma)
    if option_type == "Call":
        return K * T * np.exp(-r * T) * norm.cdf(D2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-D2)

def total_payoff_at_spot_and_time(spot, legs, days_to_expiry, r=0.01):
    """
    Calculate total payoff of strategy at given spot and days to expiry.
    Assumes options decay linearly in time (simplified).
    """
    T = max(days_to_expiry / 365, 1e-5)

    total_payoff = 0
    for leg in legs:
        K = leg['strike']
        option_type = leg['type']
        pos = 1 if leg['position'] == 'Long' else -1
        sigma = leg.get('sigma', 0.2)
        
        # Black-Scholes price of the option at time T (time to expiry)
        d1_val = (np.log(spot / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2_val = d1_val - sigma * np.sqrt(T)

        if option_type == "Call":
            price = spot * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)
        else:  # Put
            price = K * np.exp(-r * T) * norm.cdf(-d2_val) - spot * norm.cdf(-d1_val)

        # Multiply by position and subtract initial premium paid/received
        # Assume leg premium = initial price (simplification), you can improve by inputting actual premiums
        # For demo, assume leg premium = max payoff at start (can be improved)
        premium = max(0, (spot - K) if option_type == 'Call' else (K - spot))

        total_payoff += pos * (price - premium)

    return total_payoff

# ---------- Force login ----------
if not st.session_state.get("logged_in", False):
    st.warning("You must log in to access this page.")
    st.rerun()

# ---------- Payoff Functions ----------
def call_payoff(S, K, premium):
    return np.maximum(S - K, 0) - premium

def put_payoff(S, K, premium):
    return np.maximum(K - S, 0) - premium

def compute_total_payoff(spot, legs, as_of_date, S_range, r=0.01):
    payoffs = []
    for S in S_range:
        total = 0
        for leg in legs:
            K = leg['strike']
            option_type = leg['type']
            pos = 1 if leg['position'] == 'Long' else -1
            sigma = leg.get('sigma', 0.2)
            expiry = leg['expiration']
            T = max((expiry - as_of_date).days / 365, 1e-5)

            d1_val = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2_val = d1_val - sigma * np.sqrt(T)

            if option_type == "Call":
                price = S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)
            else:  # Put
                price = K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)

            premium = leg.get('premium', 0)
            total += pos * (price - premium)
        payoffs.append(total)
    return payoffs

def compute_break_even_points(S_range, payoff_array):
    break_evens = []
    for i in range(1, len(S_range)):
        y1, y2 = payoff_array[i - 1], payoff_array[i]
        if y1 * y2 < 0:
            x1, x2 = S_range[i - 1], S_range[i]
            slope = (y2 - y1) / (x2 - x1)
            x_cross = x1 - y1 / slope
            break_evens.append(x_cross)
        elif y1 == 0:
            break_evens.append(S_range[i - 1])
    return sorted(set(round(be, 2) for be in break_evens))

def compute_normal_break_even(S_range, legs):
    if not legs:
        return []

    # Final expiration for all legs
    max_expiry = max(leg['expiration'] for leg in legs)
    dte = (max_expiry - datetime.date.today()).days

    payoff = np.array([
        total_payoff_at_spot_and_time(S, legs, dte)
        for S in S_range
    ])

    return compute_break_even_points(S_range, payoff)

# ---------- Prebuilt Strategies ----------
def long_straddle(spot, expiry):
    # Buy call and put at ATM strike
    strike = round(spot, 2)
    premium_call = 5.0  # dummy premium
    premium_put = 5.0
    return [
        {"type": "Call", "position": "Long", "strike": strike, "premium": premium_call,
         "expiration": expiry, "sigma": 0.2},
        {"type": "Put", "position": "Long", "strike": strike, "premium": premium_put,
         "expiration": expiry, "sigma": 0.2}
    ]

def calculate_greeks(S, leg, r=0.01):
    K = leg['strike']
    sigma = leg.get('sigma', 0.2)
    pos = 1 if leg['position'] == 'Long' else -1
    expiry = leg['expiration']
    today = datetime.datetime.today().date()
    T = max((expiry - today).days / 365, 1e-5)

    option_type = leg['type']

    delta = black_scholes_delta(S, K, T, r, sigma, option_type)
    gamma = black_scholes_gamma(S, K, T, r, sigma)
    theta = black_scholes_theta(S, K, T, r, sigma, option_type)
    vega = black_scholes_vega(S, K, T, r, sigma)
    rho = black_scholes_rho(S, K, T, r, sigma, option_type)

    return {
        'Delta': pos * delta,
        'Gamma': pos * gamma,
        'Theta': pos * theta,
        'Vega': pos * vega,
        'Rho': pos * rho,
    }

def long_strangle(spot, expiry):
    # Buy OTM call and put
    strike_call = round(spot * 1.05, 2)
    strike_put = round(spot * 0.95, 2)
    premium_call = 3.0
    premium_put = 3.0
    return [
        {"type": "Call", "position": "Long", "strike": strike_call, "premium": premium_call,
         "expiration": expiry, "sigma": 0.25},
        {"type": "Put", "position": "Long", "strike": strike_put, "premium": premium_put,
         "expiration": expiry, "sigma": 0.25}
    ]

def bull_call_spread(spot, expiry):
    lower_strike = round(spot * 0.95, 2)
    upper_strike = round(spot * 1.05, 2)
    return [
        {"type": "Call", "position": "Long", "strike": lower_strike, "premium": 4.0,
         "expiration": expiry, "sigma": 0.2},
        {"type": "Call", "position": "Short", "strike": upper_strike, "premium": 2.0,
         "expiration": expiry, "sigma": 0.2},
    ]

def bear_put_spread(spot, expiry):
    higher_strike = round(spot * 1.05, 2)
    lower_strike = round(spot * 0.95, 2)
    return [
        {"type": "Put", "position": "Long", "strike": higher_strike, "premium": 4.0,
         "expiration": expiry, "sigma": 0.2},
        {"type": "Put", "position": "Short", "strike": lower_strike, "premium": 2.0,
         "expiration": expiry, "sigma": 0.2},
    ]

def iron_condor(spot, expiry):
    # Define strikes: put long < put short < call short < call long
    strikes = [round(spot * x, 2) for x in (0.90, 0.95, 1.05, 1.10)]
    
    # Example premiums (should be calculated or passed in real use)
    premium_put_short = 2.0    # premium received for short put
    premium_put_long = 1.0     # premium paid for long put
    premium_call_short = 2.0   # premium received for short call
    premium_call_long = 1.0    # premium paid for long call
    
    return [
        # Put spread (bull put spread)
        {"type": "Put", "position": "Long", "strike": strikes[0], "premium": premium_put_long,
         "expiration": expiry, "sigma": 0.2},
        {"type": "Put", "position": "Short", "strike": strikes[1], "premium": premium_put_short,
         "expiration": expiry, "sigma": 0.2},
        
        # Call spread (bear call spread)
        {"type": "Call", "position": "Short", "strike": strikes[2], "premium": premium_call_short,
         "expiration": expiry, "sigma": 0.2},
        {"type": "Call", "position": "Long", "strike": strikes[3], "premium": premium_call_long,
         "expiration": expiry, "sigma": 0.2}
    ]

def butterfly_spread(spot, expiry):
    # Long call spread and short call at middle strike
    strikes = [round(spot * x, 2) for x in (0.95, 1.0, 1.05)]
    premium = 2.0
    return [
        {"type": "Put", "position": "Short", "strike": strikes[1], "premium": premium,
         "expiration": expiry, "sigma": 0.2},
        {"type": "Put", "position": "Long", "strike": strikes[0], "premium": premium * 0.5,
         "expiration": expiry, "sigma": 0.2},
        {"type": "Call", "position": "Short", "strike": strikes[1], "premium": premium,
         "expiration": expiry, "sigma": 0.2},
        {"type": "Call", "position": "Long", "strike": strikes[2], "premium": premium * 0.5,
         "expiration": expiry, "sigma": 0.2}
    ]


PREBUILT_STRATEGIES = {    
    "Bull Call Spread": bull_call_spread,
    "Bear Put Spread": bear_put_spread,
    "Long Straddle": long_straddle,
    "Long Strangle": long_strangle,
    "Iron Condor": iron_condor,
    "Butterfly Spread": butterfly_spread,
}


STRATEGY_DESCRIPTIONS = {
    "Bull Call Spread": "Buy a call at a lower strike and sell a call at a higher strike to profit from a moderate rise in the underlying.",
    "Bear Put Spread": "Buy a put at a higher strike and sell a put at a lower strike to profit from a moderate decline in the underlying.",
    "Long Straddle": "Buy a call and a put at the same strike price to profit from large moves either way.",
    "Long Strangle": "Buy an OTM call and an OTM put to profit from big price swings.",
    "Iron Condor": "Sell an out-of-the-money call spread and an out-of-the-money put spread to profit from low volatility.",
    "Butterfly Spread": "Combine bull and bear spreads to profit from low volatility near a target price."
}



# ---------- Helper: Calculate payoff for one leg ----------
def option_payoff_at_price(S, leg, days_to_expiry):
    # Adjust premium by theoretical time decay? Simplified: premium fixed for now.
    # You can improve with time decay later.

    premium = leg['premium']
    K = leg['strike']
    pos = 1 if leg['position'] == 'Long' else -1
    sigma = leg.get('sigma', 0.2)
    T = max(days_to_expiry / 365, 1e-5)  # years, avoid zero

    if leg['type'] == 'Call':
        payoff = call_payoff(S, K, premium)
    else:
        payoff = put_payoff(S, K, premium)

    # For advanced, could price using BS for T days left but here just payoff at expiry.

    return pos * payoff


def calculate_net_debit_credit(legs):
    """
    Calculate net debit/credit for a list of legs.
    Long = pay premium (negative cash flow),
    Short = receive premium (positive cash flow).
    """
    net = 0.0
    for leg in legs:
        premium = leg['premium']
        pos = 1 if leg['position'] == 'Long' else -1
        net -= pos * premium  # Subtract because premium paid (long) is outflow
    return net


# ---------- Streamlit App ----------

st.title("Options Payoff Calculator with Editable Legs ")

today = datetime.datetime.today().date()

# Spot price input
spot_price = st.number_input("Current Spot Price", min_value=0.1, value=100.0, step=0.1)

# Expiration date selector (for prebuilt & legs)
expiry_date = st.date_input("Select Expiration Date", value=today + timedelta(days=30), min_value=today)

# Calculate DTE and show it inline
dte = (expiry_date - today).days
st.write(f"Days To Expiration (DTE): **{dte}** days")

# Select prebuilt strategy or none
selected_strategy = st.selectbox("Load Prebuilt Strategy (Optional)", ["None"] + list(PREBUILT_STRATEGIES.keys()))
if selected_strategy != "None":
    st.info(f"**Description for {selected_strategy}:** {STRATEGY_DESCRIPTIONS.get(selected_strategy, 'No description available.')}")

# Load prebuilt strategy legs
if selected_strategy != "None":
    if st.button(f"Load {selected_strategy} Strategy"):
        legs = PREBUILT_STRATEGIES[selected_strategy](spot_price, expiry_date)
        st.session_state.legs = legs  # Sync to main legs list
        st.session_state.leg_visibility = [False] * len(legs) # Set all new legs to visible
        st.rerun()

# Initialize legs if not present
if 'legs' not in st.session_state:
    st.session_state.legs = []

if 'strategy_legs' not in st.session_state:
    st.session_state.strategy_legs = {}

if st.session_state.legs:
    st.write("### Edit Legs")
    edited_legs = []
    remove_indices = []

    if "leg_visibility" not in st.session_state or len(st.session_state.leg_visibility) != len(st.session_state.legs):
        st.session_state.leg_visibility = [False] * len(st.session_state.legs)

    for i, leg in enumerate(st.session_state.legs):
        leg_id = f"{i}_{leg['strike']}_{leg['expiration']}"
        with st.expander(f"Leg {i+1}: {leg['position']} {leg['type']} Strike {leg['strike']:.2f} Exp {leg['expiration']}"):
            col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,2,1,1])
            option_type = col1.selectbox(f"Type {i}", ["Call", "Put"], index=["Call","Put"].index(leg['type']), key=f"type_{leg_id}")
            position = col2.selectbox(f"Position {i}", ["Long", "Short"], index=["Long","Short"].index(leg['position']), key=f"pos_{leg_id}")
            strike = col3.number_input(f"Strike {i}", min_value=0.1, value=leg['strike'], step=0.1, key=f"strike_{leg_id}")
            expiration = col4.date_input(f"Expiration {i}", value=leg['expiration'], min_value=today, key=f"exp_{leg_id}")
            premium = col5.number_input(f"Premium {i}", min_value=0.0, value=leg['premium'], step=0.1, key=f"prem_{leg_id}")
            sigma = col6.number_input(f"Volatility {i}", min_value=0.01, max_value=3.0, value=leg.get('sigma', 0.2), step=0.01, key=f"sigma_{leg_id}")

            st.session_state.leg_visibility[i] = st.checkbox("Show this leg on chart", value=st.session_state.leg_visibility[i], key=f"vis_{leg_id}")

            if st.button(f"Remove Leg {i+1}", key=f"remove_{leg_id}"):
                remove_indices.append(i)

            edited_legs.append({
                "type": option_type,
                "position": position,
                "strike": strike,
                "expiration": expiration,
                "premium": premium,
                "sigma": sigma,
            })

    if remove_indices:
        for idx in sorted(remove_indices, reverse=True):
            st.session_state.legs.pop(idx)
            st.session_state.leg_visibility.pop(idx)
        st.rerun()

    if st.button("Save Changes to Legs", key="save_changes_2"):
        st.session_state.legs = edited_legs
        st.success("Legs updated!")

with st.form("Add Leg"):
    st.write("### Add New Leg")
    col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,2,1,1])
    option_type = col1.selectbox("Type", ["Call", "Put"])
    position = col2.selectbox("Position", ["Long", "Short"])
    strike = col3.number_input("Strike", min_value=0.1, value=spot_price, step=0.1)
    expiration = col4.date_input("Expiration", value=expiry_date, min_value=today)
    premium = col5.number_input("Premium", min_value=0.0, value=5.0, step=0.1)
    sigma = col6.number_input("Volatility", min_value=0.01, max_value=3.0, value=0.2, step=0.01)
    submitted = st.form_submit_button("Add Leg")
    if submitted:
        st.session_state.legs.append({
            "type": option_type,
            "position": position,
            "strike": strike,
            "expiration": expiration,
            "premium": premium,
            "sigma": sigma,
        })
        st.session_state.leg_visibility.append(True)
        st.success("Leg added!")
        st.rerun()
if st.session_state.legs:
    net_debit_credit = calculate_net_debit_credit(st.session_state.legs)
    if net_debit_credit > 0:
        st.success(f"Net Credit Received: ${net_debit_credit:.2f}")
    elif net_debit_credit < 0:
        st.error(f"Net Debit Paid: ${-net_debit_credit:.2f}")
    else:
        st.info("Net Debit/Credit: $0.00 (Break-even)")


st.write("---")
st.write("## Payoff Chart")



def analyze_strategy_bounds(legs_grouped):
    max_profit = -float('inf')
    max_loss = float('inf')

    unlimited_profit = False
    unlimited_loss = False

    for legs in legs_grouped:
        # Separate legs by type and position
        calls_long = [leg for leg in legs if leg['type'].lower() == 'call' and leg['position'].lower() == 'long']
        calls_short = [leg for leg in legs if leg['type'].lower() == 'call' and leg['position'].lower() == 'short']
        puts_long = [leg for leg in legs if leg['type'].lower() == 'put' and leg['position'].lower() == 'long']
        puts_short = [leg for leg in legs if leg['type'].lower() == 'put' and leg['position'].lower() == 'short']

        # Calculate net premium paid/received (positive = net debit)
        net_premium = 0
        for leg in legs:
            qty = leg.get('quantity', 1)
            premium = leg.get('premium', 0)
            side = leg['position'].lower()
            if side == 'short':
                net_premium += premium * qty * 100
            else:
                net_premium -= premium * qty * 100

        qtys = [leg.get('quantity', 1) for leg in legs]
        min_qty = min(qtys) if qtys else 1

        # 1. Bull Call Spread
        if len(calls_long) == 1 and len(calls_short) == 1 and len(puts_long) == 0 and len(puts_short) == 0:
            long_call = calls_long[0]
            short_call = calls_short[0]
            if long_call['strike'] < short_call['strike']:
                strike_diff = short_call['strike'] - long_call['strike']
                max_profit_val = strike_diff * min_qty * 100 + net_premium
                max_loss_val = net_premium

                max_profit = max(max_profit, max_profit_val)
                max_loss = min(max_loss, max_loss_val)
                unlimited_profit = False
                unlimited_loss = False
                continue

        # 2. Bear Put Spread
        if len(puts_long) == 1 and len(puts_short) == 1 and len(calls_long) == 0 and len(calls_short) == 0:
            long_put = puts_long[0]
            short_put = puts_short[0]
            if long_put['strike'] > short_put['strike']:
                strike_diff = long_put['strike'] - short_put['strike']
                max_profit_val = strike_diff * min_qty * 100 + net_premium
                max_loss_val = net_premium

                max_profit = max(max_profit, max_profit_val)
                max_loss = min(max_loss, max_loss_val)
                unlimited_profit = False
                unlimited_loss = False
                continue

        # 3. Long Straddle
        if len(calls_long) == 1 and len(puts_long) == 1 and len(calls_short) == 0 and len(puts_short) == 0:
            call = calls_long[0]
            put = puts_long[0]
            if call['strike'] == put['strike']:
                max_profit = None  # unlimited upside/downside
                max_loss_val = net_premium
                max_loss = min(max_loss, max_loss_val)
                unlimited_profit = True
                unlimited_loss = False
                continue

        # 4. Long Strangle
        if len(calls_long) == 1 and len(puts_long) == 1 and len(calls_short) == 0 and len(puts_short) == 0:
            call = calls_long[0]
            put = puts_long[0]
            if put['strike'] < call['strike']:
                max_profit = None
                max_loss_val = net_premium
                max_loss = min(max_loss, max_loss_val)
                unlimited_profit = True
                unlimited_loss = False
                continue

        # 5. Iron Condor
        # 4 legs: short put, long put below short put, short call, long call above short call
        if (len(calls_long) == 1 and len(calls_short) == 1 and
            len(puts_long) == 1 and len(puts_short) == 1):
            long_call = calls_long[0]
            short_call = calls_short[0]
            long_put = puts_long[0]
            short_put = puts_short[0]

            if (long_put['strike'] < short_put['strike'] < short_call['strike'] < long_call['strike']):
                max_profit_val = net_premium
                max_loss_val = min(
                    (short_put['strike'] - long_put['strike']),
                    (long_call['strike'] - short_call['strike'])
                ) * min_qty * 100 - net_premium

                max_profit = max(max_profit, max_profit_val)
                max_loss = min(max_loss, max_loss_val)
                unlimited_profit = False
                unlimited_loss = False
                continue

        

        if (len(calls_long) == 1 and len(calls_short) == 1 and
            len(puts_long) == 1 and len(puts_short) == 1):
            long_call = calls_long[0]
            short_call = calls_short[0]
            long_put = puts_long[0]
            short_put = puts_short[0]

            if (long_put['strike'] < short_put['strike'] == short_call['strike'] < long_call['strike']):
                max_profit_val = net_premium
                max_loss_val = (short_call['strike'] - long_call['strike']) * 100 + abs(net_premium)

                max_profit = max(max_profit, max_profit_val)
                max_loss = min(max_loss, max_loss_val)
                unlimited_profit = False
                unlimited_loss = False
                continue
        # Fallback: detect unlimited profit/loss as before

        for short_call in calls_short:
            has_long_call_above = any(long_call['strike'] > short_call['strike'] for long_call in calls_long)
            if not has_long_call_above:
                unlimited_loss = True

        for long_call in calls_long:
            has_short_call_below = any(short_call['strike'] < long_call['strike'] for short_call in calls_short)
            if not has_short_call_below:
                unlimited_profit = True

        for short_put in puts_short:
            has_long_put_below = any(long_put['strike'] < short_put['strike'] for long_put in puts_long)
            if not has_long_put_below:
                unlimited_loss = True

        # Calculate profit cap fallback from net premium
        if max_profit is None or (net_premium > max_profit):
            max_profit = net_premium
        if max_loss is None or (-net_premium < max_loss):
            max_loss = -net_premium

    if unlimited_profit:
        max_profit = None
    if unlimited_loss:
        max_loss = None

    return max_profit, max_loss




legs_to_analyze = []
if st.session_state.legs:
    legs_to_analyze.append(st.session_state.legs)
if selected_strategy != "None":
    legs_to_analyze.append(st.session_state.strategy_legs.get(selected_strategy, []))

max_profit, max_loss = analyze_strategy_bounds(legs_to_analyze)

def format_bound(val):
    return "Unlimited" if val is None else f"${val:.2f}"

if st.session_state.legs or (selected_strategy != "None"):
    all_expirations = []
    if st.session_state.legs:
        all_expirations += [leg['expiration'] for leg in st.session_state.legs]
    if selected_strategy != "None":
        all_expirations += [leg['expiration'] for leg in st.session_state.strategy_legs.get(selected_strategy, [])]
    max_expiry = max(all_expirations) if all_expirations else today

    payoff_date = st.date_input("Select Payoff Date", min_value=today, max_value=max_expiry, value=max_expiry)
  
    # --- Your existing UI checkboxes ---
    show_normal_break_even = st.checkbox("Show Normal Break-even Points", value=False)

    show_future_payoff = st.checkbox("Show selected payoff (solid red line)", value=True)
    show_future_break_even = False
    if show_future_payoff:
        show_future_break_even = st.checkbox("Show selected break-even points", value=True)

    show_today_payoff = st.checkbox("Show today's payoff (magenta dashed line)", value=True)
    show_today_break_even = False
    if show_today_payoff:
        show_today_break_even = st.checkbox("Show today's break-even points", value=True)
    
    def days_to_expiry(leg):
        return max((leg['expiration'] - payoff_date).days, 0)

    S_min = max(0, spot_price * 0.5)
    S_max = spot_price * 1.5
    S_range = np.linspace(S_min, S_max, 200)

    strategy_colors = ['blue', 'orange', 'green', 'purple', 'brown', 'cyan', 'magenta']
    leg_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    strategy_payoffs = {}

    total_payoff = np.zeros_like(S_range)

    # Combine custom legs into one payoff curve
    if st.session_state.legs:
        custom_payoff = np.zeros_like(S_range)
        for leg in st.session_state.legs:
            dte = days_to_expiry(leg)
            leg_payoff = np.array([option_payoff_at_price(S, leg, dte) for S in S_range])
            custom_payoff += leg_payoff

        strategy_payoffs["Custom Legs"] = custom_payoff
        total_payoff += custom_payoff

    # Add selected preset strategy payoff if any
    if selected_strategy != "None":
        legs = st.session_state.strategy_legs.get(selected_strategy, [])
        strat_payoff = np.zeros_like(S_range)
        for leg in legs:
            dte = days_to_expiry(leg)
            strat_payoff += np.array([option_payoff_at_price(S, leg, dte) for S in S_range])
        strategy_payoffs[selected_strategy] = strat_payoff
        total_payoff += strat_payoff
    # Calculate today's payoff and selected payoff
    today = datetime.datetime.today().date()
    legs_combined = []
    if st.session_state.legs:
        legs_combined += st.session_state.legs
    if selected_strategy != "None":
        legs_combined += st.session_state.strategy_legs.get(selected_strategy, [])

    payoff_today = compute_total_payoff(spot_price, legs_combined, today, S_range)
    payoff_selected = compute_total_payoff(spot_price, legs_combined, payoff_date, S_range)

    # === LOCK ONLY normal and today's break-evens ===
    # Use session_state caching for these locked BE points so they don't move
    if "locked_be_today" not in st.session_state:
        st.session_state.locked_be_today = compute_break_even_points(S_range, payoff_today)
    if "locked_be_normal" not in st.session_state:
        st.session_state.locked_be_normal = compute_normal_break_even(S_range, payoff_selected)
    

    
    be_today = compute_break_even_points(S_range, payoff_today)
    normal_bes = compute_normal_break_even(S_range, legs_combined)

    # For future break-even, **do NOT lock** — recompute dynamically every time
    if show_future_break_even:
        be_future = compute_break_even_points(S_range, payoff_selected)
    else:
        be_future = []
    fig = go.Figure()

    # Plot combined custom legs payoff as a single line
    if "Custom Legs" in strategy_payoffs:
        fig.add_trace(go.Scatter(
            x=S_range,
            y=strategy_payoffs["Custom Legs"],
            mode='lines',
            name="Custom Legs Combined",
            line=dict(color='cyan', width=2)
        ))

    # Optionally plot individual legs for custom legs (toggle)
    show_individual_legs = st.checkbox("Show Individual Custom Legs", value=False)
    if show_individual_legs and st.session_state.legs:
        for i, leg in enumerate(st.session_state.legs):
            dte = days_to_expiry(leg)
            leg_payoff = np.array([option_payoff_at_price(S, leg, dte) for S in S_range])
            color = leg_colors[i % len(leg_colors)]
            fig.add_trace(go.Scatter(
                x=S_range,
                y=leg_payoff,
                mode='lines',
                name=f"Leg {i+1} ({leg['position']} {leg['type']})",
                line=dict(dash='dot', color=color)
            ))

    # Toggle: Today payoff
    if show_today_payoff:
        fig.add_trace(go.Scatter(
            x=S_range,
            y=payoff_today,
            mode='lines',
            name=f"Payoff @ {today.strftime('%Y-%m-%d')}",
            line=dict(color="magenta", dash="dot", width=2)
        ))

    # Toggle: Future payoff
    if show_future_payoff:
        fig.add_trace(go.Scatter(
            x=S_range,
            y=payoff_selected,
            mode='lines',
            name=f"Payoff @ {payoff_date.strftime('%Y-%m-%d')}",
            line=dict(color="green", dash="solid", width=2)
        ))

    # Toggle: Break-evens for today (locked)
    if show_today_break_even:
        for be in be_today:
            fig.add_vline(
                x=be,
                line=dict(color="magenta", dash="dot", width=1),
                annotation_text=f"BE {today.strftime('%Y-%m-%d')}",
                annotation_position="top left"
            )

    # Toggle: Break-evens for future (dynamic, no lock)
    if show_future_break_even:
        for be in be_future:
            fig.add_vline(
                x=be,
                line=dict(color="green", dash="solid", width=1),
                annotation_text=f"BE {payoff_date.strftime('%Y-%m-%d')}",
                annotation_position="top right"
            )

    # Toggle: Normal break-evens (locked)
    if show_normal_break_even:
        for be in normal_bes:
                max_expiry = max(leg['expiration'] for leg in legs_combined)
                dte = (max_expiry - datetime.date.today()).days
                payoff_normal = np.array([
                    total_payoff_at_spot_and_time(S, legs_combined, dte)
                    for S in S_range
                ])
                fig.add_trace(go.Scatter(
                    x=S_range,
                    y=payoff_normal,
                    mode='lines',
                    name="Final Expiry Payoff (Normal)",
                    line=dict(color='red', dash='dash')
                ))

    # Calculate break-evens for total payoff (only if anything plotted)
    if len(strategy_payoffs) > 0:
        signs = np.sign(total_payoff)
        zero_crossings = np.where(np.diff(signs))[0]
        break_evens = []
        for idx in zero_crossings:
            x0, x1 = S_range[idx], S_range[idx+1]
            y0, y1 = total_payoff[idx], total_payoff[idx+1]
            be = x0 - y0 * (x1 - x0) / (y1 - y0)
            break_evens.append(be)

        for be in break_evens:
            fig.add_vline(x=be, line=dict(color='red', dash='dash'), annotation_text="Break-even", annotation_position="top")

    fig.add_hline(y=0, line=dict(color='black', dash='dash'))

# Annotate max profit/loss for combined custom legs once if present
if "Custom Legs" in strategy_payoffs and st.session_state.legs:
    # Display max profit and loss above the chart
    st.markdown("---")
    st.markdown(f"### Max Profit: **:green[{format_bound(max_profit)}]**")
    st.markdown(f"### Max Loss: **:red[{format_bound(max_loss)}]**")
    st.markdown("---")


    fig.update_layout(
        title=f"Payoff at {payoff_date.strftime('%Y-%m-%d')}",
        xaxis_title="Stock Price",
        yaxis_title="Profit / Loss",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            title="Strategies/Legs",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.05,
            bordercolor="Black",
            borderwidth=1,
            font=dict(size=12)
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    df = pd.DataFrame({
        "Stock Price": np.round(S_range, 2),
        "Payoff": np.round(total_payoff, 2)
    })

    # --- Greeks Table Display ---
    if st.session_state.legs:
        st.write("---")
        st.write("## Greeks per Leg and Total")

        spot = spot_price
        greeks_list = []
        total_greeks = {'Delta': 0, 'Gamma': 0, 'Theta': 0, 'Vega': 0, 'Rho': 0}

        for i, leg in enumerate(st.session_state.legs):
            g = calculate_greeks(spot, leg)
            g['Leg'] = f"{leg['position']} {leg['type']} {leg['strike']}"
            greeks_list.append(g)
            for key in total_greeks:
                total_greeks[key] += g[key]

        greeks_list.append({**{'Leg': 'Total'}, **total_greeks})

        df_greeks = pd.DataFrame(greeks_list)
        df_greeks = df_greeks[['Leg', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho']]
        st.dataframe(df_greeks.style.format("{:.2f}", subset=pd.IndexSlice[:, df_greeks.select_dtypes(include='number').columns]))

    # Margin Requirement Estimator
    st.write("---")
    st.write("## Margin Requirement Estimator")

    def estimate_margin(legs):
        # Simple approach: sum of max loss on each leg or net debit, whichever is higher
        max_loss = 0
        total_premium = 0
        for leg in legs:
            pos = 1 if leg['position'] == 'Long' else -1
            total_premium += pos * leg['premium']
        # Max loss can be approximated as max loss in total payoff (already computed)
        return max(abs(min(total_payoff)), abs(total_premium))

    margin_required = estimate_margin(st.session_state.legs)
    st.write(f"Estimated margin requirement to hold this strategy: **${margin_required:.2f}**")

    st.write("### Payoff Table")
    st.dataframe(df)

    if st.button("Export Payoff to CSV"):
        df.to_csv("payoff.csv", index=False)
        st.success("Exported payoff.csv")

else:
    st.info("Add legs or load strategies to see payoff chart")
