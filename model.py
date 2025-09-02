# Create Streamlit app and requirements for a robust BSM pricer + strategy visualizer (Indian style)

import textwrap, json, os, sys, math, datetime as dt
from textwrap import dedent

app_code = dedent(r'''
# app_options_india.py
# Streamlit: Robust Black–Scholes–Merton Pricer + Strategy Visualizer (Indian Style)
# Author: ChatGPT (Paramjeet's assistant)
#
# Features
# - BSM call/put pricing with continuous dividend yield
# - Greeks (Delta, Gamma, Theta/day, Vega, Rho)
# - Implied Volatility solver (Brent) from market price
# - Time to expiry from calendar dates (IST)
# - Scenario charts: price vs underlying, Greeks vs price
# - Strategy Lab: 8 common strategies (long call/put, bull/bear spreads, straddle, strangle, butterfly, iron condor)
# - Payoff/P&L charts at expiry, distribution overlay (lognormal under BSM)
# - Rupee formatting and Lot size support (defaults to 50 for index options; edit as needed)
# - Downloadable scenario table CSV
#
# How to run locally:
#   pip install -r requirements.txt
#   streamlit run app_options_india.py

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, date, timedelta
import pytz

IST = pytz.timezone("Asia/Kolkata")

# -------------------------------
# Utility & Finance functions
# -------------------------------
def _to_ist_today():
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    return now_utc.astimezone(IST).date()

def year_fraction(start: date, end: date, basis: str = "ACT/365") -> float:
    """Compute year fraction. Default ACT/365 common for options; editable."""
    if end <= start:
        return 1e-8
    days = (end - start).days
    if basis.upper() == "ACT/365":
        return days / 365.0
    elif basis.upper() in ("ACT/365F", "ACT/365FIXED"):
        return days / 365.0
    elif basis.upper() == "ACT/360":
        return days / 360.0
    else:
        return days / 365.0

def bsm_price(is_call: bool, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """BSM with continuous dividend yield q. Returns option price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # Intrinsic at expiry to be robust
        if is_call:
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

def bsm_greeks(is_call: bool, S: float, K: float, T: float, r: float, q: float, sigma: float):
    """Return dict of greeks. Theta returned per calendar day."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # At expiry: Greeks are tricky; return zeros except Delta as intrinsic slope
        intrinsic = (S - K) if is_call else (K - S)
        delta = 1.0 if (is_call and S > K) else (-1.0 if (not is_call and S < K) else 0.0)
        return dict(delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)

    delta = math.exp(-q * T) * (norm.cdf(d1) if is_call else (norm.cdf(d1) - 1))
    gamma = (math.exp(-q * T) * pdf_d1) / (S * sigma * sqrtT)
    # Theta per year:
    theta_year = (- (S * math.exp(-q * T) * pdf_d1 * sigma) / (2 * sqrtT)
                  - (r * K * math.exp(-r * T) * (norm.cdf(d2) if is_call else norm.cdf(-d2)))
                  + (q * S * math.exp(-q * T) * (norm.cdf(d1) if is_call else norm.cdf(-d1))))
    # Convert to per calendar day for intuition
    theta_day = theta_year / 365.0
    vega = S * math.exp(-q * T) * pdf_d1 * sqrtT  # per 1.0 change in vol (e.g., 0.01 -> vega*0.01)
    rho = (K * T * math.exp(-r * T) * (norm.cdf(d2) if is_call else -norm.cdf(-d2)))
    return dict(delta=delta, gamma=gamma, theta=theta_day, vega=vega, rho=rho)

def implied_vol_from_price(is_call: bool, price: float, S: float, K: float, T: float, r: float, q: float,
                           low: float = 1e-6, high: float = 5.0) -> float:
    """Compute IV via Brent's method. Returns np.nan if not solvable."""
    def f(sig):
        return bsm_price(is_call, S, K, T, r, q, sig) - price
    try:
        # Ensure the function crosses zero in [low, high]
        f_low, f_high = f(low), f(high)
        if f_low * f_high > 0:
            return np.nan
        return brentq(f, low, high, maxiter=500, xtol=1e-8)
    except Exception:
        return np.nan

def fmt_money(x: float) -> str:
    try:
        return f"₹{x:,.2f}"
    except Exception:
        return f"₹{x}"

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="BSM Options (India) — Pricing & Strategy Lab", layout="wide")

st.title("📈 BSM Options (India) — Pricing & Strategy Lab")
st.caption("Robust Black–Scholes–Merton pricer with Greeks + Strategy visualizer (straddle, strangle, spreads, condor).")

with st.sidebar:
    st.header("⚙️ Global Settings")
    lot_size = st.number_input("Lot Size (contracts)", min_value=1, value=50, step=1,
                               help="Common index lot-size is often 50; update as per NSE circulars.")
    basis = st.selectbox("Day-count basis", ["ACT/365", "ACT/360"], index=0)
    default_r = st.number_input("Risk-free rate (annual, %) — e.g., 7.0", value=7.0, step=0.25, format="%.2f")
    default_q = st.number_input("Dividend yield (annual, %)", value=1.00, step=0.25, format="%.2f")
    st.info("Tip: You can adjust lot size and rates here. All amounts shown are **per share** and **per lot** where relevant.")

tab1, tab2 = st.tabs(["🧮 BSM Pricer", "🧪 Strategy Lab"])

# -------------------------------
# Tab 1: BSM Pricer
# -------------------------------
with tab1:
    st.subheader("🧮 Black–Scholes–Merton Pricer")
    colL, colR = st.columns([1.2, 1])
    with colL:
        S = st.number_input("Spot Price (₹)", min_value=0.01, value=24000.0, step=1.0)
        K = st.number_input("Strike (₹)", min_value=0.01, value=24000.0, step=1.0)
        is_call = st.selectbox("Option Type", ["Call", "Put"]) == "Call"
        today = _to_ist_today()
        exp = st.date_input("Expiry Date (IST)", value=today + timedelta(days=30), min_value=today)
        T = year_fraction(today, exp, basis=basis)
        r = default_r / 100.0
        q = default_q / 100.0

        input_mode = st.radio("Volatility Input", ["Use σ (IV)", "Solve σ from Market Price"], horizontal=True)
        if input_mode == "Use σ (IV)":
            sigma = st.number_input("Implied Volatility σ (annual, %)", min_value=0.01, value=15.0, step=0.5, format="%.2f") / 100.0
            price_mkt = None
        else:
            price_mkt = st.number_input("Market Option Price (₹ per share)", min_value=0.0, value=200.0, step=1.0)
            # Compute IV
            sigma = implied_vol_from_price(is_call, price_mkt, S, K, T, r, q)
            if np.isnan(sigma):
                st.error("Could not solve for implied volatility with given inputs. Try different initial values.")
                sigma = 0.0

        # Price & Greeks
        price = bsm_price(is_call, S, K, T, r, q, sigma) if sigma > 0 else (price_mkt or 0.0)
        greeks = bsm_greeks(is_call, S, K, T, r, q, sigma) if sigma > 0 else dict(delta=np.nan, gamma=np.nan, theta=np.nan, vega=np.nan, rho=np.nan)

        st.markdown("### 📜 Results (per share)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Option Price", fmt_money(price))
        c2.metric("Delta", f"{greeks['delta']:.4f}" if greeks['delta']==greeks['delta'] else "—")
        c3.metric("Gamma", f"{greeks['gamma']:.6f}" if greeks['gamma']==greeks['gamma'] else "—")
        c4.metric("Theta / day", f"{greeks['theta']:.4f}" if greeks['theta']==greeks['theta'] else "—")
        c1b, c2b, c3b, c4b = st.columns(4)
        c1b.metric("Vega (per 1.0 σ)", f"{greeks['vega']:.4f}" if greeks['vega']==greeks['vega'] else "—")
        c2b.metric("Rho", f"{greeks['rho']:.4f}" if greeks['rho']==greeks['rho'] else "—")
        c3b.metric("Per Lot Price", fmt_money(price * lot_size))
        if price_mkt is not None and sigma > 0:
            theo_diff = (price - price_mkt)
            c4b.metric("Theo - Market", fmt_money(theo_diff))

    with colR:
        st.markdown("### 🔧 Scenario Chart Settings")
        price_lo = st.number_input("Min Underlying Price (₹)", value=max(1.0, S * 0.7), step=1.0)
        price_hi = st.number_input("Max Underlying Price (₹)", value=S * 1.3, step=1.0)
        n_points = st.slider("Points", 50, 500, 201, step=1)

    # Charts
    grid = np.linspace(price_lo, price_hi, int(n_points))
    values = [bsm_price(is_call, s, K, T, r, q, sigma) for s in grid] if sigma > 0 else [np.nan]*len(grid)

    st.markdown("### 📊 Option Value vs Underlying")
    fig1, ax1 = plt.subplots()
    ax1.plot(grid, values, label="Option Value")
    ax1.axvline(S, linestyle="--", label="Spot")
    ax1.set_xlabel("Underlying Price (₹)")
    ax1.set_ylabel("Option Value (₹ per share)")
    ax1.legend()
    st.pyplot(fig1)

    st.markdown("### 🧪 Greeks vs Underlying")
    greeks_series = {"Delta": [], "Gamma": [], "Theta/day": [], "Vega": [], "Rho": []}
    for s in grid:
        g = bsm_greeks(is_call, s, K, T, r, q, sigma) if sigma > 0 else dict(delta=np.nan, gamma=np.nan, theta=np.nan, vega=np.nan, rho=np.nan)
        greeks_series["Delta"].append(g["delta"])
        greeks_series["Gamma"].append(g["gamma"])
        greeks_series["Theta/day"].append(g["theta"])
        greeks_series["Vega"].append(g["vega"])
        greeks_series["Rho"].append(g["rho"])

    for name, series in greeks_series.items():
        fig, ax = plt.subplots()
        ax.plot(grid, series, label=name)
        ax.axvline(S, linestyle="--", label="Spot")
        ax.set_xlabel("Underlying Price (₹)")
        ax.set_ylabel(name)
        ax.legend()
        st.pyplot(fig)

    # Download scenario
    df_scn = pd.DataFrame({"S": grid, "OptionValue": values,
                           "Delta": greeks_series["Delta"],
                           "Gamma": greeks_series["Gamma"],
                           "Theta/day": greeks_series["Theta/day"],
                           "Vega": greeks_series["Vega"],
                           "Rho": greeks_series["Rho"]})
    st.download_button("⬇️ Download Scenario CSV", df_scn.to_csv(index=False).encode("utf-8"), "bsm_scenario.csv", "text/csv")

    st.info("Note: Theta shown is per **calendar day**; Vega is per 1.0 change in annualized σ (e.g., for 1 vol point use Vega × 0.01).")

# -------------------------------
# Tab 2: Strategy Lab
# -------------------------------
with tab2:
    st.subheader("🧪 Options Strategy Lab (8 Templates)")
    st.caption("Build intuition with classic strategies. Prices from BSM given σ input; override with custom premiums if needed.")
    colA, colB = st.columns([1.3, 1])
    with colA:
        S2 = st.number_input("Spot Price (₹)", min_value=0.01, value=24000.0, step=1.0, key="S2")
        exp2 = st.date_input("Expiry Date (IST)", value=_to_ist_today() + timedelta(days=30), key="exp2")
        T2 = year_fraction(_to_ist_today(), exp2, basis=basis)
        r2 = st.number_input("Risk-free rate (annual, %)", value=default_r, step=0.25, format="%.2f", key="r2") / 100.0
        q2 = st.number_input("Dividend yield (annual, %)", value=default_q, step=0.25, format="%.2f", key="q2") / 100.0
        sigma2 = st.number_input("Implied Volatility σ (annual, %)", min_value=0.01, value=15.0, step=0.5, format="%.2f", key="sigma2") / 100.0

    with colB:
        strategy = st.selectbox("Strategy Template", [
            "Long Call",
            "Long Put",
            "Bull Call Spread",
            "Bear Put Spread",
            "Long Straddle",
            "Long Strangle",
            "Butterfly (Calls)",
            "Iron Condor"
        ])

        override = st.checkbox("Override BSM Premiums with custom prices?")
        show_dist = st.checkbox("Overlay lognormal expiry distribution", value=True,
                                help="Shows BSM-implied distribution of S(T) scaled for visualization.")

    st.markdown("### 🎯 Define Strikes / Quantities")
    # Helper to compute default strikes around spot
    K_atm = round(S2)
    K1 = round(S2 * 0.98, 0)
    K2 = round(S2 * 1.02, 0)
    K_wide1 = round(S2 * 0.95, 0)
    K_wide2 = round(S2 * 1.05, 0)

    # Input areas by strategy
    legs = []  # each leg: dict(type: 'C'|'P'|'S' for stock, side: +1/-1, K, qty, price)
    help_premium = "Premium per share (₹). Leave 0 to use BSM price if not overriding."

    if strategy == "Long Call":
        Kc = st.number_input("Call Strike", value=float(K_atm), step=1.0)
        qty = st.number_input("Lots (positive for buy)", value=1, step=1)
        prem = st.number_input("Premium (optional override)", value=0.0, step=0.5, help=help_premium)
        legs.append(dict(kind='C', side=+1, K=Kc, qty=qty, premium=prem))

    elif strategy == "Long Put":
        Kp = st.number_input("Put Strike", value=float(K_atm), step=1.0)
        qty = st.number_input("Lots (positive for buy)", value=1, step=1)
        prem = st.number_input("Premium (optional override)", value=0.0, step=0.5, help=help_premium)
        legs.append(dict(kind='P', side=+1, K=Kp, qty=qty, premium=prem))

    elif strategy == "Bull Call Spread":
        Kb = st.number_input("Buy Call Strike (lower)", value=float(K1), step=1.0)
        Ks = st.number_input("Sell Call Strike (higher)", value=float(K2), step=1.0)
        qty = st.number_input("Lots (positive for 1 spread)", value=1, step=1)
        prem_b = st.number_input("Premium Buy (optional override)", value=0.0, step=0.5, help=help_premium)
        prem_s = st.number_input("Premium Sell (optional override)", value=0.0, step=0.5, help=help_premium)
        legs += [dict(kind='C', side=+1, K=Kb, qty=qty, premium=prem_b),
                 dict(kind='C', side=-1, K=Ks, qty=qty, premium=prem_s)]

    elif strategy == "Bear Put Spread":
        Kb = st.number_input("Buy Put Strike (higher)", value=float(K2), step=1.0)
        Ks = st.number_input("Sell Put Strike (lower)", value=float(K1), step=1.0)
        qty = st.number_input("Lots (positive for 1 spread)", value=1, step=1)
        prem_b = st.number_input("Premium Buy (optional override)", value=0.0, step=0.5, help=help_premium)
        prem_s = st.number_input("Premium Sell (optional override)", value=0.0, step=0.5, help=help_premium)
        legs += [dict(kind='P', side=+1, K=Kb, qty=qty, premium=prem_b),
                 dict(kind='P', side=-1, K=Ks, qty=qty, premium=prem_s)]

    elif strategy == "Long Straddle":
        K0 = st.number_input("ATM Strike", value=float(K_atm), step=1.0)
        qty = st.number_input("Lots", value=1, step=1)
        prem_c = st.number_input("Call Premium (optional override)", value=0.0, step=0.5, help=help_premium)
        prem_p = st.number_input("Put Premium (optional override)", value=0.0, step=0.5, help=help_premium)
        legs += [dict(kind='C', side=+1, K=K0, qty=qty, premium=prem_c),
                 dict(kind='P', side=+1, K=K0, qty=qty, premium=prem_p)]

    elif strategy == "Long Strangle":
        Kl = st.number_input("Put Strike (lower)", value=float(K_wide1), step=1.0)
        Kh = st.number_input("Call Strike (higher)", value=float(K_wide2), step=1.0)
        qty = st.number_input("Lots", value=1, step=1)
        prem_p = st.number_input("Put Premium (optional override)", value=0.0, step=0.5, help=help_premium)
        prem_c = st.number_input("Call Premium (optional override)", value=0.0, step=0.5, help=help_premium)
        legs += [dict(kind='P', side=+1, K=Kl, qty=qty, premium=prem_p),
                 dict(kind='C', side=+1, K=Kh, qty=qty, premium=prem_c)]

    elif strategy == "Butterfly (Calls)":
        Kl = st.number_input("Lower Call Strike (buy)", value=float(K1), step=1.0)
        K0 = st.number_input("Middle Call Strike (sell 2x)", value=float(K_atm), step=1.0)
        Kh = st.number_input("Higher Call Strike (buy)", value=float(K2), step=1.0)
        qty = st.number_input("Lots (per butterfly)", value=1, step=1)
        prem_l = st.number_input("Premium Lower (optional override)", value=0.0, step=0.5)
        prem_m = st.number_input("Premium Middle (optional override)", value=0.0, step=0.5)
        prem_h = st.number_input("Premium Higher (optional override)", value=0.0, step=0.5)
        legs += [dict(kind='C', side=+1, K=Kl, qty=qty, premium=prem_l),
                 dict(kind='C', side=-2, K=K0, qty=qty, premium=prem_m),
                 dict(kind='C', side=+1, K=Kh, qty=qty, premium=prem_h)]

    elif strategy == "Iron Condor":
        Kl = st.number_input("Put Buy Strike (lower)", value=float(K_wide1), step=1.0)
        Ksl = st.number_input("Put Sell Strike (higher)", value=float(K1), step=1.0)
        Ksh = st.number_input("Call Sell Strike (lower)", value=float(K2), step=1.0)
        Kh = st.number_input("Call Buy Strike (higher)", value=float(K_wide2), step=1.0)
        qty = st.number_input("Lots (per condor)", value=1, step=1)
        prem_bl = st.number_input("Premium Put Buy (optional override)", value=0.0, step=0.5)
        prem_sl = st.number_input("Premium Put Sell (optional override)", value=0.0, step=0.5)
        prem_sh = st.number_input("Premium Call Sell (optional override)", value=0.0, step=0.5)
        prem_bh = st.number_input("Premium Call Buy (optional override)", value=0.0, step=0.5)
        legs += [dict(kind='P', side=+1, K=Kl, qty=qty, premium=prem_bl),
                 dict(kind='P', side=-1, K=Ksl, qty=qty, premium=prem_sl),
                 dict(kind='C', side=-1, K=Ksh, qty=qty, premium=prem_sh),
                 dict(kind='C', side=+1, K=Kh, qty=qty, premium=prem_bh)]

    # Compute premiums (BSM default or overrides)
    def leg_price(leg, S, T, r, q, sigma):
        if override and leg["premium"] > 0:
            return leg["premium"]
        is_call = (leg["kind"] == 'C')
        return bsm_price(is_call, S, leg["K"], T, r, q, sigma)

    # Scenario range
    st.markdown("### 📊 Payoff / P&L")
    s_lo = st.number_input("Min Underlying (₹)", value=max(1.0, S2 * 0.7), step=1.0, key="slo")
    s_hi = st.number_input("Max Underlying (₹)", value=S2 * 1.3, step=1.0, key="shi")
    n_pts = st.slider("Points", 50, 600, 251, step=1, key="npts")

    grid2 = np.linspace(s_lo, s_hi, int(n_pts))

    # Payoff per share at expiry & net premium paid/received
    def option_payoff(kind, side, K, ST):
        if kind == 'C':
            return side * np.maximum(ST - K, 0.0)
        else:
            return side * np.maximum(K - ST, 0.0)

    # Compute net premium (per share) and legs' premiums
    leg_premiums = [leg_price(leg, S2, T2, r2, q2, sigma2) * leg["side"] for leg in legs]
    net_premium_per_share = sum(lp for lp in leg_premiums)
    net_premium_per_lot = net_premium_per_share * lot_size

    payoff_expiry = np.zeros_like(grid2)
    for leg in legs:
        payoff_expiry += option_payoff(leg["kind"], leg["side"] * leg["qty"], leg["K"], grid2)

    pnl_expiry_per_share = payoff_expiry - net_premium_per_share * sum(abs(leg["qty"]) for leg in legs) * 0 + net_premium_per_share
    # Note: We define pnl as payoff minus net premium (entering positions). If you want per-strategy lots, multiply later.

    fig_pay, ax_pay = plt.subplots()
    ax_pay.plot(grid2, payoff_expiry, label="Payoff at Expiry (per share)")
    ax_pay.axhline(0.0, linestyle="--")
    ax_pay.axvline(S2, linestyle="--", label="Spot")
    ax_pay.set_xlabel("Underlying Price at Expiry (₹)")
    ax_pay.set_ylabel("Payoff (₹ per share)")
    ax_pay.legend()
    st.pyplot(fig_pay)

    fig_pnl, ax_pnl = plt.subplots()
    ax_pnl.plot(grid2, pnl_expiry_per_share, label="P&L at Expiry (per share)")
    ax_pnl.axhline(0.0, linestyle="--")
    ax_pnl.axvline(S2, linestyle="--", label="Spot")
    ax_pnl.set_xlabel("Underlying Price at Expiry (₹)")
    ax_pnl.set_ylabel("P&L (₹ per share)")
    ax_pnl.legend()
    if show_dist and T2 > 0 and sigma2 > 0:
        # Lognormal density scaled to 5% of y-range for visualization
        mu = math.log(S2) + (r2 - q2 - 0.5 * sigma2**2) * T2
        sd = sigma2 * math.sqrt(T2)
        pdf = (1/(grid2*sd*math.sqrt(2*math.pi))) * np.exp(-(np.log(grid2)-mu)**2/(2*sd**2))
        y_scale = (ax_pnl.get_ylim()[1] - ax_pnl.get_ylim()[0]) * 0.15 / (pdf.max() if pdf.max() > 0 else 1)
        ax_pnl.plot(grid2, pdf * y_scale + ax_pnl.get_ylim()[0], linestyle=":", label="Expiry PDF (scaled)")
    st.pyplot(fig_pnl)

    # Metrics
    st.markdown("### 🧾 Cost & Breakevens")
    st.write(f"**Net Premium (per share):** {fmt_money(net_premium_per_share)}")
    st.write(f"**Net Premium (per lot @ {lot_size}):** {fmt_money(net_premium_per_lot)}")

    # Breakeven estimation (roots of P&L curve)
    pnl_vals = pnl_expiry_per_share
    sign = np.sign(pnl_vals - 0.0)
    be_points = []
    for i in range(1, len(grid2)):
        if sign[i] == 0:
            be_points.append(grid2[i])
        elif sign[i] * sign[i-1] < 0:
            # linear interpolate
            x0, x1 = grid2[i-1], grid2[i]
            y0, y1 = pnl_vals[i-1], pnl_vals[i]
            if (y1 - y0) != 0:
                x_be = x0 - y0 * (x1 - x0) / (y1 - y0)
                be_points.append(x_be)

    if be_points:
        st.write("**Breakevens (approx):** " + ", ".join([f"₹{x:,.2f}" for x in be_points]))
    else:
        st.write("**Breakevens:** Not found in selected range.")

    # Download
    df_strategy = pd.DataFrame({"S_T": grid2,
                                "Payoff_per_share": payoff_expiry,
                                "PnL_per_share": pnl_expiry_per_share})
    st.download_button("⬇️ Download Strategy Curve CSV", df_strategy.to_csv(index=False).encode("utf-8"),
                       "strategy_curve.csv", "text/csv")

st.divider()
st.caption("Made for Indian markets. Defaults are illustrative; update lot sizes & rates per latest NSE circulars. This tool is educational, not investment advice.")
''')

reqs = dedent('''
streamlit>=1.36.0
numpy>=1.26.4
pandas>=2.2.2
scipy>=1.13.1
matplotlib>=3.9.0
pytz>=2024.1
''')

with open('/mnt/data/app_options_india.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

with open('/mnt/data/requirements.txt', 'w', encoding='utf-8') as f:
    f.write(reqs)

('/mnt/data/app_options_india.py', '/mnt/data/requirements.txt')
