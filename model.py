# app_bsm_india.py
# Streamlit — Black–Scholes–Merton Pricer (India)
# Author: ChatGPT (for Paramjeet)
#
# How to run:
#   pip install -r requirements.txt
#   streamlit run app_bsm_india.py

import math
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# -------------------------------
# Time / Day-count utilities
# -------------------------------
IST = pytz.timezone("Asia/Kolkata")

def to_ist_today() -> date:
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    return now_utc.astimezone(IST).date()

def year_fraction(start: date, end: date, basis: str = "ACT/365") -> float:
    """Simple year fraction calculator."""
    if end <= start:
        return 1e-8  # avoid zero-div/NaN
    days = (end - start).days
    if basis.upper() in ("ACT/365", "ACT/365F", "ACT/365FIXED"):
        return days / 365.0
    elif basis.upper() == "ACT/360":
        return days / 360.0
    return days / 365.0

# -------------------------------
# BSM Pricing & Greeks
# -------------------------------
def bsm_price(is_call: bool, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Black–Scholes–Merton price with continuous dividend yield (q).
    Returns theoretical price per share.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # At/near expiry with non-positive sigma -> intrinsic
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if is_call:
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

def bsm_greeks(is_call: bool, S: float, K: float, T: float, r: float, q: float, sigma: float):
    """
    Returns Greeks dict: delta, gamma, theta (per calendar day), vega, rho.
    Vega is per 1.0 change in annualized sigma (multiply by 0.01 for 1 vol point).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        delta = 1.0 if (is_call and S > K) else (-1.0 if (not is_call and S < K) else 0.0)
        return dict(delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)

    delta = math.exp(-q * T) * (norm.cdf(d1) if is_call else (norm.cdf(d1) - 1))
    gamma = (math.exp(-q * T) * pdf_d1) / (S * sigma * sqrtT)
    theta_year = (
        -(S * math.exp(-q * T) * pdf_d1 * sigma) / (2 * sqrtT)
        - (r * K * math.exp(-r * T) * (norm.cdf(d2) if is_call else norm.cdf(-d2)))
        + (q * S * math.exp(-q * T) * (norm.cdf(d1) if is_call else norm.cdf(-d1)))
    )
    theta_day = theta_year / 365.0
    vega = S * math.exp(-q * T) * pdf_d1 * sqrtT
    rho = K * T * math.exp(-r * T) * (norm.cdf(d2) if is_call else -norm.cdf(-d2))

    return dict(delta=delta, gamma=gamma, theta=theta_day, vega=vega, rho=rho)

# -------------------------------
# No-arbitrage bounds (for sanity checks / IV)
# -------------------------------
def no_arb_bounds(is_call: bool, S: float, K: float, T: float, r: float, q: float):
    """Return (lower_bound, upper_bound) for call/put price under continuous yields."""
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if is_call:
        lower = max(S * disc_q - K * disc_r, 0.0)
        upper = S * disc_q
    else:
        lower = max(K * disc_r - S * disc_q, 0.0)
        upper = K * disc_r
    return lower, upper

# -------------------------------
# Implied Volatility Solver
# -------------------------------
def implied_vol_from_price(is_call: bool, price: float, S: float, K: float, T: float, r: float, q: float,
                           lo: float = 1e-6, hi: float = 5.0) -> float:
    """
    Solve for sigma using Brent's method. Returns np.nan if:
    - Market price is outside no-arbitrage bounds, or
    - No root found in [lo, hi].
    """
    lb, ub = no_arb_bounds(is_call, S, K, T, r, q)
    if not (lb - 1e-8 <= price <= ub + 1e-8):
        return np.nan

    def f(sig):
        return bsm_price(is_call, S, K, T, r, q, sig) - price

    try:
        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi > 0:
            # Try expanding the bracket a bit
            hi2 = max(hi, 10.0)
            if f(lo) * f(hi2) > 0:
                return np.nan
            return brentq(f, lo, hi2, maxiter=600, xtol=1e-10)
        return brentq(f, lo, hi, maxiter=600, xtol=1e-10)
    except Exception:
        return np.nan

# -------------------------------
# Formatting helpers
# -------------------------------
def fmt_money(x: float) -> str:
    try:
        return f"₹{x:,.2f}"
    except Exception:
        return f"₹{x}"

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="BSM Options (India) — Pricer", layout="wide")
st.title("🧮 BSM Options (India) — Pricer")
st.caption("Robust Black–Scholes–Merton pricer with Greeks, IV solver, scenarios, and per-lot values (IST).")

# Sidebar — Global settings
with st.sidebar:
    st.header("⚙️ Global Settings")
    lot_size = st.number_input("Lot Size (contracts)", min_value=1, value=50, step=1,
                               help="E.g., NIFTY lot size ~50; update per latest NSE circulars.")
    basis = st.selectbox("Day-count basis", ["ACT/365", "ACT/360"], index=0)
    r_pct = st.number_input("Risk-free rate (annual, %)", value=7.00, step=0.25, format="%.2f")
    q_pct = st.number_input("Dividend yield (annual, %)", value=1.00, step=0.25, format="%.2f")
    r, q = r_pct / 100.0, q_pct / 100.0
    st.info("All prices shown are per share; per-lot uses the lot size above.")

# Inputs
colL, colR = st.columns([1.25, 1])
with colL:
    st.subheader("Inputs")
    S = st.number_input("Spot Price (₹)", min_value=0.01, value=24000.0, step=1.0)
    K = st.number_input("Strike (₹)", min_value=0.01, value=24000.0, step=1.0)
    is_call = (st.selectbox("Option Type", ["Call", "Put"]) == "Call")

    today = to_ist_today()
    exp = st.date_input("Expiry Date (IST)", value=today + timedelta(days=30), min_value=today)
    T = year_fraction(today, exp, basis=basis)

    iv_mode = st.radio("Volatility Input", ["Use σ (IV)", "Solve σ from Market Price"], horizontal=True)
    if iv_mode == "Use σ (IV)":
        sigma = st.number_input("Implied Volatility σ (annual, %)", min_value=0.01, value=15.0, step=0.5, format="%.2f") / 100.0
        price_mkt = None
    else:
        price_mkt = st.number_input("Market Option Price (₹ per share)", min_value=0.0, value=200.0, step=1.0)
        sigma = implied_vol_from_price(is_call, price_mkt, S, K, T, r, q)
        if np.isnan(sigma):
            lb, ub = no_arb_bounds(is_call, S, K, T, r, q)
            st.error(
                "Could not solve for implied volatility. "
                f"Check inputs. No-arb bounds: [{fmt_money(lb)}, {fmt_money(ub)}]"
            )
            sigma = 0.0  # prevent downstream NaNs in charts

    # Price & Greeks
    price = bsm_price(is_call, S, K, T, r, q, sigma) if sigma > 0 else (price_mkt or 0.0)
    greeks = bsm_greeks(is_call, S, K, T, r, q, sigma) if sigma > 0 else dict(
        delta=np.nan, gamma=np.nan, theta=np.nan, vega=np.nan, rho=np.nan
    )

    st.markdown("### 📜 Results (per share)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Option Price", fmt_money(price))
    c2.metric("Delta (Δ)", f"{greeks['delta']:.4f}" if greeks['delta']==greeks['delta'] else "—")
    c3.metric("Gamma (Γ)", f"{greeks['gamma']:.6f}" if greeks['gamma']==greeks['gamma'] else "—")
    c4.metric("Theta/day (Θ)", f"{greeks['theta']:.4f}" if greeks['theta']==greeks['theta'] else "—")

    c1b, c2b, c3b, c4b = st.columns(4)
    c1b.metric("Vega (per 1.0 σ)", f"{greeks['vega']:.4f}" if greeks['vega']==greeks['vega'] else "—")
    c2b.metric("Rho (ρ)", f"{greeks['rho']:.4f}" if greeks['rho']==greeks['rho'] else "—")
    c3b.metric("Per-Lot Price", fmt_money(price * lot_size))
    if price_mkt is not None and sigma > 0:
        c4b.metric("Theo − Market", fmt_money(price - price_mkt))

with colR:
    st.subheader("Scenario Settings")
    s_min = st.number_input("Min Underlying (₹)", value=max(1.0, S * 0.7), step=1.0)
    s_max = st.number_input("Max Underlying (₹)", value=S * 1.3, step=1.0)
    n_pts = st.slider("Points", 50, 600, 251, step=1)

# Scenario grid
grid = np.linspace(float(s_min), float(s_max), int(n_pts))
if sigma > 0:
    values = [bsm_price(is_call, s, K, T, r, q, sigma) for s in grid]
    gr_series = {"Delta": [], "Gamma": [], "Theta/day": [], "Vega": [], "Rho": []}
    for s in grid:
        g = bsm_greeks(is_call, s, K, T, r, q, sigma)
        gr_series["Delta"].append(g["delta"])
        gr_series["Gamma"].append(g["gamma"])
        gr_series["Theta/day"].append(g["theta"])
        gr_series["Vega"].append(g["vega"])
        gr_series["Rho"].append(g["rho"])
else:
    values = [np.nan] * len(grid)
    gr_series = {"Delta": [np.nan]*len(grid), "Gamma": [np.nan]*len(grid),
                 "Theta/day": [np.nan]*len(grid), "Vega": [np.nan]*len(grid), "Rho": [np.nan]*len(grid)}

# Charts
st.markdown("### 📊 Option Value vs Underlying")
fig1, ax1 = plt.subplots()
ax1.plot(grid, values, label="Option Value")
ax1.axvline(S, linestyle="--", label="Spot")
ax1.axvline(K, linestyle=":", label="Strike")
ax1.set_xlabel("Underlying Price (₹)")
ax1.set_ylabel("Option Value (₹ per share)")
ax1.legend()
st.pyplot(fig1)

st.markdown("### 🧪 Greeks vs Underlying")
for name in ["Delta", "Gamma", "Theta/day", "Vega", "Rho"]:
    fig, ax = plt.subplots()
    ax.plot(grid, gr_series[name], label=name)
    ax.axvline(S, linestyle="--", label="Spot")
    ax.axvline(K, linestyle=":", label="Strike")
    ax.set_xlabel("Underlying Price (₹)")
    ax.set_ylabel(name)
    ax.legend()
    st.pyplot(fig)

# Export
df = pd.DataFrame({
    "S": grid,
    "OptionValue": values,
    "Delta": gr_series["Delta"],
    "Gamma": gr_series["Gamma"],
    "Theta_per_day": gr_series["Theta/day"],
    "Vega": gr_series["Vega"],
    "Rho": gr_series["Rho"],
})
st.download_button("⬇️ Download Scenario CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name="bsm_scenario.csv", mime="text/csv")

st.caption("Made for Indian markets. Educational only — not investment advice. Update lot sizes & rates per latest NSE circulars.")
