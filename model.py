# # app_bsm_india.py
# Streamlit — Black–Scholes–Merton Pricer (India) with flexible chart sizing, colors, and layouts
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

# Optional Plotly (interactive)
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Optional Yahoo delayed LTP
try:
    import yfinance as yf
except Exception:
    yf = None

# -----------------------------------
# Time & day-count
# -----------------------------------
IST = pytz.timezone("Asia/Kolkata")

def to_ist_today() -> date:
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    return now_utc.astimezone(IST).date()

def year_fraction(start: date, end: date, basis: str = "ACT/365") -> float:
    if end <= start:
        return 1e-8
    days = (end - start).days
    if basis.upper() in ("ACT/365", "ACT/365F", "ACT/365FIXED"):
        return days / 365.0
    elif basis.upper() == "ACT/360":
        return days / 360.0
    return days / 365.0

# -----------------------------------
# BSM core
# -----------------------------------
def bsm_price(is_call: bool, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if is_call:
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

def bsm_greeks(is_call: bool, S: float, K: float, T: float, r: float, q: float, sigma: float):
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
    rho  = K * T * math.exp(-r * T) * (norm.cdf(d2) if is_call else -norm.cdf(-d2))
    return dict(delta=delta, gamma=gamma, theta=theta_day, vega=vega, rho=rho)

def no_arb_bounds(is_call: bool, S: float, K: float, T: float, r: float, q: float):
    disc_r = math.exp(-r * T); disc_q = math.exp(-q * T)
    if is_call:
        return max(S * disc_q - K * disc_r, 0.0), S * disc_q
    else:
        return max(K * disc_r - S * disc_q, 0.0), K * disc_r

def implied_vol_from_price(is_call: bool, price: float, S: float, K: float, T: float, r: float, q: float,
                           lo: float = 1e-6, hi: float = 5.0) -> float:
    lb, ub = no_arb_bounds(is_call, S, K, T, r, q)
    if not (lb - 1e-8 <= price <= ub + 1e-8):
        return np.nan
    def f(sig): return bsm_price(is_call, S, K, T, r, q, sig) - price
    try:
        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi > 0:
            hi2 = max(hi, 10.0)
            if f(lo) * f(hi2) > 0: return np.nan
            return brentq(f, lo, hi2, maxiter=600, xtol=1e-10)
        return brentq(f, lo, hi, maxiter=600, xtol=1e-10)
    except Exception:
        return np.nan

def fmt_money(x: float) -> str:
    try: return f"₹{x:,.2f}"
    except Exception: return f"₹{x}"

# -----------------------------------
# Delayed Yahoo LTP
# -----------------------------------
@st.cache_data(show_spinner=False, ttl=30)
def fetch_yahoo_ltp(symbol: str) -> float | None:
    if yf is None:
        return None
    try:
        t = yf.Ticker(symbol)
        price = t.fast_info.get("last_price", None)
        if price is None:
            hist = t.history(period="1d", interval="1m")
            if len(hist):
                price = float(hist["Close"].dropna().iloc[-1])
        return float(price) if price is not None else None
    except Exception:
        return None

# -----------------------------------
# UI
# -----------------------------------
st.set_page_config(page_title="BSM Options (India) — Pricer", layout="wide")
st.title("🧮 BSM Options (India) — Pricer")
st.caption("BSM with Greeks, IV solver, scenarios, and flexible visualization controls (IST).")

# Sidebar: Market & Visual Controls
with st.sidebar:
    st.header("⚙️ Global Settings")
    lot_size = st.number_input("Lot Size (contracts)", min_value=1, value=50, step=1)
    basis    = st.selectbox("Day-count basis", ["ACT/365", "ACT/360"], index=0)
    r_pct    = st.number_input("Risk-free rate (annual, %)", value=7.00, step=0.25, format="%.2f")
    q_pct    = st.number_input("Dividend yield (annual, %)", value=1.00, step=0.25, format="%.2f")
    r, q = r_pct/100.0, q_pct/100.0

    st.markdown("---")
    st.subheader("🟡 Delayed Spot (Yahoo)")
    y_symbol   = st.text_input("Symbol (e.g., RELIANCE.NS / RELIANCE.BO)", value="RELIANCE.NS")
    use_yahoo  = st.checkbox("Use delayed Yahoo LTP for Spot (S)", value=False)
    auto_refresh = st.checkbox("Auto-refresh (~30s cache)", value=True)

    st.markdown("---")
    st.subheader("🎨 Visualization")
    renderer = st.selectbox("Renderer", ["Matplotlib (static)", "Plotly (interactive)"], index=0,
                            help="Plotly supports zoom/pan & image export. Requires plotly installed.")
    layout_mode = st.selectbox("Layout", ["Single charts (stacked)", "1×5 strip", "2×3 grid"], index=2)
    use_container_width = st.checkbox("Fit width to container", value=True)
    w_in  = st.slider("Figure width (inches)", 5.0, 18.0, 12.0, 0.5)
    h_in  = st.slider("Figure height (inches)", 3.0, 8.0, 3.6, 0.2)
    dpi   = st.slider("DPI (export clarity)", 80, 240, 120, 10)
    lw    = st.slider("Line width", 1.0, 4.0, 2.0, 0.1)
    ls    = st.selectbox("Line style", ["solid", "dashed", "dotted", "dashdot"], index=0)
    show_grid = st.checkbox("Show gridlines", value=True)
    show_legend = st.checkbox("Show legends", value=True)

    st.caption("Use color pickers below to customize each series.")
    colc1, colc2 = st.columns(2)
    with colc1:
        c_opt  = st.color_picker("Option value color", "#1f77b4")
        c_d    = st.color_picker("Delta (Δ) color", "#2ca02c")
        c_g    = st.color_picker("Gamma (Γ) color", "#d62728")
    with colc2:
        c_t    = st.color_picker("Theta/day (Θ) color", "#9467bd")
        c_v    = st.color_picker("Vega color", "#ff7f0e")
        c_r    = st.color_picker("Rho (ρ) color", "#17becf")

# Inputs (left) & scenario controls (right)
colL, colR = st.columns([1.25, 1])

with colL:
    st.subheader("Inputs")
    yahoo_ltp = fetch_yahoo_ltp(y_symbol) if use_yahoo else None
    default_S = float(yahoo_ltp) if (use_yahoo and yahoo_ltp) else 24000.0
    S = st.number_input("Spot Price (₹)", min_value=0.01, value=default_S, step=1.0)
    if use_yahoo:
        st.write(f"Yahoo LTP: {fmt_money(yahoo_ltp) if yahoo_ltp else '—'}")

    K = st.number_input("Strike (₹)", min_value=0.01, value=24000.0, step=1.0)
    is_call = (st.selectbox("Option Type", ["Call", "Put"]) == "Call")

    today = to_ist_today()
    exp   = st.date_input("Expiry Date (IST)", value=today + timedelta(days=30), min_value=today)
    T     = year_fraction(today, exp, basis=basis)

    iv_mode = st.radio("Volatility Input", ["Use σ (IV)", "Solve σ from Market Price"], horizontal=True)
    if iv_mode == "Use σ (IV)":
        sigma = st.number_input("Implied Volatility σ (annual, %)", min_value=0.01,
                                value=15.0, step=0.5, format="%.2f") / 100.0
        price_mkt = None
    else:
        price_mkt = st.number_input("Market Option Price (₹ per share)", min_value=0.0, value=200.0, step=1.0)
        sigma = implied_vol_from_price(is_call, price_mkt, S, K, T, r, q)
        if np.isnan(sigma):
            lb, ub = no_arb_bounds(is_call, S, K, T, r, q)
            st.error(f"Could not solve IV. No-arb bounds: [{fmt_money(lb)}, {fmt_money(ub)}]")
            sigma = 0.0

    # Price & Greeks
    price = bsm_price(is_call, S, K, T, r, q, sigma) if sigma > 0 else (price_mkt or 0.0)
    greeks = bsm_greeks(is_call, S, K, T, r, q, sigma) if sigma > 0 else dict(
        delta=np.nan, gamma=np.nan, theta=np.nan, vega=np.nan, rho=np.nan
    )

    st.markdown("### 📜 Results (per share)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Option Price", fmt_money(price))
    c2.metric("Delta (Δ)",  f"{greeks['delta']:.4f}" if greeks['delta']==greeks['delta'] else "—")
    c3.metric("Gamma (Γ)",  f"{greeks['gamma']:.6f}" if greeks['gamma']==greeks['gamma'] else "—")
    c4.metric("Theta/day",  f"{greeks['theta']:.4f}" if greeks['theta']==greeks['theta'] else "—")
    c1b, c2b, c3b, c4b = st.columns(4)
    c1b.metric("Vega (1.0 σ)", f"{greeks['vega']:.4f}" if greeks['vega']==greeks['vega'] else "—")
    c2b.metric("Rho (ρ)",      f"{greeks['rho']:.4f}"  if greeks['rho']==greeks['rho'] else "—")
    c3b.metric("Per-Lot Price", fmt_money(price * lot_size))
    if price_mkt is not None and sigma > 0:
        c4b.metric("Theo − Market", fmt_money(price - price_mkt))

with colR:
    st.subheader("Scenario Settings")
    s_min = st.number_input("Min Underlying (₹)", value=max(1.0, S * 0.7), step=1.0)
    s_max = st.number_input("Max Underlying (₹)", value=S * 1.3, step=1.0)
    n_pts = st.slider("Points", 50, 600, 251, step=1)

# Auto-refresh note (cache ttl drives updates)
if use_yahoo and auto_refresh:
    pass  # cache ttl=30s keeps it fresh; Streamlit reruns on interactions.

# Build scenario
grid = np.linspace(float(s_min), float(s_max), int(n_pts))
if sigma > 0:
    vals = np.array([bsm_price(is_call, s, K, T, r, q, sigma) for s in grid])
    G = {"Delta": [], "Gamma": [], "Theta/day": [], "Vega": [], "Rho": []}
    for s in grid:
        g = bsm_greeks(is_call, s, K, T, r, q, sigma)
        G["Delta"].append(g["delta"]);  G["Gamma"].append(g["gamma"])
        G["Theta/day"].append(g["theta"]);  G["Vega"].append(g["vega"]);  G["Rho"].append(g["rho"])
else:
    vals = np.array([np.nan]*len(grid))
    G = {k: [np.nan]*len(grid) for k in ["Delta","Gamma","Theta/day","Vega","Rho"]}

# -----------------------------------
# PLOTTING HELPERS
# -----------------------------------
def mpl_make_fig():
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    return fig

def mpl_style(ax):
    if show_grid: ax.grid(True, alpha=0.3)
    if show_legend: ax.legend()

def draw_mpl_single():
    # Option value
    fig = mpl_make_fig()
    ax = fig.add_subplot(111)
    ax.plot(grid, vals, color=c_opt, linewidth=lw, linestyle=ls, label="Option Value")
    ax.axvline(S, color="#666666", linestyle="--", linewidth=1, label="Spot")
    ax.axvline(K, color="#999999", linestyle=":",  linewidth=1, label="Strike")
    ax.set_xlabel("Underlying Price (₹)"); ax.set_ylabel("Option Value (₹ per share)")
    mpl_style(ax)
    st.pyplot(fig, use_container_width=use_container_width)

    # Greeks stacked
    for name, col in [("Delta", c_d), ("Gamma", c_g), ("Theta/day", c_t), ("Vega", c_v), ("Rho", c_r)]:
        fig = mpl_make_fig()
        ax = fig.add_subplot(111)
        ax.plot(grid, G[name], color=col, linewidth=lw, linestyle=ls, label=name)
        ax.axvline(S, color="#666666", linestyle="--", linewidth=1, label="Spot")
        ax.axvline(K, color="#999999", linestyle=":",  linewidth=1, label="Strike")
        ax.set_xlabel("Underlying Price (₹)"); ax.set_ylabel(name)
        mpl_style(ax)
        st.pyplot(fig, use_container_width=use_container_width)

def draw_mpl_strip():
    # 1x5 strip (may be wide; use container width)
    names_cols = [("Delta", c_d), ("Gamma", c_g), ("Theta/day", c_t), ("Vega", c_v), ("Rho", c_r)]
    st.markdown("### 🧪 Greeks — strip")
    cols = st.columns(5)
    for (name, colc), host in zip(names_cols, cols):
        with host:
            fig = mpl_make_fig()
            ax = fig.add_subplot(111)
            ax.plot(grid, G[name], color=colc, linewidth=lw, linestyle=ls, label=name)
            ax.axvline(S, color="#666666", linestyle="--", linewidth=1, label="Spot")
            ax.axvline(K, color="#999999", linestyle=":",  linewidth=1, label="Strike")
            ax.set_xlabel("S (₹)"); ax.set_ylabel(name)
            mpl_style(ax)
            st.pyplot(fig, use_container_width=True)

def draw_mpl_grid():
    # 2x3 grid: Option value + 5 greeks
    st.markdown("### 📊 All charts — 2×3 grid")
    rows = [st.columns(3), st.columns(3)]
    # (0,0) Option value
    with rows[0][0]:
        fig = mpl_make_fig()
        ax = fig.add_subplot(111)
        ax.plot(grid, vals, color=c_opt, linewidth=lw, linestyle=ls, label="Option Value")
        ax.axvline(S, color="#666666", linestyle="--", linewidth=1, label="Spot")
        ax.axvline(K, color="#999999", linestyle=":",  linewidth=1, label="Strike")
        ax.set_xlabel("S (₹)"); ax.set_ylabel("Value (₹/sh)")
        mpl_style(ax)
        st.pyplot(fig, use_container_width=True)

    targets = [("Delta", c_d), ("Gamma", c_g), ("Theta/day", c_t), ("Vega", c_v), ("Rho", c_r)]
    positions = [(0,1), (0,2), (1,0), (1,1), (1,2)]
    for (name, color), (ri, ci) in zip(targets, positions):
        with rows[ri][ci]:
            fig = mpl_make_fig()
            ax = fig.add_subplot(111)
            ax.plot(grid, G[name], color=color, linewidth=lw, linestyle=ls, label=name)
            ax.axvline(S, color="#666666", linestyle="--", linewidth=1, label="Spot")
            ax.axvline(K, color="#999999", linestyle=":",  linewidth=1, label="Strike")
            ax.set_xlabel("S (₹)"); ax.set_ylabel(name)
            mpl_style(ax)
            st.pyplot(fig, use_container_width=True)

def draw_plotly_all():
    # Option value
    st.markdown("### 📊 Option Value vs Underlying")
    figv = go.Figure()
    figv.add_trace(go.Scatter(x=grid, y=vals, name="Option Value", line=dict(color=c_opt, width=lw)))
    figv.add_vline(x=S, line_dash="dash", line_color="#666666", annotation_text="Spot")
    figv.add_vline(x=K, line_dash="dot",  line_color="#999999", annotation_text="Strike")
    figv.update_layout(height=int(h_in*80), width=None, template="plotly_white" if show_grid else None,
                       showlegend=show_legend, margin=dict(l=10,r=10,t=40,b=10))
    figv.update_xaxes(title="Underlying Price (₹)", showgrid=show_grid)
    figv.update_yaxes(title="Option Value (₹ per share)", showgrid=show_grid)
    st.plotly_chart(figv, use_container_width=use_container_width)

    # Greeks — choose layout
    figs = []
    for name, color in [("Delta", c_d), ("Gamma", c_g), ("Theta/day", c_t), ("Vega", c_v), ("Rho", c_r)]:
        f = go.Figure()
        f.add_trace(go.Scatter(x=grid, y=G[name], name=name, line=dict(color=color, width=lw)))
        f.add_vline(x=S, line_dash="dash", line_color="#666666")
        f.add_vline(x=K, line_dash="dot",  line_color="#999999")
        f.update_layout(height=int(h_in*80), template="plotly_white" if show_grid else None,
                        showlegend=show_legend, margin=dict(l=10,r=10,t=40,b=10))
        f.update_xaxes(title="Underlying Price (₹)", showgrid=show_grid)
        f.update_yaxes(title=name, showgrid=show_grid)
        figs.append(f)

    if layout_mode == "1×5 strip":
        cols = st.columns(5)
        for f, host in zip(figs, cols):
            with host:
                st.plotly_chart(f, use_container_width=True)
    elif layout_mode == "2×3 grid":
        rows = [st.columns(3), st.columns(3)]
        order = [0,1,2,3,4]
        pos = [(0,0),(0,1),(0,2),(1,0),(1,1)]
        for idx, (ri,ci) in zip(order, pos):
            with rows[ri][ci]:
                st.plotly_chart(figs[idx], use_container_width=True)
    else:
        for f in figs:
            st.plotly_chart(f, use_container_width=use_container_width)

# -----------------------------------
# Render charts
# -----------------------------------
# First: Option value main chart (for Matplotlib single/strip/grid we’ll include it in those sections)
if renderer.startswith("Plotly"):
    if not PLOTLY_OK:
        st.error("Plotly not installed. Add `plotly>=5.24` to requirements.txt or switch to Matplotlib.")
    else:
        draw_plotly_all()
else:
    # Matplotlib
    st.markdown("### 📊 Option Value vs Underlying")
    fig0 = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    ax0 = fig0.add_subplot(111)
    ax0.plot(grid, vals, color=c_opt, linewidth=lw, linestyle=ls, label="Option Value")
    ax0.axvline(S, color="#666666", linestyle="--", linewidth=1, label="Spot")
    ax0.axvline(K, color="#999999", linestyle=":",  linewidth=1, label="Strike")
    ax0.set_xlabel("Underlying Price (₹)"); ax0.set_ylabel("Option Value (₹ per share)")
    if show_grid: ax0.grid(True, alpha=0.3)
    if show_legend: ax0.legend()
    st.pyplot(fig0, use_container_width=use_container_width)

    # Greek layouts
    if layout_mode == "Single charts (stacked)":
        draw_mpl_single()
    elif layout_mode == "1×5 strip":
        draw_mpl_strip()
    else:
        draw_mpl_grid()

# -----------------------------------
# Download data
# -----------------------------------
df = pd.DataFrame({
    "S": grid,
    "OptionValue": vals,
    "Delta": G["Delta"],
    "Gamma": G["Gamma"],
    "Theta_per_day": G["Theta/day"],
    "Vega": G["Vega"],
    "Rho": G["Rho"],
})
st.download_button("⬇️ Download Scenario CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name="bsm_scenario.csv", mime="text/csv")

st.caption("Made for Indian markets. Yahoo prices are delayed ~15–20 minutes. Educational use — not investment advice.")
