# pages/2_Options_Arcade_Game.py
# 🎮 Options Arcade — a simple, visual options trading game (Indian style)
#
# Play rounds, place call/put trades at strikes near spot, watch P&L as spot evolves.
# Uses BSM for option pricing (European), IST day-count, lot sizes, strike steps, and simple costs.
#
# Requirements (already in your app):
#   streamlit, numpy, pandas, scipy, plotly, pytz
# Optional (for delayed spot lookup): yfinance

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import pytz
import streamlit as st
from scipy.stats import norm
import plotly.graph_objects as go

# Optional Yahoo delayed prices
try:
    import yfinance as yf
except Exception:
    yf = None

# =========================
# Helpers: time & finance
# =========================
IST = pytz.timezone("Asia/Kolkata")

def to_ist_today() -> date:
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    return now_utc.astimezone(IST).date()

def year_fraction_days(days: float, basis: str = "ACT/365") -> float:
    if days <= 0:
        return 1e-8
    if basis.upper().startswith("ACT/360"):
        return days / 360.0
    return days / 365.0

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

def fmt_money(x: float) -> str:
    try:
        return f"₹{x:,.2f}"
    except Exception:
        return f"₹{x}"

@st.cache_data(show_spinner=False, ttl=45)
def yahoo_ltp(symbol: str) -> Optional[float]:
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

# =========================
# Game Data Structures
# =========================
@dataclass
class Position:
    kind: str               # 'C' or 'P'
    side: int               # +1 buy (long), -1 sell (short)
    K: float                # strike
    lots: int               # lots (each lot has lot_size units)
    lot_size: int
    open_price: float       # premium per share at entry
    sigma: float            # IV assumed at entry (can reprice using current IV if you want)
    r: float
    q: float
    ttm_days: float         # time to maturity in days (decreases each turn)
    id: int = field(default_factory=lambda: random.randint(100000, 999999))

    def value_mark(self, S: float, basis: str) -> float:
        T = year_fraction_days(max(self.ttm_days, 0.0), basis)
        is_call = (self.kind == 'C')
        px = bsm_price(is_call, S, self.K, T, self.r, self.q, self.sigma)
        return px * self.lots * self.lot_size * self.side  # signed mark-to-market value

    def payoff_expire(self, S: float) -> float:
        intrinsic = max(S - self.K, 0.0) if self.kind == 'C' else max(self.K - S, 0.0)
        return intrinsic * self.lots * self.lot_size * self.side  # signed payout at expiry

# =========================
# Streamlit Page Setup
# =========================
st.set_page_config(page_title="🎮 Options Arcade", layout="wide")
st.title("🎮 Options Arcade — Learn Options by Playing")

st.caption(
    "Buy & sell calls/puts, watch **spot** evolve, and manage your P&L. "
    "European options (BSM), IST calendar, Indian lot sizes & simple costs. "
    "This is a **game** for learning — not investment advice."
)

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("⚙️ Game Settings")
    seed = st.number_input("Random seed", value=42, step=1)
    st.session_state.setdefault("seed", seed)
    random.seed(seed); np.random.seed(seed)

    # Live/Delayed Spot Setup
    st.subheader("🟡 Spot Source (Delayed OK)")
    spot_source = st.selectbox("Spot Source", ["Manual", "Yahoo (delayed)"], index=1)
    symbol = st.text_input("Yahoo symbol", value="RELIANCE.NS")
    manual_spot = st.number_input("Manual Spot (₹)", min_value=1.0, value=24000.0, step=1.0)

    # Market model
    st.subheader("📈 Underlying Model")
    mu_annual = st.slider("Drift μ (annual, %)", -10.0, 15.0, 5.0, 0.5)
    vol_annual = st.slider("Underlying vol σ (annual, %)", 5.0, 60.0, 20.0, 1.0)
    dt_minutes = st.slider("Minutes per turn", 1, 120, 15, 1)
    basis = st.selectbox("Day-count basis", ["ACT/365", "ACT/360"], index=0)

    # Option market (pricing inputs)
    st.subheader("🧮 Option Market (BSM)")
    r_pct = st.number_input("Risk-free (annual, %)", value=7.00, step=0.25, format="%.2f")
    q_pct = st.number_input("Dividend yield (annual, %)", value=1.00, step=0.25, format="%.2f")
    iv_pct = st.slider("Implied Volatility σ (annual, %)", 5.0, 80.0, 20.0, 1.0)
    strike_step = st.selectbox("Strike step (₹)", [10, 25, 50, 100], index=2)
    lot_size = st.number_input("Lot size", min_value=1, value=50, step=1)

    # Costs (simplified Indian style)
    st.subheader("💸 Costs (per leg)")
    brokerage = st.number_input("Brokerage (₹/lot)", value=20.0, step=1.0)
    stt_rate_sell = st.number_input("STT (sell only, % of premium)", value=0.05, step=0.01, format="%.2f")
    gst_rate = st.number_input("GST (on brokerage, %)", value=18.0, step=1.0, format="%.1f")
    stamp_rate_buy = st.number_input("Stamp duty (buy only, % of premium)", value=0.003, step=0.001, format="%.3f")

    # Theme / visuals
    st.subheader("🎨 Visuals")
    theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
    gridlines = st.checkbox("Show gridlines", value=True)

# =========================
# Session State
# =========================
def reset_game(S0: float):
    st.session_state["spot"] = float(S0)
    st.session_state["t_minutes"] = 0
    st.session_state["cash"] = 1_000_000.0  # starting cash
    st.session_state["nav_history"] = []
    st.session_state["spot_history"] = [float(S0)]
    st.session_state["time_history"] = [0]
    st.session_state["positions"] = []  # list[Position]
    st.session_state["round"] = 0
    st.session_state["game_on"] = True
    st.session_state["best_nav"] = st.session_state.get("best_nav", st.session_state["cash"])

if "game_on" not in st.session_state:
    # initialize with current spot
    S0 = yahoo_ltp(symbol) if (spot_source == "Yahoo (delayed)") else manual_spot
    reset_game(S0 or manual_spot)

# Resolve current spot
current_spot = yahoo_ltp(symbol) if (spot_source == "Yahoo (delayed)") else None
if current_spot:
    st.session_state["spot"] = float(current_spot)
S = float(st.session_state["spot"])

# =========================
# UI: Top KPIs
# =========================
def portfolio_value() -> float:
    r, q, iv = r_pct/100.0, q_pct/100.0, iv_pct/100.0
    total = st.session_state["cash"]
    for pos in st.session_state["positions"]:
        # mark using current IV & rates but remaining T
        T = year_fraction_days(max(pos.ttm_days, 0.0), basis)
        px = bsm_price(pos.kind == 'C', S, pos.K, T, r, q, iv)
        total += px * pos.lot_size * pos.lots * pos.side
    return float(total)

nav = portfolio_value()
st.session_state["best_nav"] = max(st.session_state.get("best_nav", nav), nav)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Spot (₹)", fmt_money(S))
kpi2.metric("Cash", fmt_money(st.session_state["cash"]))
kpi3.metric("Portfolio NAV", fmt_money(nav))
kpi4.metric("Best NAV", fmt_money(st.session_state["best_nav"]))

# =========================
# Trade ticket
# =========================
st.markdown("## 🎯 Trade Ticket")

colA, colB, colC, colD, colE = st.columns([1,1,1,1,1])
with colA:
    side = st.selectbox("Side", ["Buy", "Sell"], index=0)
with colB:
    kind = st.selectbox("Option", ["Call", "Put"], index=0)
with colC:
    # nearest strike aligned to step
    baseK = int(round(S / strike_step) * strike_step)
    K = st.number_input("Strike (₹)", value=float(baseK), step=float(strike_step))
with colD:
    lots = st.number_input("Lots", min_value=1, value=1, step=1)
with colE:
    ttm_days = st.selectbox("Expiry", [7, 14, 30], index=0)

r, q, iv = r_pct/100.0, q_pct/100.0, iv_pct/100.0
is_call = (kind == "Call")
theo_px = bsm_price(is_call, S, K, year_fraction_days(ttm_days, basis), r, q, iv)
gross_leg_value = theo_px * lots * lot_size

# cost calc (simplified)
stt = (theo_px * lot_size * lots) * (stt_rate_sell/100.0) if side == "Sell" else 0.0
stamp = (theo_px * lot_size * lots) * (stamp_rate_buy/100.0) if side == "Buy" else 0.0
broker = brokerage * lots
gst = broker * (gst_rate/100.0)
total_costs = stt + stamp + broker + gst
cash_change = (-1 if side == "Buy" else +1) * (gross_leg_value)  # premium cashflow
cash_change -= total_costs  # costs always reduce cash

c1, c2, c3, c4 = st.columns(4)
c1.metric("Theo Premium (₹/sh)", f"{theo_px:,.2f}")
c2.metric("Gross Trade (₹)", fmt_money(gross_leg_value))
c3.metric("Costs (₹)", fmt_money(total_costs))
c4.metric("Cash Δ (₹)", fmt_money(cash_change))

place_col, _ = st.columns([1,3])
with place_col:
    if st.button("✅ Place Order", use_container_width=True):
        pos = Position(
            kind='C' if is_call else 'P',
            side=+1 if side == "Buy" else -1,
            K=float(K),
            lots=int(lots),
            lot_size=int(lot_size),
            open_price=float(theo_px),
            sigma=float(iv),
            r=float(r),
            q=float(q),
            ttm_days=float(ttm_days),
        )
        st.session_state["positions"].append(pos)
        st.session_state["cash"] += cash_change
        st.success(f"Order filled: {side} {lots} lot(s) of {kind} {int(K)} @ ₹{theo_px:.2f} (per share)")

# =========================
# Positions table
# =========================
st.markdown("## 📋 Open Positions")
if not st.session_state["positions"]:
    st.info("No positions yet. Use the trade ticket above to enter.")
else:
    rows = []
    for p in st.session_state["positions"]:
        T = year_fraction_days(p.ttm_days, basis)
        mkt = bsm_price(p.kind == 'C', S, p.K, T, r, q, iv)
        rows.append(dict(
            ID=p.id, Side="Long" if p.side>0 else "Short", Type="Call" if p.kind=='C' else "Put",
            Strike=int(p.K), Lots=p.lots, LotSize=p.lot_size, DaysLeft=max(round(p.ttm_days,1),0),
            EntryPrice=round(p.open_price,2), MktPrice=round(mkt,2),
            MTM=round((mkt - p.open_price) * p.lot_size * p.lots * p.side,2)
        ))
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Close button per position
    close_id = st.text_input("Close by Position ID", value="")
    if st.button("❌ Close Position", disabled=(close_id.strip()=="")):
        try:
            pid = int(close_id.strip())
            idx = next(i for i,p in enumerate(st.session_state["positions"]) if p.id == pid)
            p = st.session_state["positions"].pop(idx)
            T = year_fraction_days(max(p.ttm_days,0.0), basis)
            mkt = bsm_price(p.kind == 'C', S, p.K, T, r, q, iv)
            gross = mkt * p.lot_size * p.lots
            # Closing: reverse trade direction
            stt_close = gross * (stt_rate_sell/100.0) if p.side<0 else 0.0  # sell to close short -> STT
            stamp_close = gross * (stamp_rate_buy/100.0) if p.side>0 else 0.0  # buy to close long -> stamp
            broker_close = brokerage * p.lots
            gst_close = broker_close * (gst_rate/100.0)
            costs_close = stt_close + stamp_close + broker_close + gst_close

            cashflow = (+1 if p.side>0 else -1) * gross  # receiving premium if selling (closing long); paying if buying (closing short)
            st.session_state["cash"] += cashflow - costs_close
            st.success(f"Closed {pid}. Cash Δ {fmt_money(cashflow - costs_close)}")
        except StopIteration:
            st.error("Position ID not found.")
        except ValueError:
            st.error("Enter a valid numeric Position ID.")

# =========================
# Advance the game
# =========================
st.markdown("## ⏭️ Next Turn / Sim")
colN, colReset = st.columns([1,1])
with colN:
    if st.button("➡️ Next Turn", use_container_width=True):
        # GBM step for underlying
        dt_days = dt_minutes / (60*24)        # minutes to days
        T = year_fraction_days(dt_days, basis)
        mu = mu_annual/100.0
        sig = vol_annual/100.0
        z = np.random.normal()
        S_next = S * math.exp((mu - 0.5*sig*sig)*T + sig*math.sqrt(T)*z)

        # decay positions / expire
        expired_cash = 0.0
        for p in st.session_state["positions"][:]:
            p.ttm_days -= dt_minutes/ (60*24)
            if p.ttm_days <= 0:
                payoff = p.payoff_expire(S_next)
                # Expiry costs minimal: STT on ITM sell? Keep it simple: no extra costs at settlement in the game.
                expired_cash += payoff
                st.session_state["positions"].remove(p)

        st.session_state["cash"] += expired_cash
        st.session_state["t_minutes"] += dt_minutes
        st.session_state["spot"] = float(S_next)
        st.session_state["round"] += 1
        # track history
        st.session_state["spot_history"].append(float(S_next))
        new_nav = portfolio_value()
        st.session_state["nav_history"].append(float(new_nav))
        st.session_state["time_history"].append(st.session_state["t_minutes"])

with colReset:
    if st.button("🔁 Reset Game", use_container_width=True):
        S0 = yahoo_ltp(symbol) if (spot_source == "Yahoo (delayed)") else manual_spot
        reset_game(S0 or manual_spot)
        st.success("Game reset!")

# =========================
# Charts (Plotly)
# =========================
st.markdown("## 📊 Charts")

template = "plotly_dark" if theme == "Dark" else "plotly_white"

cSpot, cNAV = st.columns(2)
with cSpot:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state["time_history"], y=st.session_state["spot_history"],
                             name="Spot", line=dict(color="#1f77b4", width=3)))
    fig.update_layout(template=template, title="Spot Path", margin=dict(l=10,r=10,t=40,b=10))
    fig.update_xaxes(title="Minutes", showgrid=gridlines)
    fig.update_yaxes(title="₹", showgrid=gridlines)
    st.plotly_chart(fig, use_container_width=True)

with cNAV:
    fig2 = go.Figure()
    nav_series = st.session_state["nav_history"] or [portfolio_value()]
    time_series = st.session_state["time_history"][1:] if len(st.session_state["time_history"])>1 else [0]
    if len(nav_series) != len(time_series):
        # align lengths
        nav_series = [portfolio_value() for _ in range(len(time_series))]
    fig2.add_trace(go.Scatter(x=time_series, y=nav_series, name="NAV",
                              line=dict(color="#2ca02c", width=3)))
    fig2.update_layout(template=template, title="Portfolio NAV", margin=dict(l=10,r=10,t=40,b=10))
    fig2.update_xaxes(title="Minutes", showgrid=gridlines)
    fig2.update_yaxes(title="₹", showgrid=gridlines)
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# Game Tips & Rules
# =========================
with st.expander("📜 Game Rules & Tips (click to view)", expanded=False):
    st.markdown("""
- **Objective:** Grow your **Portfolio NAV** by trading European **Calls & Puts** before expiry.
- **Model:** Option prices come from **BSM** with the IV slider you set. Spot evolves by a **GBM** step each turn.
- **Lot size & Strikes:** You trade in lots. Strikes are rounded to the chosen **strike step** (₹10/₹25/₹50/₹100).
- **Costs (simplified):** Brokerage per lot, **STT on sell**, **Stamp duty on buy**, **GST on brokerage**.
- **Expiry:** When days-to-maturity hits 0, options **cash settle** to intrinsic; the game credits/debits your cash.
- **No margin simulation** (to keep it simple). You **can short** options; just mind your P&L swings.
- **Yahoo source is delayed** ~15–20 minutes — use for demos. For true realtime plug a broker API (Zerodha/Upstox/Angel).
- **Educational only** — not investment advice.
    """)

st.success("Have fun! Try different IVs, strike steps and vol regimes — you’ll *feel* how pricing reacts. 🚀")
