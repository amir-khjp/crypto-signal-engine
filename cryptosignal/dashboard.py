"""Streamlit dashboard. Run with ``cryptosignal dashboard`` or ``streamlit run cryptosignal/dashboard.py``."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cryptosignal.backtest import bars_per_year, run_backtest
from cryptosignal.config import load_config
from cryptosignal.data import MarketData
from cryptosignal.paper import PaperPortfolio
from cryptosignal.pipeline import history_with_signals, scan, strategy_params, timeframe_delta
from cryptosignal.strategy import BUY, SELL

warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="Crypto Signal Engine", layout="wide")

UP, DOWN, MUTED = "#16a34a", "#dc2626", "#64748b"


@st.cache_data(ttl=600, show_spinner="Scanning markets...")
def cached_scan(mode: str, offline: bool):
    cfg = load_config()
    md = MarketData(cfg, offline=offline)
    signals, errors = scan(md, cfg, mode)
    return [s.to_dict() for s in signals], errors, md.exchange_name


@st.cache_data(ttl=600, show_spinner="Building signal history (walk-forward)...")
def cached_history(symbol: str, mode: str, offline: bool):
    cfg = load_config()
    md = MarketData(cfg, offline=offline)
    df, proba = history_with_signals(md, symbol, cfg, mode)
    return df, proba


def price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name=symbol,
        increasing_line_color=UP, decreasing_line_color=DOWN,
    ), row=1, col=1)
    for col, color in [("ema_fast", "#f59e0b"), ("ema_mid", "#3b82f6"), ("ema_slow", "#a855f7")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col.upper().replace("_", " "), line=dict(width=1.2, color=color)), row=1, col=1)

    # Mark only the first candle of each new signal so the chart stays readable.
    changed = df["signal"].ne(df["signal"].shift())
    buys = df[changed & (df["signal"] == BUY)]
    sells = df[changed & (df["signal"] == SELL)]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["low"] * 0.985, mode="markers", name="BUY",
                             marker=dict(symbol="triangle-up", size=11, color=UP)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells["high"] * 1.015, mode="markers", name="SELL",
                             marker=dict(symbol="triangle-down", size=11, color=DOWN)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(width=1, color=MUTED)), row=2, col=1)
    for level in (30, 70):
        fig.add_hline(y=level, line_dash="dot", line_color=MUTED, row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["score"], name="Signal score",
                         marker_color=[UP if v > 0 else DOWN for v in df["score"]]), row=3, col=1)
    fig.update_layout(height=760, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.02))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Score", row=3, col=1, range=[-1, 1])
    return fig


def main() -> None:
    cfg = load_config()
    st.title("Crypto Signal Engine")
    st.caption("Probabilistic signals for research. Not financial advice - always paper-trade first.")

    with st.sidebar:
        mode = st.selectbox("Strategy mode", ["hybrid", "rules", "ml"], help="hybrid blends rules with the ML model, "
                            "down-weighting ML when its out-of-sample AUC shows no edge")
        offline = st.toggle("Offline (cached data only)", value=False)
        if st.button("Refresh data"):
            st.cache_data.clear()

    tab_signals, tab_chart, tab_backtest, tab_paper = st.tabs(["Signals", "Chart", "Backtest", "Paper trading"])

    with tab_signals:
        signals, errors, exchange = cached_scan(mode, offline)
        st.subheader(f"Latest signals - {exchange}, {cfg['timeframe']} candles")
        if signals:
            cols = st.columns(len(signals))
            for col, sig in zip(cols, signals):
                col.metric(sig["symbol"], sig["action"], f"{sig['confidence']:.0%} confidence",
                           delta_color="normal" if sig["action"] == "BUY" else "inverse" if sig["action"] == "SELL" else "off")
            table = pd.DataFrame(signals)[["symbol", "action", "confidence", "price", "stop_loss", "take_profit",
                                           "rule_score", "ml_proba", "ml_auc", "regime", "time"]]
            st.dataframe(table, hide_index=True, width="stretch")
            for sig in signals:
                with st.expander(f"Why {sig['action']} on {sig['symbol']}?"):
                    st.write("\n".join(f"- {r}" for r in sig["reasons"]))
        for symbol, err in errors.items():
            st.error(f"{symbol}: {err}")

    with tab_chart:
        symbol = st.selectbox("Symbol", cfg["symbols"], key="chart_symbol")
        df, _ = cached_history(symbol, mode, offline)
        candles = st.slider("Candles shown", 100, len(df), min(500, len(df)), step=50)
        st.plotly_chart(price_chart(df.tail(candles), symbol), width="stretch")
        st.caption("History uses only out-of-sample ML predictions, so markers reflect what the engine would have said at the time.")

    with tab_backtest:
        symbol = st.selectbox("Symbol", cfg["symbols"], key="bt_symbol")
        df, proba = cached_history(symbol, mode, offline)
        start = proba.first_valid_index() if proba is not None and proba.notna().any() else df.index[0]
        params = strategy_params(cfg, mode)
        bpy = bars_per_year(int(timeframe_delta(cfg["timeframe"]).total_seconds()))
        result = run_backtest(df.loc[start:], cfg["backtest"], params, bpy)
        m = result.metrics
        c = st.columns(6)
        c[0].metric("Strategy return", f"{m['total_return']:+.1%}")
        c[1].metric("Buy & hold", f"{m['buy_hold_return']:+.1%}")
        c[2].metric("Max drawdown", f"{m['max_drawdown']:.1%}")
        c[3].metric("Sharpe", f"{m['sharpe']:.2f}")
        c[4].metric("Trades / win rate", f"{m['trades']} / {m['win_rate']:.0%}")
        c[5].metric("Time in market", f"{m['exposure']:.0%}")

        closes = df.loc[start:, "close"]
        bench = closes / closes.iloc[0] * cfg["backtest"]["initial_capital"]
        fig = go.Figure([
            go.Scatter(x=result.equity.index, y=result.equity, name="Strategy", line=dict(color="#3b82f6")),
            go.Scatter(x=bench.index, y=bench, name="Buy & hold", line=dict(color=MUTED, dash="dot")),
        ])
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Equity")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Out-of-sample window from {start:%Y-%m-%d}. Fees {cfg['backtest']['fee']:.2%}/side, "
                   f"slippage {cfg['backtest']['slippage']:.2%}.")
        if result.trades:
            st.dataframe(result.trades_frame(), hide_index=True, width="stretch")

    with tab_paper:
        path = Path(cfg["paper"]["state_file"])
        portfolio = PaperPortfolio.load(path, cfg["paper"]["initial_capital"])
        prices = {s["symbol"]: s["price"] for s in signals}
        equity = portfolio.equity(prices)
        start_cap = cfg["paper"]["initial_capital"]
        c = st.columns(3)
        c[0].metric("Paper equity", f"{equity:,.2f}", f"{equity / start_cap - 1:+.2%}")
        c[1].metric("Cash", f"{portfolio.cash:,.2f}")
        c[2].metric("Open positions", len(portfolio.positions))
        if portfolio.positions:
            st.dataframe(pd.DataFrame([{"symbol": k, **vars(v), "price": prices.get(k)} for k, v in portfolio.positions.items()]),
                         hide_index=True, width="stretch")
        if portfolio.closed:
            st.dataframe(pd.DataFrame(portfolio.closed), hide_index=True, width="stretch")
        if not portfolio.positions and not portfolio.closed:
            st.info("No paper trades yet. Start with `cryptosignal watch --paper`.")


main()
