"""Long-only (spot) backtester with fees, slippage, ATR stops and take-profits.

Signals are evaluated at the close of a candle and executed at the next candle's
open, so a strategy can never trade on information it would not have had.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from cryptosignal.strategy import BUY, SELL

SECONDS_PER_YEAR = 365 * 24 * 3600


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str
    pnl_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade]
    metrics: dict[str, float]

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_dict() for t in self.trades])


def run_backtest(
    df: pd.DataFrame, bt: dict[str, Any], strategy: dict[str, Any], bars_per_year: float
) -> BacktestResult:
    """Simulate ``df['signal']`` on the candles in ``df``."""
    opens, highs, lows, closes = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    atrs = df["atr"].to_numpy(float)
    signals = df["signal"].to_numpy(int)
    times = df.index
    fee, slip = bt["fee"], bt["slippage"]

    cash = float(bt["initial_capital"])
    qty = 0.0
    entry_price = stop = take = cost_basis = 0.0
    entry_time = None
    trades: list[Trade] = []
    equity = np.empty(len(df))

    def close_position(i: int, raw_price: float, reason: str) -> None:
        nonlocal cash, qty
        price = raw_price * (1 - slip)
        cash = qty * price * (1 - fee)
        trades.append(Trade(entry_time, times[i], entry_price, price, reason, cash / cost_basis - 1))
        qty = 0.0

    for i in range(len(df)):
        if i > 0:
            prev = signals[i - 1]
            if qty == 0 and prev == BUY and not np.isnan(atrs[i - 1]):
                entry_price = opens[i] * (1 + slip)
                cost_basis = cash
                qty = cash * (1 - fee) / entry_price
                cash = 0.0
                entry_time = times[i]
                stop = entry_price - strategy["stop_loss_atr"] * atrs[i - 1]
                take = entry_price + strategy["take_profit_atr"] * atrs[i - 1]
            elif qty > 0 and prev == SELL:
                close_position(i, opens[i], "signal")

        if qty > 0:
            # Assume the worst case when a candle touches both levels: the stop fills first.
            if lows[i] <= stop:
                close_position(i, min(opens[i], stop), "stop_loss")
            elif highs[i] >= take:
                close_position(i, max(opens[i], take), "take_profit")

        equity[i] = cash + qty * closes[i]

    if qty > 0:
        close_position(len(df) - 1, closes[-1], "end_of_data")
        equity[-1] = cash

    equity_series = pd.Series(equity, index=times, name="equity")
    return BacktestResult(equity_series, trades, compute_metrics(equity_series, trades, closes, bars_per_year))


def compute_metrics(equity: pd.Series, trades: list[Trade], closes: np.ndarray, bars_per_year: float) -> dict[str, float]:
    returns = equity.pct_change().fillna(0.0)
    years = len(equity) / bars_per_year
    total = equity.iloc[-1] / equity.iloc[0] - 1
    drawdown = equity / equity.cummax() - 1
    pnl = np.array([t.pnl_pct for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    std = returns.std()
    in_market = sum(((t.exit_time - t.entry_time) for t in trades), pd.Timedelta(0))
    span = equity.index[-1] - equity.index[0]
    return {
        "total_return": float(total),
        "buy_hold_return": float(closes[-1] / closes[0] - 1),
        "cagr": float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(returns.mean() / std * np.sqrt(bars_per_year)) if std > 0 else 0.0,
        "trades": len(trades),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf") if len(wins) else 0.0,
        "avg_trade": float(pnl.mean()) if len(pnl) else 0.0,
        "exposure": float(in_market / span) if span > pd.Timedelta(0) else 0.0,
    }


def bars_per_year(timeframe_seconds: int) -> float:
    return SECONDS_PER_YEAR / timeframe_seconds
