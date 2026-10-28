"""Turn indicators (and optional ML probabilities) into BUY / SELL / HOLD signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

BUY, HOLD, SELL = 1, 0, -1
ACTION_NAMES = {BUY: "BUY", HOLD: "HOLD", SELL: "SELL"}


def component_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Score each indicator family in [-1, 1]; positive means bullish."""
    close = df["close"]
    trend = 0.5 * np.sign(df["ema_fast"] - df["ema_mid"]) + 0.5 * np.sign(close - df["ema_slow"])

    hist = df["macd_hist"]
    rising = hist.diff() > 0
    momentum = pd.Series(np.select(
        [(hist > 0) & rising, hist > 0, (hist < 0) & ~rising, hist < 0],
        [1.0, 0.5, -1.0, -0.5],
        0.0,
    ), index=df.index)

    # Healthy momentum zone is bullish; stretched extremes lean towards reversal.
    rsi = df["rsi"]
    rsi_score = ((rsi - 50) / 20).clip(-1, 1)
    rsi_score = rsi_score.mask(rsi > 75, -0.5).mask(rsi < 25, 0.5)

    # Unusual volume confirms the direction of the candle it appears on.
    volume = (np.sign(df["ret_1"]) * (df["vol_z"].clip(0, 3) / 3)).fillna(0.0)

    bb = df["bb_pct"]
    bollinger = pd.Series(np.select([bb < 0, bb > 1], [0.5, -0.5], 0.0), index=df.index)

    scores = pd.DataFrame(
        {"trend": trend, "momentum": momentum, "rsi": rsi_score, "volume": volume, "bollinger": bollinger}
    )
    return scores.fillna(0.0)


def rule_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    scores = component_scores(df)
    total = sum(weights.values())
    return sum(scores[name] * weight for name, weight in weights.items()) / total


def generate_signals(df: pd.DataFrame, params: dict[str, Any], ml_proba: pd.Series | None = None) -> pd.DataFrame:
    """Return ``df`` with ``rule_score``, ``ml_proba``, ``score`` and ``signal`` columns.

    Rows where the ML probability is missing fall back to the rule score alone.
    """
    out = df.copy()
    out["rule_score"] = rule_score(out, params["weights"])
    proba = ml_proba.reindex(out.index) if ml_proba is not None else pd.Series(np.nan, index=out.index)
    out["ml_proba"] = proba

    ml_score = ((proba - 0.5) / params["ml_scale"]).clip(-1, 1)
    weight = params["ml_weight"]
    blended = (1 - weight) * out["rule_score"] + weight * ml_score
    out["score"] = blended.where(proba.notna(), out["rule_score"]).clip(-1, 1)

    buy = out["score"] >= params["buy_threshold"]
    if params["require_bull_regime"] and "regime" in out:
        buy &= out["regime"] > 0
    sell = out["score"] <= params["sell_threshold"]
    out["signal"] = np.select([buy, sell], [BUY, SELL], HOLD)
    # Indicators need a warm-up period; never trade on incomplete inputs.
    out.loc[out[["ema_slow", "atr", "rsi"]].isna().any(axis=1), "signal"] = HOLD
    return out


@dataclass
class Signal:
    symbol: str
    timeframe: str
    time: str
    action: str
    confidence: float
    price: float
    stop_loss: float | None
    take_profit: float | None
    score: float
    rule_score: float
    ml_proba: float | None
    regime: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain(row: pd.Series) -> list[str]:
    reasons = []
    reasons.append("EMA20 above EMA50" if row["ema_fast"] > row["ema_mid"] else "EMA20 below EMA50")
    reasons.append("price above EMA200" if row["close"] > row["ema_slow"] else "price below EMA200")
    reasons.append(f"RSI {row['rsi']:.0f}" + (" (overbought)" if row["rsi"] > 70 else " (oversold)" if row["rsi"] < 30 else ""))
    reasons.append("MACD histogram positive" if row["macd_hist"] > 0 else "MACD histogram negative")
    if row.get("vol_z", 0) > 1.5:
        reasons.append(f"volume spike ({row['vol_z']:.1f} sd)")
    if pd.notna(row.get("ml_proba")):
        reasons.append(f"ML up-probability {row['ml_proba']:.0%}")
    if "regime" in row:
        reasons.append({1.0: "daily trend bullish", -1.0: "daily trend bearish"}.get(row["regime"], "daily trend unknown"))
    return reasons


def latest_signal(df: pd.DataFrame, symbol: str, timeframe: str, params: dict[str, Any]) -> Signal:
    """Build a human-readable signal from the last closed candle of ``generate_signals`` output."""
    row = df.iloc[-1]
    action = int(row["signal"])
    price = float(row["close"])
    stop = take = None
    if action == BUY:
        stop = price - params["stop_loss_atr"] * row["atr"]
        take = price + params["take_profit_atr"] * row["atr"]
    elif action == SELL:
        stop = price + params["stop_loss_atr"] * row["atr"]
        take = price - params["take_profit_atr"] * row["atr"]
    regime = {1.0: "bull", -1.0: "bear"}.get(row.get("regime", 0.0), "unknown")
    return Signal(
        symbol=symbol,
        timeframe=timeframe,
        time=df.index[-1].isoformat(),
        action=ACTION_NAMES[action],
        confidence=round(abs(float(row["score"])), 3),
        price=price,
        stop_loss=None if stop is None else round(float(stop), 8),
        take_profit=None if take is None else round(float(take), 8),
        score=round(float(row["score"]), 3),
        rule_score=round(float(row["rule_score"]), 3),
        ml_proba=None if pd.isna(row["ml_proba"]) else round(float(row["ml_proba"]), 3),
        regime=regime,
        reasons=explain(row),
    )
