"""End-to-end analysis for one symbol: candles -> indicators -> ML -> signals."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from cryptosignal import model
from cryptosignal.data import MarketData
from cryptosignal.indicators import add_indicators, add_regime
from cryptosignal.strategy import Signal, generate_signals, latest_signal

log = logging.getLogger(__name__)


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    import ccxt

    return pd.Timedelta(seconds=ccxt.Exchange.parse_timeframe(timeframe))


def prepare(md: MarketData, symbol: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Candles with indicators and the higher-timeframe regime attached."""
    df = md.candles(symbol, cfg["timeframe"], cfg["history_candles"])
    df = add_indicators(df, cfg["indicators"])
    htf = md.candles(symbol, cfg["trend_timeframe"], cfg["trend_candles"])
    return add_regime(df, htf, timeframe_delta(cfg["trend_timeframe"]), cfg["indicators"]["trend_ema"])


def strategy_params(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    """Strategy settings for ``rules``, ``ml`` or ``hybrid`` (the configured blend)."""
    params = dict(cfg["strategy"])
    if mode == "rules":
        params["ml_weight"] = 0.0
    elif mode == "ml":
        params["ml_weight"] = 1.0
    return params


def edge_factor(auc: float | None, model_cfg: dict[str, Any]) -> float:
    """0..1 multiplier for the ML weight based on out-of-sample AUC."""
    if auc is None:
        return 0.0
    span = model_cfg["full_auc"] - model_cfg["min_auc"]
    return float(min(max((auc - model_cfg["min_auc"]) / span, 0.0), 1.0))


AUC_CACHE_SECONDS = 24 * 3600


def walk_forward_auc(df: pd.DataFrame, cfg: dict[str, Any], key: str) -> float | None:
    """Out-of-sample AUC, cached for a day because walk-forward training is slow."""
    path = Path(cfg["data_dir"]) / "model_quality.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    entry = cache.get(key)
    if entry and time.time() - entry["computed_at"] < AUC_CACHE_SECONDS:
        return entry["auc"]
    auc = model.evaluate(df, model.walk_forward_proba(df, cfg["model"]), cfg["model"]).get("auc")
    cache[key] = {"auc": auc, "computed_at": time.time()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))
    return auc


def live_proba(df: pd.DataFrame, cfg: dict[str, Any], key: str) -> tuple[pd.Series | None, float | None]:
    """Score the latest candle with a model trained on all history.

    Also returns the walk-forward AUC, which measures how much the model can be trusted.
    """
    if not cfg["model"]["enabled"]:
        return None, None
    auc = walk_forward_auc(df, cfg, key)
    fitted = model.train(df, cfg["model"])
    if fitted is None:
        return None, auc
    proba = pd.Series(float("nan"), index=df.index)
    proba.iloc[-1] = model.predict_proba(fitted, df.iloc[[-1]]).iloc[-1]
    return proba, auc


def analyze(md: MarketData, symbol: str, cfg: dict[str, Any], mode: str = "hybrid") -> tuple[pd.DataFrame, Signal]:
    df = prepare(md, symbol, cfg)
    params = strategy_params(cfg, mode)
    key = f"{symbol}|{cfg['timeframe']}"
    proba, auc = live_proba(df, cfg, key) if params["ml_weight"] > 0 else (None, None)
    if proba is not None and mode != "ml":
        params["ml_weight"] *= edge_factor(auc, cfg["model"])
    signals = generate_signals(df, params, proba)
    sig = latest_signal(signals, symbol, cfg["timeframe"], params)
    sig.ml_auc = None if auc is None else round(auc, 3)
    if auc is not None:
        sig.reasons.append(f"ML out-of-sample AUC {auc:.2f} -> weight {params['ml_weight']:.0%}")
    return signals, sig


def history_with_signals(
    md: MarketData, symbol: str, cfg: dict[str, Any], mode: str = "hybrid"
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Historical signals using only out-of-sample ML predictions (safe for charts and backtests)."""
    df = prepare(md, symbol, cfg)
    params = strategy_params(cfg, mode)
    proba = model.walk_forward_proba(df, cfg["model"]) if params["ml_weight"] > 0 and cfg["model"]["enabled"] else None
    return generate_signals(df, params, proba), proba


def scan(md: MarketData, cfg: dict[str, Any], mode: str = "hybrid") -> tuple[list[Signal], dict[str, str]]:
    """Latest signal for every configured symbol; failures are reported, not raised."""
    signals, errors = [], {}
    for symbol in cfg["symbols"]:
        try:
            signals.append(analyze(md, symbol, cfg, mode)[1])
        except Exception as exc:
            log.warning("%s failed: %s", symbol, exc)
            errors[symbol] = f"{type(exc).__name__}: {exc}"
    return signals, errors
