"""Configuration loading with sane defaults.

Values in ``config.yaml`` override the defaults below key by key, so the file
only needs to contain the settings you want to change.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    # Exchanges are tried in order; the first one reachable from your network wins.
    "exchanges": ["kucoin", "binance", "bybit", "okx", "kraken"],
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
    "timeframe": "4h",
    "history_candles": 3000,
    # Higher timeframe used as a market-regime filter.
    "trend_timeframe": "1d",
    "trend_candles": 600,
    "data_dir": "data",
    "indicators": {
        "ema_fast": 20,
        "ema_mid": 50,
        "ema_slow": 200,
        "rsi": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_window": 20,
        "bb_std": 2.0,
        "atr": 14,
        "adx": 14,
        "volume_window": 20,
        "trend_ema": 200,
    },
    "strategy": {
        "weights": {"trend": 0.35, "momentum": 0.25, "rsi": 0.20, "volume": 0.10, "bollinger": 0.10},
        "buy_threshold": 0.35,
        "sell_threshold": -0.35,
        "require_bull_regime": True,
        # Share of the final score coming from the ML model (0 = rules only, 1 = ML only).
        "ml_weight": 0.5,
        # ML probability distance from 0.5 that maps to a full-strength (+/-1) score.
        "ml_scale": 0.15,
        "stop_loss_atr": 2.0,
        "take_profit_atr": 4.0,
    },
    "model": {
        "enabled": True,
        "horizon": 6,
        "label_threshold": 0.005,
        "min_train": 1000,
        "retrain_every": 250,
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 40,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
        },
    },
    "backtest": {
        "initial_capital": 10_000.0,
        "fee": 0.001,
        "slippage": 0.0005,
    },
    "alerts": {
        "telegram_token": "",
        "telegram_chat_id": "",
    },
    "paper": {
        "initial_capital": 10_000.0,
        "position_fraction": 0.2,
        "state_file": "data/paper_portfolio.json",
    },
    "watch_interval_minutes": 60,
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Return the defaults merged with ``path`` (or ``./config.yaml`` if present)."""
    cfg = copy.deepcopy(DEFAULTS)
    file = Path(path) if path else Path(os.environ.get("CRYPTOSIGNAL_CONFIG", "config.yaml"))
    if file.exists():
        with file.open(encoding="utf-8") as fh:
            cfg = _merge(cfg, yaml.safe_load(fh) or {})
    elif path:
        raise FileNotFoundError(f"Config file not found: {file}")

    # Secrets are better kept in the environment than in a committed file.
    alerts = cfg["alerts"]
    alerts["telegram_token"] = os.environ.get("TELEGRAM_BOT_TOKEN", alerts["telegram_token"])
    alerts["telegram_chat_id"] = os.environ.get("TELEGRAM_CHAT_ID", alerts["telegram_chat_id"])
    return cfg
