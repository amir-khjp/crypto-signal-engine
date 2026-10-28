"""Technical indicators implemented with plain pandas so every formula is visible."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, used by RSI, ATR and ADX."""
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), period)
    loss = wilder(-delta.clip(upper=0), period)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # No losses in the window means maximum strength.
    return out.mask(loss.eq(0) & gain.gt(0), 100.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift()
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1, skipna=False).fillna(df["high"] - df["low"])


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return wilder(true_range(df), period)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = wilder(true_range(df), period)
    plus_di = 100 * wilder(plus_dm, period) / tr
    minus_di = 100 * wilder(minus_dm, period) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder(dx, period)


def add_indicators(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Return a copy of ``df`` with indicator and feature columns appended."""
    out = df.copy()
    close, volume = out["close"], out["volume"]

    out["ema_fast"] = ema(close, params["ema_fast"])
    out["ema_mid"] = ema(close, params["ema_mid"])
    out["ema_slow"] = ema(close, params["ema_slow"])
    out["rsi"] = rsi(close, params["rsi"])

    macd_line = ema(close, params["macd_fast"]) - ema(close, params["macd_slow"])
    out["macd"] = macd_line
    out["macd_signal"] = ema(macd_line, params["macd_signal"])
    out["macd_hist"] = macd_line - out["macd_signal"]
    out["macd_hist_pct"] = out["macd_hist"] / close

    mid = close.rolling(params["bb_window"]).mean()
    std = close.rolling(params["bb_window"]).std(ddof=0)
    out["bb_high"] = mid + params["bb_std"] * std
    out["bb_low"] = mid - params["bb_std"] * std
    out["bb_pct"] = (close - out["bb_low"]) / (out["bb_high"] - out["bb_low"]).replace(0, np.nan)
    out["bb_width"] = (out["bb_high"] - out["bb_low"]) / mid

    out["atr"] = atr(out, params["atr"])
    out["atr_pct"] = out["atr"] / close
    out["adx"] = adx(out, params["adx"])

    vol_mean = volume.rolling(params["volume_window"]).mean()
    vol_std = volume.rolling(params["volume_window"]).std()
    out["vol_z"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    out["ret_1"] = close.pct_change()
    out["ret_6"] = close.pct_change(6)
    out["ret_24"] = close.pct_change(24)
    out["volatility_24"] = out["ret_1"].rolling(24).std()
    out["dist_ema_fast"] = close / out["ema_fast"] - 1
    out["dist_ema_mid"] = close / out["ema_mid"] - 1
    out["dist_ema_slow"] = close / out["ema_slow"] - 1
    out["ema_fast_slope"] = out["ema_fast"].pct_change(3)
    return out


def add_regime(df: pd.DataFrame, htf: pd.DataFrame, htf_period: pd.Timedelta, ema_len: int = 200) -> pd.DataFrame:
    """Attach the higher-timeframe trend regime (+1 bull, -1 bear, 0 unknown).

    A higher-timeframe candle only becomes known once it closes, so its values are
    aligned on ``open time + htf_period``. This prevents look-ahead bias.
    """
    trend = pd.DataFrame(
        {"htf_close": htf["close"].to_numpy(), "htf_ema": ema(htf["close"], ema_len).to_numpy()},
        index=htf.index + htf_period,
    ).dropna()
    out = df.copy()
    if trend.empty:
        out["htf_close"] = np.nan
        out["htf_ema"] = np.nan
    else:
        merged = pd.merge_asof(
            out[[]].reset_index(), trend.rename_axis("timestamp").reset_index(), on="timestamp", direction="backward"
        ).set_index("timestamp")
        out["htf_close"] = merged["htf_close"].to_numpy()
        out["htf_ema"] = merged["htf_ema"].to_numpy()
    out["regime"] = np.sign(out["htf_close"] - out["htf_ema"]).fillna(0.0)
    return out
