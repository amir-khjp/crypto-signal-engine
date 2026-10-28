"""Market data: download OHLCV candles through ccxt and cache them locally."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd

log = logging.getLogger(__name__)

COLUMNS = ["open", "high", "low", "close", "volume"]


def connect(names: list[str]) -> ccxt.Exchange:
    """Return the first exchange in ``names`` whose public API is reachable."""
    errors = []
    for name in names:
        try:
            exchange = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 20_000})
            exchange.load_markets()
            log.info("Connected to %s", name)
            return exchange
        except Exception as exc:  # network blocks show up as many different ccxt errors
            errors.append(f"{name}: {type(exc).__name__}")
    raise RuntimeError("No exchange reachable - " + "; ".join(errors))


def to_frame(rows: list[list[Any]]) -> pd.DataFrame:
    """Convert raw ccxt OHLCV rows into a UTC-indexed, de-duplicated DataFrame."""
    df = pd.DataFrame(rows, columns=["timestamp", *COLUMNS])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    return df.astype(float)


def drop_incomplete(df: pd.DataFrame, timeframe_ms: int, now_ms: int) -> pd.DataFrame:
    """Remove the candle that is still forming - signals must only use closed candles."""
    if df.empty:
        return df
    open_ms = df.index.as_unit("ms").asi8
    return df[open_ms + timeframe_ms <= now_ms]


def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int, limit: int = 1000) -> pd.DataFrame:
    """Page through the exchange API from ``since_ms`` up to now."""
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    rows: list[list[Any]] = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        if last < cursor or last + tf_ms > exchange.milliseconds():
            break
        cursor = last + tf_ms
    df = to_frame(rows) if rows else pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    return drop_incomplete(df, tf_ms, exchange.milliseconds())


class MarketData:
    """Candle provider with a parquet cache in ``data_dir``.

    With ``offline=True`` only cached data is used, which keeps the dashboard and
    tests usable without network access.
    """

    def __init__(self, cfg: dict[str, Any], offline: bool = False):
        self.cfg = cfg
        self.offline = offline
        self.data_dir = Path(cfg["data_dir"])
        self._exchange: ccxt.Exchange | None = None

    @property
    def exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            self._exchange = connect(self.cfg["exchanges"])
        return self._exchange

    @property
    def exchange_name(self) -> str:
        if self._exchange is not None:
            return self._exchange.id
        if self.offline:
            cached = sorted(self.data_dir.glob("*.parquet"))
            if cached:
                return cached[0].name.split("_", 1)[0]
        return self.exchange.id

    def cache_path(self, symbol: str, timeframe: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9]+", "-", symbol)
        return self.data_dir / f"{self.exchange_name}_{safe}_{timeframe}.parquet"

    def candles(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Return the latest ``count`` closed candles, updating the cache incrementally."""
        path = self.cache_path(symbol, timeframe)
        cached = pd.read_parquet(path) if path.exists() else None
        if self.offline:
            if cached is None:
                raise FileNotFoundError(f"No cached data for {symbol} {timeframe}; run `cryptosignal fetch` first")
            return cached.tail(count)

        tf_ms = self.exchange.parse_timeframe(timeframe) * 1000
        wanted_from = self.exchange.milliseconds() - count * tf_ms
        if cached is not None and not cached.empty and cached.index[0].value // 1_000_000 <= wanted_from:
            since = int(cached.index[-1].value // 1_000_000) + tf_ms
        else:
            cached, since = None, wanted_from

        fresh = fetch_ohlcv(self.exchange, symbol, timeframe, since)
        df = fresh if cached is None else pd.concat([cached, fresh])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        log.debug("%s %s: %d candles (%d new)", symbol, timeframe, len(df), len(fresh))
        return df.tail(count)

    def prefetch(self) -> None:
        for symbol in self.cfg["symbols"]:
            self.candles(symbol, self.cfg["timeframe"], self.cfg["history_candles"])
            self.candles(symbol, self.cfg["trend_timeframe"], self.cfg["trend_candles"])
            time.sleep(0.2)
