import numpy as np
import pandas as pd
import pytest

from cryptosignal.config import DEFAULTS


def make_ohlcv(n: int = 1500, drift: float = 0.0, seed: int = 0, freq: str = "4h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, 0.01, n)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0, 0.005, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(100, 1000, n)
    index = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def cfg() -> dict:
    import copy

    return copy.deepcopy(DEFAULTS)
