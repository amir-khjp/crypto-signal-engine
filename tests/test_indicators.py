import numpy as np
import pandas as pd

from cryptosignal.indicators import add_indicators, add_regime, atr, ema, rsi
from tests.conftest import make_ohlcv


def test_ema_matches_pandas(ohlcv):
    expected = ohlcv["close"].ewm(span=20, adjust=False).mean()
    pd.testing.assert_series_equal(ema(ohlcv["close"], 20).iloc[20:], expected.iloc[20:])


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi(up).dropna().eq(100).all()
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert rsi(down).dropna().lt(1).all()
    values = rsi(make_ohlcv()["close"]).dropna()
    assert values.between(0, 100).all()


def test_atr_positive(ohlcv):
    values = atr(ohlcv).dropna()
    assert (values > 0).all()


def test_add_indicators_columns(ohlcv, cfg):
    out = add_indicators(ohlcv, cfg["indicators"])
    for col in ["ema_fast", "rsi", "macd_hist", "bb_pct", "atr", "adx", "vol_z", "dist_ema_slow"]:
        assert col in out
    assert out.iloc[300:][["rsi", "atr", "adx", "bb_pct"]].notna().all().all()
    assert len(out) == len(ohlcv)


def test_regime_has_no_lookahead():
    daily = make_ohlcv(400, freq="1D", seed=1)
    intraday = make_ohlcv(400 * 6, freq="4h", seed=2)
    out = add_regime(intraday, daily, pd.Timedelta("1D"), ema_len=50)
    # The 4h candle opening at midnight of day D may only see daily candles closed by then,
    # i.e. the one that opened on day D-1.
    day = daily.index[100]
    row = out.loc[day]
    assert row["htf_close"] == daily["close"].iloc[99]
    assert set(out["regime"].unique()) <= {-1.0, 0.0, 1.0}
