import numpy as np
import pandas as pd

from cryptosignal.indicators import add_indicators
from cryptosignal.strategy import BUY, SELL, generate_signals, latest_signal
from tests.conftest import make_ohlcv


def prepared(cfg, drift=0.0, seed=0):
    df = add_indicators(make_ohlcv(1200, drift=drift, seed=seed), cfg["indicators"])
    df["regime"] = 1.0
    return df


def test_signals_are_valid_and_warmup_is_hold(cfg):
    out = generate_signals(prepared(cfg), cfg["strategy"])
    assert set(out["signal"].unique()) <= {-1, 0, 1}
    assert out["score"].between(-1, 1).all()
    assert (out["signal"].iloc[:199] == 0).all()


def test_uptrend_produces_more_buys_than_downtrend(cfg):
    up = generate_signals(prepared(cfg, drift=0.003), cfg["strategy"])
    down = generate_signals(prepared(cfg, drift=-0.003), cfg["strategy"])
    assert (up["signal"] == BUY).sum() > (down["signal"] == BUY).sum()
    assert (down["signal"] == SELL).sum() > (up["signal"] == SELL).sum()


def test_bear_regime_blocks_buys(cfg):
    df = prepared(cfg, drift=0.003)
    df["regime"] = -1.0
    out = generate_signals(df, cfg["strategy"])
    assert (out["signal"] != BUY).all()


def test_ml_probability_moves_score(cfg):
    df = prepared(cfg)
    bullish = generate_signals(df, cfg["strategy"], pd.Series(0.9, index=df.index))
    bearish = generate_signals(df, cfg["strategy"], pd.Series(0.1, index=df.index))
    assert (bullish["score"] >= bearish["score"]).all()
    assert (bullish["score"] > bearish["score"]).any()


def test_missing_ml_falls_back_to_rules(cfg):
    df = prepared(cfg)
    proba = pd.Series(np.nan, index=df.index)
    out = generate_signals(df, cfg["strategy"], proba)
    pd.testing.assert_series_equal(out["score"], out["rule_score"].clip(-1, 1), check_names=False)


def test_latest_signal_levels(cfg):
    df = generate_signals(prepared(cfg, drift=0.003), cfg["strategy"])
    df.loc[df.index[-1], ["signal", "score"]] = [BUY, 0.6]
    sig = latest_signal(df, "BTC/USDT", "4h", cfg["strategy"])
    assert sig.action == "BUY"
    assert sig.stop_loss < sig.price < sig.take_profit
    assert sig.reasons
