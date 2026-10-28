import numpy as np

from cryptosignal.indicators import add_indicators
from cryptosignal.model import evaluate, make_labels, predict_proba, train, walk_forward_proba
from tests.conftest import make_ohlcv


def model_cfg(cfg):
    m = dict(cfg["model"])
    m.update(min_train=600, retrain_every=200, params={**m["params"], "n_estimators": 50})
    return m


def test_labels_use_future_close_and_leave_tail_unknown(ohlcv):
    labels = make_labels(ohlcv, horizon=3, threshold=0.0)
    assert labels.iloc[-3:].isna().all()
    expected = float(ohlcv["close"].iloc[3] > ohlcv["close"].iloc[0])
    assert labels.iloc[0] == expected


def test_walk_forward_is_out_of_sample(cfg):
    df = add_indicators(make_ohlcv(1500), cfg["indicators"])
    df["regime"] = 1.0
    mc = model_cfg(cfg)
    proba = walk_forward_proba(df, mc)
    assert proba.iloc[: mc["min_train"]].isna().all()
    assert proba.dropna().between(0, 1).all()
    assert proba.notna().sum() > 500
    metrics = evaluate(df, proba, mc)
    assert 0 <= metrics["auc"] <= 1


def test_random_walk_has_no_real_edge(cfg):
    df = add_indicators(make_ohlcv(3000, seed=7), cfg["indicators"])
    df["regime"] = 0.0
    mc = model_cfg(cfg)
    metrics = evaluate(df, walk_forward_proba(df, mc), mc)
    # Pure noise should not be predictable; guards against look-ahead leaks.
    assert abs(metrics["auc"] - 0.5) < 0.08


def test_train_and_predict(cfg):
    df = add_indicators(make_ohlcv(1000), cfg["indicators"])
    df["regime"] = 1.0
    model = train(df, model_cfg(cfg))
    proba = predict_proba(model, df)
    assert np.isfinite(proba.iloc[-1])
