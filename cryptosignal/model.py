"""LightGBM model estimating the probability of an up-move over the next candles.

``walk_forward_proba`` produces strictly out-of-sample predictions: every
prediction comes from a model trained only on earlier data, with a purge gap of
``horizon`` candles so training labels never overlap the prediction window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

FEATURES = [
    "rsi",
    "macd_hist_pct",
    "bb_pct",
    "bb_width",
    "atr_pct",
    "adx",
    "vol_z",
    "ret_1",
    "ret_6",
    "ret_24",
    "volatility_24",
    "dist_ema_fast",
    "dist_ema_mid",
    "dist_ema_slow",
    "ema_fast_slope",
    "regime",
]


def make_labels(df: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    """1 if close rises by more than ``threshold`` after ``horizon`` candles, else 0 (NaN when unknown)."""
    forward = df["close"].shift(-horizon) / df["close"] - 1
    return (forward > threshold).astype(float).where(forward.notna())


def _features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURES if c in df]
    return df[cols].replace([np.inf, -np.inf], np.nan)


def _new_model(params: dict[str, Any]) -> LGBMClassifier:
    return LGBMClassifier(**params)


def train(df: pd.DataFrame, cfg: dict[str, Any]) -> LGBMClassifier | None:
    """Fit on every labelled row of ``df``; returns None if there is too little data."""
    X = _features(df)
    y = make_labels(df, cfg["horizon"], cfg["label_threshold"])
    mask = X.notna().all(axis=1) & y.notna()
    if mask.sum() < 200 or y[mask].nunique() < 2:
        return None
    model = _new_model(cfg["params"])
    model.fit(X[mask], y[mask].astype(int))
    return model


def predict_proba(model: LGBMClassifier, df: pd.DataFrame) -> pd.Series:
    X = _features(df)
    valid = X.notna().all(axis=1)
    out = pd.Series(np.nan, index=df.index, name="ml_proba")
    if valid.any():
        out[valid] = model.predict_proba(X[valid])[:, 1]
    return out


def walk_forward_proba(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """Out-of-sample probabilities, retraining every ``retrain_every`` candles."""
    X = _features(df)
    y = make_labels(df, cfg["horizon"], cfg["label_threshold"])
    valid = X.notna().all(axis=1).to_numpy()
    out = pd.Series(np.nan, index=df.index, name="ml_proba")
    n, horizon = len(df), cfg["horizon"]

    for start in range(cfg["min_train"], n, cfg["retrain_every"]):
        end = min(start + cfg["retrain_every"], n)
        train_mask = valid[: start - horizon] & y.iloc[: start - horizon].notna().to_numpy()
        if train_mask.sum() < 200:
            continue
        y_train = y.iloc[: start - horizon][train_mask].astype(int)
        if y_train.nunique() < 2:
            continue
        model = _new_model(cfg["params"])
        model.fit(X.iloc[: start - horizon][train_mask], y_train)
        test_mask = valid[start:end]
        if test_mask.any():
            idx = df.index[start:end][test_mask]
            out.loc[idx] = model.predict_proba(X.loc[idx])[:, 1]
    return out


def evaluate(df: pd.DataFrame, proba: pd.Series, cfg: dict[str, Any]) -> dict[str, float]:
    """Compare out-of-sample probabilities against realised labels."""
    y = make_labels(df, cfg["horizon"], cfg["label_threshold"])
    mask = proba.notna() & y.notna()
    if mask.sum() == 0 or y[mask].nunique() < 2:
        return {"samples": int(mask.sum())}
    y_true, p = y[mask].astype(int), proba[mask]
    return {
        "samples": int(mask.sum()),
        "base_rate": float(y_true.mean()),
        "auc": float(roc_auc_score(y_true, p)),
        "accuracy": float(accuracy_score(y_true, p > 0.5)),
        "majority_accuracy": float(max(y_true.mean(), 1 - y_true.mean())),
    }


def feature_importance(model: LGBMClassifier) -> pd.Series:
    return pd.Series(model.feature_importances_, index=model.feature_name_).sort_values(ascending=False)


def save(model: LGBMClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path) -> LGBMClassifier:
    return joblib.load(path)
