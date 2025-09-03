
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def _eps(x: float = 1e-12) -> float:
    return x


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt = y_true.replace(0, np.nan)
    return float(np.mean(np.abs((yt - y_pred) / yt)))


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) + _eps()
    return float(200.0 * np.mean(np.abs(y_true - y_pred) / denom))


def mase(y_true: pd.Series, y_pred: pd.Series, m: int = 1) -> float:
    # scale by naive seasonal forecast
    diffs = np.abs(y_true[m:] - y_true[:-m])
    denom = np.mean(diffs) + _eps()
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


METRICS = {
"mae": mae,
"rmse": rmse,
"mape": mape,
"smape": smape,
"mase": mase,
}


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, metrics: tuple[str, ...] = ("mae", "rmse", "mape")) -> Dict[str, float]:
    return {name: METRICS[name](y_true, y_pred) for name in metrics}


