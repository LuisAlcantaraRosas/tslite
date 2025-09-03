
from __future__ import annotations
import pandas as pd
from typing import Optional, Dict, Any

from .models import make_model
from .metrics import compute_metrics


class BacktestResult:
    def __init__(self, metrics_by_fold: pd.DataFrame, predictions: pd.Series):
        self.metrics_by_fold = metrics_by_fold
        self.predictions = predictions


def backtest(
    y: pd.Series,
    model_name: str,
    model_params: Dict[str, Any],
    splitter,
    X: Optional[pd.DataFrame] = None,
    X_fut_builder=None,
    metrics: tuple[str, ...] = ("mae", "rmse", "mape"),
    ) -> BacktestResult:
    """Simple walk-forward backtest.

    Parameters
    ----------
    y : pd.Series
    Target series (datetime index recommended).
    model_name : str
    One of: "holt", "hw", "sarima".
    model_params : dict
    Params for the chosen model class.
    splitter : object
    Has .split(n) yielding (train_idx, val_idx).
    X : pd.DataFrame, optional
    Exogenous regressors aligned with y (for SARIMA only).
    X_fut_builder : callable, optional
    Function to build future exog per fold: f(y, tr_idx, vl_idx) -> X_future
    metrics : tuple[str, ...]
    Metric names.

    Returns
    -------
    BacktestResult
    """
    model = make_model(model_name, **(model_params or {}))

    recs = []
    preds = []

    for tr_idx, vl_idx in splitter.split(len(y)):
        tr_idx = list(tr_idx)
        vl_idx = list(vl_idx)
        y_tr, y_vl = y.iloc[tr_idx], y.iloc[vl_idx]

        X_tr = X.iloc[tr_idx] if X is not None else None
        X_vf = X_fut_builder(y, tr_idx, vl_idx) if X_fut_builder is not None else None

        model.fit(y_tr, X=X_tr)
        yhat = model.predict(h=len(vl_idx), X_fut=X_vf)

        # normaliza la salida a Series y asigna índice de validación
        if yhat is None:
            raise ValueError(f"Model '{model_name}' returned None on predict(). "
                            f"Check model params or the predict() implementation.")

        if not isinstance(yhat, pd.Series):
            yhat = pd.Series(yhat, index=y.iloc[vl_idx].index, name="yhat")
        else:
            yhat = yhat.copy()
            yhat.index = y.iloc[vl_idx].index

        preds.append(yhat)

        recs.append(compute_metrics(y.iloc[vl_idx], yhat, metrics))

    metrics_df = pd.DataFrame(recs)
    predictions = pd.concat(preds).sort_index()

    return BacktestResult(metrics_df, predictions)
