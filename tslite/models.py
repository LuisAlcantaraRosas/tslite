from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseTSModel(ABC):
    def __init__(self, **kwargs):
        self.kw = kwargs

    #@abstractmethod
    #def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "BaseTSModel":
    

    #@abstractmethod
    #def predict(self, h: int, X_fut: Optional[pd.DataFrame] = None) -> pd.Series:



# --- Registry ---
MODEL_REGISTRY: Dict[str, type[BaseTSModel]] = {}


def register_model(name: str):
    def deco(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return deco



def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def make_model(name: str, **kwargs) -> BaseTSModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    return MODEL_REGISTRY[name](**kwargs)

def _future_index_like(last_index, h: int):
    import pandas as pd
    # intenta respetar freq del índice si existe
    freq = getattr(last_index, "freq", None) or pd.infer_freq(last_index)
    if freq:
        return pd.date_range(last_index[-1], periods=h+1, freq=freq)[1:]
    # fallback: RangeIndex
    return pd.RangeIndex(len(last_index), len(last_index) + h)


# --- Holt (double), Holt-Winters, and ARIMA/SARIMA adapters ---
@register_model("holt")
class HoltModel(BaseTSModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from statsmodels.tsa.holtwinters import Holt
        self.Holt = Holt
        self._fit_res = None
        self._index = None

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None):
        self._index = y.index
        self._fit_res = self.Holt(y, **self.kw).fit()
        return self
    
    def predict(self, h: int, X_fut: Optional[pd.DataFrame] = None) -> pd.Series:
            # statsmodels holt/ES soportan forecast(h)
            pred = self._fit_res.forecast(h)
            fut_idx = _future_index_like(self._index, h)
            return pd.Series(pd.Series(pred).values, index=fut_idx, name="yhat")


@register_model("hw")
class HoltWintersModel(BaseTSModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self.ES = ExponentialSmoothing
        self._fit_res = None
        self._index = None

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None):
        self._index = y.index
        self._fit_res = self.ES(y, **self.kw).fit()
        return self

    def predict(self, h: int, X_fut: Optional[pd.DataFrame] = None) -> pd.Series:
            pred = self._fit_res.forecast(h)
            fut_idx = _future_index_like(self._index, h)
            return pd.Series(pd.Series(pred).values, index=fut_idx, name="yhat")


@register_model("sarima")
class SARIMAModel(BaseTSModel):
    """ARIMA/SARIMA via statsmodels SARIMAX.
    Use (p,d,q) and (P,D,Q,s) via order=() and seasonal_order=().
    """
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), trend=None, **kwargs):
        super().__init__(order=order, seasonal_order=seasonal_order, trend=trend, **kwargs)
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        self.SARIMAX = SARIMAX
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self._fit_res = None
        self._index = None

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None):
        self._index = y.index
        self._fit_res = self.SARIMAX(
        y,
        exog=X,
        order=self.order,
        seasonal_order=self.seasonal_order,
        trend=self.trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, h: int, X_fut: Optional[pd.DataFrame] = None) -> pd.Series:
        fc = self._fit_res.get_forecast(steps=h, exog=X_fut)
        mean = pd.Series(fc.predicted_mean)
        fut_idx = _future_index_like(self._index, h)
        return pd.Series(mean.values, index=fut_idx, name="yhat")

