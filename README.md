# tslite

Minimal sliding-window time series forecasting toolkit using statsmodels.

Features
- Rolling/expanding splitters
- Backtesting loop
- Models: Holt (double), Holt-Winters (ExponentialSmoothing), ARIMA/SARIMA (SARIMAX)
- Basic metrics (MAE, RMSE, MAPE, sMAPE, MASE)

Quickstart

```python
import pandas as pd
from tslite.splitters import RollingSplitter
from tslite.backtest import backtest

# y must be a pandas Series indexed by datetime
# y = pd.Series(...)

splitter = RollingSplitter(window=365, horizon=20, step=20, expanding=True)

res = backtest(
y=y,
model_name="hw",
model_params={"trend": "add", "seasonal": "mul", "seasonal_periods": 7},
splitter=splitter,
)

print(res.metrics_by_fold)
ax = y.plot(figsize=(12,5), label="actual")
res.predictions.plot(ax=ax, label="pred")
```
