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

res_holt = backtest(y=y, model_name="holt",
                    model_params={"exponential": False, "damped_trend": True},
                    splitter=splitter)
```
