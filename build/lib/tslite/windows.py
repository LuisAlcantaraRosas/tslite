
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple


def rolling_window(series: pd.Series, window: int, step: int = 1) -> list[pd.Series]:
    """Yield rolling windows over a Series.

    Parameters
    ----------
    series : pd.Series
    Time-indexed series.
    window : int
    Number of observations per window.
    step : int, default 1
    Step size between windows.

    Returns
    -------
    list[pd.Series]
    List of windowed series (views/copies by slice).
    """
    out = []
    n = len(series)
    i = window
    while i <= n:
        out.append(series.iloc[i - window : i])
        i += step
    return out

