
from __future__ import annotations
from typing import Iterator, Tuple


def _as_range(idx):
    return range(idx.start, idx.stop) if hasattr(idx, "start") else idx


class RollingSplitter:
    """Generate (train_idx, val_idx) for rolling/expanding backtests.

    Parameters
    ----------
    window : int
    Training window size (ignored if expanding=True, where it becomes minimum window).
    horizon : int
    Forecast horizon for validation fold.
    step : int, default 1
    Shift between fold starts.
    expanding : bool, default False
    If True, use expanding train set; else fixed-size rolling train window.
    """

    def __init__(self, window: int, horizon: int, step: int = 1, expanding: bool = False):
        self.window = int(window)
        self.h = int(horizon)
        self.step = int(step)
        self.expanding = bool(expanding)

    def split(self, n: int):
        start = self.window
        while start + self.h <= n:
            if self.expanding:
              tr = range(0, start)
            else:
                tr = range(start - self.window, start)
            vl = range(start, start + self.h)
            yield tr, vl
            start += self.step

