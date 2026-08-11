from __future__ import annotations
import numpy as np
import pandas as pd
from metal_predictor.core import ColumnConfig

class SilverDatasetValidator:
    def __init__(self, columns: ColumnConfig) -> None:
        self._c = columns

    def validate(self, frame: pd.DataFrame) -> None:
        missing = [name for name in self._c.required if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if frame.empty:
            raise ValueError("Input dataset is empty.")
        ts = pd.to_datetime(frame[self._c.timestamp], utc=True, errors="coerce")
        if ts.isna().any() or ts.duplicated().any() or not ts.is_monotonic_increasing:
            raise ValueError("Timestamps must be valid, unique, and chronological.")
        prices = frame[[self._c.open, self._c.high, self._c.low, self._c.close]].apply(pd.to_numeric, errors="coerce")
        values = prices.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("OHLC must be finite and strictly positive.")
        invalid = (
            (prices[self._c.high] < prices[self._c.low])
            | (prices[self._c.high] < prices[[self._c.open, self._c.close]].max(axis=1))
            | (prices[self._c.low] > prices[[self._c.open, self._c.close]].min(axis=1))
        )
        if invalid.any():
            raise ValueError("Invalid OHLC invariants detected.")
