from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.core import ColumnConfig
from metal_predictor.precious_metals_features import PlatinumPalladiumCrossAssetFeatures


def frame(start: str, closes: list[float]) -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(closes), freq="h", tz="UTC")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "timestamp_utc": ts,
        "open_usd_per_kg": c * 0.999,
        "high_usd_per_kg": c * 1.002,
        "low_usd_per_kg": c * 0.998,
        "close_usd_per_kg": c,
        "quality_flag": ["COMPLETE_SOURCE_HOUR"] * len(c),
    })


def test_exact_clock_returns_and_breadth() -> None:
    silver = frame("2026-01-01", [100, 101, 103, 102, 104, 106])
    platinum = frame("2026-01-01", [200, 202, 204, 205, 207, 210])
    palladium = frame("2026-01-01", [300, 299, 301, 303, 302, 304])
    out = PlatinumPalladiumCrossAssetFeatures(
        platinum, palladium, ExactTimestampAligner(), ColumnConfig(),
        lags=(1, 3), windows=(3,)
    ).transform(silver)
    i = 4
    sr = np.log(104 / 102)
    pr = np.log(207 / 205)
    dr = np.log(302 / 303)
    assert out.loc[i, "platinum_log_return_1h"] == pytest.approx(pr)
    assert out.loc[i, "palladium_log_return_1h"] == pytest.approx(dr)
    assert out.loc[i, "platinum_silver_relative_return_1h"] == pytest.approx(sr - pr)
    assert out.loc[i, "precious_metals_return_breadth_1h"] == pytest.approx(2 / 3)


def test_missing_timestamp_is_not_forward_filled() -> None:
    silver = frame("2026-01-01", [100, 101, 102, 103, 104])
    platinum = frame("2026-01-01", [200, 201, 202, 203, 204]).drop(index=2).reset_index(drop=True)
    palladium = frame("2026-01-01", [300, 301, 302, 303, 304])
    out = PlatinumPalladiumCrossAssetFeatures(
        platinum, palladium, ExactTimestampAligner(), ColumnConfig(), lags=(1,), windows=(3,)
    ).transform(silver)
    assert out.loc[2, "platinum_has_exact_current"] == 0
    assert pd.isna(out.loc[2, "log_platinum_silver_ratio"])
    assert out.loc[3, "platinum_has_exact_1h"] == 0
    assert pd.isna(out.loc[3, "platinum_log_return_1h"])


def test_future_rows_cannot_change_past_features() -> None:
    silver = frame("2026-01-01", [100, 101, 102, 103, 104, 105])
    platinum = frame("2026-01-01", [200, 201, 202, 203, 204, 205])
    palladium = frame("2026-01-01", [300, 301, 302, 303, 304, 305])
    build = lambda pt, pd_: PlatinumPalladiumCrossAssetFeatures(
        pt, pd_, ExactTimestampAligner(), ColumnConfig(), lags=(1, 3), windows=(3,)
    )
    full_component = build(platinum, palladium)
    full = full_component.transform(silver)
    truncated = build(platinum.iloc[:4], palladium.iloc[:4]).transform(silver.iloc[:4].copy())
    for name in full_component.feature_names:
        pd.testing.assert_series_equal(
            full.loc[:3, name].reset_index(drop=True),
            truncated[name].reset_index(drop=True),
            check_names=False,
        )
