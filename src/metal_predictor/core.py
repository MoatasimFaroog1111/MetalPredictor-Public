from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ColumnConfig:
    timestamp: str = "timestamp_utc"
    open: str = "open_usd_per_kg"
    high: str = "high_usd_per_kg"
    low: str = "low_usd_per_kg"
    close: str = "close_usd_per_kg"
    quality: str = "quality_flag"

    @property
    def required(self) -> tuple[str, ...]:
        return (self.timestamp, self.open, self.high, self.low, self.close)

@dataclass(frozen=True)
class FeatureConfig:
    return_lags: tuple[int, ...] = (1, 3, 6, 12, 24, 72, 168)
    volatility_windows: tuple[int, ...] = (6, 12, 24, 72, 168)
    trend_windows: tuple[int, ...] = (12, 24, 72, 168)
    rsi_window: int = 14
    atr_window: int = 14

@dataclass(frozen=True)
class PipelineConfig:
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
