from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from typing import Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from metal_predictor.precious_metals.contracts import PreciousMetalInstrument
from metal_predictor.price_normalization import TROY_OZ_PER_KG


class PublicFeedTransport(Protocol):
    def get_json(self, url: str) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class DukascopyPublicFeedSpec:
    instrument_name: str
    feed_code: str
    earliest_h1_utc: datetime


@dataclass(frozen=True)
class DukascopyPublicFeedBucket:
    url: str
    first_hour_utc: datetime
    last_hour_utc: datetime
    active_bucket: bool


_FEED_SPECS = {
    "XPT.CMD/USD": DukascopyPublicFeedSpec(
        "XPT.CMD/USD", "XPT.CMD-USD", datetime(2021, 11, 1, tzinfo=timezone.utc)
    ),
    "XPD.CMD/USD": DukascopyPublicFeedSpec(
        "XPD.CMD/USD", "XPD.CMD-USD", datetime(2021, 7, 4, 22, tzinfo=timezone.utc)
    ),
}


class UrllibPublicFeedTransport:
    def __init__(self, timeout_seconds: float = 30.0, max_response_bytes: int = 5_000_000) -> None:
        self._timeout = float(timeout_seconds)
        self._max_response_bytes = int(max_response_bytes)
        if not math.isfinite(self._timeout) or self._timeout <= 0:
            raise ValueError("timeout_seconds must be finite and positive.")
        if self._max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive.")

    def get_json(self, url: str) -> Mapping[str, object]:
        if not url.startswith(DukascopyPublicH1UrlPlanner.DATA_API_ROOT + "/"):
            raise ValueError("Dukascopy public-feed URL must use the fixed data API root.")
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "MetalPredictor-Research/1.0"})
        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read(self._max_response_bytes + 1)
        except HTTPError as exc:
            raise RuntimeError(f"Dukascopy public feed returned HTTP {exc.code}.") from None
        except URLError:
            raise RuntimeError("Dukascopy public feed request failed.") from None
        if len(raw) > self._max_response_bytes:
            raise RuntimeError("Dukascopy public feed response exceeded the safety limit.")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("Dukascopy public feed returned invalid JSON.") from None
        if not isinstance(payload, Mapping):
            raise RuntimeError("Dukascopy public feed returned a non-object payload.")
        return payload


class DukascopyPublicH1UrlPlanner:
    DATA_API_ROOT = "https://jetta.dukascopy.com/v1"

    @classmethod
    def feed_spec(cls, instrument: PreciousMetalInstrument) -> DukascopyPublicFeedSpec:
        try:
            return _FEED_SPECS[instrument.dukascopy_name.upper()]
        except KeyError:
            raise ValueError(f"No keyless Dukascopy feed spec registered for {instrument.dukascopy_name}.") from None

    def plan(self, instrument: PreciousMetalInstrument, start_utc: datetime, end_utc: datetime, *, now_utc: datetime | None = None) -> tuple[DukascopyPublicFeedBucket, ...]:
        start = self._exact_hour(start_utc, "start_utc")
        end = self._exact_hour(end_utc, "end_utc")
        if end < start:
            raise ValueError("end_utc must be on or after start_utc.")
        now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        spec = self.feed_spec(instrument)
        effective_start = max(start, spec.earliest_h1_utc)
        if effective_start > end:
            return ()
        cursor = datetime(effective_start.year, effective_start.month, 1, tzinfo=timezone.utc)
        buckets = []
        while cursor <= end:
            next_month = self._next_month(cursor)
            active = cursor <= now < next_month
            if active:
                url = f"{self.DATA_API_ROOT}/candles/hour/{spec.feed_code}/BID?from={int(cursor.timestamp()*1000)}"
            else:
                url = f"{self.DATA_API_ROOT}/candles/hour/{spec.feed_code}/BID/{cursor.year}/{cursor.month}"
            buckets.append(DukascopyPublicFeedBucket(url, max(effective_start, cursor), min(end, next_month-pd.Timedelta(hours=1).to_pytimedelta()), active))
            cursor = next_month
        return tuple(buckets)

    @staticmethod
    def _next_month(value: datetime) -> datetime:
        return datetime(value.year + (value.month == 12), 1 if value.month == 12 else value.month + 1, 1, tzinfo=timezone.utc)

    @staticmethod
    def _exact_hour(value: datetime, name: str) -> datetime:
        if value.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware.")
        utc = value.astimezone(timezone.utc)
        if utc.minute or utc.second or utc.microsecond:
            raise ValueError(f"{name} must align to an exact UTC hour.")
        return utc


class DukascopyCompressedH1Decoder:
    H1_SHIFT_MS = 3_600_000

    def decode(self, payload: Mapping[str, object]) -> tuple[tuple[datetime, float, float, float, float], ...]:
        times = self._array(payload, "times", non_negative=True)
        if not times:
            return ()
        multiplier = self._positive_float(payload.get("multiplier"), "multiplier")
        timestamp = self._integer(payload.get("timestamp"), "timestamp")
        shift = self._integer(payload.get("shift"), "shift")
        if shift != self.H1_SHIFT_MS:
            raise RuntimeError(f"Dukascopy public feed returned shift={shift}; expected exact H1 shift.")
        base = {name: self._positive_float(payload.get(name), name) for name in ("open", "high", "low", "close")}
        deltas = {name: self._array(payload, plural) for name, plural in (("open", "opens"), ("high", "highs"), ("low", "lows"), ("close", "closes"))}
        if any(len(values) != len(times) for values in deltas.values()):
            raise RuntimeError("Dukascopy public feed candle columns have mismatched lengths.")
        units = {name: int(round(value / multiplier)) for name, value in base.items()}
        rows = []
        current = timestamp
        for index, delta_time in enumerate(times):
            current += delta_time * shift
            for name in units:
                units[name] += deltas[name][index]
            prices = tuple(units[name] * multiplier for name in ("open", "high", "low", "close"))
            o, h, l, c = prices
            if not all(math.isfinite(v) and v > 0 for v in prices) or h < l or h < max(o, c) or l > min(o, c):
                raise RuntimeError("Dukascopy public feed produced an invalid OHLC candle.")
            ts = datetime.fromtimestamp(current / 1000.0, tz=timezone.utc)
            if ts.minute or ts.second or ts.microsecond:
                raise RuntimeError("Dukascopy public feed produced a non-exact UTC H1 timestamp.")
            rows.append((ts, o, h, l, c))
        return tuple(rows)

    @staticmethod
    def _positive_float(value: object, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            raise RuntimeError(f"Dukascopy public feed field {name} is invalid.") from None
        if isinstance(value, bool) or not math.isfinite(result) or result <= 0:
            raise RuntimeError(f"Dukascopy public feed field {name} must be positive.")
        return result

    @staticmethod
    def _integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"Dukascopy public feed field {name} must be numeric.")
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise RuntimeError(f"Dukascopy public feed field {name} must be an integer.")
        return int(numeric)

    @classmethod
    def _array(cls, payload: Mapping[str, object], name: str, *, non_negative: bool = False) -> tuple[int, ...]:
        raw = payload.get(name)
        if not isinstance(raw, list):
            raise RuntimeError(f"Dukascopy public feed field {name} must be an array.")
        values = []
        for value in raw:
            number = cls._integer(value, name)
            if non_negative and number < 0:
                raise RuntimeError(f"Dukascopy public feed field {name} must be non-negative.")
            values.append(number)
        return tuple(values)


class DukascopyPublicHistoricalMetalSource:
    def __init__(self, transport: PublicFeedTransport | None = None, planner: DukascopyPublicH1UrlPlanner | None = None, decoder: DukascopyCompressedH1Decoder | None = None) -> None:
        self._transport = transport or UrllibPublicFeedTransport()
        self._planner = planner or DukascopyPublicH1UrlPlanner()
        self._decoder = decoder or DukascopyCompressedH1Decoder()

    def fetch_hourly(self, instrument: PreciousMetalInstrument, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
        parsed = []
        for bucket in self._planner.plan(instrument, start_utc, end_utc):
            for ts, o, h, l, c in self._decoder.decode(self._transport.get_json(bucket.url)):
                if not bucket.first_hour_utc <= ts <= bucket.last_hour_utc:
                    continue
                parsed.append({"timestamp_utc": ts, "open_usd_per_kg": o*TROY_OZ_PER_KG, "high_usd_per_kg": h*TROY_OZ_PER_KG, "low_usd_per_kg": l*TROY_OZ_PER_KG, "close_usd_per_kg": c*TROY_OZ_PER_KG, "open_usd_per_oz": o, "high_usd_per_oz": h, "low_usd_per_oz": l, "close_usd_per_oz": c, "quality_flag": "PROVIDER_H1_BID", "source_provider": "Dukascopy Public Historical Feed", "source_symbol": instrument.dukascopy_name, "market_type": "commodity_cfd_cross_feed"})
        columns = ["timestamp_utc","open_usd_per_kg","high_usd_per_kg","low_usd_per_kg","close_usd_per_kg","open_usd_per_oz","high_usd_per_oz","low_usd_per_oz","close_usd_per_oz","quality_flag","source_provider","source_symbol","market_type"]
        if not parsed:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(parsed, columns=columns).sort_values("timestamp_utc").reset_index(drop=True)
        duplicates = frame[frame["timestamp_utc"].duplicated(keep=False)]
        for _, group in duplicates.groupby("timestamp_utc", sort=False):
            if len(group[["open_usd_per_kg","high_usd_per_kg","low_usd_per_kg","close_usd_per_kg"]].drop_duplicates()) > 1:
                raise RuntimeError("Dukascopy public feed returned conflicting duplicate H1 candles.")
        return frame.drop_duplicates(subset=["timestamp_utc"], keep="first").reset_index(drop=True)
