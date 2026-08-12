from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx2 as httpx
import pytest

from metal_predictor.live.gold_api_source import GoldApiSilverOhlcSource
from metal_predictor.price_normalization import TROY_OZ_PER_KG


def test_gold_api_adapter_fetches_exact_hour_and_converts_to_usd_per_kg() -> None:
    start = datetime(2026, 8, 12, 5, tzinfo=timezone.utc)
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["path"] = request.url.path
        observed["params"] = dict(request.url.params)
        observed["key"] = request.headers.get("x-api-key")
        return httpx.Response(
            200,
            request=request,
            json={"open": 38.0, "high": 38.4, "low": 37.8, "close": 38.2},
        )

    source = GoldApiSilverOhlcSource(
        "gold-test-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    bar = source.fetch_completed_hour(start)

    assert observed["path"] == "/ohlc/XAG"
    assert observed["key"] == "gold-test-secret"
    params = observed["params"]
    assert isinstance(params, dict)
    assert int(params["startTimestamp"]) == int(start.timestamp())
    assert int(params["endTimestamp"]) == int(
        (start + timedelta(hours=1) - timedelta(seconds=1)).timestamp()
    )
    assert bar.timestamp_utc == start
    assert bar.source_provider == "GoldAPI"
    assert bar.quality_flag == "PROVIDER_AGGREGATED_H1"
    assert bar.open_usd_per_kg == pytest.approx(38.0 * TROY_OZ_PER_KG)
    assert bar.close_usd_per_kg == pytest.approx(38.2 * TROY_OZ_PER_KG)


def test_gold_api_range_policy_uses_one_request_for_newest_hour() -> None:
    requested: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(dict(request.url.params))
        return httpx.Response(
            200,
            request=request,
            json={"open": 40.0, "high": 40.2, "low": 39.8, "close": 40.1},
        )

    source = GoldApiSilverOhlcSource(
        "gold-test-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    start = datetime(2026, 8, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=30)
    bars = source.fetch_completed_range(start, end)

    assert len(requested) == 1
    assert len(bars) == 1
    assert bars[0].timestamp_utc == end
    assert int(requested[0]["startTimestamp"]) == int(end.timestamp())


def test_gold_api_http_error_redacts_secret() -> None:
    secret = "gold-super-secret-key"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, request=request, json={"error": "rate limit"})

    source = GoldApiSilverOhlcSource(
        secret,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(RuntimeError) as captured:
        source.fetch_completed_hour(datetime(2026, 8, 12, 5, tzinfo=timezone.utc))

    assert secret not in str(captured.value)
    assert str(captured.value) == "Gold API request failed with HTTP 429."
