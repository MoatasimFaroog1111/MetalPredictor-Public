from __future__ import annotations

import pytest

from metal_predictor.live.bullionvault_quote import BullionVaultQuoteProvider


MARKET_XML = """<?xml version="1.0" encoding="UTF-8"?>
<envelope><message type="MARKET_DEPTH_A"><market><pitches>
<pitch securityId="AGXLN" considerationCurrency="USD">
<buyPrices><price actionIndicator="B" quantity="2.500" limit="2120"/><price actionIndicator="B" quantity="4.000" limit="2119"/></buyPrices>
<sellPrices><price actionIndicator="S" quantity="1.750" limit="2122"/><price actionIndicator="S" quantity="3.000" limit="2123"/></sellPrices>
</pitch>
</pitches></market></message></envelope>
"""


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None


class FakeClient:
    def __init__(self, market_xml: str = MARKET_XML) -> None:
        self.market_xml = market_xml
        self.get_calls = []
        self.post_calls = []

    def get(self, url, params=None):
        self.get_calls.append((url, params))
        if "view_market_xml.do" in url:
            return FakeResponse(self.market_xml)
        return FakeResponse("<html/>")

    def post(self, url, data=None):
        self.post_calls.append((url, data))
        return FakeResponse("<html/>")


def test_public_top_of_book() -> None:
    client = FakeClient()
    quote = BullionVaultQuoteProvider(client=client, access_mode="public").fetch_quote()
    assert quote.best_bid_usd_per_kg == 2120.0
    assert quote.best_ask_usd_per_kg == 2122.0
    assert quote.spread_usd_per_kg == 2.0
    assert quote.best_bid_quantity_kg == 2.5
    assert quote.best_ask_quantity_kg == 1.75
    assert quote.access_mode == "PUBLIC_CACHED_READ_ONLY"
    assert client.post_calls == []


def test_authenticated_mode_uses_secure_market_and_has_no_order_api() -> None:
    client = FakeClient()
    provider = BullionVaultQuoteProvider(
        username="user",
        password="secret",
        access_mode="authenticated",
        client=client,
    )
    quote = provider.fetch_quote()
    assert quote.access_mode == "AUTHENTICATED_READ_ONLY"
    assert len(client.post_calls) == 1
    assert client.get_calls[-1][0].endswith("/secure/api/v2/view_market_xml.do")
    assert not hasattr(provider, "place_order")
    assert not hasattr(provider, "cancel_order")


def test_entity_declarations_are_rejected() -> None:
    provider = BullionVaultQuoteProvider(
        client=FakeClient('<!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><envelope/>'),
        access_mode="public",
    )
    with pytest.raises(RuntimeError, match="forbidden declarations"):
        provider.fetch_quote()
