import pytest
from metal_predictor.precious_metals.provenance import HistoricalBootstrapManifest,HistoricalBootstrapProvenanceGate

def fixed_manifest(asset):
    return HistoricalBootstrapManifest(asset=asset,provider="Yahoo Finance via yfinance",source_symbol="PL=F" if asset=="XPT" else "PA=F",instrument_semantics="FUTURES",interval="1h",requested_history_days=729,has_open=False,has_high=True,has_low=True,has_close=True,exact_utc_hour_guaranteed=False,forward_fill_used=False,interpolation_used=False,provenance_manifest_embedded=False,source_repository="MoatasimFaroog1111/fixed",source_commit="672845b854cbfae2284f68df28369dd6f3de94bb")

def test_fixed_sources_are_rejected_before_model_fit():
    gate=HistoricalBootstrapProvenanceGate()
    for asset in ("XPT","XPD"):
        result=gate.assess(fixed_manifest(asset))
        assert not result.accepted
        assert set(result.failures)=={"INSTRUMENT_SEMANTICS_NOT_ALLOWED","INSUFFICIENT_HOURLY_HISTORY","INCOMPLETE_OHLC_SCHEMA","EXACT_UTC_HOUR_NOT_GUARANTEED","PROVENANCE_MANIFEST_MISSING"}
    with pytest.raises(ValueError,match="provenance gate rejected"):
        gate.validate_many((fixed_manifest("XPT"),fixed_manifest("XPD")))

def test_valid_exact_hour_cross_feed_manifest_passes():
    m=HistoricalBootstrapManifest(asset="XPT",provider="research-provider",source_symbol="XPT/USD",instrument_semantics="COMMODITY_CFD_CROSS_FEED",interval="1h",requested_history_days=1826,has_open=True,has_high=True,has_low=True,has_close=True,exact_utc_hour_guaranteed=True,forward_fill_used=False,interpolation_used=False,provenance_manifest_embedded=True)
    r=HistoricalBootstrapProvenanceGate().assess(m)
    assert r.accepted and r.failures==()
