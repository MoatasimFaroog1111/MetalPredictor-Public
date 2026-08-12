from datetime import datetime, timezone
import pandas as pd
import pytest
from metal_predictor.precious_metals.contracts import PLATINUM
from metal_predictor.precious_metals.dukascopy_public_source import DukascopyCompressedH1Decoder,DukascopyPublicH1UrlPlanner,DukascopyPublicHistoricalMetalSource
from metal_predictor.price_normalization import TROY_OZ_PER_KG
UTC=timezone.utc

class FakeTransport:
    def __init__(self,payloads):self.payloads=payloads;self.calls=[]
    def get_json(self,url):self.calls.append(url);return self.payloads[url]

def p(ts,times):
    n=len(times);return {"timestamp":ts,"multiplier":0.01,"open":1000.0,"high":1010.0,"low":990.0,"close":1005.0,"shift":3600000,"times":times,"opens":[0]*n,"highs":[0]*n,"lows":[0]*n,"closes":[0]*n,"volumes":[1.0]*n}

def test_keyless_urls_and_provider_start():
    planner=DukascopyPublicH1UrlPlanner(); rows=planner.plan(PLATINUM,datetime(2021,8,1,tzinfo=UTC),datetime(2021,12,1,tzinfo=UTC),now_utc=datetime(2026,1,1,tzinfo=UTC))
    assert rows[0].first_hour_utc==datetime(2021,11,1,tzinfo=UTC)
    assert rows[0].url=="https://jetta.dukascopy.com/v1/candles/hour/XPT.CMD-USD/BID/2021/11"
    assert "key=" not in rows[0].url.lower()

def test_decoder_preserves_gap_and_rejects_wrong_shift():
    d=DukascopyCompressedH1Decoder(); base=int(datetime(2025,1,1,tzinfo=UTC).timestamp()*1000); rows=d.decode(p(base,[0,1,2])); assert [r[0].hour for r in rows]==[0,1,3]
    bad=p(base,[0]);bad["shift"]=60000
    with pytest.raises(RuntimeError,match="expected exact H1 shift"):d.decode(bad)

def test_source_normalizes_and_is_read_only():
    planner=DukascopyPublicH1UrlPlanner(); start=datetime(2025,1,1,1,tzinfo=UTC);end=datetime(2025,1,1,2,tzinfo=UTC);bucket=planner.plan(PLATINUM,start,end,now_utc=datetime(2026,1,1,tzinfo=UTC))[0];base=int(datetime(2025,1,1,tzinfo=UTC).timestamp()*1000);t=FakeTransport({bucket.url:p(base,[0,1,1,1])});s=DukascopyPublicHistoricalMetalSource(t,planner);f=s.fetch_hourly(PLATINUM,start,end)
    assert list(pd.to_datetime(f.timestamp_utc,utc=True))==list(pd.date_range(start,end,freq="h",tz="UTC"))
    assert f.iloc[0].open_usd_per_kg==pytest.approx(1000*TROY_OZ_PER_KG)
    assert f.source_provider.eq("Dukascopy Public Historical Feed").all()
    for name in ("place_order","cancel_order","submit_order","execute","buy","sell"):assert not hasattr(s,name)
