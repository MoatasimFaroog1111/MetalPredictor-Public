from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest
from metal_predictor.alignment import ExactTimestampAligner
from metal_predictor.core import ColumnConfig
from metal_predictor.precious_metals.contracts import PLATINUM
from metal_predictor.precious_metals.dukascopy_source import DukascopyHistoricalMetalSource
from metal_predictor.precious_metals.features import PlatinumPalladiumCrossAssetFeatures
from metal_predictor.price_normalization import TROY_OZ_PER_KG

C=ColumnConfig()

class FakeTransport:
    def __init__(self):self.calls=[]
    def get_json(self,params):
        self.calls.append(dict(params))
        if params["path"]=="api/instrumentList":return [{"id":101,"name":"XPT.CMD/USD"},{"id":102,"name":"XPD.CMD/USD"}]
        start=int(params["start"]); return [{"timestamp":start,"open":1000,"high":1010,"low":995,"close":1005},{"timestamp":start+3600000,"open":1005,"high":1012,"low":1001,"close":1008}]

def test_source_contract_and_normalization():
    t=FakeTransport(); s=DukascopyHistoricalMetalSource("fake-secret",t); start=datetime(2026,1,1,tzinfo=timezone.utc); frame=s.fetch_hourly(PLATINUM,start,start.replace(hour=1))
    assert len(frame)==2; assert frame.loc[0,"open_usd_per_kg"]==pytest.approx(1000*TROY_OZ_PER_KG)
    q=[c for c in t.calls if c["path"]=="api/historicalPrices"][0]; assert q["timeFrame"]=="1hour" and q["dayStartTime"]=="UTC" and q["offerSide"]=="B"
    assert not hasattr(s,"place_order") and not hasattr(s,"cancel_order")

def silver(n=400):
    ts=pd.date_range("2025-01-01",periods=n,freq="h",tz="UTC"); close=2100+np.arange(n)*.02+np.sin(np.arange(n)/13)
    return pd.DataFrame({C.timestamp:ts,C.open:close-.2,C.high:close+.5,C.low:close-.5,C.close:close,C.quality:"OK"})
def metal(asset,n=400):
    ts=pd.date_range("2025-01-01",periods=n,freq="h",tz="UTC"); close=(31000 if asset=="XPT" else 33000)+np.arange(n)*.5+np.sin(np.arange(n)/(9 if asset=="XPT" else 10))*8
    return pd.DataFrame({"timestamp_utc":ts,"open_usd_per_kg":close-2,"high_usd_per_kg":close+4,"low_usd_per_kg":close-4,"close_usd_per_kg":close,"quality_flag":"PROVIDER_H1_BID"})

def test_pre_registered_family_and_missing_hour():
    pt=metal("XPT",100).drop(index=[40]).reset_index(drop=True); pdm=metal("XPD",100); comp=PlatinumPalladiumCrossAssetFeatures(pt,pdm,ExactTimestampAligner(),C); out=comp.transform(silver(100))
    assert len(comp.feature_names)==43 and len(set(comp.feature_names))==43
    assert out.loc[40,"xpt_has_exact_current"]==0 and np.isnan(out.loc[40,"log_xpt_silver_ratio"])
    assert out.loc[40,"xpd_has_exact_current"]==1 and out.loc[40,"both_metals_have_exact_current"]==0

def test_future_auxiliary_changes_cannot_change_past_features():
    s=silver(); pt=metal("XPT"); pdm=metal("XPD"); cutoff=250; base_comp=PlatinumPalladiumCrossAssetFeatures(pt,pdm,ExactTimestampAligner(),C); base=base_comp.transform(s)
    pt2=pt.copy(); pd2=pdm.copy(); cols=["open_usd_per_kg","high_usd_per_kg","low_usd_per_kg","close_usd_per_kg"]; pt2.loc[cutoff+1:,cols]*=3; pd2.loc[cutoff+1:,cols]*=4
    changed_comp=PlatinumPalladiumCrossAssetFeatures(pt2,pd2,ExactTimestampAligner(),C); changed=changed_comp.transform(s)
    pd.testing.assert_frame_equal(base.loc[:cutoff,base_comp.feature_names],changed.loc[:cutoff,changed_comp.feature_names],check_dtype=False,rtol=0,atol=1e-14)
