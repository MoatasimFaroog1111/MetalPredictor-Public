from __future__ import annotations
from datetime import datetime,timedelta,timezone
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,pytest
import httpx2 as httpx
from fastapi.testclient import TestClient
from metal_predictor.future_features import SilverFeatureAssembler
from metal_predictor.live.app import create_app
from metal_predictor.live.catchup import LiveMarketCatchUpService
from metal_predictor.live.contracts import HourlySilverBar
from metal_predictor.live.inference import LiveForecastOrchestrator,LivePredictionEngine
from metal_predictor.live.market_sources import TwelveDataSilverMinuteSource
from metal_predictor.live.notifications import TelegramForecastPublisher
from metal_predictor.live.repository import SQLiteForecastRepository
from metal_predictor.live.scheduler import HourlyCollectionScheduler
from metal_predictor.live.settings import LiveSettings

ROOT=Path(__file__).resolve().parents[1]

def _hash_payload(payload:dict[str,object])->dict[str,object]:
    clean=dict(payload); clean.pop('model_payload_sha256',None); payload=dict(clean); payload['model_payload_sha256']=hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest(); return payload

def _artifact_bundle(path:Path)->Path:
    path.mkdir(parents=True,exist_ok=True); assembler=SilverFeatureAssembler(); names=list(assembler.feature_names); n=len(names)
    start=datetime(2026,7,20,0,tzinfo=timezone.utc); rows=[]
    for i in range(500):
        ts=start+timedelta(hours=i); base=2000+12*np.sin(i/18)+i*0.05; op=base; cl=base+0.8*np.sin(i/5); hi=max(op,cl)+2; lo=min(op,cl)-2
        rows.append({'timestamp_utc':ts.isoformat(),'open_usd_per_kg':op,'high_usd_per_kg':hi,'low_usd_per_kg':lo,'close_usd_per_kg':cl,'minute_count':60,'quality_flag':'OK'})
    pd.DataFrame(rows).to_csv(path/'live_context.csv',index=False)
    base={'schema_version':1,'feature_names':names,'imputer_median':[0.0]*n,'missing_indicator_feature_indices':[],'scaler_mean':[0.0]*n,'scaler_scale':[1.0]*n,'ridge_coefficients':[0.0]*n,'training_rows':1000,'training_first_timestamp_utc':'2024-01-01T00:00:00+00:00','training_last_timestamp_utc':'2026-01-01T00:00:00+00:00','training_last_target_timestamp_utc':'2026-01-01T01:00:00+00:00','source_dataset_git_blob_sha':'synthetic-ci-fixture','research_code_cutoff_commit':'public-ci','target':'target_log_return_1h','prediction_semantics':'next exact-hour XAG/USD log return'}
    for name,alpha,intercept in [('ridge_alpha_100',100.0,0.001),('ridge_alpha_10',10.0,-0.0005)]:
        p=dict(base); p.update({'model_name':name,'family':'frozen_standardized_ridge','alpha':alpha,'ridge_intercept':intercept}); (path/f'{name}.json').write_text(json.dumps(_hash_payload(p),indent=2),encoding='utf-8')
    return path

def _bar(ts:datetime,close:float=2050.0)->HourlySilverBar:
    return HourlySilverBar(ts,close,close+2,close-2,close+0.5,60,'OK','Synthetic','XAGUSD','ci_fixture')

def test_feature_contract_is_52_and_causal_names_are_stable():
    names=SilverFeatureAssembler().feature_names; assert len(names)==52; assert 'log_return_168h' in names; assert 'gap_from_previous_hours' in names

def test_frozen_engine_and_repository_are_idempotent(tmp_path:Path):
    artifacts=_artifact_bundle(tmp_path/'artifacts'); engine=LivePredictionEngine(artifacts); repo=SQLiteForecastRepository(tmp_path/'db.sqlite3'); orch=LiveForecastOrchestrator(repo,engine)
    ts=engine.historical_last_datetime_utc+timedelta(hours=1); bar=_bar(ts)
    assert repo.put_bar(bar) is True; assert repo.put_bar(bar) is False
    snapshot,created=orch.materialize_latest_forecast(); assert created is True; assert snapshot.baseline_model=='ridge_alpha_100'; assert snapshot.challenger_model=='ridge_alpha_10'; assert snapshot.research_only is True; assert snapshot.buy_sell_enabled is False; assert np.isfinite(snapshot.baseline_log_return_1h)
    with pytest.raises(ValueError,match='LIVE_BAR_REVISION_CONFLICT'): repo.put_bar(_bar(ts,2100.0))

def test_fastapi_auth_pwa_and_security_headers(tmp_path:Path):
    artifacts=_artifact_bundle(tmp_path/'artifacts'); settings=LiveSettings(artifact_dir=artifacts,database_path=tmp_path/'api.sqlite3',admin_token='test-admin')
    with TestClient(create_app(settings,web_root=ROOT/'live_web')) as client:
        health=client.get('/api/v1/health'); assert health.status_code==200; assert health.json()['buy_sell_enabled'] is False; assert health.headers['cache-control']=='no-store'
        page=client.get('/'); assert page.status_code==200; assert page.headers['x-frame-options']=='DENY'; assert "frame-ancestors 'none'" in page.headers['content-security-policy']
        assert client.get('/sw.js').headers['service-worker-allowed']=='/'
        engine=client.app.state.engine; ts=engine.historical_last_datetime_utc+timedelta(hours=1); payload={'timestamp_utc':ts.isoformat(),'open_usd_per_kg':2050,'high_usd_per_kg':2052,'low_usd_per_kg':2048,'close_usd_per_kg':2050.5,'minute_count':60,'quality_flag':'OK'}
        assert client.post('/api/v1/market/silver/hourly',json=payload).status_code==401
        r=client.post('/api/v1/market/silver/hourly',json=payload,headers={'X-Admin-Token':'test-admin'}); assert r.status_code==200,r.text; assert r.json()['forecast_created'] is True

def test_twelvedata_m1_aggregation_is_bounded_and_redacts_key():
    start=datetime(2026,8,10,10,tzinfo=timezone.utc); values=[]
    for m in range(60):
        ts=start+timedelta(minutes=m); p=70+m*0.001; values.append({'datetime':ts.strftime('%Y-%m-%d %H:%M:%S'),'open':str(p),'high':str(p+0.01),'low':str(p-0.01),'close':str(p+0.002)})
    def handler(req:httpx.Request)->httpx.Response:
        assert req.url.params['outputsize']=='4320'; return httpx.Response(200,json={'status':'ok','values':values})
    source=TwelveDataSilverMinuteSource('synthetic-test-key',client=httpx.Client(transport=httpx.MockTransport(handler))); bars=source.fetch_hours(start,start); assert len(bars)==1; assert bars[0].minute_count==60; assert bars[0].source_provider=='TwelveData'
    with pytest.raises(ValueError,match='72-hour'): source.fetch_hours(start,start+timedelta(hours=72))
    def bad(req:httpx.Request)->httpx.Response:return httpx.Response(401,request=req,json={'status':'error'})
    bad_source=TwelveDataSilverMinuteSource('synthetic-test-key',client=httpx.Client(transport=httpx.MockTransport(bad)))
    with pytest.raises(RuntimeError) as cap: bad_source.fetch_hours(start,start)
    assert 'synthetic-test-key' not in str(cap.value)

def test_catchup_fills_context_without_retroactive_forecasts(tmp_path:Path):
    artifacts=_artifact_bundle(tmp_path/'artifacts'); engine=LivePredictionEngine(artifacts); repo=SQLiteForecastRepository(tmp_path/'catch.sqlite3'); orch=LiveForecastOrchestrator(repo,engine)
    first=engine.historical_last_datetime_utc+timedelta(hours=1); target=first+timedelta(hours=4)
    class Source:
        calls=[]
        def fetch_hours(self,start,end):
            self.calls.append((start,end)); out=[]; cur=start
            while cur<=end: out.append(_bar(cur,2050+(cur-first).total_seconds()/3600)); cur+=timedelta(hours=1)
            return out
    src=Source(); service=LiveMarketCatchUpService(src,repo,engine,orch,batch_hours=2); result=service.catch_up(target)
    assert len(src.calls)==3; assert result.created_bars==5; assert result.forecast_created is True; assert len(repo.forecast_history(20))==1; assert repo.latest_forecast().feature_timestamp_utc==target

def test_scheduler_time_and_telegram_secret_transport():
    now=datetime(2026,8,11,8,4,59,tzinfo=timezone.utc); assert HourlyCollectionScheduler.next_due_utc(now,5)==datetime(2026,8,11,8,5,tzinfo=timezone.utc)
    observed={}
    def handler(req:httpx.Request)->httpx.Response:
        observed['path']=req.url.path; observed['json']=json.loads(req.content.decode()); return httpx.Response(200,json={'ok':True})
    pub=TelegramForecastPublisher('000000:synthetic-test-token',('42',),client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ValueError,match='HTTPS'): pub.configure_webhook('http://example.test','synthetic-hook-secret')
    out=pub.configure_webhook('https://example.test','synthetic-hook-secret'); assert out['configured'] is True; assert observed['json']['secret_token']=='synthetic-hook-secret'
    def fail(req:httpx.Request)->httpx.Response:return httpx.Response(500,request=req)
    bad=TelegramForecastPublisher('000000:synthetic-test-token',('42',),client=httpx.Client(transport=httpx.MockTransport(fail)))
    with pytest.raises(RuntimeError) as cap: bad.send_text('42','x')
    assert '000000:synthetic-test-token' not in str(cap.value)
