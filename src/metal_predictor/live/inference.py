from __future__ import annotations
from datetime import datetime, timedelta, timezone
import logging, math
from pathlib import Path
import numpy as np, pandas as pd
from metal_predictor.frozen_ridge import FrozenRidgeRegressor
from metal_predictor.future_features import SilverFeatureAssembler
from metal_predictor.live.contracts import ForecastSnapshot, HourlySilverBar

logger=logging.getLogger(__name__)

class LivePredictionEngine:
    """Frozen causal inference only; no fitting, tuning, scoring, or trade execution."""
    def __init__(self,artifact_dir:Path)->None:
        root=Path(artifact_dir)
        self._historical=pd.read_csv(root/'live_context.csv')
        self._historical['timestamp_utc']=pd.to_datetime(self._historical['timestamp_utc'],utc=True,errors='raise')
        self._historical=self._historical.sort_values('timestamp_utc').reset_index(drop=True)
        self._historical_last=pd.Timestamp(self._historical['timestamp_utc'].iloc[-1])
        self._baseline=FrozenRidgeRegressor.from_path(root/'ridge_alpha_100.json'); self._challenger=FrozenRidgeRegressor.from_path(root/'ridge_alpha_10.json')
        self._assembler=SilverFeatureAssembler()
        if self._baseline.feature_names!=self._challenger.feature_names or tuple(self._assembler.feature_names)!=self._baseline.feature_names: raise ValueError('Frozen model feature contract mismatch.')
    @property
    def historical_last_datetime_utc(self)->datetime:return self._historical_last.to_pydatetime()
    @property
    def baseline_model_name(self)->str:return self._baseline.model_name
    @property
    def challenger_model_name(self)->str:return self._challenger.model_name
    def model_status(self)->dict[str,object]:
        return {'baseline_model':self._baseline.model_name,'baseline_model_sha256':self._baseline.model_payload_sha256,'challenger_model':self._challenger.model_name,'challenger_model_sha256':self._challenger.model_payload_sha256,'feature_count':len(self._baseline.feature_names),'target':'next exact-hour XAG/USD log return','historical_context_last_timestamp_utc':self._historical_last.isoformat(),'edge_status':'NOT_PROVEN','research_only':True,'buy_sell_enabled':False}
    def predict(self,live_bars:list[HourlySilverBar])->ForecastSnapshot:
        if not live_bars: raise ValueError('At least one live bar is required.')
        ordered=sorted(live_bars,key=lambda x:x.timestamp_utc); rows=[]
        for b in ordered:
            if b.timestamp_utc.tzinfo is None: raise ValueError('Live bar timestamps must be timezone-aware.')
            ts=b.timestamp_utc.astimezone(timezone.utc)
            if ts<=self._historical_last: raise ValueError('Live bars must be newer than frozen historical context.')
            rows.append({'timestamp_utc':ts,'open_usd_per_kg':b.open_usd_per_kg,'high_usd_per_kg':b.high_usd_per_kg,'low_usd_per_kg':b.low_usd_per_kg,'close_usd_per_kg':b.close_usd_per_kg,'minute_count':b.minute_count,'quality_flag':b.quality_flag})
        live=pd.DataFrame(rows); combined=pd.concat([self._historical,live],ignore_index=True,sort=False).sort_values('timestamp_utc').reset_index(drop=True); featured=self._assembler.transform(combined); newest=pd.Timestamp(live['timestamp_utc'].iloc[-1]); row=featured.loc[featured['timestamp_utc'].eq(newest)]
        if len(row)!=1: raise ValueError('Could not resolve latest feature row.')
        values=row.loc[:,self._baseline.feature_names].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        # NaN is expected around market gaps. The sealed model reproduces training-time
        # median imputation and missing indicators. Never forward-fill exact-clock lags.
        if np.isinf(values).any(): raise ValueError('LIVE_FEATURES_INVALID_INFINITE '+newest.isoformat())
        br=float(self._baseline.predict(row)[0]); cr=float(self._challenger.predict(row)[0]); close=float(row['close_usd_per_kg'].iloc[0]); latest=ordered[-1]
        return ForecastSnapshot(feature_timestamp_utc=newest.to_pydatetime(),decision_time_utc=(newest+pd.Timedelta(hours=1)).to_pydatetime(),current_price_usd_per_kg=close,baseline_model=self._baseline.model_name,baseline_log_return_1h=br,baseline_predicted_price_usd_per_kg=close*math.exp(br),baseline_direction=self._direction(br),challenger_model=self._challenger.model_name,challenger_log_return_1h=cr,challenger_predicted_price_usd_per_kg=close*math.exp(cr),challenger_direction=self._direction(cr),data_quality=latest.quality_flag,source_provider=latest.source_provider,source_compatible_with_training=(latest.source_provider=='HistData' and latest.market_type=='spot_bid'))
    @staticmethod
    def _direction(v:float)->str:return 'UP' if v>0 else 'DOWN' if v<0 else 'FLAT'

class LiveForecastOrchestrator:
    def __init__(self,repository,engine:LivePredictionEngine,notifier=None)->None:self._repo=repository; self._engine=engine; self._notifier=notifier
    def ingest_bar(self,bar:HourlySilverBar)->bool:return self._repo.put_bar(bar)
    def materialize_latest_forecast(self)->tuple[ForecastSnapshot,bool]:
        snapshot=self._engine.predict(self._repo.recent_bars(limit=5000)); created=self._repo.put_forecast(snapshot)
        if created and self._notifier is not None:
            try:self._notifier.publish_forecast(snapshot)
            except Exception: logger.exception('Telegram notification failed after forecast persistence.')
        return snapshot,created
    def ingest_and_forecast(self,bar:HourlySilverBar)->tuple[ForecastSnapshot,bool,bool]:
        created=self.ingest_bar(bar); snapshot,fcreated=self.materialize_latest_forecast(); return snapshot,created,fcreated
    def latest(self):return self._repo.latest_forecast()
    @staticmethod
    def previous_completed_hour(now_utc:datetime|None=None)->datetime:
        now=(now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc); return now.replace(minute=0,second=0,microsecond=0)-timedelta(hours=1)
