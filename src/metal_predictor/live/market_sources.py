from __future__ import annotations
from datetime import datetime, timedelta, timezone
import httpx2 as httpx
import numpy as np
import pandas as pd
from metal_predictor.live.contracts import HourlySilverBar

OZ_PER_KG = 32.15074656862798

class TwelveDataSilverMinuteSource:
    """Operational XAG/USD adapter. Provider data is never treated as training-equivalent."""
    _URL='https://api.twelvedata.com/time_series'
    def __init__(self,api_key:str,symbol:str='XAG/USD',client:httpx.Client|None=None)->None:
        if not api_key.strip(): raise ValueError('Twelve Data API key is required.')
        self._key=api_key.strip(); self._symbol=symbol.strip() or 'XAG/USD'; self._client=client or httpx.Client(timeout=30.0)
    def fetch_hours(self,start_utc:datetime,end_utc:datetime)->list[HourlySilverBar]:
        start=self._hour(start_utc); end=self._hour(end_utc)
        if end<start: raise ValueError('end_utc must be >= start_utc.')
        possible=int((end-start).total_seconds()/60)+60
        if possible>4320: raise ValueError('Twelve Data range exceeds 72-hour safety bound.')
        request_end=end+timedelta(minutes=59,seconds=59)
        try:
            response=self._client.get(self._URL,params={'symbol':self._symbol,'interval':'1min','start_date':start.strftime('%Y-%m-%dT%H:%M:%S'),'end_date':request_end.strftime('%Y-%m-%dT%H:%M:%S'),'timezone':'UTC','order':'asc','outputsize':4320,'format':'JSON','apikey':self._key}); response.raise_for_status()
        except httpx.HTTPError: raise RuntimeError('Twelve Data transport request failed.') from None
        payload=response.json()
        if payload.get('status')=='error': raise RuntimeError(f"Twelve Data API rejected the request with code {payload.get('code','unknown')}.")
        values=payload.get('values')
        if not isinstance(values,list): raise RuntimeError('Twelve Data response does not contain minute values.')
        if not values:return []
        return self._aggregate(values,start,end)
    def _aggregate(self,values:list[dict[str,object]],start:datetime,end:datetime)->list[HourlySilverBar]:
        df=pd.DataFrame(values); required={'datetime','open','high','low','close'}; missing=required.difference(df.columns)
        if missing: raise RuntimeError(f'Twelve Data minute response missing fields: {sorted(missing)}')
        df['timestamp_utc']=pd.to_datetime(df['datetime'],utc=True,errors='coerce')
        for c in ('open','high','low','close'):df[c]=pd.to_numeric(df[c],errors='coerce')
        if df['timestamp_utc'].isna().any(): raise RuntimeError('Twelve Data returned invalid timestamps.')
        prices=df[['open','high','low','close']]; finite=np.isfinite(prices.to_numpy(float)).all(axis=1); positive=prices.gt(0).all(axis=1); invariants=df['high'].ge(df['low']) & df['high'].ge(df[['open','close']].max(axis=1)) & df['low'].le(df[['open','close']].min(axis=1)); df=df.loc[finite & positive & invariants].copy()
        grouped=df.groupby('timestamp_utc')[['open','high','low','close']]
        conflicts=grouped.nunique(dropna=False).max(axis=1); conflict_ts=conflicts[conflicts>1].index
        conflict_hours=set(pd.DatetimeIndex(conflict_ts).floor('h'))
        df=df.loc[~df['timestamp_utc'].isin(conflict_ts)].drop_duplicates('timestamp_utc',keep='first').sort_values('timestamp_utc'); df['hour']=df['timestamp_utc'].dt.floor('h')
        out=[]
        for hour,g in df.groupby('hour',sort=True):
            dt=hour.to_pydatetime()
            if dt<start or dt>end or hour in conflict_hours:continue
            n=len(g)
            if not 1<=n<=60:continue
            out.append(HourlySilverBar(timestamp_utc=dt,open_usd_per_kg=float(g.iloc[0]['open'])*OZ_PER_KG,high_usd_per_kg=float(g['high'].max())*OZ_PER_KG,low_usd_per_kg=float(g['low'].min())*OZ_PER_KG,close_usd_per_kg=float(g.iloc[-1]['close'])*OZ_PER_KG,minute_count=n,quality_flag='OK' if n==60 else 'PARTIAL_SOURCE_HOUR',source_provider='TwelveData',source_symbol=self._symbol,market_type='spot_quote'))
        return out
    @staticmethod
    def _hour(v:datetime)->datetime:
        if v.tzinfo is None:raise ValueError('UTC timestamps must be timezone-aware.')
        u=v.astimezone(timezone.utc)
        if u.minute or u.second or u.microsecond:raise ValueError('Timestamps must align to exact UTC hours.')
        return u
