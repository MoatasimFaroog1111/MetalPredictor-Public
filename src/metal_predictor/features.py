from __future__ import annotations
import numpy as np
import pandas as pd
from metal_predictor.core import ColumnConfig, FeatureConfig


def _exact_lag(series: pd.Series, timestamps: pd.Series, hours: int) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    keyed = pd.Series(series.to_numpy(dtype=float), index=idx)
    wanted = idx - pd.Timedelta(hours=hours)
    return pd.Series(keyed.reindex(wanted).to_numpy(), index=series.index, dtype=float)

class PriceActionFeatures:
    def __init__(self, c: ColumnConfig) -> None:
        self._c = c
        self.feature_names = ("candle_range_pct","candle_body_pct","upper_wick_pct","lower_wick_pct","close_location_value","log_hl_range")
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out=frame.copy(); o,h,l,c=(out[self._c.open],out[self._c.high],out[self._c.low],out[self._c.close])
        safe_o=o.replace(0,np.nan); span=h-l; safe_span=span.replace(0,np.nan)
        out["candle_range_pct"]=span/safe_o; out["candle_body_pct"]=(c-o)/safe_o
        out["upper_wick_pct"]=(h-pd.concat([o,c],axis=1).max(axis=1))/safe_o
        out["lower_wick_pct"]=(pd.concat([o,c],axis=1).min(axis=1)-l)/safe_o
        out["close_location_value"]=((c-l)/safe_span).fillna(0.5); out["log_hl_range"]=np.log(h/l); return out

class MomentumFeatures:
    def __init__(self,c:ColumnConfig,cfg:FeatureConfig)->None:
        self._c,self._cfg=c,cfg; names=[]
        for lag in cfg.return_lags:
            names.extend([f"has_exact_{lag}h",f"log_return_{lag}h"])
            if lag>1:names.append(f"momentum_{lag}h")
        names.append(f"rsi_{cfg.rsi_window}h"); self.feature_names=tuple(names)
    def transform(self,frame:pd.DataFrame)->pd.DataFrame:
        out=frame.copy(); ts=pd.to_datetime(out[self._c.timestamp],utc=True); close=out[self._c.close].astype(float); log_close=np.log(close)
        for lag in self._cfg.return_lags:
            prior_log=_exact_lag(log_close,ts,lag); prior_close=_exact_lag(close,ts,lag)
            out[f"has_exact_{lag}h"]=prior_close.notna().astype("int8"); out[f"log_return_{lag}h"]=log_close-prior_log
            if lag>1: out[f"momentum_{lag}h"]=close/prior_close-1.0
        prev=_exact_lag(close,ts,1); delta=close-prev; gains=delta.clip(lower=0); losses=-delta.clip(upper=0); n=self._cfg.rsi_window
        idx=pd.DatetimeIndex(ts); minp=max(2,int(np.ceil(n*0.5)))
        ag=pd.Series(gains.to_numpy(float),index=idx).rolling(f"{n}h",min_periods=minp).mean(); al=pd.Series(losses.to_numpy(float),index=idx).rolling(f"{n}h",min_periods=minp).mean()
        rs=ag/al.replace(0,np.nan); rsi=100-100/(1+rs); rsi=rsi.mask(ag.eq(0)&al.eq(0),50).mask(al.eq(0)&ag.gt(0),100); out[f"rsi_{n}h"]=rsi.to_numpy(); return out

class VolatilityFeatures:
    def __init__(self,c:ColumnConfig,cfg:FeatureConfig)->None:
        self._c,self._cfg=c,cfg; self.feature_names=tuple([f"realized_vol_{w}h" for w in cfg.volatility_windows]+[f"atr_pct_{cfg.atr_window}h"])
    def transform(self,frame:pd.DataFrame)->pd.DataFrame:
        out=frame.copy(); ts=pd.to_datetime(out[self._c.timestamp],utc=True); idx=pd.DatetimeIndex(ts); close=out[self._c.close].astype(float); log_close=np.log(close)
        ret=log_close-_exact_lag(log_close,ts,1); keyed=pd.Series(ret.to_numpy(float),index=idx)
        for w in self._cfg.volatility_windows: out[f"realized_vol_{w}h"]=keyed.rolling(f"{w}h",min_periods=max(2,int(np.ceil(w*0.5)))).std(ddof=0).to_numpy()
        high=out[self._c.high].astype(float); low=out[self._c.low].astype(float); prev=_exact_lag(close,ts,1)
        tr=pd.concat([high-low,(high-prev).abs(),(low-prev).abs()],axis=1).max(axis=1); n=self._cfg.atr_window
        atr=pd.Series(tr.to_numpy(float),index=idx).rolling(f"{n}h",min_periods=max(2,int(np.ceil(n*0.5)))).mean(); out[f"atr_pct_{n}h"]=atr.to_numpy()/close.to_numpy(); return out

class TrendFeatures:
    def __init__(self,c:ColumnConfig,cfg:FeatureConfig)->None:
        self._c,self._cfg=c,cfg; names=[]
        for w in cfg.trend_windows:names.extend([f"close_vs_sma_{w}h",f"sma_slope_1h_{w}h",f"range_position_{w}h"])
        self.feature_names=tuple(names)
    def transform(self,frame:pd.DataFrame)->pd.DataFrame:
        out=frame.copy(); ts=pd.to_datetime(out[self._c.timestamp],utc=True); idx=pd.DatetimeIndex(ts); close=out[self._c.close].astype(float); keyed=pd.Series(close.to_numpy(float),index=idx)
        for w in self._cfg.trend_windows:
            roll=keyed.rolling(f"{w}h",min_periods=max(2,int(np.ceil(w*0.5)))); sma=roll.mean(); lo=roll.min(); hi=roll.max(); sma_series=pd.Series(sma.to_numpy(float),index=out.index); prev=_exact_lag(sma_series,ts,1)
            out[f"close_vs_sma_{w}h"]=close.to_numpy()/sma.to_numpy()-1; out[f"sma_slope_1h_{w}h"]=sma.to_numpy()/prev.to_numpy()-1
            span=(hi-lo).replace(0,np.nan); out[f"range_position_{w}h"]=((keyed-lo)/span).fillna(0.5).to_numpy()
        return out

class TemporalFeatures:
    def __init__(self,c:ColumnConfig)->None:
        self._c=c; self.feature_names=("hour_sin","hour_cos","weekday_sin","weekday_cos","gap_from_previous_hours","is_contiguous_from_previous")
    def transform(self,frame:pd.DataFrame)->pd.DataFrame:
        out=frame.copy(); ts=pd.to_datetime(out[self._c.timestamp],utc=True); hour=ts.dt.hour.astype(float); wd=ts.dt.dayofweek.astype(float)
        out["hour_sin"]=np.sin(2*np.pi*hour/24); out["hour_cos"]=np.cos(2*np.pi*hour/24); out["weekday_sin"]=np.sin(2*np.pi*wd/7); out["weekday_cos"]=np.cos(2*np.pi*wd/7)
        gap=ts.diff().dt.total_seconds().div(3600); out["gap_from_previous_hours"]=gap; out["is_contiguous_from_previous"]=gap.eq(1).astype("int8"); return out

class QualityFeatures:
    def __init__(self,c:ColumnConfig)->None:self._c=c; self.feature_names=("is_partial_source_hour",)
    def transform(self,frame:pd.DataFrame)->pd.DataFrame:
        out=frame.copy(); q=out[self._c.quality].astype("string").fillna("") if self._c.quality in out.columns else pd.Series("",index=out.index,dtype="string"); out["is_partial_source_hour"]=q.eq("PARTIAL_SOURCE_HOUR").astype("int8"); return out
