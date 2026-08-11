from __future__ import annotations
from datetime import datetime, timezone
import json, sqlite3
from pathlib import Path
from metal_predictor.live.contracts import ForecastSnapshot, HourlySilverBar

class SQLiteForecastRepository:
    def __init__(self,path:Path)->None:
        self._path=Path(path); self._path.parent.mkdir(parents=True,exist_ok=True); self._init()
    def _connect(self):
        conn=sqlite3.connect(self._path,timeout=30); conn.row_factory=sqlite3.Row; conn.execute('PRAGMA journal_mode=WAL'); conn.execute('PRAGMA synchronous=NORMAL'); return conn
    def _init(self)->None:
        with self._connect() as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS bars(
              timestamp_utc TEXT PRIMARY KEY, open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
              minute_count INTEGER NOT NULL, quality_flag TEXT NOT NULL, source_provider TEXT NOT NULL, source_symbol TEXT NOT NULL, market_type TEXT NOT NULL,
              payload_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS forecasts(
              feature_timestamp_utc TEXT PRIMARY KEY, payload_json TEXT NOT NULL);
            ''')
    @staticmethod
    def _canon(data:dict[str,object])->str:return json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    def put_bar(self,bar:HourlySilverBar)->bool:
        payload=bar.as_dict(); key=bar.timestamp_utc.astimezone(timezone.utc).isoformat(); encoded=self._canon(payload)
        with self._connect() as c:
            existing=c.execute('SELECT payload_json FROM bars WHERE timestamp_utc=?',(key,)).fetchone()
            if existing:
                if existing['payload_json']!=encoded: raise ValueError(f'LIVE_BAR_REVISION_CONFLICT {key}')
                return False
            c.execute('INSERT INTO bars VALUES(?,?,?,?,?,?,?,?,?,?,?)',(key,bar.open_usd_per_kg,bar.high_usd_per_kg,bar.low_usd_per_kg,bar.close_usd_per_kg,bar.minute_count,bar.quality_flag,bar.source_provider,bar.source_symbol,bar.market_type,encoded)); return True
    def recent_bars(self,limit:int=500)->list[HourlySilverBar]:
        with self._connect() as c: rows=c.execute('SELECT payload_json FROM bars ORDER BY timestamp_utc DESC LIMIT ?', (int(limit),)).fetchall()
        out=[]
        for row in reversed(rows):
            d=json.loads(row['payload_json']); d['timestamp_utc']=datetime.fromisoformat(d['timestamp_utc']); out.append(HourlySilverBar(**d))
        return out
    def put_forecast(self,snapshot:ForecastSnapshot)->bool:
        payload=snapshot.as_dict(); key=snapshot.feature_timestamp_utc.astimezone(timezone.utc).isoformat(); encoded=self._canon(payload)
        with self._connect() as c:
            existing=c.execute('SELECT payload_json FROM forecasts WHERE feature_timestamp_utc=?',(key,)).fetchone()
            if existing:
                if existing['payload_json']!=encoded: raise ValueError(f'LIVE_FORECAST_REVISION_CONFLICT {key}')
                return False
            c.execute('INSERT INTO forecasts VALUES(?,?)',(key,encoded)); return True
    @staticmethod
    def _forecast(encoded:str)->ForecastSnapshot:
        d=json.loads(encoded); d['feature_timestamp_utc']=datetime.fromisoformat(d['feature_timestamp_utc']); d['decision_time_utc']=datetime.fromisoformat(d['decision_time_utc']); return ForecastSnapshot(**d)
    def latest_forecast(self)->ForecastSnapshot|None:
        with self._connect() as c: row=c.execute('SELECT payload_json FROM forecasts ORDER BY feature_timestamp_utc DESC LIMIT 1').fetchone()
        return None if row is None else self._forecast(row['payload_json'])
    def forecast_history(self,limit:int=100)->list[ForecastSnapshot]:
        with self._connect() as c: rows=c.execute('SELECT payload_json FROM forecasts ORDER BY feature_timestamp_utc DESC LIMIT ?', (int(limit),)).fetchall()
        return [self._forecast(row['payload_json']) for row in rows]
