from __future__ import annotations
from datetime import datetime,timedelta,timezone
from metal_predictor.live.contracts import CatchUpResult

class LiveMarketCatchUpService:
    """Fills missing H1 feature context, then materializes only the latest requested forecast."""
    def __init__(self,source,repository,engine,orchestrator,batch_hours:int=72)->None:
        if not 1<=batch_hours<=72:raise ValueError('batch_hours must be between 1 and 72.')
        self._source=source; self._repo=repository; self._engine=engine; self._orchestrator=orchestrator; self._batch_hours=batch_hours
    def catch_up(self,through_utc:datetime)->CatchUpResult:
        target=self._hour(through_utc); recent=self._repo.recent_bars(limit=1)
        start=(recent[-1].timestamp_utc.astimezone(timezone.utc)+timedelta(hours=1)) if recent else (self._engine.historical_last_datetime_utc+timedelta(hours=1))
        fetched=created=0; cursor=start
        while cursor<=target:
            batch_end=min(target,cursor+timedelta(hours=self._batch_hours-1)); bars=self._source.fetch_hours(cursor,batch_end); fetched+=len(bars)
            for bar in bars:
                if cursor<=bar.timestamp_utc.astimezone(timezone.utc)<=batch_end: created+=int(self._orchestrator.ingest_bar(bar))
            cursor=batch_end+timedelta(hours=1)
        latest=self._repo.recent_bars(limit=1); latest_ts=latest[-1].timestamp_utc.astimezone(timezone.utc) if latest else None
        if latest_ts!=target:return CatchUpResult(target,fetched,created,latest_ts,False,'LATEST_HOUR_UNAVAILABLE')
        try:
            _,fcreated=self._orchestrator.materialize_latest_forecast(); status='FORECAST_CREATED' if fcreated else 'FORECAST_ALREADY_EXISTS'
            return CatchUpResult(target,fetched,created,latest_ts,fcreated,status)
        except ValueError as exc:
            if str(exc).startswith('LIVE_FEATURES_INCOMPLETE'):return CatchUpResult(target,fetched,created,latest_ts,False,'FEATURES_INCOMPLETE')
            raise
    @staticmethod
    def _hour(v:datetime)->datetime:
        if v.tzinfo is None:raise ValueError('through_utc must be timezone-aware.')
        u=v.astimezone(timezone.utc)
        if u.minute or u.second or u.microsecond:raise ValueError('through_utc must align to an exact UTC hour.')
        return u
