from __future__ import annotations
import asyncio,logging
from datetime import datetime,timedelta,timezone
logger=logging.getLogger(__name__)

class HourlyCollectionScheduler:
    def __init__(self,catchup_service,delay_minutes:int=5)->None:
        if not 1<=int(delay_minutes)<=30:raise ValueError('delay_minutes must be between 1 and 30.')
        self._service=catchup_service; self._delay=int(delay_minutes)
    async def run_forever(self)->None:
        while True:
            now=datetime.now(timezone.utc); due=self.next_due_utc(now,self._delay); await asyncio.sleep(max(0,(due-now).total_seconds()))
            try:await asyncio.to_thread(self._service.catch_up,due.replace(minute=0)-timedelta(hours=1))
            except asyncio.CancelledError:raise
            except Exception:logger.exception('Hourly catch-up failed.')
    @staticmethod
    def next_due_utc(now_utc:datetime,delay_minutes:int=5)->datetime:
        if now_utc.tzinfo is None:raise ValueError('now_utc must be timezone-aware.')
        now=now_utc.astimezone(timezone.utc); due=now.replace(minute=delay_minutes,second=0,microsecond=0)
        return due+timedelta(hours=1) if due<=now else due
