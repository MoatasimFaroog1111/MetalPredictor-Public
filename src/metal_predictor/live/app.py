from __future__ import annotations
import asyncio,os,secrets
from contextlib import asynccontextmanager,suppress
from datetime import datetime,timezone
from pathlib import Path
from typing import Annotated,Any
from fastapi import Depends,FastAPI,Header,HTTPException,Query,Request
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,ConfigDict,Field
from metal_predictor.live.catchup import LiveMarketCatchUpService
from metal_predictor.live.contracts import HourlySilverBar
from metal_predictor.live.inference import LiveForecastOrchestrator,LivePredictionEngine
from metal_predictor.live.market_sources import TwelveDataSilverMinuteSource
from metal_predictor.live.notifications import TelegramForecastPublisher
from metal_predictor.live.repository import SQLiteForecastRepository
from metal_predictor.live.scheduler import HourlyCollectionScheduler
from metal_predictor.live.settings import LiveSettings

class HourlyBarRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    timestamp_utc:datetime
    open_usd_per_kg:float=Field(gt=0); high_usd_per_kg:float=Field(gt=0); low_usd_per_kg:float=Field(gt=0); close_usd_per_kg:float=Field(gt=0)
    minute_count:int=Field(ge=1,le=60); quality_flag:str=Field(min_length=1,max_length=64)
    source_provider:str=Field(default='Manual',min_length=1,max_length=64); source_symbol:str=Field(default='XAGUSD',min_length=1,max_length=64); market_type:str=Field(default='manual_input',min_length=1,max_length=64)
    def to_contract(self)->HourlySilverBar:
        if self.timestamp_utc.tzinfo is None:raise ValueError('timestamp_utc must include a timezone.')
        return HourlySilverBar(timestamp_utc=self.timestamp_utc.astimezone(timezone.utc),open_usd_per_kg=self.open_usd_per_kg,high_usd_per_kg=self.high_usd_per_kg,low_usd_per_kg=self.low_usd_per_kg,close_usd_per_kg=self.close_usd_per_kg,minute_count=self.minute_count,quality_flag=self.quality_flag,source_provider=self.source_provider,source_symbol=self.source_symbol,market_type=self.market_type)

def create_app(settings:LiveSettings|None=None,web_root:Path|None=None)->FastAPI:
    config=settings or LiveSettings.from_environment(); repo=SQLiteForecastRepository(config.database_path); engine=LivePredictionEngine(config.artifact_dir)
    notifier=TelegramForecastPublisher(config.telegram_bot_token,config.telegram_allowed_chat_ids) if config.telegram_enabled else None
    orchestrator=LiveForecastOrchestrator(repo,engine,notifier); source=TwelveDataSilverMinuteSource(config.twelvedata_api_key,config.twelvedata_symbol) if config.market_source_enabled else None
    catchup=LiveMarketCatchUpService(source,repo,engine,orchestrator) if source else None
    scheduler=HourlyCollectionScheduler(catchup,config.collection_delay_minutes) if config.auto_collection_enabled and catchup else None
    @asynccontextmanager
    async def lifespan(_:FastAPI):
        task=None
        if scheduler:task=asyncio.create_task(scheduler.run_forever(),name='silver-hourly-catch-up')
        try:yield
        finally:
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):await task
    app=FastAPI(title='Silver AI Forecast API',version='1.0.0',description='Research-only hourly XAG/USD forecasts. Predictive edge is not proven; BUY/SELL execution is disabled.',lifespan=lifespan)
    app.state.settings=config; app.state.repository=repo; app.state.engine=engine; app.state.orchestrator=orchestrator; app.state.catchup=catchup; app.state.telegram=notifier
    static_dir=(web_root or Path(os.getenv('LIVE_WEB_ROOT','live_web'))).resolve()
    app.mount('/static',StaticFiles(directory=static_dir),name='static')
    @app.middleware('http')
    async def security_headers(request:Request,call_next):
        response=await call_next(request); response.headers['X-Content-Type-Options']='nosniff'; response.headers['X-Frame-Options']='DENY'; response.headers['Referrer-Policy']='no-referrer'; response.headers['Permissions-Policy']='camera=(), microphone=(), geolocation=()'; response.headers['Cross-Origin-Opener-Policy']='same-origin'
        if request.url.path=='/' or request.url.path=='/sw.js' or request.url.path.startswith('/static/'):
            response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; manifest-src 'self'; worker-src 'self'"
        if request.url.path.startswith('/api/'):response.headers['Cache-Control']='no-store'
        return response
    admin_header=APIKeyHeader(name='X-Admin-Token',auto_error=False)
    def require_admin(supplied:str|None=Depends(admin_header))->None:
        if not config.admin_token:raise HTTPException(status_code=503,detail='LIVE_ADMIN_TOKEN is not configured.')
        if not supplied or not secrets.compare_digest(supplied,config.admin_token):raise HTTPException(status_code=401,detail='Invalid admin token.')
    @app.get('/',include_in_schema=False)
    def dashboard()->FileResponse:return FileResponse(static_dir/'index.html')
    @app.get('/sw.js',include_in_schema=False)
    def sw()->FileResponse:return FileResponse(static_dir/'sw.js',media_type='application/javascript',headers={'Service-Worker-Allowed':'/','Cache-Control':'no-cache'})
    @app.get('/api/v1/health')
    def health()->dict[str,object]:return {'status':'ok','service':'silver-ai-live','edge_status':'NOT_PROVEN','buy_sell_enabled':False}
    @app.get('/api/v1/status')
    def status()->dict[str,object]:
        latest=orchestrator.latest(); bars=repo.recent_bars(1)
        return {'service':'silver-ai-live','model':engine.model_status(),'market_source':{'configured':config.market_source_enabled,'provider':'TwelveData' if config.market_source_enabled else None,'symbol':config.twelvedata_symbol if config.market_source_enabled else None},'automatic_collection':{'enabled':config.auto_collection_enabled,'catch_up_enabled':catchup is not None,'delay_minutes_after_hour':config.collection_delay_minutes},'telegram':{'notifications_enabled':config.telegram_enabled,'webhook_ready':config.telegram_webhook_enabled,'allowed_chat_count':len(config.telegram_allowed_chat_ids)},'latest_live_bar_timestamp_utc':bars[-1].timestamp_utc.isoformat() if bars else None,'latest_forecast_timestamp_utc':latest.feature_timestamp_utc.isoformat() if latest else None}
    @app.get('/api/v1/model/status')
    def model_status()->dict[str,object]:return engine.model_status()
    @app.get('/api/v1/forecast/latest')
    def latest_forecast()->dict[str,object]:
        s=orchestrator.latest()
        if s is None:raise HTTPException(status_code=404,detail='No live forecast has been materialized yet.')
        return s.as_dict()
    @app.get('/api/v1/forecast/history')
    def history(limit:Annotated[int,Query(ge=1,le=500)]=100)->list[dict[str,object]]:return [x.as_dict() for x in repo.forecast_history(limit)]
    @app.get('/api/v1/market/silver/recent')
    def market(limit:Annotated[int,Query(ge=1,le=1000)]=168)->list[dict[str,object]]:return [x.as_dict() for x in repo.recent_bars(limit)]
    @app.post('/api/v1/market/silver/hourly',dependencies=[Depends(require_admin)])
    def ingest(payload:HourlyBarRequest)->dict[str,object]:
        try:
            bar=payload.to_contract(); created=orchestrator.ingest_bar(bar)
            try:snapshot,fcreated=orchestrator.materialize_latest_forecast(); return {'bar_created':created,'forecast_created':fcreated,'forecast_status':'CREATED' if fcreated else 'ALREADY_EXISTS','forecast':snapshot.as_dict()}
            except ValueError as exc:
                if str(exc).startswith('LIVE_FEATURES_INCOMPLETE'):return {'bar_created':created,'forecast_created':False,'forecast_status':'FEATURES_INCOMPLETE','forecast':None}
                raise
        except ValueError as exc:raise HTTPException(status_code=409,detail=str(exc)) from exc
    @app.post('/api/v1/admin/collect',dependencies=[Depends(require_admin)])
    def collect(hour_start_utc:datetime|None=None)->dict[str,object]:
        if catchup is None:raise HTTPException(status_code=503,detail='TWELVEDATA_API_KEY is not configured.')
        target=hour_start_utc or orchestrator.previous_completed_hour()
        if target.tzinfo is None:raise HTTPException(status_code=422,detail='hour_start_utc must include a timezone.')
        try:return {'catch_up':catchup.catch_up(target.astimezone(timezone.utc)).as_dict()}
        except (ValueError,RuntimeError) as exc:raise HTTPException(status_code=409,detail=str(exc)) from exc
    @app.post('/api/v1/admin/telegram/configure-webhook',dependencies=[Depends(require_admin)])
    def configure_webhook()->dict[str,object]:
        if notifier is None or not config.telegram_webhook_enabled:raise HTTPException(status_code=503,detail='Telegram webhook is not fully configured.')
        try:return notifier.configure_webhook(config.public_base_url,config.telegram_webhook_secret)
        except (ValueError,RuntimeError) as exc:raise HTTPException(status_code=409,detail=str(exc)) from exc
    @app.post('/api/v1/telegram/webhook',include_in_schema=False)
    def telegram_webhook(update:dict[str,Any],x_telegram_bot_api_secret_token:Annotated[str|None,Header()]=None)->dict[str,bool]:
        if notifier is None or not config.telegram_webhook_secret:raise HTTPException(status_code=503,detail='Telegram webhook is not configured.')
        if not secrets.compare_digest(x_telegram_bot_api_secret_token or '',config.telegram_webhook_secret):raise HTTPException(status_code=401,detail='Invalid Telegram webhook secret.')
        msg=update.get('message')
        if not isinstance(msg,dict):return {'ok':True}
        chat=msg.get('chat')
        if not isinstance(chat,dict) or 'id' not in chat:return {'ok':True}
        chat_id=str(chat['id'])
        if chat_id not in config.telegram_allowed_chat_ids:return {'ok':True}
        cmd=str(msg.get('text','')).strip().split(maxsplit=1)[0].lower()
        if cmd in {'/start','/help'}:notifier.send_text(chat_id,'🥈 <b>Silver AI Forecast</b>\n/latest — آخر توقع\n/status — حالة النظام\n⚠️ Research only; BUY/SELL disabled.')
        elif cmd=='/latest':
            s=orchestrator.latest(); notifier.send_text(chat_id,notifier.format_forecast(s) if s else 'لا يوجد Forecast حي حتى الآن.')
        elif cmd=='/status':notifier.send_text(chat_id,f'✅ النظام يعمل\nBaseline: {engine.baseline_model_name}\nChallenger: {engine.challenger_model_name}\nEdge: NOT_PROVEN\nBUY/SELL: DISABLED')
        return {'ok':True}
    return app
