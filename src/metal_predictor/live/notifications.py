from __future__ import annotations
import httpx2 as httpx
from metal_predictor.live.contracts import ForecastSnapshot

class TelegramForecastPublisher:
    def __init__(self,bot_token:str,allowed_chat_ids:tuple[str,...],client:httpx.Client|None=None)->None:
        if not bot_token.strip():raise ValueError('Telegram bot token is required.')
        self._token=bot_token.strip(); self._allowed=tuple(map(str,allowed_chat_ids)); self._client=client or httpx.Client(timeout=20.0)
    def _url(self,method:str)->str:return f'https://api.telegram.org/bot{self._token}/{method}'
    def send_text(self,chat_id:str,text:str)->None:
        if str(chat_id) not in self._allowed:return
        try:r=self._client.post(self._url('sendMessage'),json={'chat_id':str(chat_id),'text':text,'parse_mode':'HTML','disable_web_page_preview':True}); r.raise_for_status()
        except httpx.HTTPError:raise RuntimeError('Telegram sendMessage transport request failed.') from None
    def format_forecast(self,s:ForecastSnapshot)->str:
        return (f'🥈 <b>Silver AI Forecast</b>\n\nالسعر: ${s.current_price_usd_per_kg:,.2f}/kg\nBaseline: <b>{s.baseline_direction}</b> → ${s.baseline_predicted_price_usd_per_kg:,.2f}\nResearch: <b>{s.challenger_direction}</b> → ${s.challenger_predicted_price_usd_per_kg:,.2f}\nالجودة: {s.data_quality}\nالمصدر: {s.source_provider}\n\n⚠️ Research only — BUY/SELL disabled.')
    def publish_forecast(self,snapshot:ForecastSnapshot)->None:
        text=self.format_forecast(snapshot)
        for chat in self._allowed:self.send_text(chat,text)
    def configure_webhook(self,public_base_url:str,secret:str)->dict[str,object]:
        base=public_base_url.rstrip('/')
        if not base.startswith('https://'):raise ValueError('Telegram webhook requires HTTPS.')
        if not secret.strip():raise ValueError('Telegram webhook secret is required.')
        url=base+'/api/v1/telegram/webhook'
        try:r=self._client.post(self._url('setWebhook'),json={'url':url,'secret_token':secret,'allowed_updates':['message']}); r.raise_for_status()
        except httpx.HTTPError:raise RuntimeError('Telegram setWebhook transport request failed.') from None
        return {'configured':True,'url':url}
