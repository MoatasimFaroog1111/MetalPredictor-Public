from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path


def _bool(name:str,default:bool=False)->bool:
    raw=os.getenv(name)
    return default if raw is None else raw.strip().lower() in {'1','true','yes','on'}

@dataclass(frozen=True)
class LiveSettings:
    artifact_dir: Path
    database_path: Path
    admin_token: str = ''
    twelvedata_api_key: str = ''
    twelvedata_symbol: str = 'XAG/USD'
    auto_collection_enabled: bool = False
    collection_delay_minutes: int = 5
    telegram_bot_token: str = ''
    telegram_webhook_secret: str = ''
    telegram_allowed_chat_ids: tuple[str,...] = ()
    public_base_url: str = ''

    @property
    def market_source_enabled(self)->bool: return bool(self.twelvedata_api_key.strip())
    @property
    def telegram_enabled(self)->bool: return bool(self.telegram_bot_token.strip() and self.telegram_allowed_chat_ids)
    @property
    def telegram_webhook_enabled(self)->bool: return bool(self.telegram_enabled and self.telegram_webhook_secret.strip() and self.public_base_url.startswith('https://'))

    @classmethod
    def from_environment(cls)->'LiveSettings':
        allowed=tuple(v.strip() for v in os.getenv('TELEGRAM_ALLOWED_CHAT_IDS','').split(',') if v.strip())
        delay=int(os.getenv('LIVE_COLLECTION_DELAY_MINUTES','5'))
        if not 1<=delay<=30: raise ValueError('LIVE_COLLECTION_DELAY_MINUTES must be between 1 and 30.')
        return cls(artifact_dir=Path(os.getenv('LIVE_ARTIFACT_DIR','runtime/artifacts')),database_path=Path(os.getenv('LIVE_DB_PATH','runtime/live_predictions.sqlite3')),admin_token=os.getenv('LIVE_ADMIN_TOKEN',''),twelvedata_api_key=os.getenv('TWELVEDATA_API_KEY',''),twelvedata_symbol=os.getenv('TWELVEDATA_SYMBOL','XAG/USD'),auto_collection_enabled=_bool('LIVE_AUTO_COLLECT',False),collection_delay_minutes=delay,telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN',''),telegram_webhook_secret=os.getenv('TELEGRAM_WEBHOOK_SECRET',''),telegram_allowed_chat_ids=allowed,public_base_url=os.getenv('PUBLIC_BASE_URL','').rstrip('/'))
