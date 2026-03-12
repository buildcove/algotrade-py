from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    is_live: bool = False
    live_confirm: Optional[str] = None

    alpaca_key_id: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    tradingview_sessionid: Optional[str] = None
    tradingview_username: Optional[str] = None
    tradingview_password: Optional[str] = None
    ib_account: Optional[str] = None

    # When both are set, the API enables HTTP Basic auth for protected routes.
    # When unset, auth is disabled (useful for local-only runs).
    basic_username: Optional[str] = None
    basic_password: Optional[str] = None
    user_tz: str = "Asia/Manila"


settings = Settings()
