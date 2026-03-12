import re
import time
from typing import Optional

import requests
from algotrade.core import utils
from algotrade.settings import settings
from structlog import get_logger
from tvDatafeed import TvDatafeed

logger = get_logger()
CHART_URL = "https://www.tradingview.com/chart/M7kV72q4/"

HEADERS = {
    # A normal desktop UA is enough
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) " "AppleWebKit/537.36 (KHTML, like Gecko) " "Chrome/123.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def cookie_dict(raw: str) -> dict[str, str]:
    return {k.strip(): v for k, v in (p.split("=", 1) for p in raw.split(";") if "=" in p)}


def fetch_html(cookies: dict[str, str]) -> tuple[str, dict[str, str]]:
    r = requests.get(CHART_URL, headers=HEADERS, cookies=cookies, timeout=30)
    r.raise_for_status()
    return r.text, r.cookies.get_dict()


def find_token(html: str) -> Optional[str]:
    m = re.search(r'"auth_token"\s*:\s*"([^"]+)"', html)
    return m.group(1) if m else None


def get_token(raw_cookie: str) -> Optional[str]:
    start = time.time()
    cookies = cookie_dict(raw_cookie)
    html, new_cookies = fetch_html(cookies)

    # token = new_cookies.get("auth_token") or cookies.get("auth_token") or find_token(html)
    token = find_token(html)
    logger.debug("get_token", elapsed=time.time() - start)
    return token


def _get_tv_via_password():
    tv = TvDatafeed(username=settings.tradingview_username, password=settings.tradingview_password)
    if not tv.token or utils.is_jwt_expired_no_sig(tv.token):
        raise ValueError("Failed to get TradingView token")

    logger.debug("TradingView token obtained via username/password", token_present=bool(tv.token))
    return tv


def get_tv():
    tv = TvDatafeed()
    tv.token = get_token(settings.tradingview_sessionid)
    return tv
