import os
import time

from structlog import get_logger

logger = get_logger()

# Set default timezone to UTC once per interpreter
os.environ.setdefault("TZ", "UTC")
time.tzset()

if os.environ.get("ALGO_TZ_LOGGED") != "1":
    # Only emit the noisy debug log when explicitly requested
    if os.environ.get("ALGO_TZ_DEBUG") == "1":
        logger.debug("Default timezone: UTC")
    os.environ["ALGO_TZ_LOGGED"] = "1"
