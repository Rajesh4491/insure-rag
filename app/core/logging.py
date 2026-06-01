"""Structured logging setup."""
import logging
import sys
from app.core.config import settings


def setup_logging():
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(stream=sys.stdout, level=level, format=fmt, force=True)
    # Quiet noisy libraries
    for noisy in ["httpx", "httpcore", "azure", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
