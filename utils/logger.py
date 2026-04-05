"""Centralised logging for OPAC."""
import logging
import sys
from pathlib import Path
from config.settings import LOGS_DIR

_initialised = False

def get_logger(name: str) -> logging.Logger:
    global _initialised
    if not _initialised:
        log_file = LOGS_DIR / "opac.log"
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, encoding="utf-8"),
            ],
        )
        _initialised = True
    return logging.getLogger(name)
