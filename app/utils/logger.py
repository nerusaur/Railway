"""
ChildFocus - Logger Utility
backend/app/utils/logger.py
"""

import logging
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
handlers = [logging.StreamHandler()]  # always log to stdout

# Only write to file locally (not on Railway/production)
if os.name == "nt" or os.getenv("FLASK_ENV") != "production":
    os.makedirs(LOG_DIR, exist_ok=True)
    handlers.append(
        logging.FileHandler(os.path.join(LOG_DIR, "childfocus.log"))
    )

logging.basicConfig(
    level   = logging.INFO,
    format  = "[%(asctime)s] %(levelname)s %(message)s",
    handlers= handlers
)

logger = logging.getLogger("childfocus")


def log_classification(video_id: str, label: str, mode: str = "full"):
    """Log a classification event."""
    logger.info(f"CLASSIFY [{mode.upper()}] video_id={video_id} label={label}")


def log_error(context: str, error: Exception):
    """Log an error with context."""
    logger.error(f"ERROR [{context}] {type(error).__name__}: {error}")