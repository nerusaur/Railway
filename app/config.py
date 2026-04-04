"""
ChildFocus - Flask App Configuration
backend/app/config.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration."""
    SECRET_KEY         = os.getenv("SECRET_KEY", "childfocus-dev-key-change-in-prod")
    YOUTUBE_API_KEY    = os.getenv("YOUTUBE_API_KEY", "")

    # SQLAlchemy
    SQLALCHEMY_DATABASE_URI            = os.getenv("DATABASE_URL", "sqlite:///childfocus.db")
    SQLALCHEMY_TRACK_MODIFICATIONS     = False

    # CORS — allow Android app to connect
    CORS_ORIGINS = ["*"]

    # Classification thresholds (matches thesis + hybrid_fusion.py)
    THRESHOLD_BLOCK = 0.75
    THRESHOLD_ALLOW = 0.35

    # Fusion weights
    ALPHA_NB         = 0.4
    BETA_HEURISTIC   = 0.6

    # Video analysis limits
    MAX_VIDEO_DURATION_SEC = 90
    SEGMENT_DURATION_SEC   = 20
    NUM_SEGMENTS           = 3


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_map = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
    "default":     DevelopmentConfig,
}