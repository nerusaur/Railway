"""
ChildFocus - Flask App Factory
backend/app/__init__.py
"""

from flask import Flask
from app.config import DevelopmentConfig


def create_app(config=DevelopmentConfig):
    app = Flask(__name__)
    app.config.from_object(config)

    # Register blueprints
    from app.routes.classify import classify_bp
    from app.routes.metadata import metadata_bp

    app.register_blueprint(classify_bp)
    app.register_blueprint(metadata_bp)

    return app