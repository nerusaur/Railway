"""
ChildFocus - Metadata Route
backend/app/routes/metadata.py

Endpoints:
  GET /metadata?video_url=...  → fetch YouTube video metadata
  GET /health                  → health check
  GET /config                  → current fusion config
"""

from flask import Blueprint, request, jsonify
from app.modules.youtube_api  import get_video_metadata, extract_video_id
from app.modules.hybrid_fusion import get_fusion_config
from app.modules.heuristic     import get_feature_weights
from app.utils.validators      import validate_video_url

metadata_bp = Blueprint("metadata", __name__)


@metadata_bp.route("/metadata", methods=["GET"])
def get_metadata():
    """
    Fetch YouTube metadata for a video.

    Query params:
        video_url: full YouTube URL or video ID

    Returns:
        title, description, tags, channel, thumbnail_url, duration, view_count
    """
    video_url = request.args.get("video_url", "")
    if not video_url:
        return jsonify({"error": "Missing video_url query parameter"}), 400

    error = validate_video_url(video_url)
    if error:
        return jsonify({"error": error}), 400

    video_id = extract_video_id(video_url)
    metadata = get_video_metadata(video_id)

    if "error" in metadata:
        return jsonify(metadata), 404

    return jsonify(metadata), 200


@metadata_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Android app connectivity test."""
    return jsonify({
        "status":  "ok",
        "service": "ChildFocus Backend",
        "version": "1.0.0",
    }), 200


@metadata_bp.route("/config", methods=["GET"])
def config():
    """Returns current fusion config and heuristic weights for transparency."""
    return jsonify({
        "fusion":    get_fusion_config(),
        "heuristic": get_feature_weights(),
    }), 200