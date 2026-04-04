"""
ChildFocus — Test Suite: Heuristic Module
backend/tests/test_heuristic.py

Tests compute_heuristic_score() and get_feature_weights() from heuristic.py.
NOTE: Tests that require video download are mocked to avoid network calls.
Run: py -m pytest tests/test_heuristic.py -v
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.modules.heuristic import compute_heuristic_score, get_feature_weights


class TestGetFeatureWeights:

    def test_returns_dict(self):
        """get_feature_weights must return a dict."""
        weights = get_feature_weights()
        assert isinstance(weights, dict)

    def test_required_weight_keys(self):
        """Must contain all four feature weight keys."""
        weights = get_feature_weights()
        required = ["w_fcr", "w_csv", "w_att", "w_thumb"]
        for key in required:
            assert key in weights, f"Missing weight key: {key}"

    def test_weights_are_positive(self):
        """All weights must be positive floats."""
        weights = get_feature_weights()
        for key in ["w_fcr", "w_csv", "w_att", "w_thumb"]:
            assert weights[key] > 0, f"{key} must be > 0"

    def test_weights_sum_to_one(self):
        """Feature weights must sum to 1.0 per thesis formula."""
        weights = get_feature_weights()
        total = weights["w_fcr"] + weights["w_csv"] + weights["w_att"] + weights["w_thumb"]
        assert abs(total - 1.0) < 0.01, (
            f"Feature weights sum to {total}, expected 1.0. "
            f"Check heuristic.py constants."
        )

    def test_thresholds_present(self):
        """Threshold keys must be present."""
        weights = get_feature_weights()
        assert "threshold_high" in weights
        assert "threshold_low"  in weights

    def test_threshold_ordering(self):
        """threshold_high must be greater than threshold_low."""
        weights = get_feature_weights()
        assert weights["threshold_high"] > weights["threshold_low"]


class TestComputeHeuristicScore:

    def test_unavailable_video_returns_error_status(self):
        """A clearly invalid video ID should return a non-success status."""
        result = compute_heuristic_score("INVALID_VIDEO_ID_12345", "")
        assert isinstance(result, dict)
        assert result.get("status") in ("unavailable", "error"), (
            f"Expected error status for invalid video, got: {result.get('status')}"
        )

    def test_returns_dict_always(self):
        """Must always return a dict, never raise an exception."""
        try:
            result = compute_heuristic_score("nonexistent_video", "")
            assert isinstance(result, dict)
        except Exception as e:
            assert False, f"compute_heuristic_score raised an exception: {e}"

    def test_mocked_success_response_structure(self):
        """
        Mock sample_video to return a known result and verify
        compute_heuristic_score correctly maps it.
        """
        mock_sample = {
            "status": "success",
            "video_id": "test123",
            "video_duration_sec": 60.0,
            "thumbnail_intensity": 0.45,
            "segments": [
                {
                    "segment_id": "S1",
                    "offset_seconds": 0,
                    "length_seconds": 20,
                    "fcr": 0.30,
                    "csv": 0.25,
                    "att": 0.40,
                    "score_h": 0.315,
                },
                {
                    "segment_id": "S2",
                    "offset_seconds": 20,
                    "length_seconds": 20,
                    "fcr": 0.50,
                    "csv": 0.40,
                    "att": 0.60,
                    "score_h": 0.49,
                },
                {
                    "segment_id": "S3",
                    "offset_seconds": 40,
                    "length_seconds": 20,
                    "fcr": 0.20,
                    "csv": 0.15,
                    "att": 0.30,
                    "score_h": 0.215,
                },
            ],
            "aggregate_heuristic_score": 0.49,
        }

        with patch("app.modules.heuristic.sample_video", return_value=mock_sample):
            result = compute_heuristic_score("test123", "http://example.com/thumb.jpg")

        assert result["status"] == "success"
        assert "score_h"   in result
        assert "segments"  in result
        assert "thumbnail" in result
        assert isinstance(result["score_h"], float)
        assert 0.0 <= result["score_h"] <= 1.0

    def test_mocked_unavailable_video(self):
        """Mock an unavailable video and verify clean error handling."""
        mock_unavailable = {
            "status":  "unavailable",
            "reason":  "Video is private",
            "message": "Video cannot be analyzed: Video is private",
            "video_id": "private123",
        }

        with patch("app.modules.heuristic.sample_video", return_value=mock_unavailable):
            result = compute_heuristic_score("private123", "")

        assert result["status"] in ("unavailable", "error")
        assert isinstance(result, dict)
