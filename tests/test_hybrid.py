"""
ChildFocus — Test Suite: Hybrid Fusion Module
backend/tests/test_hybrid.py

Tests classify_fast() and classify_full() from hybrid_fusion.py.
classify_full tests mock the heuristic to avoid video downloads.
Run: py -m pytest tests/test_hybrid.py -v
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.modules.hybrid_fusion import (
    classify_fast,
    classify_full,
    get_fusion_config,
    ALPHA, BETA, THRESHOLD_BLOCK, THRESHOLD_ALLOW,
)


class TestFusionConfig:

    def test_alpha_beta_sum_to_one(self):
        """Alpha + Beta must equal 1.0 (thesis formula requirement)."""
        assert abs(ALPHA + BETA - 1.0) < 0.001, (
            f"ALPHA ({ALPHA}) + BETA ({BETA}) = {ALPHA + BETA}, must equal 1.0"
        )

    def test_threshold_ordering(self):
        """Block threshold must be higher than allow threshold."""
        assert THRESHOLD_BLOCK > THRESHOLD_ALLOW

    def test_get_fusion_config_returns_dict(self):
        """get_fusion_config must return a valid dict."""
        config = get_fusion_config()
        assert isinstance(config, dict)
        assert "alpha_nb"        in config
        assert "beta_heuristic"  in config
        assert "threshold_block" in config
        assert "threshold_allow" in config

    def test_fusion_config_values_match_constants(self):
        """Config values must match the module-level constants."""
        config = get_fusion_config()
        assert config["alpha_nb"]        == ALPHA
        assert config["beta_heuristic"]  == BETA
        assert config["threshold_block"] == THRESHOLD_BLOCK
        assert config["threshold_allow"] == THRESHOLD_ALLOW


class TestClassifyFast:

    def test_returns_dict(self):
        """classify_fast must always return a dict."""
        result = classify_fast(
            video_id="test123",
            title="Kids Learn ABC Phonics",
            tags=["educational", "kids"],
            description="Learning video for toddlers."
        )
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        """Response must contain all required keys."""
        result = classify_fast(
            video_id="test123",
            title="Kids Learn ABC Phonics",
            tags=[],
            description=""
        )
        required = [
            "video_id", "score_nb", "nb_label",
            "preliminary_label", "action", "status"
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_video_id_echoed(self):
        """video_id in response must match the input."""
        result = classify_fast(video_id="abc123", title="test", tags=[], description="")
        assert result["video_id"] == "abc123"

    def test_action_values_are_valid(self):
        """action must be one of the three valid values."""
        result = classify_fast(
            video_id="test123",
            title="kids cartoon episode",
            tags=[],
            description=""
        )
        valid_actions = {"block", "allow", "pending_full_analysis"}
        assert result["action"] in valid_actions

    def test_score_nb_in_range(self):
        """score_nb must be between 0.0 and 1.0."""
        result = classify_fast(
            video_id="test",
            title="surprise egg unboxing fast compilation",
            tags=["surprise", "unboxing"],
            description=""
        )
        assert 0.0 <= result["score_nb"] <= 1.0

    def test_block_action_when_score_high(self):
        """
        If NB returns score >= THRESHOLD_BLOCK, action must be 'block'.
        Mock score_metadata to return a high score.
        """
        mock_nb = {
            "score_nb": 0.90,
            "label": "Overstimulating",
            "confidence": 0.90,
            "probabilities": {},
            "status": "success",
        }
        with patch("app.modules.hybrid_fusion.score_metadata", return_value=mock_nb):
            result = classify_fast(video_id="over123", title="x", tags=[], description="")
        assert result["action"] == "block"
        assert result["preliminary_label"] == "Overstimulating"

    def test_allow_action_when_score_low(self):
        """
        If NB returns score <= THRESHOLD_ALLOW, action must be 'allow'.
        """
        mock_nb = {
            "score_nb": 0.10,
            "label": "Educational",
            "confidence": 0.90,
            "probabilities": {},
            "status": "success",
        }
        with patch("app.modules.hybrid_fusion.score_metadata", return_value=mock_nb):
            result = classify_fast(video_id="edu123", title="x", tags=[], description="")
        assert result["action"] == "allow"
        assert result["preliminary_label"] == "Educational"

    def test_pending_when_score_uncertain(self):
        """
        If NB score is between thresholds, action must be 'pending_full_analysis'.
        """
        mock_nb = {
            "score_nb": 0.55,
            "label": "Neutral",
            "confidence": 0.55,
            "probabilities": {},
            "status": "success",
        }
        with patch("app.modules.hybrid_fusion.score_metadata", return_value=mock_nb):
            result = classify_fast(video_id="neu123", title="x", tags=[], description="")
        assert result["action"] == "pending_full_analysis"


class TestClassifyFull:

    def _mock_nb_result(self, score=0.40):
        return {
            "score_nb":      score,
            "label":         "Neutral",
            "confidence":    0.60,
            "probabilities": {"Educational": 0.30, "Neutral": 0.50, "Overstimulating": 0.20},
            "status":        "success",
        }

    def _mock_heuristic_result(self, score_h=0.45):
        return {
            "status":         "success",
            "score_h":        score_h,
            "video_title":    "Test Video",
            "video_duration": 60.0,
            "thumbnail":      0.35,
            "runtime_seconds": 10.0,
            "segments": [
                {"segment_id": "S1", "fcr": 0.3, "csv": 0.2, "att": 0.4, "score_h": 0.30},
                {"segment_id": "S2", "fcr": 0.5, "csv": 0.4, "att": 0.5, "score_h": 0.45},
                {"segment_id": "S3", "fcr": 0.4, "csv": 0.3, "att": 0.4, "score_h": 0.37},
            ],
        }

    def test_returns_dict(self):
        """classify_full must return a dict."""
        with patch("app.modules.hybrid_fusion.score_metadata",
                   return_value=self._mock_nb_result()), \
             patch("app.modules.hybrid_fusion.compute_heuristic_score",
                   return_value=self._mock_heuristic_result()):
            result = classify_full(video_id="test123")
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        """Response must have all required keys."""
        with patch("app.modules.hybrid_fusion.score_metadata",
                   return_value=self._mock_nb_result()), \
             patch("app.modules.hybrid_fusion.compute_heuristic_score",
                   return_value=self._mock_heuristic_result()):
            result = classify_full(video_id="test123")

        required = [
            "video_id", "score_nb", "score_h", "score_final",
            "oir_label", "action", "fusion_weights", "status"
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_fusion_formula_correct(self):
        """
        score_final must equal (ALPHA × score_nb) + (BETA × score_h).
        Thesis formula: Score_final = (0.4 × NB) + (0.6 × Heuristic)
        """
        nb_score = 0.40
        h_score  = 0.60

        with patch("app.modules.hybrid_fusion.score_metadata",
                   return_value=self._mock_nb_result(nb_score)), \
             patch("app.modules.hybrid_fusion.compute_heuristic_score",
                   return_value=self._mock_heuristic_result(h_score)):
            result = classify_full(video_id="formula_test")

        expected = round((ALPHA * nb_score) + (BETA * h_score), 4)
        actual   = result["score_final"]
        assert abs(actual - expected) < 0.01, (
            f"Fusion formula wrong: expected {expected}, got {actual}. "
            f"Formula: ({ALPHA} × {nb_score}) + ({BETA} × {h_score})"
        )

    def test_overstimulating_label_when_score_high(self):
        """score_final >= 0.75 must produce oir_label='Overstimulating' and action='block'."""
        with patch("app.modules.hybrid_fusion.score_metadata",
                   return_value=self._mock_nb_result(0.85)), \
             patch("app.modules.hybrid_fusion.compute_heuristic_score",
                   return_value=self._mock_heuristic_result(0.90)):
            result = classify_full(video_id="over_test")

        assert result["oir_label"] == "Overstimulating"
        assert result["action"]    == "block"

    def test_educational_label_when_score_low(self):
        """score_final <= 0.35 must produce oir_label='Educational' and action='allow'."""
        with patch("app.modules.hybrid_fusion.score_metadata",
                   return_value=self._mock_nb_result(0.10)), \
             patch("app.modules.hybrid_fusion.compute_heuristic_score",
                   return_value=self._mock_heuristic_result(0.10)):
            result = classify_full(video_id="edu_test")

        assert result["oir_label"] == "Educational"
        assert result["action"]    == "allow"

    def test_heuristic_failure_falls_back_to_nb(self):
        """If heuristic fails, classify_full must fall back to NB score only."""
        mock_fail = {"status": "error", "message": "Video unavailable"}

        with patch("app.modules.hybrid_fusion.score_metadata",
                   return_value=self._mock_nb_result(0.80)), \
             patch("app.modules.hybrid_fusion.compute_heuristic_score",
                   return_value=mock_fail):
            result = classify_full(video_id="fallback_test")

        assert isinstance(result, dict)
        assert result.get("status") == "success"
        # When heuristic fails, score_final should be driven by NB
        assert result["score_final"] >= 0.70

    def test_score_final_in_range(self):
        """score_final must always be between 0.0 and 1.0."""
        with patch("app.modules.hybrid_fusion.score_metadata",
                   return_value=self._mock_nb_result(0.50)), \
             patch("app.modules.hybrid_fusion.compute_heuristic_score",
                   return_value=self._mock_heuristic_result(0.50)):
            result = classify_full(video_id="range_test")

        assert 0.0 <= result["score_final"] <= 1.0

    def test_fusion_weights_in_response(self):
        """fusion_weights must report alpha and beta used."""
        with patch("app.modules.hybrid_fusion.score_metadata",
                   return_value=self._mock_nb_result()), \
             patch("app.modules.hybrid_fusion.compute_heuristic_score",
                   return_value=self._mock_heuristic_result()):
            result = classify_full(video_id="weights_test")

        assert result["fusion_weights"]["alpha_nb"]       == ALPHA
        assert result["fusion_weights"]["beta_heuristic"] == BETA
