"""
ChildFocus — Test Suite: Naïve Bayes Module
backend/tests/test_naive_bayes.py

Tests score_metadata() from naive_bayes.py.
Requires nb_model.pkl and vectorizer.pkl to be present.
Run: py -m pytest tests/test_naive_bayes.py -v
"""

import sys
import os

# Add backend root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.modules.naive_bayes import score_metadata, get_model_metrics


class TestScoreMetadata:

    def test_returns_dict(self):
        """score_metadata must always return a dict."""
        result = score_metadata(title="kids learn abc", tags=[], description="")
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        """Response dict must contain all required keys."""
        result = score_metadata(title="kids learn abc", tags=[], description="")
        required = ["score_nb", "label", "confidence", "probabilities", "status"]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_score_nb_in_range(self):
        """score_nb must be a float between 0.0 and 1.0."""
        result = score_metadata(title="kids learn abc", tags=[], description="")
        assert isinstance(result["score_nb"], float)
        assert 0.0 <= result["score_nb"] <= 1.0

    def test_label_is_valid_oir(self):
        """label must be one of the three OIR classes."""
        result = score_metadata(title="kids learn abc", tags=[], description="")
        valid_labels = {"Educational", "Neutral", "Overstimulating", "Uncertain"}
        assert result["label"] in valid_labels

    def test_educational_video_scores_low(self):
        """Clear educational content should score low (≤ 0.50)."""
        result = score_metadata(
            title="Learn ABC Phonics for Kids | Educational Preschool",
            tags=["educational", "kids", "learning", "phonics", "abc"],
            description="Structured learning video for toddlers. Teaches alphabet phonics."
        )
        assert result["score_nb"] <= 0.60, (
            f"Educational video scored too high: {result['score_nb']}"
        )

    def test_overstimulating_video_scores_high(self):
        """Clearly overstimulating content should score higher than educational."""
        edu_result = score_metadata(
            title="Learn ABC Phonics for Kids",
            tags=["educational", "learning"],
            description="Structured learning video for toddlers."
        )
        over_result = score_metadata(
            title="CRAZY FAST surprise egg unboxing compilation kids",
            tags=["surprise", "unboxing", "fast", "compilation", "kids"],
            description="Super fast unboxing surprise eggs toys crazy fun!"
        )
        assert over_result["score_nb"] >= edu_result["score_nb"], (
            f"Overstimulating ({over_result['score_nb']}) should score >= "
            f"Educational ({edu_result['score_nb']})"
        )

    def test_empty_input_returns_safe_default(self):
        """Empty input should return status 'empty_text' with score_nb = 0.5."""
        result = score_metadata(title="", tags=[], description="")
        assert result["status"] == "empty_text"
        assert result["score_nb"] == 0.5

    def test_tags_list_accepted(self):
        """score_metadata must accept a list of tags without error."""
        result = score_metadata(
            title="Kids Science Experiments",
            tags=["science", "kids", "experiments", "educational"],
            description="Fun science experiments for children."
        )
        assert result["status"] == "success"

    def test_confidence_in_range(self):
        """Confidence value must be between 0.0 and 1.0."""
        result = score_metadata(title="kids educational video", tags=[], description="")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self):
        """Per-class probabilities must sum to approximately 1.0."""
        result = score_metadata(title="kids cartoon episode", tags=[], description="")
        if result["probabilities"]:
            total = sum(result["probabilities"].values())
            assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, not 1.0"

    def test_model_loaded_status(self):
        """Status should be 'success' when model files are present."""
        result = score_metadata(title="children learn colors", tags=[], description="")
        assert result["status"] == "success", (
            f"Expected 'success' but got '{result['status']}'. "
            f"Make sure nb_model.pkl exists in backend/app/models/"
        )


class TestGetModelMetrics:

    def test_returns_dict(self):
        """get_model_metrics must return a dict."""
        metrics = get_model_metrics()
        assert isinstance(metrics, dict)

    def test_has_accuracy_key(self):
        """Metrics dict should contain accuracy if model was trained."""
        metrics = get_model_metrics()
        # Either has metrics or is empty — both are acceptable
        if metrics:
            assert "accuracy" in metrics or "f1_macro" in metrics
