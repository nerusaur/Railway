"""
ChildFocus - Naïve Bayes Metadata Classifier
backend/app/modules/naive_bayes.py

score_metadata() returns a plain dict — compatible with tests and hybrid_fusion.py
score_from_metadata_dict() returns a SimpleNamespace for dot-notation (classify.py)
"""

import os
import pickle
import re
import types
import numpy as np

# ── Model paths (primary: backend/app/models, fallback: ml_training/outputs) ──
_MODULE_DIR          = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR_PRIMARY  = os.path.normpath(os.path.join(_MODULE_DIR, "..", "models"))
_MODELS_DIR_FALLBACK = os.path.normpath(
    os.path.join(_MODULE_DIR, "..", "..", "..", "ml_training", "outputs")
)

_MODEL_PATH    = None
_VEC_PATH      = None

# ── Lazy-loaded globals ────────────────────────────────────────────────────────
_model         = None
_vectorizer    = None
_label_encoder = None
_label_names   = None
_OVER_IDX      = -1
_metrics_cache = {}


def _resolve_paths() -> bool:
    global _MODEL_PATH, _VEC_PATH
    for directory in [_MODELS_DIR_PRIMARY, _MODELS_DIR_FALLBACK]:
        mp = os.path.join(directory, "nb_model.pkl")
        vp = os.path.join(directory, "vectorizer.pkl")
        if os.path.exists(mp) and os.path.exists(vp):
            _MODEL_PATH = mp
            _VEC_PATH   = vp
            return True
    return False


def _load_models() -> bool:
    global _model, _vectorizer, _label_encoder, _label_names, _OVER_IDX, _metrics_cache

    if _model is not None:
        return True

    if not _resolve_paths():
        print(f"[NB] ✗ Model files not found. Run: cd ml_training/scripts && py train_nb.py")
        return False

    try:
        with open(_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)

        if isinstance(bundle, dict):
            print(f"[NB] ✓ Unwrapped model from dict. Keys: {list(bundle.keys())}")
            _model         = bundle["model"]
            _label_encoder = bundle["label_encoder"]
            raw_names      = bundle.get("label_names", list(_label_encoder.classes_))
            _label_names   = [str(x) for x in raw_names]
            _metrics_cache = bundle.get("metrics", {})
        else:
            _model = bundle
            from sklearn.preprocessing import LabelEncoder
            _label_encoder = LabelEncoder()
            _label_encoder.fit(["Educational", "Neutral", "Overstimulating"])
            _label_names = ["Educational", "Neutral", "Overstimulating"]

        with open(_VEC_PATH, "rb") as f:
            _vectorizer = pickle.load(f)

        classes   = [str(c) for c in _label_encoder.classes_]
        _OVER_IDX = classes.index("Overstimulating") if "Overstimulating" in classes else -1

        print(f"[NB] ✓ Label names: {_label_names}")
        print(f"[NB] ✓ Model loaded from {os.path.dirname(_MODEL_PATH)}")
        print(f"[NB] ✓ Classes: {list(_label_encoder.classes_)}")
        return True

    except Exception as e:
        print(f"[NB] ✗ Failed to load model: {e}")
        return False


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ════════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════════

def score_metadata(
    title:       str  = "",
    tags:        list = None,
    description: str  = "",
) -> dict:
    """
    Compute Score_NB from video metadata.

    Returns a plain dict:
        score_nb      (float)  — P(Overstimulating) in [0.0, 1.0]
        label         (str)    — Predicted OIR class
        confidence    (float)  — Max class probability
        probabilities (dict)   — Per-class probability map
        status        (str)    — "success" | "empty_text" | "model_not_loaded" | "error:..."
    """
    if not _load_models():
        return {
            "score_nb":      0.5,
            "label":         "Uncertain",
            "confidence":    0.0,
            "probabilities": {},
            "status":        "model_not_loaded",
        }

    tags_str     = " ".join(str(t) for t in tags) if isinstance(tags, list) else (tags or "")
    desc_trimmed = (description or "")[:500]
    combined     = _clean_text(f"{title} {desc_trimmed} {tags_str}")

    if not combined.strip():
        return {
            "score_nb":      0.5,
            "label":         "Uncertain",
            "confidence":    0.0,
            "probabilities": {},
            "status":        "empty_text",
        }

    try:
        X          = _vectorizer.transform([combined])
        proba      = _model.predict_proba(X)[0]
        classes    = [str(c) for c in _label_encoder.classes_]
        proba_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        score_nb   = float(proba[_OVER_IDX]) if _OVER_IDX >= 0 else 0.5
        pred_idx   = int(np.argmax(proba))
        pred_label = str(_label_encoder.classes_[pred_idx])

        print(f"[NB] {title[:45]!r} → {pred_label} | Score_NB={round(score_nb,4)} | P(over)={round(score_nb,3)}")

        return {
            "score_nb":      round(score_nb, 4),
            "label":         pred_label,
            "confidence":    round(float(np.max(proba)), 4),
            "probabilities": proba_dict,
            "status":        "success",
        }

    except Exception as e:
        print(f"[NB] ✗ Scoring error: {e}")
        return {
            "score_nb":      0.5,
            "label":         "Uncertain",
            "confidence":    0.0,
            "probabilities": {},
            "status":        f"error: {e}",
        }


def score_from_metadata_dict(metadata: dict):
    """
    Wrapper for classify.py — accepts a metadata dict, returns a SimpleNamespace
    so classify.py can use dot notation: result.score_nb, result.predicted_label
    """
    result = score_metadata(
        title       = metadata.get("title", ""),
        tags        = metadata.get("tags", []),
        description = metadata.get("description", ""),
    )
    obj                 = types.SimpleNamespace()
    obj.score_nb        = result["score_nb"]
    obj.predicted_label = result["label"]
    obj.confidence      = result["confidence"]
    obj.probabilities   = result["probabilities"]
    obj.status          = result["status"]
    return obj


def get_model_metrics() -> dict:
    """
    Returns training metrics from nb_model.pkl bundle.
    Used by /health endpoint and test_naive_bayes.py.
    """
    _load_models()
    return dict(_metrics_cache) if _metrics_cache else {}


def model_status() -> dict:
    """Returns model loading status for /health endpoint."""
    loaded = _load_models()
    return {
        "loaded":     loaded,
        "model_path": _MODEL_PATH or "not found",
        "classes":    [str(c) for c in _label_encoder.classes_] if loaded and _label_encoder else [],
        "over_idx":   _OVER_IDX,
    }
