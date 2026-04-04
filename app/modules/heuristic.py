"""
ChildFocus - Heuristic Analysis Module
backend/app/modules/heuristic.py

This module supports two call styles for compatibility across the codebase/tests:
  1) compute_heuristic_score(video_id_or_url: str, thumbnail_url: str)
     - Used by unit tests and by modules that want heuristic-only scoring.
  2) compute_heuristic_score(sample: dict)
     - Used by routes that already called frame_sampler.sample_video().
"""

from app.modules.frame_sampler import sample_video  # re-exported for tests to patch

# ── Heuristic weights (from thesis) ───────────────────────────────────────────
W_FCR   = 0.35
W_CSV   = 0.25
W_ATT   = 0.20
W_THUMB = 0.20

# ── Thresholds (from thesis) ──────────────────────────────────────────────────
THRESHOLD_HIGH = 0.75   # Overstimulating
THRESHOLD_LOW  = 0.35   # Safe / Educational


def _compute_from_sample(sample: dict) -> dict:
    """
    Compute the final heuristic score from a pre-sampled video dict.

    Accepts the dict already returned by frame_sampler.sample_video().
    Does NOT call sample_video() again — classify.py already did that.

    Args:
        sample: dict returned by sample_video(), containing:
                  - segments: list of dicts with fcr, csv, att, score_h
                  - thumbnail_intensity: float
                  - aggregate_heuristic_score: float
                  - status: "success" | "thumbnail_only" | "unavailable"

    Returns:
        dict with:
            score_h (float):  Aggregate heuristic score [0.0, 1.0]
            details (dict):   Segment breakdown + thumbnail for logging
    """
    segments = sample.get("segments", [])
    thumb    = float(sample.get("thumbnail_intensity", 0.0))
    status   = sample.get("status", "success")

    # Use pre-computed aggregate if available (fastest path)
    if "aggregate_heuristic_score" in sample:
        score_h = float(sample["aggregate_heuristic_score"])

    elif segments:
        seg_scores = []
        for seg in segments:
            if not seg:
                continue
            fcr = float(seg.get("fcr", 0.0))
            csv = float(seg.get("csv", 0.0))
            att = float(seg.get("att", 0.0))
            seg_scores.append(round(W_FCR * fcr + W_CSV * csv + W_ATT * att, 4))

        if seg_scores:
            max_seg = max(seg_scores)
            score_h = round(0.80 * max_seg + 0.20 * thumb, 4)
        else:
            score_h = round(W_THUMB * thumb, 4)

    else:
        score_h = round(W_THUMB * thumb, 4)

    score_h = round(min(1.0, max(0.0, score_h)), 4)

    details = {
        "segments":            segments,
        "thumbnail_intensity": thumb,
        "status":              status,
        "weights": {
            "fcr":   W_FCR,
            "csv":   W_CSV,
            "att":   W_ATT,
            "thumb": W_THUMB,
        }
    }

    return {"score_h": score_h, "details": details}


def compute_heuristic_score(video_id_or_sample, thumbnail_url: str = "") -> dict:
    """
    Compatibility wrapper.

    - If passed a dict, it is treated as the output of sample_video().
    - If passed a string, sample_video() is called (tests patch this call).

    Returns a dict that contains the keys expected by tests and by
    hybrid_fusion/classify routes.
    """
    try:
        if isinstance(video_id_or_sample, dict):
            sample = video_id_or_sample
        else:
            # Treat as a video identifier/url. Unit tests patch sample_video().
            sample = sample_video(str(video_id_or_sample), thumbnail_url=thumbnail_url)

        status = sample.get("status", "error")
        if status not in ("success", "thumbnail_only", "thumbnail-only"):
            # Preserve a clean failure shape for callers/tests.
            return {
                "status": status,
                "message": sample.get("message", sample.get("reason", "unavailable")),
                "video_id": sample.get("video_id", str(video_id_or_sample)),
            }

        base = _compute_from_sample(sample)
        details = base.get("details", {})

        # Provide the flat keys expected by tests (and used by hybrid_fusion).
        return {
            "status": "success",
            "score_h": float(base["score_h"]),
            "segments": details.get("segments", []),
            "thumbnail": float(details.get("thumbnail_intensity", 0.0)),
            "video_id": sample.get("video_id", str(video_id_or_sample)),
            "video_title": sample.get("video_title", ""),
            "video_duration": float(sample.get("video_duration_sec", 0.0) or 0.0),
            "runtime_seconds": float(sample.get("runtime_seconds", 0.0) or 0.0),
            # Keep the detailed structure for classify.py logging.
            "details": details,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def compute_segment_score(fcr: float, csv: float, att: float) -> float:
    """
    Compute heuristic score for a single segment.
    Thesis formula: Score_H = (w1*FCR) + (w2*CSV) + (w3*ATT)
    """
    return round(
        (W_FCR * fcr) + (W_CSV * csv) + (W_ATT * att),
        4
    )


def _label_from_score(score: float) -> str:
    """Map a numeric score to an OIR label using thesis thresholds."""
    if score >= THRESHOLD_HIGH:
        return "Overstimulating"
    elif score <= THRESHOLD_LOW:
        return "Safe"
    else:
        return "Uncertain"


def get_feature_weights() -> dict:
    """Return the heuristic feature weights for transparency/logging."""
    return {
        "w_fcr":           W_FCR,
        "w_csv":           W_CSV,
        "w_att":           W_ATT,
        "w_thumb":         W_THUMB,
        "threshold_high":  THRESHOLD_HIGH,
        "threshold_low":   THRESHOLD_LOW,
    }