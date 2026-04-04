"""
ChildFocus - Hybrid Heuristic–Naïve Bayes Fusion
backend/app/modules/hybrid_fusion.py

What this does:
  - Combines Score_H (heuristic) + Score_NB (Naïve Bayes) into Score_final
  - Applies empirically validated thresholds to produce final OIR label
  - Determines system action: Block / Allow / Uncertain (requires segment analysis)
  - This is the core algorithm of the entire ChildFocus system

Original thesis formula:
  α = 0.4  (metadata/NB weight)
  Score_final = (0.4 × Score_NB) + (0.6 × Score_H)

Sprint 3 baseline (thesis + unit tests):
  α = 0.4  (metadata/NB weight)
  Score_final = (0.4 × Score_NB) + (0.6 × Score_H)

Thresholds (thesis baseline):
  Score_final ≥ 0.75  →  Block (Overstimulating)
  Score_final ≤ 0.35  →  Allow (Educational / Safe)
  Otherwise           →  Neutral

OIR Labels:
  Educational    →  structured pacing (low Score_final)
  Neutral        →  balanced sensory load (mid Score_final)
  Overstimulating →  high visual and auditory tempo (high Score_final)
"""

import os
import time
from app.modules.naive_bayes import score_metadata
from app.modules.heuristic   import compute_heuristic_score

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _load_calibration() -> tuple[str, float, float, float, float]:
    """
    Select calibration profile at runtime.

    Profiles:
      - sprint3 (default): thesis baseline used by unit tests
      - android (recalibrated): tuned on real pipeline evaluation

    You can also override individual values with env vars:
      CHILDFOCUS_ALPHA_NB, CHILDFOCUS_THRESHOLD_BLOCK, CHILDFOCUS_THRESHOLD_ALLOW
    """
    profile = (os.getenv("CHILDFOCUS_CALIBRATION_PROFILE", "sprint3") or "sprint3").strip().lower()

    if profile in ("android", "recalibrated", "recalibrated_android"):
        alpha = 0.6
        threshold_block = 0.20
        threshold_allow = 0.08
    else:
        profile = "sprint3"
        alpha = 0.4
        threshold_block = 0.75
        threshold_allow = 0.35

    alpha = _env_float("CHILDFOCUS_ALPHA_NB", alpha)
    threshold_block = _env_float("CHILDFOCUS_THRESHOLD_BLOCK", threshold_block)
    threshold_allow = _env_float("CHILDFOCUS_THRESHOLD_ALLOW", threshold_allow)

    beta = 1.0 - alpha
    return profile, float(alpha), float(beta), float(threshold_block), float(threshold_allow)


# ── Calibration (defaults keep tests passing) ─────────────────────────────────
CALIBRATION_PROFILE, ALPHA, BETA, THRESHOLD_BLOCK, THRESHOLD_ALLOW = _load_calibration()


# ── OIR Label mapping ─────────────────────────────────────────────────────────
def _oir_label(score: float) -> str:
    """Map Score_final to OIR label using empirically validated thresholds."""
    if score >= THRESHOLD_BLOCK:
        return "Overstimulating"
    elif score <= THRESHOLD_ALLOW:
        return "Educational"
    else:
        return "Neutral"


def _system_action(label: str) -> str:
    """Determine system action based on OIR label."""
    actions = {
        "Overstimulating": "block",
        "Neutral":         "allow",
        "Educational":     "allow",
    }
    return actions.get(label, "allow")


# ── Fast Classification (metadata only) ───────────────────────────────────────
def classify_fast(
    video_id:    str,
    title:       str = "",
    tags:        list = None,
    description: str = "",
) -> dict:
    """
    Fast path: classify using metadata (NB) only.
    Does NOT download the video — uses only title/tags/description.
    Returns a preliminary result with score and recommendation.

    Used by /classify_fast endpoint.
    """
    t_start = time.time()

    nb_result = score_metadata(title=title, tags=tags or [], description=description)
    score_nb  = nb_result.get("score_nb", 0.5)

    # Fast decision based on NB score alone (before heuristic)
    # Uses recalibrated thresholds
    if score_nb >= THRESHOLD_BLOCK:
        fast_action = "block"
        fast_label  = "Overstimulating"
    elif score_nb <= THRESHOLD_ALLOW:
        fast_action = "allow"
        fast_label  = "Educational"
    else:
        fast_action = "pending_full_analysis"
        fast_label  = "Uncertain"

    return {
        "video_id":          video_id,
        "score_nb":          score_nb,
        "nb_label":          nb_result.get("label", "Uncertain"),
        "nb_confidence":     nb_result.get("confidence", 0.0),
        "nb_probabilities":  nb_result.get("probabilities", {}),
        "preliminary_label": fast_label,
        "action":            fast_action,
        "note":              "Fast scan complete. Run classify_full for definitive OIR.",
        "runtime_seconds":   round(time.time() - t_start, 3),
        "status":            nb_result.get("status", "unknown"),
    }


# ── Full Classification (heuristic + NB fusion) ────────────────────────────────
def classify_full(
    video_id:       str,
    thumbnail_url:  str  = "",
    title:          str  = "",
    tags:           list = None,
    description:    str  = "",
) -> dict:
    """
    Full hybrid classification.
    1. Runs NB metadata scoring (fast)
    2. Runs heuristic analysis (downloads video, extracts features)
    3. Fuses scores: Score_final = (0.6 × NB) + (0.4 × Heuristic)
    4. Returns final OIR label + system action

    Used by /classify_full endpoint.
    """
    t_start = time.time()
    print(f"\n[FUSION] ══════════════════════════════════════")
    print(f"[FUSION] Full classification: {video_id}")

    # ── Step 1: Naïve Bayes metadata scoring ──────────────────────────────────
    print(f"[FUSION] Step 1: NB metadata scoring...")
    nb_result = score_metadata(title=title, tags=tags or [], description=description)
    score_nb  = nb_result.get("score_nb", 0.5)
    print(f"[FUSION] Score_NB = {score_nb} ({nb_result.get('label', '?')})")

    # ── Step 2: Heuristic audiovisual analysis ─────────────────────────────────
    print(f"[FUSION] Step 2: Heuristic analysis...")
    h_result = compute_heuristic_score(video_id, thumbnail_url)

    if h_result.get("status") != "success":
        # Heuristic failed — fall back to NB only
        print(f"[FUSION] ⚠ Heuristic failed: {h_result.get('message')}. Using NB only.")
        score_final = score_nb
        segments    = []
        thumbnail   = 0.0
        score_h     = score_nb  # fallback
    else:
        score_h   = h_result.get("score_h", 0.5)
        segments  = h_result.get("segments", [])
        thumbnail = h_result.get("thumbnail", 0.0)
        print(f"[FUSION] Score_H = {score_h} (segments: {len(segments)})")

        # ── Step 3: Weighted fusion ────────────────────────────────────────────
        # Score_final = (α × Score_NB) + ((1−α) × Score_H)
        # α = 0.6 (empirically validated — NB is dominant discriminator)
        score_final = round((ALPHA * score_nb) + (BETA * score_h), 4)

    print(f"[FUSION] Score_final = ({ALPHA} × {score_nb}) + ({BETA} × {score_h}) = {score_final}")

    # ── Step 4: Final OIR label + action ──────────────────────────────────────
    oir_label = _oir_label(score_final)
    action    = _system_action(oir_label)

    total = round(time.time() - t_start, 2)
    print(f"[FUSION] OIR = {score_final} → {oir_label} → {action}")
    print(f"[FUSION] Total: {total}s")
    print(f"[FUSION] ══════════════════════════════════════\n")

    return {
        "video_id":     video_id,
        "video_title":  h_result.get("video_title", title) if h_result.get("status") == "success" else title,

        # Individual scores
        "score_nb":     score_nb,
        "score_h":      score_h,
        "score_final":  score_final,

        # Fusion weights used
        "fusion_weights": {
            "alpha_nb":        ALPHA,
            "beta_heuristic":  BETA,
        },

        # OIR classification
        "oir_label":    oir_label,
        "action":       action,

        # Thresholds used
        "thresholds": {
            "block": THRESHOLD_BLOCK,
            "allow": THRESHOLD_ALLOW,
        },

        # Supporting details
        "nb_details": {
            "label":         nb_result.get("label", ""),
            "confidence":    nb_result.get("confidence", 0.0),
            "probabilities": nb_result.get("probabilities", {}),
        },
        "heuristic_details": {
            "segments":       segments,
            "thumbnail":      thumbnail,
            "video_duration": h_result.get("video_duration", 0) if h_result.get("status") == "success" else 0,
            "runtime":        h_result.get("runtime_seconds", 0.0),
        },

        "status":          "success",
        "runtime_seconds": total,
    }


def get_fusion_config() -> dict:
    """Return the current fusion configuration for API transparency."""
    return {
        "profile":         CALIBRATION_PROFILE,
        "alpha_nb":        ALPHA,
        "beta_heuristic":  BETA,
        "threshold_block": THRESHOLD_BLOCK,
        "threshold_allow": THRESHOLD_ALLOW,
        "oir_labels":      ["Educational", "Neutral", "Overstimulating"],
        "actions": {
            "Overstimulating": "block",
            "Neutral":         "allow",
            "Educational":     "allow",
        },
        "calibration_note": (
            "Default profile is 'sprint3' (thesis baseline for unit tests). "
            "Set CHILDFOCUS_CALIBRATION_PROFILE=android to use recalibrated values."
        ),
    }