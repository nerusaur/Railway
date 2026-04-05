import os
import sqlite3
import time

from flask import Blueprint, jsonify, request

from app.modules.frame_sampler import sample_video
from app.modules.heuristic import compute_heuristic_score
from app.modules.naive_bayes import score_from_metadata_dict, score_metadata

classify_bp = Blueprint("classify", __name__)

DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "database", "childfocus.db"
)

# ── Confidence-gated hybrid fusion config (v3) ───────────────────────────────
# When NB confidence < CONF_THRESH the heuristic gets more weight because
# audiovisual pacing is a more reliable signal than an uncertain text
# classification.  H_OVERRIDE caps any extremely slow-paced video as
# non-Overstimulating regardless of Score_NB.
# Validated on 30-video dataset: calibration acc=60%, held-out test acc=60%.
BASE_ALPHA      = 0.40   # NB weight when NB confidence >= CONF_THRESH
LOW_ALPHA       = 0.15   # NB weight when NB confidence <  CONF_THRESH
CONF_THRESH     = 0.40   # confidence boundary
H_OVERRIDE      = 0.10   # if Score_H < this → cannot be Overstimulating
THRESHOLD_BLOCK = 0.20   # >= Overstimulating
THRESHOLD_ALLOW = 0.18   # <= Educational

def _fuse(score_nb: float, score_h: float, nb_confidence: float) -> tuple[float, str]:
    eff_alpha   = LOW_ALPHA if nb_confidence < CONF_THRESH else BASE_ALPHA
    final       = round((eff_alpha * score_nb) + ((1 - eff_alpha) * score_h), 4)
    if H_OVERRIDE > 0 and score_h < H_OVERRIDE:
        label = "Educational" if final <= THRESHOLD_ALLOW else "Neutral"
    elif final >= THRESHOLD_BLOCK:
        label = "Overstimulating"
    elif final <= THRESHOLD_ALLOW:
        label = "Educational"
    else:
        label = "Neutral"
    return final, label


def extract_video_id(url: str) -> str:
    import re
    for pattern in [r"(?:v=)([a-zA-Z0-9_-]{11})",
                    r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
                    r"(?:shorts/)([a-zA-Z0-9_-]{11})",
                    r"(?:embed/)([a-zA-Z0-9_-]{11})"]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url.strip()):
        return url.strip()
    return url.strip()


def _nb_only_result(video_id: str, metadata: dict, reason: str, t_start: float) -> dict:
    """Last resort fallback: NB only when video and thumbnail both unavailable."""
    nb_obj      = score_from_metadata_dict(metadata)
    score_nb    = nb_obj.score_nb
    score_final = round(score_nb, 4)
    # NB-only path: no heuristic score available, classify directly on Score_NB
    if   score_final >= THRESHOLD_BLOCK: oir_label = "Overstimulating"
    elif score_final <= THRESHOLD_ALLOW: oir_label = "Educational"
    else:                                oir_label = "Neutral"
    action      = "block" if oir_label == "Overstimulating" else "allow"
    runtime     = round(time.time() - t_start, 3)
    print(f"[ROUTE] NB-only ({reason[:60]}) → {video_id} {oir_label} ({score_final}) in {runtime}s")
    return {
        "video_id":        video_id,
        "video_title":     metadata.get("title", ""),
        "oir_label":       oir_label,
        "score_nb":        round(score_nb, 4),
        "score_h":         None,
        "score_final":     score_final,
        "cached":          False,
        "action":          action,
        "runtime_seconds": runtime,
        "status":          "success",
        "fallback_reason": reason[:120],
        "nb_details": {
            "predicted":  nb_obj.predicted_label,
            "confidence": round(nb_obj.confidence, 4),
        },
    }


def _fetch_metadata_only(video_url: str) -> dict:
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                                "skip_download": True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
        return {
            "title":       info.get("title", ""),
            "tags":        info.get("tags", []) or [],
            "description": info.get("description", "") or "",
        }
    except Exception as e:
        print(f"[META] ✗ {e}")
        return {"title": "", "tags": [], "description": ""}


def _save_to_db(result: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO videos
            (video_id, label, final_score, last_checked, checked_by,
             video_title, nb_score, heuristic_score, runtime_seconds)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
        """, (
            result["video_id"], result.get("oir_label", ""),
            result.get("score_final", 0.0), "hybrid_full",
            result.get("video_title", ""), result.get("score_nb", 0.0),
            result.get("score_h") or 0.0, result.get("runtime_seconds", 0.0),
        ))
        for seg in result.get("heuristic_details", {}).get("segments", []):
            cur.execute("""
                INSERT INTO segments
                (video_id, segment_id, offset_seconds, length_seconds,
                 fcr, csv, att, score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (result["video_id"], seg.get("segment_id"),
                  seg.get("offset_seconds"), seg.get("length_seconds"),
                  seg.get("fcr"), seg.get("csv"), seg.get("att"), seg.get("score_h")))
        conn.commit()
        conn.close()
        print(f"[DB] ✓ Saved {result['video_id']}")
    except Exception as e:
        print(f"[DB] ✗ {e}")


def _check_cache(video_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("SELECT label, final_score, last_checked FROM videos WHERE video_id = ?",
                    (video_id,))
        row = cur.fetchone()
        conn.close()
        return row
    except Exception as e:
        print(f"[CACHE] {e}")
        return None


# ── /classify_fast ────────────────────────────────────────────────────────────

@classify_bp.route("/classify_fast", methods=["POST"])
def classify_fast():
    data  = request.get_json(silent=True) or {}
    title = data.get("title", "")
    if not title:
        return jsonify({"error": "title is required", "status": "error"}), 400
    try:
        result = score_metadata(title=title, tags=data.get("tags", []),
                                description=data.get("description", ""))
        return jsonify({
            "score_nb":   result["score_nb"],
            "oir_label":  result["label"],
            "label":      result["label"],
            "confidence": result.get("confidence", 0.0),
            "status":     "success",
        }), 200
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


# ── /classify_full ────────────────────────────────────────────────────────────

@classify_bp.route("/classify_full", methods=["POST"])
def classify_full():
    """
    Full Hybrid Heuristic-Naïve Bayes classification.

    Always runs BOTH algorithms:
      1. Naïve Bayes — metadata scoring (title, tags, description)
      2. Heuristic   — audiovisual analysis (FCR, CSV, ATT, thumbnail)

    Fusion (v3 confidence-gated):
      eff_alpha   = LOW_ALPHA (0.15) if NB confidence < 0.40 else BASE_ALPHA (0.40)
      Score_final = (eff_alpha × Score_NB) + ((1 − eff_alpha) × Score_H)
      H-override  : Score_H < 0.10 → cannot be Overstimulating
    Thresholds: Block >= 0.20, Allow <= 0.18

    Fallback chain inside sample_video():
      Normal download
        → Cookie-authenticated retry  (age-restricted videos)
          → Thumbnail-only heuristic  (CSV + thumbnail, FCR/ATT = 0)
            → NB-only                 (video and thumbnail both unavailable)
    """
    data          = request.get_json(silent=True) or {}
    video_url     = data.get("video_url", "").strip()
    thumbnail_url = data.get("thumbnail_url", "")
    hint_title    = data.get("hint_title", "").strip()
    hint_desc     = (data.get("hint_description") or "").strip()
    hint_tags     = data.get("hint_tags") or []
    if hint_desc:
        print(f"[CLASSIFY_FULL] hint_description={repr(hint_desc[:120])}")
    else:
        print(f"[CLASSIFY_FULL] hint_description=EMPTY")

    if not video_url:
        return jsonify({"error": "video_url is required", "status": "error"}), 400

    video_id = extract_video_id(video_url)

    # ── Cache check ───────────────────────────────────────────────────────────
    cached = _check_cache(video_id)
    if cached:
        label, final_score, last_checked = cached
        print(f"[CACHE] ✓ Hit for {video_id} → {label}")
        return jsonify({
            "video_id":     video_id,
            "oir_label":    label,
            "score_final":  final_score,
            "last_checked": last_checked,
            "cached":       True,
            "action":       "block" if label == "Overstimulating" else "allow",
            "status":       "success",
        }), 200

    t_start = time.time()

    try:
        print(f"[ROUTE] /classify_full → {video_id}")

        # ── Run full pipeline: download + heuristic + NB ──────────────────────
        sample        = sample_video(video_url, thumbnail_url=thumbnail_url,
                                     hint_title=hint_title)
        sample_status = sample.get("status", "error")

        # ── Absolute fallback: video AND thumbnail both failed ────────────────
        if sample_status in ("unavailable", "error"):
            reason   = sample.get("reason", sample.get("message", "unavailable"))
            print(f"[ROUTE] ✗ Fully unavailable — NB-only for {video_id}")
            metadata = _fetch_metadata_only(video_url)
            if not metadata["title"] and hint_title:
                metadata["title"] = hint_title
            result = _nb_only_result(video_id, metadata, reason, t_start)
            _save_to_db(result)
            return jsonify(result), 200

        # ── Heuristic score (full video or thumbnail-only) ────────────────────
        h_result  = compute_heuristic_score(sample)
        score_h   = h_result["score_h"]
        h_details = h_result.get("details", {})

        # ── NB score (prefer downloaded title over hint) ──────────────────────
        nb_title = sample.get("video_title", "") or hint_title
        nb_desc  = sample.get("description", "") or hint_desc
        nb_tags  = sample.get("tags") or hint_tags
        print(f"[NB_INPUT] title={repr(nb_title[:80])}")
        print(f"[NB_INPUT] description={'EMPTY' if not nb_desc else repr(nb_desc[:120])}")
        print(f"[NB_INPUT] tags={nb_tags[:5]}")
        nb_obj = score_from_metadata_dict({
            "title":       nb_title,
            "tags":        nb_tags,
            "description": nb_desc,
        })
        score_nb        = nb_obj.score_nb
        predicted_label = nb_obj.predicted_label

        # ── Hybrid fusion ─────────────────────────────────────────────────────
        score_final, oir_label = _fuse(score_nb, score_h, nb_obj.confidence)
        action      = "block" if oir_label == "Overstimulating" else "allow"

        path_label = "full" if sample_status == "success" else "thumbnail-only"
        runtime    = round(time.time() - t_start, 3)
        print(f"[ROUTE] [{path_label}] nb={score_nb:.4f} h={score_h:.4f} "
              f"final={score_final:.4f}")

        result = {
            "video_id":          video_id,
            "video_title":       sample.get("video_title", "") or hint_title,
            "oir_label":         oir_label,
            "score_nb":          round(score_nb, 4),
            "score_h":           round(score_h, 4),
            "score_final":       score_final,
            "cached":            False,
            "action":            action,
            "runtime_seconds":   runtime,
            "status":            "success",
            "sample_path":       path_label,
            "fusion_weights": {
                "version":       "v3-confidence-gated",
                "base_alpha_nb": BASE_ALPHA,
                "low_alpha_nb":  LOW_ALPHA,
                "conf_thresh":   CONF_THRESH,
                "h_override":    H_OVERRIDE,
                "eff_alpha":     LOW_ALPHA if nb_obj.confidence < CONF_THRESH else BASE_ALPHA,
            },
            "heuristic_details": h_details,
            "nb_details": {
                "predicted":  predicted_label,
                "confidence": round(nb_obj.confidence, 4),
            },
        }
        _save_to_db(result)
        print(f"[ROUTE] /classify_full {video_id} → {oir_label} "
              f"({score_final}) [{path_label}] in {runtime}s")
        return jsonify(result), 200

    except Exception as e:
        print(f"[ROUTE] /classify_full error for {video_id}: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "status": "error"}), 500


# ── /classify_by_title ────────────────────────────────────────────────────────

@classify_bp.route("/classify_by_title", methods=["POST"])
def classify_by_title():
    data    = request.get_json(silent=True) or {}
    title   = data.get("title", "").strip()
    channel = data.get("channel", "").strip().lstrip("@")

    if not title:
        return jsonify({"error": "title is required", "status": "error"}), 400
    if len(title.split()) < 2:
        print(f"[TITLE_ROUTE] Rejected: {title!r}")
        return jsonify({"error": "Title too short", "status": "error"}), 400

    # Build a Shorts-specific search query using title + channel handle (if available)
    query = f"{title} {channel} #shorts".strip() if channel else f"{title} #shorts"
    print(f"[TITLE_ROUTE] Searching for: {query!r}")

    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                                "extract_flat": "in_playlist"}) as ydl:
            info = ydl.extract_info(f"ytsearch3:{query}", download=False)

        entries = info.get("entries", [])
        if not entries:
            return jsonify({"error": "No video found", "status": "error"}), 404

        # Prefer the first result that is actually a Short (duration <= 65s)
        entry = next(
            (e for e in entries if (e.get("duration") or 999) <= 65),
            entries[0]  # fallback to top result if none match duration
        )

        video_id  = entry.get("id", "")
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        thumb_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        duration  = entry.get("duration")

        print(f"[TITLE_ROUTE] Resolved: {title!r} -> {video_id} (duration={duration}s, is_short={'YES' if duration and duration <= 65 else 'NO/unknown'})")

        # ── Fetch full metadata for description + tags ────────────────────────
        # extract_flat mode above intentionally skips description/tags,
        # so we do a second targeted fetch on the resolved video URL.
        hint_desc = ""
        hint_tags = []
        try:
            with yt_dlp.YoutubeDL({
                "quiet":         True,
                "no_warnings":   True,
                "skip_download": True,
            }) as ydl_full:
                full_info = ydl_full.extract_info(video_url, download=False)
            hint_desc = (full_info.get("description") or "").strip()
            hint_tags = full_info.get("tags") or []
            print(f"[TITLE_ROUTE] description={'EMPTY' if not hint_desc else repr(hint_desc[:120])}")
            print(f"[TITLE_ROUTE] tags={hint_tags[:5]}")
        except Exception as e:
            print(f"[TITLE_ROUTE] ✗ full metadata fetch failed: {e}")
            hint_desc = (entry.get("description") or "").strip()  # flat fallback
            hint_tags = entry.get("tags") or []
        # ─────────────────────────────────────────────────────────────────────

        cached = _check_cache(video_id)
        if cached:
            label, final_score, last_checked = cached
            return jsonify({
                "video_id":     video_id,
                "oir_label":    label,
                "score_final":  final_score,
                "last_checked": last_checked,
                "cached":       True,
                "action":       "block" if label == "Overstimulating" else "allow",
                "status":       "success",
            }), 200

        from flask import current_app
        with current_app.test_request_context(
            "/classify_full", method="POST",
            json={"video_url": video_url, "thumbnail_url": thumb_url,
                  "hint_title": title, "hint_description": hint_desc,
                  "hint_tags": hint_tags},
        ):
            return classify_full()

    except Exception as e:
        print(f"[TITLE_ROUTE] Error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "status": "error"}), 500


# ── /health ───────────────────────────────────────────────────────────────────

@classify_bp.route("/health", methods=["GET"])
def health():
    from app.modules.naive_bayes import model_status
    from app.modules.frame_sampler import COOKIES_PATH, _has_cookies
    return jsonify({
        "status":       "ok",
        "nb_model":     model_status(),
        "db_path":      DB_PATH,
        "db_exists":    os.path.exists(DB_PATH),
        "cookies_path": COOKIES_PATH,
        "cookies_ok":   _has_cookies(),
        "fusion_config": {
            "version":         "v3-confidence-gated",
            "base_alpha_nb":   BASE_ALPHA,
            "low_alpha_nb":    LOW_ALPHA,
            "conf_thresh":     CONF_THRESH,
            "h_override":      H_OVERRIDE,
            "threshold_block": THRESHOLD_BLOCK,
            "threshold_allow": THRESHOLD_ALLOW,
            "neutral_range":   f"{THRESHOLD_ALLOW} < score < {THRESHOLD_BLOCK}",
        },
    }), 200
