import os
import sqlite3
import time

from flask import Blueprint, jsonify, request

from app.modules.frame_sampler import sample_video
from app.modules.heuristic import compute_heuristic_score
from app.modules.naive_bayes import score_from_metadata_dict, score_metadata
from app.modules.hybrid_fusion import ALPHA, BETA, THRESHOLD_BLOCK, THRESHOLD_ALLOW

classify_bp = Blueprint("classify", __name__)

DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "database", "childfocus.db"
)

def _oir_label(score: float) -> str:
    if score >= THRESHOLD_BLOCK: return "Overstimulating"
    if score <= THRESHOLD_ALLOW: return "Educational"
    return "Neutral"


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
    oir_label   = _oir_label(score_final)
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


# ══════════════════════════════════════════════════════════════════════════════
# /classify_fast
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# /classify_full
# ══════════════════════════════════════════════════════════════════════════════

@classify_bp.route("/classify_full", methods=["POST"])
def classify_full():
    """
    Full Hybrid Heuristic-Naïve Bayes classification.

    Always runs BOTH algorithms:
      1. Naïve Bayes — metadata scoring (title, tags, description)
      2. Heuristic   — audiovisual analysis (FCR, CSV, ATT, thumbnail)

    Fusion: Score_final = (0.6 × Score_NB) + (0.4 × Score_H)
    Thresholds: Block >= 0.75, Allow <= 0.35

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
        nb_obj = score_from_metadata_dict({
            "title":       sample.get("video_title", "") or hint_title,
            "tags":        sample.get("tags", []),
            "description": sample.get("description", ""),
        })
        score_nb        = nb_obj.score_nb
        predicted_label = nb_obj.predicted_label

        # ── Hybrid fusion ─────────────────────────────────────────────────────
        # Score_final = (0.4 × Score_NB) + (0.6 × Score_H)
        score_final = round((ALPHA * score_nb) + (BETA * score_h), 4)
        oir_label   = _oir_label(score_final)
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
                "alpha_nb":       ALPHA,
                "beta_heuristic": BETA,
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


# ══════════════════════════════════════════════════════════════════════════════
# /classify_by_title
# ══════════════════════════════════════════════════════════════════════════════

@classify_bp.route("/classify_by_title", methods=["POST"])
def classify_by_title():
    data             = request.get_json(silent=True) or {}
    title            = data.get("title", "").strip()
    channel          = data.get("channel", "").strip()
    channel_id       = data.get("channel_id", "").strip()
    duration_seconds = int(data.get("duration_seconds", 0))
    is_verified      = bool(data.get("is_verified", False))

    if not title or len(title.split()) < 2:
        return jsonify({"error": "Title too short", "status": "error"}), 400

    # Build search query — channel first for precision
    query = f"{channel} {title}".strip() if channel else title
    print(f"[TITLE_ROUTE] query={query!r} dur={duration_seconds}s verified={is_verified}")

    try:
        import yt_dlp

        # Search 5 candidates instead of 1
        with yt_dlp.YoutubeDL({
            "quiet": True, "no_warnings": True,
            "extract_flat": True,
        }) as ydl:
            info    = ydl.extract_info(f"ytsearch5:{query}", download=False)
            entries = info.get("entries", [])

        if not entries:
            return jsonify({"error": "No video found", "status": "error"}), 404

        # ── Duration filter (most important disambiguator) ──────────────────
        best_entry = None
        if duration_seconds > 0:
            # Prefer videos within ±5s of the known duration
            TOLERANCE = 5
            duration_matches = [
                e for e in entries
                if e.get("duration") and
                   abs(int(e["duration"]) - duration_seconds) <= TOLERANCE
            ]
            if duration_matches:
                # Among matches, prefer verified channel if flag set
                if is_verified:
                    verified = [
                        e for e in duration_matches
                        if is_verified_channel(e)
                    ]
                    best_entry = verified[0] if verified else duration_matches[0]
                else:
                    best_entry = duration_matches[0]
                print(f"[TITLE_ROUTE] Duration match: "
                      f"{best_entry['id']} dur={best_entry.get('duration')}s")
            else:
                # No exact match — pick closest by duration
                best_entry = min(
                    entries,
                    key=lambda e: abs(
                        int(e.get("duration") or 9999) - duration_seconds
                    )
                )
                print(f"[TITLE_ROUTE] Closest duration: "
                      f"{best_entry['id']} dur={best_entry.get('duration')}s "
                      f"(wanted {duration_seconds}s)")
        else:
            # No duration — use first result (lower accuracy, log warning)
            best_entry = entries[0]
            print(f"[TITLE_ROUTE] ⚠ No duration supplied — accuracy ~30-40%")

        video_id  = best_entry.get("id", "")
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        thumb_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        print(f"[TITLE_ROUTE] Resolved: {title!r} → {video_id} "
              f"(dur={best_entry.get('duration')}s)")

        # Continue to classify_full as before...
        cached = _check_cache(video_id)
        if cached:
            label, final_score, last_checked = cached
            return jsonify({
                "video_id":     video_id,
                "video_title":  title,
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
            json={
                "video_url":     video_url,
                "thumbnail_url": thumb_url,
                "hint_title":    title,
            },
        ):
            return classify_full()

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "status": "error"}), 500


def is_verified_channel(entry: dict) -> bool:
    """Heuristic — yt-dlp flat extract doesn't always expose verified badge."""
    channel = (entry.get("channel") or entry.get("uploader") or "").lower()
    # Channel name matching @Handle is a proxy for verified in flat results
    return len(channel) > 2


# ══════════════════════════════════════════════════════════════════════════════
# /safe_suggestions — KEEP THIS ONE ONLY
# ══════════════════════════════════════════════════════════════════════════════

@classify_bp.route("/safe_suggestions", methods=["GET"])
def safe_suggestions():
    """
    Returns up to `limit` Educational videos from the DB.
    Falls back to hardcoded safe videos if the DB has no Educational entries.

    Query params:
      limit (int, default 3) — number of suggestions to return
      exclude (str, optional) — video_id to exclude (the one just blocked)

    Response:
      {
        "suggestions": [
          {"video_id": "...", "video_title": "...", "final_score": 0.04},
          ...
        ],
        "source": "db" | "fallback"
      }
    """
    limit      = request.args.get("limit",   3,   type=int)
    exclude_id = request.args.get("exclude", "",  type=str).strip()

    # Clamp limit between 1 and 5
    limit = max(1, min(5, limit))

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # ← Makes rows dict-like
        cur  = conn.cursor()

        if exclude_id:
            cur.execute("""
                SELECT video_id, video_title, final_score
                FROM   videos
                WHERE  label = 'Educational'
                AND    video_id != ?
                ORDER  BY last_checked DESC
                LIMIT  ?
            """, (exclude_id, limit))
        else:
            cur.execute("""
                SELECT video_id, video_title, final_score
                FROM   videos
                WHERE  label = 'Educational'
                ORDER  BY last_checked DESC
                LIMIT  ?
            """, (limit,))

        rows = cur.fetchall()
        conn.close()

        if rows:
            suggestions = [
                {
                    "video_id":    row["video_id"],
                    "video_title": row["video_title"] or f"Educational video ({row['video_id']})",
                    "final_score": round(row["final_score"], 4),
                }
                for row in rows
            ]
            print(f"[SAFE] ✓ Returning {len(suggestions)} suggestions from DB")
            return jsonify({"suggestions": suggestions, "source": "db"}), 200

        # DB has no Educational entries yet — use hardcoded fallback
        print("[SAFE] DB empty — using fallback suggestions")
        fallback = [
            {
                "video_id":    "WaO3dBiC0kI",
                "video_title": "Khan Academy – Introduction to Fractions",
                "final_score": 0.04,
            },
            {
                "video_id":    "09maaUaRT4M",
                "video_title": "National Geographic Kids – Amazing Animals",
                "final_score": 0.05,
            },
            {
                "video_id":    "Ck-AMBxM2ww",
                "video_title": "Science for Kids – How Plants Grow",
                "final_score": 0.06,
            },
        ]
        return jsonify({"suggestions": fallback[:limit], "source": "fallback"}), 200

    except Exception as e:
        print(f"[SAFE] ✗ {e}")
        # Even if DB fails, return something useful
        fallback = [
            {"video_id": "WaO3dBiC0kI", "video_title": "Khan Academy – Fractions", "final_score": 0.04},
            {"video_id": "09maaUaRT4M", "video_title": "Nat Geo Kids – Animals",   "final_score": 0.05},
        ]
        return jsonify({"suggestions": fallback[:limit], "source": "fallback"}), 200


# ══════════════════════════════════════════════════════════════════════════════
# /health
# ══════════════════════════════════════════════════════════════════════════════

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
            "alpha_nb":        ALPHA,
            "beta_heuristic":  BETA,
            "threshold_block": THRESHOLD_BLOCK,
            "threshold_allow": THRESHOLD_ALLOW,
        },
    }), 200
