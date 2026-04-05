"""
ChildFocus - Frame Sampling Module
backend/app/modules/frame_sampler.py

Fallback chain for unavailable / age-restricted videos
───────────────────────────────────────────────────────
  1. fetch_video()            — normal download (no cookies)
  2. fetch_video(cookies)     — retry with cookies.txt (bypasses age-gate)
  3. _sample_thumbnail_only() — CSV from thumbnail image, FCR=0, ATT=0
  4. status="unavailable"     — classify.py falls back to NB-only

Cookies setup (one-time)
────────────────────────
  Export cookies from a logged-in YouTube tab using the browser extension
  "Get cookies.txt LOCALLY" (Chrome/Firefox), save as:
      backend/cookies.txt
  That path is controlled by COOKIES_PATH below.

Optimizations (Sprint 2 + empirical updates):
  1. fetch_video()        → validate + download in ONE yt-dlp call (saves ~8-15s)
  2. ThreadPoolExecutor   → S1, S2, S3, thumbnail all run concurrently (saves ~10-20s)
  3. librosa direct read  → no ffmpeg subprocess per segment (saves ~3-6s)
  4. Frame resize 320px   → faster numpy ops on smaller frames
  5. Runtime timer        → printed on completion + returned in response
  6. Short video fix      → segments deduplicated for videos < 20s (e.g. Shorts)

Performance updates (empirically validated):
  7. max_duration reduced 90 → 63s  — covers 3×20s segments exactly, avoids
     downloading excess video that is never analyzed (saves ~20-60s on long videos)
  8. AD_SKIP_SECONDS = 15    — all segment offsets start after the first 15s to
     avoid scanning YouTube pre-roll ads which inflate FCR and ATT readings
  9. FRAME_SAMPLE_RATE 1 → 0.5 — 1 frame per 2 seconds instead of 1fps,
     halves frame extraction time while still capturing pacing changes accurately
"""

import os
import re
import time
import warnings
import cv2
import numpy as np
import tempfile
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from PIL import Image
    from io import BytesIO
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    print("[ERROR] yt-dlp not installed.")

# ── Constants ──────────────────────────────────────────────────────────────────
SEGMENT_DURATION  = 20       # seconds per segment (unchanged)
FRAME_SAMPLE_RATE = 0.5      # frames per second — was 1, reduced to halve processing time
C_MAX             = 4.0      # max cuts/sec normalization
S_MAX             = 128.0    # max saturation normalization
FRAME_WIDTH       = 320      # resize width for faster numpy ops

# ── Ad-skip offset ─────────────────────────────────────────────────────────────
# YouTube injects pre-roll ads in the first 0-15 seconds of many videos.
# All segment start offsets are pushed forward by this amount so the
# heuristic analyzes actual video content rather than ad content.
# For Shorts and videos shorter than AD_SKIP_SECONDS + SEGMENT_DURATION,
# this is automatically reduced to 0 (handled in sample_video()).
AD_SKIP_SECONDS   = 15

# ── Download limit ─────────────────────────────────────────────────────────────
# Maximum seconds of video to download per request.
# 3 segments × 20s + 1 AD_SKIP_SECONDS buffer = 75s is sufficient.
# Reduced from 90 to 63 — covers exactly 3 non-overlapping 20s segments
# after the ad skip, without downloading unused video data.
MAX_DOWNLOAD_SECONDS = 63    # was 90

# ── Cookies path ───────────────────────────────────────────────────────────────
COOKIES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "cookies.txt")


def _has_cookies() -> bool:
    """Returns True if a non-empty cookies.txt exists at COOKIES_PATH."""
    return os.path.isfile(COOKIES_PATH) and os.path.getsize(COOKIES_PATH) > 0


def _extract_video_id(url_or_id: str) -> str:
    """
    Normalise input: accepts a full YouTube URL or a bare 11-char video ID.
    Returns the 11-character video ID.
    """
    for pattern in [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]:
        m = re.search(pattern, url_or_id)
        if m:
            return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url_or_id.strip()):
        return url_or_id.strip()
    return url_or_id.strip()


# ── yt-dlp shared options ──────────────────────────────────────────────────────
def _ydl_opts(extra: dict = None, cookies_file: str = None) -> dict:
    opts = {
        "quiet":              True,
        "no_warnings":        True,
        "noprogress":         True,
        "geo_bypass":         True,
        "geo_bypass_country": "US",
        "xff":                "US",           # explicit XFF header (newer yt-dlp)
        "extract_flat":       False,
        "retries":            3,
        "fragment_retries":   3,
        "extractor_args": {
            "youtube": {
                # "android" is the key geo-bypass client — it uses InnerTube with
                # softer regional enforcement than the web client.
                # Do NOT replace with android_vr — that client only serves DASH
                # streams and does NOT bypass geo-restrictions.
                "player_client": ["web", "tv_embedded", "android"]
            }
        },
        "http_headers": {
            "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if extra:
        opts.update(extra)
    return opts


# ── Download: validate AND fetch in one yt-dlp call ───────────────────────────
def fetch_video(video_id: str, max_duration: int = MAX_DOWNLOAD_SECONDS,
                cookies_file: str = None) -> dict:
    """
    Single yt-dlp call — validates availability AND downloads.
    Downloads only max_duration seconds (default 63s) to minimize wait time.
    """
    if not YTDLP_AVAILABLE:
        return {"ok": False, "reason": "yt-dlp not installed"}

    output_path = tempfile.mktemp(suffix=".mp4")

    urls_to_try = [
        f"https://www.youtube.com/watch?v={video_id}",
        f"https://www.youtube.com/shorts/{video_id}",
    ]

    last_error = None
    for url in urls_to_try:
        try:
            opts = _ydl_opts(
                extra={
                    "format":            "worst[ext=mp4]/worst",
                    "outtmpl":           output_path,
                    "download_sections": [f"*0-{max_duration}"],
                    "postprocessors":    [],
                },
                cookies_file=cookies_file,
            )
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)

            if not os.path.exists(output_path):
                raise FileNotFoundError("Downloaded file missing")

            return {
                "ok":          True,
                "path":        output_path,
                "title":       info.get("title",       "Unknown"),
                "tags":        info.get("tags",         []) or [],
                "description": info.get("description", "") or "",
                "duration":    info.get("duration",     0),
                "uploader":    info.get("uploader",     "Unknown"),
            }
        except Exception as e:
            last_error = e
            if os.path.exists(output_path):
                try: os.remove(output_path)
                except Exception: pass
            continue

    msg = str(last_error).lower()
    if "not available" in msg:
        reason = "Video is not available in this region or has been removed"
    elif "private" in msg:
        reason = "Video is private"
    elif "age" in msg or "restricted" in msg:
        reason = "Video is age-restricted"
    elif "members" in msg:
        reason = "Video is members-only"
    elif "copyright" in msg:
        reason = "Video is unavailable due to copyright"
    else:
        reason = str(last_error)
    return {"ok": False, "reason": reason}


# ── Frame extraction ───────────────────────────────────────────────────────────
def extract_frames(video_path: str, start_sec: int, duration: int) -> list:
    """
    Extracts frames at FRAME_SAMPLE_RATE fps, resized to FRAME_WIDTH.
    0.5fps = 1 frame every 2 seconds — adequate for pacing detection,
    half the processing time of 1fps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame    = int(min((start_sec + duration) * fps, total_frames))
    start_frame  = int(start_sec * fps)
    step         = max(1, int(fps / FRAME_SAMPLE_RATE))  # frames to skip between samples

    frames = []
    idx = start_frame
    while idx < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)),
                           interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
        idx += step

    cap.release()
    return frames


# ── FCR ───────────────────────────────────────────────────────────────────────
def compute_fcr(frames: list) -> float:
    if len(frames) < 2:
        return 0.0
    cuts = 0
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for f in frames[1:]:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if np.mean(cv2.absdiff(prev, gray)) > 25:
            cuts += 1
        prev = gray
    return round(min(1.0, (cuts / max(len(frames), 1)) / C_MAX), 4)


# ── CSV ───────────────────────────────────────────────────────────────────────
def compute_csv(frames: list) -> float:
    if not frames:
        return 0.0
    sats = [np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:, :, 1]) for f in frames]
    return round(min(1.0, float(np.std(sats)) / S_MAX), 4)


# ── ATT ───────────────────────────────────────────────────────────────────────
def compute_att(video_path: str, start_sec: int, duration: int) -> float:
    """
    Fast path: librosa reads directly from MP4.
    Fallback: ffmpeg WAV extraction if direct read fails.
    """
    if LIBROSA_AVAILABLE:
        try:
            y, sr = librosa.load(
                video_path,
                offset=float(start_sec),
                duration=float(duration),
                sr=22050,
                mono=True,
            )
            if len(y) > 100:
                return round(min(1.0, float(np.mean(
                    librosa.onset.onset_strength(y=y, sr=sr)
                )) / 10.0), 4)
        except Exception:
            pass

    # Fallback: ffmpeg WAV
    wav_path = tempfile.mktemp(suffix=".wav")
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start_sec), "-t", str(duration),
             "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "22050", "-ac", "1", wav_path],
            capture_output=True, timeout=30,
        )
        if r.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 500:
            if LIBROSA_AVAILABLE:
                y, sr = librosa.load(wav_path, sr=None, mono=True)
                if len(y) > 100:
                    return round(min(1.0, float(np.mean(
                        librosa.onset.onset_strength(y=y, sr=sr)
                    )) / 10.0), 4)
            import wave
            with wave.open(wav_path, "rb") as wf:
                samples = np.frombuffer(
                    wf.readframes(wf.getnframes()), dtype=np.int16
                ).astype(np.float32) / 32768.0
            if len(samples) > 100:
                chunk = 2205
                rms = [float(np.sqrt(np.mean(samples[i:i+chunk]**2)))
                       for i in range(0, len(samples) - chunk, chunk)]
                if rms:
                    return round(min(1.0, float(np.std(rms)) * 10.0), 4)
    except Exception as e:
        print(f"[ATT] Fallback error: {e}")
    finally:
        if os.path.exists(wav_path):
            try: os.remove(wav_path)
            except Exception: pass
    return 0.0


# ── Thumbnail ─────────────────────────────────────────────────────────────────
def compute_thumbnail_intensity(url: str) -> float:
    if not url:
        return 0.0
    try:
        raw = requests.get(url, timeout=6).content
        img = (cv2.cvtColor(np.array(Image.open(BytesIO(raw)).convert("RGB")),
                            cv2.COLOR_RGB2BGR)
               if PILLOW_AVAILABLE
               else cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))
        if img is None:
            return 0.0
        mean_sat = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1])) / 255.0
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge_den = float(np.sum(cv2.Canny(gray, 100, 200) > 0)) / float(gray.size)
        return round(min(1.0, 0.7 * mean_sat + 0.3 * edge_den), 4)
    except Exception as e:
        print(f"[THUMB] {e}")
        return 0.0


# ── Thumbnail-only heuristic ──────────────────────────────────────────────────
def _sample_thumbnail_only(video_id: str, thumbnail_url: str,
                            hint_title: str = "") -> dict:
    """
    Partial heuristic when video download is impossible.
    Computes CSV + THUMB from thumbnail only. FCR=0, ATT=0.
    """
    t_start = time.time()
    print(f"[SAMPLER] ── Thumbnail-only fallback for {video_id} ──────────")

    if not thumbnail_url:
        return {"video_id": video_id, "status": "unavailable",
                "reason": "No thumbnail URL available"}

    try:
        raw = requests.get(thumbnail_url, timeout=6).content
        img = (cv2.cvtColor(np.array(Image.open(BytesIO(raw)).convert("RGB")),
                            cv2.COLOR_RGB2BGR)
               if PILLOW_AVAILABLE
               else cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))

        if img is None:
            raise ValueError("Thumbnail decode returned None")

        hsv      = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        step     = max(1, hsv.shape[0] // 20)
        row_sats = [float(np.mean(hsv[r, :, 1]))
                    for r in range(0, hsv.shape[0], step)]
        csv      = round(min(1.0, float(np.std(row_sats)) / S_MAX), 4)

        mean_sat = float(np.mean(hsv[:, :, 1])) / 255.0
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge_den = float(np.sum(cv2.Canny(gray, 100, 200) > 0)) / float(gray.size)
        thumb    = round(min(1.0, 0.7 * mean_sat + 0.3 * edge_den), 4)

        fcr         = 0.0
        att         = 0.0
        score_h_seg = round(0.35 * fcr + 0.25 * csv + 0.20 * att, 4)
        seg = {
            "segment_id": "S_THUMB", "offset_seconds": 0, "length_seconds": 0,
            "fcr": fcr, "csv": csv, "att": att, "score_h": score_h_seg,
        }
        segments  = [seg, seg, seg]
        agg_score = round(0.80 * score_h_seg + 0.20 * thumb, 4)
        label     = ("Overstimulating" if agg_score >= 0.20
                     else "Safe"       if agg_score <= 0.08
                     else "Neutral")

        print(f"[SAMPLER] Thumbnail-only: CSV={csv} | THUMB={thumb} | "
              f"seg_h={score_h_seg} | agg={agg_score} → {label}")

        return {
            "video_id":                  video_id,
            "video_title":               hint_title,
            "video_duration_sec":        0,
            "thumbnail_url":             thumbnail_url,
            "thumbnail_intensity":       thumb,
            "segments":                  segments,
            "tags":                      [],
            "description":               "",
            "aggregate_heuristic_score": agg_score,
            "preliminary_label":         label,
            "status":                    "thumbnail_only",
            "runtime_seconds":           round(time.time() - t_start, 2),
        }

    except Exception as e:
        print(f"[SAMPLER] ✗ Thumbnail-only failed: {e}")
        return {"video_id": video_id, "status": "unavailable", "reason": str(e)}


# ── Process one segment ───────────────────────────────────────────────────────
def _process_segment(video_path: str, seg_id: str, start: int, seg_dur: int = 20) -> dict:
    t       = time.time()
    frames  = extract_frames(video_path, start, seg_dur)
    fcr     = compute_fcr(frames)
    csv     = compute_csv(frames)
    att     = compute_att(video_path, start, seg_dur)
    
    # Apply High-Sensitivity Segment Scoring
    multiplier = 1.25 if (fcr > 0.5 or att > 0.5) else 1.0
    score_h = round((0.45 * fcr + 0.15 * csv + 0.30 * att) * multiplier, 4)
    
    return {
        "segment_id": seg_id,
        "offset_seconds": start,
        "length_seconds": seg_dur,
        "fcr": fcr, "csv": csv, "att": att, "score_h": score_h,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def sample_video(video_url_or_id: str, thumbnail_url: str = "",
                 hint_title: str = "") -> dict:
    """
    Full fallback chain:
      1. Normal download (no cookies)
      2. Cookie-authenticated download  [if cookies.txt exists]
      3. Thumbnail-only heuristic       [CSV + thumb intensity, FCR/ATT = 0]
      4. status="unavailable"           [classify.py falls back to NB-only]

    Ad-skip: all segment offsets start at AD_SKIP_SECONDS (15s) minimum
    to avoid scanning YouTube pre-roll ads.

    For short videos (Shorts, clips < AD_SKIP_SECONDS + SEGMENT_DURATION),
    ad-skip is automatically disabled.
    """
    t_start  = time.time()
    video_id = _extract_video_id(video_url_or_id)

    try:
        print(f"\n[SAMPLER] ══════════════════════════════════════")
        print(f"[SAMPLER] Analyzing: {video_id}")

        # ── Step 1 & 2: Download with preemptive cookies ──────────────────────
        # Using cookies on the FIRST attempt is critical: YouTube authenticates
        # the session upfront and serves a wider format list (including pre-muxed
        # streams). A no-cookie first attempt negotiates a restricted format list,
        # and a subsequent cookie retry cannot recover the format selection.
        t0 = time.time()
        use_cookies = COOKIES_PATH if _has_cookies() else None

        if use_cookies:
            print("[SAMPLER] Found cookies.txt, using preemptively to bypass bot-blocks.")

        result = fetch_video(video_id, max_duration=MAX_DOWNLOAD_SECONDS, cookies_file=use_cookies)
        print(f"[SAMPLER] Download attempt 1: {time.time()-t0:.1f}s")

        # Retry without cookies only for non-geo-block failures (e.g. expired cookie).
        # For true geo-blocks ("not available"), a cookieless retry wastes ~6s and
        # returns the same error — skip it.
        _reason = result.get("reason", "").lower()
        _is_geo_block = "not available" in _reason or "region" in _reason
        if not result["ok"] and not use_cookies and _has_cookies():
            print(f"[SAMPLER] ⚠ Failed ({result['reason']}) — retrying with cookies.txt")
            t0 = time.time()
            result = fetch_video(video_id, max_duration=MAX_DOWNLOAD_SECONDS, cookies_file=COOKIES_PATH)
            print(f"[SAMPLER] Cookie retry: {time.time()-t0:.1f}s")
        elif not result["ok"] and _is_geo_block:
            print(f"[SAMPLER] ✗ True geo-block — skipping cookieless retry (android client + geo_bypass already tried)")

        # ── Step 3: Thumbnail-only fallback ───────────────────────────────────
        if not result["ok"]:
            print(f"[SAMPLER] ✗ {result['reason']} — thumbnail-only fallback")
            return _sample_thumbnail_only(video_id, thumbnail_url,
                                          hint_title=hint_title)

        # ── Full pipeline ──────────────────────────────────────────────────────
        video_path     = result["path"]
        cap            = cv2.VideoCapture(video_path)
        video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
        cap.release()
        print(f"[SAMPLER] ✓ '{result['title']}' ({video_duration:.1f}s)")

        actual_dur = min(video_duration, MAX_DOWNLOAD_SECONDS)

        # ── Determine ad-skip offset ───────────────────────────────────────────
        # Disable ad-skip for short videos where skipping 15s would leave
        # too little content (Shorts, clips under 35 seconds total)

        min_duration_for_ad_skip = AD_SKIP_SECONDS + SEGMENT_DURATION
        if actual_dur < min_duration_for_ad_skip:
            ad_skip = 0
            print(f"[SAMPLER] Short video ({actual_dur:.0f}s) — ad-skip disabled")
        else:
            ad_skip = AD_SKIP_SECONDS
            print(f"[SAMPLER] Ad-skip: first {ad_skip}s excluded from analysis")

        content_dur = actual_dur - ad_skip

        # >>> REPLACE YOUR IF/ELSE BLOCK WITH THIS: <<<
        if content_dur <= SEGMENT_DURATION:
            # User Insight: Analyze the whole video as ONE block for better data stability.
            effective_seg_dur = max(1, int(content_dur))
            # We only prepare S1 here. We will duplicate it later to save CPU time.
            seg_starts = [("S1", ad_skip)]
            is_short_video = True
        else:
            effective_seg_dur = SEGMENT_DURATION
            mid = ad_skip + max(0, int(content_dur / 2) - effective_seg_dur // 2)
            end = ad_skip + max(0, int(content_dur) - effective_seg_dur)
            seg_starts = [("S1", ad_skip), ("S2", mid), ("S3", end)]
            is_short_video = False

        print(f"[SAMPLER] Segments: {[(s, o) for s, o in seg_starts]} "
              f"| seg_dur={effective_seg_dur}s | ad_skip={ad_skip}s")

        # ── Concurrent analysis ────────────────────────────────────────────────
        t0       = time.time()
        segments = [None, None, None]
        thumb    = 0.0

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_process_segment, video_path, sid, start, effective_seg_dur): i
                for i, (sid, start) in enumerate(seg_starts)
            }
            futures[pool.submit(compute_thumbnail_intensity, thumbnail_url)] = "thumb"

            for future in as_completed(futures):
                key = futures[future]
                if key == "thumb":
                    thumb = future.result()
                    print(f"[SAMPLER] Thumbnail: {thumb}")
                else:
                    segments[key] = future.result()

        # >>> ADD THIS DUPLICATION LOGIC RIGHT AFTER THE THREADPOOL <<<
        if is_short_video and segments[0] is not None:
            # We computed the whole video perfectly once. Now clone it for the DB.
            s1_data = segments[0]
            segments[1] = {**s1_data, "segment_id": "S2"}
            segments[2] = {**s1_data, "segment_id": "S3"}

        print(f"[SAMPLER] Analysis: {time.time()-t0:.1f}s")

        max_seg   = max(s["score_h"] for s in segments if s is not None)
        agg_score = round(0.90 * max_seg + 0.10 * thumb, 4)
        # Updated Label Thresholds (v2)
        label     = ("Overstimulating" if agg_score >= 0.30
                     else "Safe"       if agg_score <= 0.12
                     else "Neutral")

        total = time.time() - t_start
        print(f"[SAMPLER] ✓ Score: {agg_score} → {label}")
        print(f"[SAMPLER] ✓ Total runtime: {total:.1f}s")
        print(f"[SAMPLER] ══════════════════════════════════════\n")

        return {
            "video_id":                  video_id,
            "video_title":               result.get("title", ""),
            "tags":                      result.get("tags", []),
            "description":               result.get("description", ""),
            "video_duration_sec":        round(video_duration, 1),
            "thumbnail_url":             thumbnail_url,
            "thumbnail_intensity":       thumb,
            "segments":                  segments,
            "aggregate_heuristic_score": agg_score,
            "preliminary_label":         label,
            "status":                    "success",
            "runtime_seconds":           round(total, 2),
            "ad_skip_seconds":           ad_skip,
        }

    except Exception as e:
        print(f"[SAMPLER] ✗ Fatal: {e}")
        import traceback; traceback.print_exc()
        return {"video_id": video_id, "status": "error", "message": str(e)}
    finally:
        if "result" in dir() and isinstance(result, dict) and result.get("ok"):
            path = result.get("path", "")
            if path and os.path.exists(path):
                try: os.remove(path)
                except Exception: pass
