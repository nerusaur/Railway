import os
import re
import json
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

THUMBNAIL_QUALITY = ["maxres", "standard", "high", "medium", "default"]

_SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def get_best_thumbnail_url(thumbnails: dict) -> str:
    """Returns the highest quality thumbnail URL from a thumbnails dict."""
    for quality in THUMBNAIL_QUALITY:
        if quality in thumbnails:
            return thumbnails[quality]["url"]
    return ""


def extract_video_id(url: str) -> str:
    """Extracts video ID from a YouTube URL. Always returns an 11-char ID."""
    import re
    url = url.strip()
    for pattern in [r"[?&]v=([A-Za-z0-9_-]{11})",
                    r"youtu\.be/([A-Za-z0-9_-]{11})",
                    r"shorts/([A-Za-z0-9_-]{11})"]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    # Raw ID passed (possibly repeated) — extract first valid 11-char sequence
    m = re.search(r"([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    return url


def scrape_ytInitialData_keywords(video_id: str) -> list:
    """
    Scrapes the hidden 'keywords' array embedded in YouTube's ytInitialData
    JavaScript blob — invisible to viewers but present in raw page HTML.

    This is the most complete tag source: the YouTube Data API v3 often returns
    an empty tags list when creators mark them private, and yt-dlp may also miss
    them. ytInitialData always contains the full keyword set used for SEO.

    Returns a list of keyword strings, or [] on failure.
    """
    urls_to_try = [
        f"https://www.youtube.com/watch?v={video_id}",
        f"https://www.youtube.com/shorts/{video_id}",
    ]
    for url in urls_to_try:
        try:
            resp = requests.get(url, headers=_SCRAPE_HEADERS, timeout=10)
            resp.raise_for_status()
            match = re.search(r'"keywords"\s*:\s*(\[[^\]]*\])', resp.text)
            if match:
                keywords = json.loads(match.group(1))
                print(f"[YTAPI] ✓ Scraped {len(keywords)} hidden keywords for {video_id}")
                return [str(k) for k in keywords if k]
        except Exception as e:
            print(f"[YTAPI] ✗ Scrape failed ({url}): {e}")
            continue
    print(f"[YTAPI] ✗ No keywords array found in ytInitialData for {video_id}")
    return []


def _merge_tags(*tag_sources) -> list:
    """
    Merges multiple tag lists, deduplicating case-insensitively.
    Preserves original casing from the first occurrence of each tag.
    """
    seen  = set()
    merged = []
    for source in tag_sources:
        for tag in (source or []):
            key = str(tag).lower().strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(str(tag).strip())
    return merged


def get_video_metadata(video_id: str) -> dict:

    """
    Fetches title, tags, description, and duration from YouTube Data API v3.
    Returns a dict or raises an exception on failure.
    """
    if not API_KEY:
        return {"error": "YOUTUBE_API_KEY not set in .env"}

    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,contentDetails,statistics",
        "id": video_id,
        "key": API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}

    if not data.get("items"):
        return {"error": f"Video not found: {video_id}"}

    item    = data["items"][0]
    snippet = item["snippet"]
    stats   = item.get("statistics", {})

    thumbnail_url = get_best_thumbnail_url(snippet.get("thumbnails", {}))

    api_tags      = snippet.get("tags", []) or []
    scraped_tags  = scrape_ytInitialData_keywords(video_id)
    merged_tags   = _merge_tags(api_tags, scraped_tags)

    if not api_tags and scraped_tags:
        print(f"[YTAPI] API returned 0 tags — using {len(scraped_tags)} scraped keywords")
    elif scraped_tags:
        added = len(merged_tags) - len(api_tags)
        if added > 0:
            print(f"[YTAPI] Merged: {len(api_tags)} API tags + {added} new from ytInitialData = {len(merged_tags)} total")

    return {
        "video_id":        video_id,
        "title":           snippet.get("title", ""),
        "description":     snippet.get("description", "")[:1000],
        "tags":            merged_tags,
        "tags_api":        api_tags,
        "tags_scraped":    scraped_tags,
        "channel":         snippet.get("channelTitle", ""),
        "duration":        item["contentDetails"].get("duration", ""),
        "view_count":      int(stats.get("viewCount", 0)),
        "like_count":      int(stats.get("likeCount", 0)),
        "comment_count":   int(stats.get("commentCount", 0)),
        "thumbnail_url":   thumbnail_url,
    }


def get_thumbnail_url(video_id: str) -> str:
    """Returns best available thumbnail URL for a video without using API quota."""
    direct_urls = [
        f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/default.jpg",
    ]
    for thumb_url in direct_urls:
        try:
            resp = requests.head(thumb_url, timeout=5)
            if resp.status_code == 200 and int(resp.headers.get("content-length", 0)) > 5000:
                return thumb_url
        except Exception:
            continue
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


def search_child_videos(query: str, max_results: int = 50) -> list:
    """Searches YouTube for child-directed videos matching the query."""
    if not API_KEY:
        return []

    url    = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part":       "snippet",
        "q":          query,
        "type":       "video",
        "maxResults": min(max_results, 50),
        "relevanceLanguage": "en",
        "key":        API_KEY,
    }
    try:
        resp  = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [item["id"]["videoId"] for item in items if "videoId" in item.get("id", {})]
    except Exception as e:
        print(f"[SEARCH] Error: {e}")
        return []
