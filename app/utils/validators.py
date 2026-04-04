"""
ChildFocus - Input Validators
backend/app/utils/validators.py
"""

import re


YOUTUBE_URL_PATTERNS = [
    r"youtube\.com/watch\?v=[\w-]{11}",
    r"youtu\.be/[\w-]{11}",
    r"youtube\.com/shorts/[\w-]{11}",
    r"^[\w-]{11}$",   # raw video ID
]


def validate_video_url(url: str) -> str:
    """
    Validate a YouTube URL or video ID.
    Returns an error string if invalid, or empty string if valid.
    """
    if not url or not isinstance(url, str):
        return "video_url must be a non-empty string"

    url = url.strip()
    for pattern in YOUTUBE_URL_PATTERNS:
        if re.search(pattern, url):
            return ""   # valid

    return f"Invalid YouTube URL or video ID: '{url}'"