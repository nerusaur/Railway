"""
ChildFocus - Shared Text Builder
backend/app/modules/text_builder.py

Single source of truth for build_nb_text().

⚠  THIS FILE IS THE AUTHORITATIVE DEFINITION.
   preprocess.py (training) imports or must stay BYTE-FOR-BYTE IDENTICAL to it.
   naive_bayes.py (inference) imports directly from here.

   Training formula = Inference formula = consistent NB scoring.

Any change to build_nb_text() or STOP_WORDS requires:
  1. Update this file.
  2. Update preprocess.py to match (or import from here).
  3. Retrain the model: python preprocess.py && python train_nb.py.
"""

import re

# ── Stop words (must match preprocess.py exactly) ─────────────────────────────
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "was", "are",
    "be", "as", "so", "we", "he", "she", "they", "you", "i", "my", "your",
    "his", "her", "its", "our", "their", "what", "which", "who", "will",
    "would", "could", "should", "has", "have", "had", "do", "does", "did",
    "not", "no", "if", "then", "than", "when", "where", "how", "all",
    "each", "more", "also", "just", "can", "up", "out", "about", "into",
    "too", "very", "s", "t", "re", "ve", "ll", "d",
}


def build_nb_text(title: str = "", tags=None, description: str = "") -> str:
    """
    Canonical text representation for NB classification.

    Formula:
      - title repeated 3× (high signal density)
      - tags joined with spaces (medium signal)
      - description truncated to 300 chars (low signal)
    Then: lowercase → strip URLs → strip non-alpha → remove stop words

    Args:
        title:       Video title string.
        tags:        List of tag strings, or a raw space/comma-separated string.
        description: Video description (truncated to first 300 chars).

    Returns:
        Cleaned, tokenised string ready for TF-IDF vectorisation.
    """
    title_part = f"{title} " * 3
    tags_str   = " ".join(str(t) for t in tags) if isinstance(tags, list) else (tags or "")
    desc_part  = (description or "")[:300]

    raw = f"{title_part}{tags_str} {desc_part}".lower()

    # Remove URLs
    raw = re.sub(r"https?://\S+|www\.\S+", " ", raw)
    # Remove non-alpha characters (no digits — consistent with preprocess.py)
    raw = re.sub(r"[^a-z\s]", " ", raw)
    # Remove stop words and single-char tokens
    tokens = [t for t in raw.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)
