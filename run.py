import os
import sqlite3

# ── Windows-only: set ffmpeg path for local dev ───────────────────────────────
if os.name == "nt":
    os.environ["PATH"] = r"C:\ffmpeg\bin" + os.pathsep + os.environ.get("PATH", "")

# ── Database setup ─────────────────────────────────────────────────────────────
DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "database", "childfocus.db")
)

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        video_id        TEXT PRIMARY KEY,
        label           TEXT,
        final_score     REAL,
        last_checked    TEXT,
        checked_by      TEXT,
        video_title     TEXT,
        nb_score        REAL,
        heuristic_score REAL,
        runtime_seconds REAL
    )
""")
conn.execute("""
    CREATE TABLE IF NOT EXISTS segments (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id        TEXT,
        segment_id      TEXT,
        offset_seconds  REAL,
        length_seconds  REAL,
        fcr             REAL,
        csv             REAL,
        att             REAL,
        score           REAL
    )
""")
conn.commit()
conn.close()

# ── Create Flask app ───────────────────────────────────────────────────────────
from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1",
            host="0.0.0.0",
            port=port)