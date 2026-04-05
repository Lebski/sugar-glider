"""
Persistent history of analyzed ads.

Stored in outputs/history.json — a list of entries, newest first.
Each entry: hash, name, description, video_path, timestamp, and key scores.
Thumbnails (first frame JPEGs) live alongside the client_cache.
Uploaded videos are copied to outputs/history_videos/ for permanence.
"""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

HISTORY_FILE = Path("outputs/history.json")
CACHE_DIR = Path("outputs/client_cache")
VIDEOS_DIR = Path("outputs/history_videos")
ADS_DIR = Path("ads").resolve()


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _thumbnail_path(video_hash: str) -> Path:
    return CACHE_DIR / video_hash / "thumbnail.jpg"


def _permanent_video_path(video_path: str, video_hash: str) -> str:
    """
    Return a path to a permanently stored copy of the video.
    Library ads (ads/) are already permanent and are not copied.
    Uploaded videos are copied to outputs/history_videos/.
    """
    src = Path(video_path).resolve()
    try:
        src.relative_to(ADS_DIR)
        return video_path  # library video — already permanent
    except ValueError:
        pass

    target = VIDEOS_DIR / f"{video_hash}.mp4"
    if not target.exists():
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_path, target)
    return str(target)


def _load_raw() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save(index: list[dict]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(index, indent=2))


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def extract_thumbnail(video_path: str, video_hash: str) -> str:
    """Extract the first video frame as JPEG using ffmpeg. Returns the path."""
    out = _thumbnail_path(video_hash)
    if out.exists():
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vframes", "1", "-ss", "0",
            "-q:v", "3", str(out),
        ],
        capture_output=True,
    )
    return str(out)


def load_index() -> list[dict]:
    """Return all history entries, newest first."""
    return _load_raw()


def get_entry(video_hash: str) -> dict | None:
    """Return a single history entry by hash, or None."""
    return next((e for e in _load_raw() if e["hash"] == video_hash), None)


def add_or_update(video_hash: str, name: str, video_path: str, stats: dict) -> None:
    """
    Upsert a history entry.
    - New entries are prepended (newest first).
    - Existing entries: scores and video_path are updated; name/description are preserved.
    """
    permanent_path = _permanent_video_path(video_path, video_hash)
    extract_thumbnail(permanent_path, video_hash)

    scores = {
        "brain_score": float(stats.get("overall_score", 0)),
        "impact_score": float(stats.get("impact_score") or 0),
        "early_attention_score": float(stats.get("early_attention_score") or 0),
        "duration_trs": int(len(stats.get("engagement_over_time", []))),
    }

    index = _load_raw()
    existing = next((e for e in index if e["hash"] == video_hash), None)

    if existing is None:
        index.insert(0, {
            "hash": video_hash,
            "name": name,
            "description": "",
            "video_path": permanent_path,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **scores,
        })
    else:
        existing["video_path"] = permanent_path
        existing.update(scores)

    _save(index)


def update_meta(video_hash: str, name: str | None = None, description: str | None = None) -> None:
    """Update the editable name and/or description of an entry."""
    index = _load_raw()
    for entry in index:
        if entry["hash"] == video_hash:
            if name is not None:
                entry["name"] = name
            if description is not None:
                entry["description"] = description
            break
    _save(index)


def thumbnail_path(video_hash: str) -> Path:
    return _thumbnail_path(video_hash)
