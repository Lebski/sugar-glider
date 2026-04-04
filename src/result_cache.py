"""
Client-side persistent cache for analysis results.

Results are stored under outputs/client_cache/<video_hash>/:
  stats.json      — all numeric stats (engagement_over_time as list)
  preds_mean.npy  — raw vertex activations needed for delta rendering
  brain.png       — rendered cortical surface image

This means results survive browser refreshes and new Streamlit sessions.
If the user compares X+Y then Y+Z, Y is loaded from disk — no API call.
"""

import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path("outputs/client_cache")


def _dir(video_hash: str) -> Path:
    return CACHE_DIR / video_hash


def exists(video_hash: str) -> bool:
    d = _dir(video_hash)
    return (d / "stats.json").exists() and (d / "preds_mean.npy").exists()


def load(video_hash: str) -> dict | None:
    """Return cached result dict, or None if not cached."""
    d = _dir(video_hash)
    if not exists(video_hash):
        return None

    stats = json.loads((d / "stats.json").read_text())
    # Restore numpy arrays
    stats["engagement_over_time"] = np.array(stats["engagement_over_time"])
    stats["preds_mean"] = np.load(d / "preds_mean.npy")

    return {
        "stats": stats,
        "brain_png": str(d / "brain.png"),
    }


def save(video_hash: str, stats: dict, brain_png_bytes: bytes) -> None:
    """Persist an analysis result to disk."""
    d = _dir(video_hash)
    d.mkdir(parents=True, exist_ok=True)

    # Separate out the non-JSON-serialisable numpy arrays before writing
    preds_mean = stats.pop("preds_mean")
    engagement = stats["engagement_over_time"]

    serialisable = {
        **stats,
        "engagement_over_time": engagement.tolist()
        if hasattr(engagement, "tolist")
        else engagement,
    }
    (d / "stats.json").write_text(json.dumps(serialisable))
    np.save(d / "preds_mean.npy", preds_mean)
    (d / "brain.png").write_bytes(brain_png_bytes)

    # Put preds_mean back so the caller's dict is unchanged
    stats["preds_mean"] = preds_mean
