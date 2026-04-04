"""
Ad library manifest. Scans the ads/ folder for .mp4 files.
Falls back to examples/ if ads/ is empty.
"""

from pathlib import Path

ADS_DIR = Path("ads")
EXAMPLES_DIR = Path("examples")


def get_library_entries() -> list[dict]:
    """Return list of {"name": str, "path": str} dicts for all available ads."""
    entries = []

    mp4_files = sorted(ADS_DIR.glob("*.mp4")) if ADS_DIR.exists() else []

    if not mp4_files:
        # Fall back to examples/ so there's always something to show
        mp4_files = sorted(EXAMPLES_DIR.glob("*.mp4")) if EXAMPLES_DIR.exists() else []

    for path in mp4_files:
        entries.append({"name": path.stem, "path": str(path)})

    return entries


def get_library_names() -> list[str]:
    return [e["name"] for e in get_library_entries()]


def get_library_path(name: str) -> str:
    for entry in get_library_entries():
        if entry["name"] == name:
            return entry["path"]
    raise ValueError(f"Ad '{name}' not found in library")
