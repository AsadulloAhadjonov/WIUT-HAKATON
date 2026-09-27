"""Skriptlar uchun: repo ildizini import yo'liga qo'shish (python scripts/xxx.py ko'rinishida ishlashi uchun)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src  # noqa: E402,F401  — oflayn rejim

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def list_videos(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(x for x in p.iterdir() if x.suffix.lower() in VIDEO_EXT)
