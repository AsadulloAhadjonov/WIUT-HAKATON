"""Hakaton yechimi — starter kit interfeysi.

!!! 1-KUN, 1-SOAT: starter kitdagi solution.py ni oching va quyidagilarni AYNAN ko'chiring:
!!!   1) CLASSES ro'yxati (14 ta nom) — pastdagi namunani almashtiring;
!!!   2) detect_events va RiskEstimator imzolari (argumentlar, qaytish turi).
!!! Mantiq src/ ichida; bu fayl faqat yupqa "adapter".
!!! run_submission.py va evaluate.py O'ZGARTIRILMAYDI.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
import numpy as np

# run_submission.py solution.py ni boshqa papkadan yoki importlib bilan yuklasa ham src/ topilsin
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src  # noqa: F401  — oflayn rejimni yoqadi (ultralytics importidan oldin)
from src.io_format import to_events
from src.pipeline import detect_segments
from src.risk import OnlineRisk
from src.utils import load_config

# !!! Starter kitdagi ro'yxat bilan ALMASHTIRING (tartib va yozilishi bir xil bo'lsin).
CLASSES = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke"
]

_CFG = None


def _cfg() -> dict:
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


# Xato bo'lsa butun topshiriq yiqilmasin: log yoziladi va bo'sh natija qaytadi.
# Dev paytida xatolarni ko'rish uchun: TRAFFIC_STRICT=1
_STRICT = os.environ.get("TRAFFIC_STRICT", "0") == "1"


def detect_events(video_path: str) -> list[list]:
    """Part A: butun video bo'yicha hodisalar ro'yxati (kelajak kadrlari ishlatilishi mumkin)."""
    try:
        segs, info = detect_segments(video_path, _cfg())
        return to_events(segs, info["fps"], CLASSES)
    except Exception:
        if _STRICT:
            raise
        traceback.print_exc()
        return []


class RiskEstimator:
    """Part B (optional). Causal: step() sees frames in order and nothing else."""

    def __init__(self, fps: float = 25.0, **kwargs):
        self._impl = OnlineRisk(fps=fps, cfg=_cfg())
        
    def reset(self, meta: dict) -> None:
        self._impl = OnlineRisk(fps=meta.get("fps", 25.0), cfg=_cfg())

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        try:
            return self._impl.step(frame, t_sec)
        except Exception:
            if _STRICT:
                raise
            traceback.print_exc()
            return 0.0
