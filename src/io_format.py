"""Ichki segment formatini starter kit formatiga o'tkazish — BITTA joy.

!!! 1-KUN, 1-SOAT: examples/predictions.json ni oching va shu fayldagi
!!! `to_event()` ni aynan o'sha formatga moslang. Boshqa hech qayerni o'zgartirish shart emas.

Hozirgi taxminiy format: {"class": "...", "start": 12.3, "end": 15.0, "score": 0.9}
(vaqt — soniyada). Agar kit kadr raqami kutsa — `start_frame`/`end_frame` ga o'tkazing.
"""
from __future__ import annotations


def to_event(seg: dict, fps: float) -> list:
    return [
        round(float(seg["start"]), 3),
        round(float(seg["end"]), 3),
        seg["cls"]
    ]


def to_events(segs: list[dict], fps: float, allowed: list[str] | None = None) -> list[list]:
    """allowed — CLASSES ro'yxati; unda yo'q sinflar tashlanadi (run_submission ham tashlaydi)."""
    out = [to_event(s, fps) for s in segs]
    if allowed:
        out = [e for e in out if e[2] in allowed]
    return out
