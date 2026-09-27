"""Official starter-kit submission harness (run_submission.py).

Walks a folder of videos, calls detect_events on each, streams every frame
through RiskEstimator, enforces the 3x wall-clock time budget, drops malformed
or same-class overlapping events, and writes predictions.json.

Usage:
    python run_submission.py --videos samples/ --out predictions_samples.json
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import cv2

import solution

OFFICIAL_CLASSES = [
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
    "fire_smoke",
]


def sanitize_events(raw_events: list, duration: float, video_name: str) -> list[list]:
    """Filter out invalid labels, start >= end, and same-class overlaps (earlier start kept)."""
    valid = []
    allowed = set(getattr(solution, "CLASSES", OFFICIAL_CLASSES)) & set(OFFICIAL_CLASSES)
    if not isinstance(raw_events, list):
        print(f"[DROP] {video_name}: detect_events did not return a list")
        return []

    for item in raw_events:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            print(f"[DROP] {video_name}: malformed event {item!r}")
            continue
        try:
            s = float(item[0])
            e = float(item[1])
            label = str(item[2])
        except (ValueError, TypeError):
            print(f"[DROP] {video_name}: non-numeric timestamps in {item!r}")
            continue

        if label not in allowed:
            print(f"[DROP] {video_name}: label {label!r} outside CLASSES")
            continue
        s = max(0.0, s)
        if duration > 0:
            e = min(duration, e)
        if s >= e:
            print(f"[DROP] {video_name}: start_sec ({s:.3f}) >= end_sec ({e:.3f}) for {label}")
            continue
        valid.append([round(s, 3), round(e, 3), label])

    # Sort by (start_sec, end_sec) and drop overlapping segments of the same class
    valid.sort(key=lambda x: (x[0], x[1], x[2]))
    cleaned = []
    last_end_by_class: dict[str, float] = {}
    for s, e, label in valid:
        prev_end = last_end_by_class.get(label, -1.0)
        if s < prev_end:
            print(f"[DROP] {video_name}: overlapping {label} segment [{s:.3f}, {e:.3f}] (prev ends at {prev_end:.3f})")
            continue
        cleaned.append([s, e, label])
        last_end_by_class[label] = e

    return cleaned


def process_video(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path.name}")
        return {"events": [], "risk": []}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0 or fps > 240:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_frames / fps if n_frames > 0 else 0.0
    cap.release()

    budget_s = 3.0 * duration if duration > 0 else 3600.0
    t_start = time.perf_counter()

    # Part A: Event detection
    try:
        raw_events = solution.detect_events(str(video_path))
    except Exception:
        print(f"[ERROR] Crash inside detect_events on {video_path.name}:")
        traceback.print_exc()
        raw_events = []

    # Part B: Causal accident anticipation
    risk_curve: list[list[float]] = []
    if hasattr(solution, "RiskEstimator"):
        try:
            estimator = solution.RiskEstimator()
            meta = {
                "video_id": video_path.name,
                "fps": fps,
                "width": width,
                "height": height,
                "n_frames": n_frames,
            }
            estimator.reset(meta)

            cap = cv2.VideoCapture(str(video_path))
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                t_sec = round(frame_idx / fps, 4)
                try:
                    score = float(estimator.step(frame, t_sec))
                    score = max(0.0, min(1.0, score))
                except Exception:
                    print(f"[ERROR] Crash inside RiskEstimator.step on {video_path.name} frame {frame_idx}:")
                    traceback.print_exc()
                    score = 0.0
                risk_curve.append([round(t_sec, 3), round(score, 4)])
                frame_idx += 1

                if (time.perf_counter() - t_start) > budget_s and duration > 0:
                    print(f"[TIMEOUT] {video_path.name}: exceeded 3x time budget ({budget_s:.1f}s)")
                    cap.release()
                    return {"events": [], "risk": []}
            cap.release()
            if frame_idx > 0:
                duration = frame_idx / fps
        except Exception:
            print(f"[ERROR] Crash initializing/running RiskEstimator on {video_path.name}:")
            traceback.print_exc()
            risk_curve = []

    elapsed = time.perf_counter() - t_start
    if duration > 0 and elapsed > 3.0 * duration:
        print(f"[TIMEOUT] {video_path.name}: elapsed {elapsed:.1f}s > budget {3.0 * duration:.1f}s; scoring as empty.")
        return {"events": [], "risk": []}

    events = sanitize_events(raw_events, duration, video_path.name)
    print(f"[OK] {video_path.name}: {len(events)} events, {len(risk_curve)} risk frames in {elapsed:.1f}s (video {duration:.1f}s)")
    return {"events": events, "risk": risk_curve}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run traffic CV submission on a directory of videos.")
    parser.add_argument("--videos", required=True, help="Directory containing .mp4 videos or a single .mp4 file")
    parser.add_argument("--out", required=True, help="Output predictions JSON path")
    parser.add_argument("--team", default="Prodigy", help="Team name")
    args = parser.parse_args()

    vpath = Path(args.videos)
    if vpath.is_file():
        videos = [vpath]
    else:
        videos = sorted(p for p in vpath.iterdir() if p.suffix.lower() == ".mp4")

    output = {
        "team": args.team,
        "videos": {},
    }

    for video in videos:
        output["videos"][video.name] = process_video(video)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote predictions for {len(videos)} video(s) to {out_path}")


if __name__ == "__main__":
    main()
