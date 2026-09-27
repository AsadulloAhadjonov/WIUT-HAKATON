"""Mahalliy ishga tushirish (DEV uchun): papkadagi videolar → predictions_local.json + vaqt o'lchovi.

Rasmiy topshiriq run_submission.py orqali ishlaydi; bu skript uning o'rnini BOSMAYDI,
faqat starter kit bo'lmagan joyda tez tekshirish va vaqtni o'lchash uchun.

    python scripts/run_local.py samples/ --out dev/predictions_local.json
    python scripts/run_local.py samples/ --no-risk      # faqat Part A
"""
import argparse
import json
import time

import _common  # noqa: F401
import cv2

import solution


def run_risk(video, fps):
    """Part B: kadr ketma-ketligi bo'yicha xavf qiymatlarini hisoblash."""
    est = solution.RiskEstimator(fps=fps)
    est.reset({"fps": fps})
    cap = cv2.VideoCapture(str(video))
    out, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(round(est.step(frame, i / fps), 4))
        i += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos")
    ap.add_argument("--out", default="dev/predictions_local.json")
    ap.add_argument("--no-risk", action="store_true")
    args = ap.parse_args()

    result, timing = {}, {}
    for v in _common.list_videos(args.videos):
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        t0 = time.perf_counter()
        events = solution.detect_events(str(v))
        ta = time.perf_counter() - t0
        
        risk = [] if args.no_risk else run_risk(v, fps)
        tb = time.perf_counter() - t0 - ta
        
        # Format: events = [[start, end, label], ...], risk = [float, ...]
        result[v.name] = {"events": events, "risk": risk}
        
        dur = n / fps if fps else 0
        timing[v.name] = {
            "video_s": round(dur, 1),
            "part_a_s": round(ta, 1),
            "part_b_s": round(tb, 1),
            "x_realtime": round((ta + tb) / dur, 2) if dur else None,
            "n_events": len(events),
        }
        print(f"{v.name}: {len(events)} hodisa | A {ta:.1f}s, B {tb:.1f}s, video {dur:.1f}s")
        
        # Topilgan hodisalarni ko'rsatish
        for e in events:
            if isinstance(e, (list, tuple)) and len(e) >= 3:
                print(f"  [{e[0]:.1f}s - {e[1]:.1f}s] {e[2]}")
            elif isinstance(e, dict):
                print(f"  [{e.get('start', 0):.1f}s - {e.get('end', 0):.1f}s] {e.get('class', '?')}")

    _common.ROOT.joinpath(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(_common.ROOT / args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"\nNatijalar: {_common.ROOT / args.out}")
    print(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
