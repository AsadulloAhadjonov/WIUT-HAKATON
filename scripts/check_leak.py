"""RiskEstimator kelajakdan ma'lumot olmayotganini tekshiradi.

G'oya: videoning birinchi K kadrini bergandagi xavf qiymatlari, butun videoni bergandagi
birinchi K qiymat bilan AYNAN bir xil bo'lishi kerak. Farq bo'lsa — step() kelajakni ko'ryapti
(yoki holat videolar/obyektlar orasida aralashyapti).

    python scripts/check_leak.py samples/video.mp4 --frames 300
Chiqish kodi: 0 — o'tdi, 1 — oqish bor.
"""
import argparse
import sys

import _common  # noqa: F401
import cv2
import numpy as np

import solution


def risks(video, n_max, fps):
    est = solution.RiskEstimator(fps=fps)
    cap = cv2.VideoCapture(str(video))
    out = []
    while len(out) < n_max:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(est.step(frame, len(out) / fps))
    cap.release()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=300, help="to'liq ketma-ketlik uzunligi")
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    full = risks(args.video, args.frames, fps)
    half = risks(args.video, len(full) // 2, fps)
    diff = float(np.abs(full[:len(half)] - half).max()) if len(half) else 0.0
    print(f"kadrlar: to'liq={len(full)}, yarmi={len(half)}, maksimal farq={diff:.2e}, "
          f"xavf oralig'i=[{full.min():.3f}, {full.max():.3f}]")
    if diff > 1e-6:
        print("XATO: prefiks natijasi farq qiladi — kelajakdan ma'lumot oqishi mumkin!")
        sys.exit(1)
    print("OK: oqish yo'q.")


if __name__ == "__main__":
    main()
