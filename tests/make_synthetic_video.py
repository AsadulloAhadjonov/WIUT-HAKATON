"""Sintetik test video yasaydi (faqat pipeline'ni tekshirish uchun, trening uchun EMAS).

Ultralytics paketi ichidagi bus.jpg dan avtobus va piyoda kesib olinadi va
kulrang yo'l ustida harakatlantiriladi. Internet kerak emas.

    python tests/make_synthetic_video.py --out samples/synthetic.mp4
"""
import argparse
import os

import cv2
import numpy as np


def _asset(name: str) -> str:
    import ultralytics

    return os.path.join(os.path.dirname(ultralytics.__file__), "assets", name)


def _paste(canvas, sprite, x, y):
    """sprite ni (x, y) chap-yuqori burchakka joylaydi, chegaradan chiqqan qismini kesadi."""
    h, w = sprite.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0 + w, W), min(y0 + h, H)
    if x1 >= x2 or y1 >= y2:
        return
    canvas[y1:y2, x1:x2] = sprite[y1 - y0:y2 - y0, x1 - x0:x2 - x0]


def make_background(W=1280, H=720):
    bg = np.full((H, W, 3), (70, 110, 60), np.uint8)          # o't
    cv2.rectangle(bg, (0, 220), (W, 560), (90, 90, 90), -1)   # yo'l
    for x in range(0, W, 80):                                  # o'rta chiziq
        cv2.line(bg, (x, 390), (x + 40, 390), (230, 230, 230), 4)
    for x in range(560, 720, 24):                              # piyoda o'tish joyi
        cv2.rectangle(bg, (x, 225), (x + 12, 555), (220, 220, 220), -1)
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples/synthetic.mp4")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--seconds", type=float, default=14.0)
    args = ap.parse_args()

    img = cv2.imread(_asset("bus.jpg"))
    bus = cv2.resize(img[228:600, 14:808], (300, 140))
    bus_rev = cv2.flip(bus, 1)
    person = cv2.resize(img[401:901, 49:244], (56, 140))

    W, H = 1280, 720
    bg = make_background(W, H)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    n = int(args.seconds * args.fps)
    for i in range(n):
        t = i / args.fps
        f = bg.copy()
        # 1-avtobus: pastki polosada chapdan o'ngga; 3.0-7.0 s oralig'ida to'xtaydi
        if t < 3.0:
            bx = -300 + 180 * t
        elif t < 7.0:
            bx = -300 + 180 * 3.0
        else:
            bx = -300 + 180 * 3.0 + 180 * (t - 7.0)
        _paste(f, bus, bx, 400)
        # 2-avtobus: yuqori polosada o'ngdan chapga, 8 s dan keyin
        if t >= 8.0:
            _paste(f, bus_rev, W - 150 * (t - 8.0), 240)
        # piyoda: 1-6 s oralig'ida o'tish joyidan yuqoridan pastga
        if 1.0 <= t <= 6.0:
            _paste(f, person, 610, 150 + 90 * (t - 1.0))
        vw.write(f)
    vw.release()
    print(f"yozildi: {args.out} ({n} kadr, {args.fps} fps)")


if __name__ == "__main__":
    main()
