"""Sahnani (zonalar, polosalar, chiziqlar) kadr ustiga chizish va koordinatalarni olish.

    python scripts/draw_scene.py samples/v1.mp4 --out scene_check.png         # tekshirish
    python scripts/draw_scene.py samples/v1.mp4 --pick                         # nuqtalarni olish
    python scripts/draw_scene.py samples/v1.mp4 --frame 500 --scene configs/scene.yaml

--pick rejimi: oynada chap tugma — nuqta qo'shish, "n" — keyingi poligon, "q" — chiqish.
Nuqtalar scene.yaml ga ko'chirish uchun tayyor formatda terminalga chiqadi.
(Ekran kerak; serverda --out bilan ishlating.)
"""
import argparse

import _common  # noqa: F401
import cv2

from src.scene import Scene, draw
from src.utils import load_yaml


def read_frame(video, idx):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Kadrni o'qib bo'lmadi: {video} #{idx}")
    return frame


def pick(frame):
    polys, cur = [], []
    win = "pick: chap tugma=nuqta, n=keyingi, q=chiqish"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            cur.append([x, y])
            print(f"  [{x}, {y}]")

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    print(f"image_size: [{frame.shape[1]}, {frame.shape[0]}]")
    while True:
        view = frame.copy()
        for p in polys + [cur]:
            for i, pt in enumerate(p):
                cv2.circle(view, tuple(pt), 4, (0, 255, 255), -1)
                if i:
                    cv2.line(view, tuple(p[i - 1]), tuple(pt), (0, 255, 255), 2)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("n"), ord("q")) and cur:
            polys.append(cur)
            print(f"poligon_{len(polys)}: {cur}")
            cur = []
        if k == ord("q"):
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--scene", default="configs/scene.yaml")
    ap.add_argument("--out", default="scene_check.png")
    ap.add_argument("--pick", action="store_true")
    args = ap.parse_args()
    frame = read_frame(args.video, args.frame)
    if args.pick:
        pick(frame)
        return
    scene = Scene(load_yaml(args.scene), (frame.shape[1], frame.shape[0]))
    cv2.imwrite(args.out, draw(scene, frame))
    print(f"yozildi: {args.out}")


if __name__ == "__main__":
    main()
