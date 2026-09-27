"""Ikki marta ishga tushirib natijalarni solishtiradi (kesh o'chirilgan holda).

    python scripts/check_determinism.py samples/
Chiqish kodi: 0 — bir xil, 1 — farq bor.
"""
import argparse
import json
import sys

import _common  # noqa: F401

import solution
from run_local import run_risk


def once(videos, with_risk):
    solution._cfg()["use_cache"] = False       # kesh natijani "bir xil" qilib ko'rsatmasin
    res = {}
    for v in videos:
        res[v.name] = {"events": solution.detect_events(str(v))}
        if with_risk:
            import cv2
            cap = cv2.VideoCapture(str(v))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            res[v.name]["risk"] = run_risk(v, fps)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos")
    ap.add_argument("--no-risk", action="store_true")
    args = ap.parse_args()
    vids = _common.list_videos(args.videos)
    a = once(vids, not args.no_risk)
    b = once(vids, not args.no_risk)
    if json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True):
        print(f"OK: {len(vids)} video, ikki ishga tushirish bir xil.")
        return
    for k in a:
        if a[k] != b[k]:
            print(f"FARQ: {k}")
    sys.exit(1)


if __name__ == "__main__":
    main()
