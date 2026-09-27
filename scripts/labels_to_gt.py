"""Annotatsiya jadvali (CSV) → dev ground truth (JSON).

dev/labels.csv ustunlari:  video,class,start_s,end_s,izoh
    sample_01.mp4,near_miss,12.4,14.0,oq mashina va piyoda

    python scripts/labels_to_gt.py dev/labels.csv --out dev/ground_truth_dev.json

Chiqish formati: {video: [{class, start, end}, ...]}
Bu format quick_eval.py va rasmiy evaluate.py bilan mos keladi.
"""
import argparse
import csv
import json
import sys

import _common  # noqa: F401

from solution import CLASSES


def to_gt(rows_by_video: dict) -> dict:
    """Ichki formatdan rasmiy GT formatiga o'tkazish.
    
    Chiqish: {video: [{class, start, end}, ...]}
    """
    return {v: [{"class": r["class"], "start": r["start"], "end": r["end"]} for r in rows]
            for v, rows in sorted(rows_by_video.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="dev/ground_truth_dev.json")
    args = ap.parse_args()

    by_video, errors = {}, []
    with open(args.csv, encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            try:
                cls = r["class"].strip()
                s, e = float(r["start_s"]), float(r["end_s"])
            except (KeyError, ValueError) as ex:
                errors.append(f"{i}-qator: {ex}")
                continue
            if cls not in CLASSES:
                errors.append(f"{i}-qator: noma'lum sinf '{cls}' (CLASSES: {CLASSES})")
            if e <= s:
                errors.append(f"{i}-qator: end <= start ({s} → {e})")
            by_video.setdefault(r["video"].strip(), []).append({"class": cls, "start": s, "end": e})
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    
    gt = to_gt(by_video)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=1, ensure_ascii=False)
    n = sum(len(v) for v in by_video.values())
    print(f"{args.out}: {len(by_video)} video, {n} hodisa")
    print("Sinflar:", sorted({r['class'] for rows in by_video.values() for r in rows}))


if __name__ == "__main__":
    main()
