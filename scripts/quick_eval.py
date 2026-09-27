"""Tezkor dev-baholash: har sinf bo'yicha P / R / F1 (vaqt IoU 0.5 va 0.7) + sinflar chalkashligi.

Rasmiy ball — evaluate.py. Bu skript XATOLAR TAHLILI uchun: qaysi sinf yomon, nima bilan chalkashadi.

    python scripts/quick_eval.py dev/ground_truth_dev.json dev/predictions_local.json
    python scripts/quick_eval.py GT PRED --csv dev/eval_by_class.csv
Formatlar:
  GT   = {video: [{class, start, end}]}  yoki  {video: [[start, end, label], ...]}
  PRED = {video: {"events": [[start, end, label], ...]}}  yoki  {video: [[start, end, label], ...]}
"""
import argparse
import csv
import json
from collections import Counter, defaultdict


def tiou(a, b):
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def _normalize_event(e):
    """Har xil formatlarni dict ga aylantiradi."""
    if isinstance(e, dict):
        # {class, start, end} yoki {label, start, end}
        cls = e.get("class") or e.get("label") or e.get("cls", "unknown")
        return {"class": cls, "start": float(e.get("start", e.get("start_s", 0))),
                "end": float(e.get("end", e.get("end_s", 0)))}
    elif isinstance(e, (list, tuple)) and len(e) >= 3:
        # [start, end, label] formatini qo'llab-quvvatlaydi
        return {"class": str(e[2]), "start": float(e[0]), "end": float(e[1])}
    return None


def _events(x):
    """Video yozuvidan hodisalar ro'yxatini olish."""
    if isinstance(x, dict):
        raw = x.get("events", x)
        if isinstance(raw, dict):
            # Har xil ichki dict bo'lishi mumkin
            return []
        return [_normalize_event(e) for e in raw if _normalize_event(e) is not None]
    elif isinstance(x, list):
        return [_normalize_event(e) for e in x if _normalize_event(e) is not None]
    return []


def match(gt, pred, thr):
    """Har sinf ichida ochko'z moslashtirish (eng yuqori IoU birinchi)."""
    stats = defaultdict(Counter)
    for video in sorted(set(gt) | set(pred)):
        G = _events(gt.get(video, []))
        P = _events(pred.get(video, {}))
        for cls in {e["class"] for e in G} | {e["class"] for e in P}:
            g = [e for e in G if e["class"] == cls]
            p = [e for e in P if e["class"] == cls]
            pairs = sorted(((tiou(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)), reverse=True)
            ug, up = set(), set()
            for iou, i, j in pairs:
                if iou < thr or i in ug or j in up:
                    continue
                ug.add(i)
                up.add(j)
            stats[cls]["tp"] += len(ug)
            stats[cls]["fn"] += len(g) - len(ug)
            stats[cls]["fp"] += len(p) - len(up)
    return stats


def confusion(gt, pred, thr=0.3):
    """Bashorat qilingan sinf ≠ haqiqiy sinf, lekin vaqt bo'yicha ustma-ust (IoU ≥ thr)."""
    c = Counter()
    for video in set(gt) & set(pred):
        for p in _events(pred[video]):
            best = max(_events(gt[video]), key=lambda g: tiou(g, p), default=None)
            if best is not None and tiou(best, p) >= thr and best["class"] != p["class"]:
                c[(best["class"], p["class"])] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("pred")
    ap.add_argument("--csv")
    args = ap.parse_args()
    gt = json.load(open(args.gt, encoding="utf-8"))
    pred = json.load(open(args.pred, encoding="utf-8"))

    rows = []
    s5, s7 = match(gt, pred, 0.5), match(gt, pred, 0.7)
    all_cls = sorted(set(s5) | set(s7))
    
    if not all_cls:
        print("Mos keladigan sinflar topilmadi. GT va PRED formatlarini tekshiring.")
        return
    
    print(f"{'sinf':24s} {'GT':>4s} {'PR':>4s} | {'P@.5':>5s} {'R@.5':>5s} {'F1@.5':>5s} | "
          f"{'P@.7':>5s} {'R@.7':>5s} {'F1@.7':>5s}")
    for cls in all_cls:
        r = {"class": cls}
        for tag, s in (("5", s5[cls]), ("7", s7[cls])):
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            p = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            r.update({f"P@.{tag}": p, f"R@.{tag}": rc, f"F1@.{tag}": 2 * p * rc / (p + rc) if p + rc else 0.0})
        r["gt"] = s5[cls]["tp"] + s5[cls]["fn"]
        r["pred"] = s5[cls]["tp"] + s5[cls]["fp"]
        rows.append(r)
        print(f"{cls:24s} {r['gt']:4d} {r['pred']:4d} | {r['P@.5']:5.2f} {r['R@.5']:5.2f} {r['F1@.5']:5.2f} | "
              f"{r['P@.7']:5.2f} {r['R@.7']:5.2f} {r['F1@.7']:5.2f}")
    if rows:
        f1_5 = sum(r["F1@.5"] for r in rows) / len(rows)
        f1_7 = sum(r["F1@.7"] for r in rows) / len(rows)
        print(f"{'O`RTACHA F1':24s} {'':9s} | {'':11s} {f1_5:5.2f} | {'':11s} {f1_7:5.2f}")
        print(f"\nScore A taxminan: {0.7 * f1_7 + 0.3 * f1_5:.4f}")

    conf = confusion(gt, pred)
    if conf:
        print("\nChalkashliklar (haqiqiy → bashorat):")
        for (g, p), n in conf.most_common(15):
            print(f"  {g:22s} → {p:22s} {n}")
    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
