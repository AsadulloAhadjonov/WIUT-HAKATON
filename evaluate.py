"""Official evaluation and format validation script (evaluate.py).

Implements the exact metrics from WIUT Hackathon 2026 CV Track specification:
  - Format validation (--validate-only)
  - Part A (Score_A): Macro F1 averaged over temporal IoU thresholds {0.3, 0.5, 0.7}
    across active classes C, plus diagnostic Micro F1, Class-agnostic F1, and per-class table.
  - Part B (Score_B): Accident anticipation (H = 5s, W = 10s, theta = 0.5)
    Score_B = 0.4 * AP + 0.4 * F1_alarm + 0.2 * (mTTA / W)
  - Model score: M = 0.7 * Score_A + 0.3 * Score_B

Usage:
    python evaluate.py --pred predictions.json --validate-only
    python evaluate.py --pred predictions.json --gt ground_truth.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

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
    "fire_smoke",
]
ALLOWED_CLASSES = set(CLASSES)
THRESHOLDS = (0.3, 0.5, 0.7)

# Part B constants
H_HORIZON = 5.0
W_WINDOW = 10.0
THETA = 0.5
ALARM_MERGE_GAP = 2.0


def temporal_iou(seg1: tuple[float, float], seg2: tuple[float, float]) -> float:
    inter = max(0.0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))
    union = max(seg1[1], seg2[1]) - min(seg1[0], seg2[0])
    return inter / union if union > 0 else 0.0


def validate_predictions(pred_data: dict) -> list[str]:
    errors = []
    if not isinstance(pred_data, dict):
        return ["Root of predictions JSON must be an object."]
    if "videos" not in pred_data or not isinstance(pred_data["videos"], dict):
        return ["Predictions JSON must contain a 'videos' object mapping video filenames to predictions."]

    for vname, vobj in pred_data["videos"].items():
        if "/" in vname or "\\" in vname:
            errors.append(f"{vname}: video key must be a file name, not a path.")
        if not isinstance(vobj, dict):
            errors.append(f"{vname}: value must be an object with 'events' and 'risk'.")
            continue
        events = vobj.get("events")
        if not isinstance(events, list):
            errors.append(f"{vname}: 'events' must be a list of [start_sec, end_sec, label].")
            continue

        by_class: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for idx, ev in enumerate(events):
            if not isinstance(ev, list) or len(ev) != 3:
                errors.append(f"{vname} event #{idx}: must be a 3-element list [start_sec, end_sec, label], got {ev!r}")
                continue
            s, e, label = ev
            if not isinstance(s, (int, float)) or not isinstance(e, (int, float)):
                errors.append(f"{vname} event #{idx}: start_sec and end_sec must be numbers.")
                continue
            if label not in ALLOWED_CLASSES:
                errors.append(f"{vname} event #{idx}: unknown class label {label!r}.")
            if s < 0 or s >= e:
                errors.append(f"{vname} event #{idx}: invalid interval [{s}, {e}] (must have 0 <= start < end).")
            by_class[label].append((float(s), float(e)))

        for label, segs in by_class.items():
            segs_sorted = sorted(segs)
            for i in range(1, len(segs_sorted)):
                if segs_sorted[i][0] < segs_sorted[i - 1][1]:
                    errors.append(
                        f"{vname}: overlapping segments of class {label!r}: "
                        f"{segs_sorted[i - 1]} and {segs_sorted[i]}"
                    )

        risk = vobj.get("risk", [])
        if not isinstance(risk, list):
            errors.append(f"{vname}: 'risk' must be a list of [t_sec, score] pairs.")
        else:
            for idx, item in enumerate(risk):
                if not isinstance(item, list) or len(item) != 2:
                    errors.append(f"{vname} risk #{idx}: must be [t_sec, score], got {item!r}")
                    break
                t_sec, score = item
                if not isinstance(t_sec, (int, float)) or not isinstance(score, (int, float)):
                    errors.append(f"{vname} risk #{idx}: t_sec and score must be numbers.")
                    break
                if not (0.0 <= float(score) <= 1.0):
                    errors.append(f"{vname} risk #{idx}: score {score} outside [0, 1].")
                    break

    return errors


def greedy_match_counts(
    gt_segs: list[tuple[float, float]],
    pred_segs: list[tuple[float, float]],
    tau: float,
) -> tuple[int, int, int]:
    pairs = []
    for gi, g in enumerate(gt_segs):
        for pi, p in enumerate(pred_segs):
            iou = temporal_iou(g, p)
            if iou >= tau:
                pairs.append((iou, gi, pi))
    pairs.sort(key=lambda x: x[0], reverse=True)
    matched_g, matched_p = set(), set()
    for _, gi, pi in pairs:
        if gi not in matched_g and pi not in matched_p:
            matched_g.add(gi)
            matched_p.add(pi)
    tp = len(matched_g)
    fp = len(pred_segs) - len(matched_p)
    fn = len(gt_segs) - len(matched_g)
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def evaluate_part_a(gt_videos: dict, pred_videos: dict) -> tuple[float, dict]:
    active_classes = set()
    for vname, gobj in gt_videos.items():
        for ev in gobj.get("events", []):
            active_classes.add(ev[2])
    for vname, pobj in pred_videos.items():
        for ev in pobj.get("events", []):
            active_classes.add(ev[2])

    active_classes_sorted = [c for c in CLASSES if c in active_classes]
    per_class_scores = {}

    pooled_micro = {tau: [0, 0, 0] for tau in THRESHOLDS}
    pooled_agnostic = {tau: [0, 0, 0] for tau in THRESHOLDS}

    for vname in set(gt_videos) | set(pred_videos):
        g_all = [(float(e[0]), float(e[1])) for e in gt_videos.get(vname, {}).get("events", [])]
        p_all = [(float(e[0]), float(e[1])) for e in pred_videos.get(vname, {}).get("events", [])]
        for tau in THRESHOLDS:
            tp, fp, fn = greedy_match_counts(g_all, p_all, tau)
            pooled_agnostic[tau][0] += tp
            pooled_agnostic[tau][1] += fp
            pooled_agnostic[tau][2] += fn

    for cls in active_classes_sorted:
        f1_by_tau = {}
        for tau in THRESHOLDS:
            tp_total = fp_total = fn_total = 0
            for vname in set(gt_videos) | set(pred_videos):
                g_segs = [(float(e[0]), float(e[1])) for e in gt_videos.get(vname, {}).get("events", []) if e[2] == cls]
                p_segs = [(float(e[0]), float(e[1])) for e in pred_videos.get(vname, {}).get("events", []) if e[2] == cls]
                tp, fp, fn = greedy_match_counts(g_segs, p_segs, tau)
                tp_total += tp
                fp_total += fp
                fn_total += fn
            pooled_micro[tau][0] += tp_total
            pooled_micro[tau][1] += fp_total
            pooled_micro[tau][2] += fn_total
            _, _, f1_tau = f1_from_counts(tp_total, fp_total, fn_total)
            f1_by_tau[tau] = f1_tau
        per_class_scores[cls] = {
            "f1_0.3": f1_by_tau[0.3],
            "f1_0.5": f1_by_tau[0.5],
            "f1_0.7": f1_by_tau[0.7],
            "mean_f1": sum(f1_by_tau.values()) / 3.0,
        }

    score_a = (
        sum(v["mean_f1"] for v in per_class_scores.values()) / len(active_classes_sorted)
        if active_classes_sorted
        else 0.0
    )
    micro_f1 = sum(f1_from_counts(*pooled_micro[tau])[2] for tau in THRESHOLDS) / 3.0
    agnostic_f1 = sum(f1_from_counts(*pooled_agnostic[tau])[2] for tau in THRESHOLDS) / 3.0

    return score_a, {
        "active_classes": active_classes_sorted,
        "per_class": per_class_scores,
        "micro_f1": micro_f1,
        "agnostic_f1": agnostic_f1,
    }


def compute_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Pooled frame-level Average Precision, chance-normalized."""
    if len(labels) == 0:
        return 0.0
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return 0.0
    r = n_pos / len(labels)
    if float(np.max(scores) - np.min(scores)) < 1e-12:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp_cum = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, len(sorted_labels) + 1)
    precisions = tp_cum / ranks
    ap_raw = float(precisions[sorted_labels == 1].sum() / n_pos)
    return max(0.0, (ap_raw - r) / (1.0 - r))


def extract_alarms(risk_pairs: list[list[float]], ignored_intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Maximal runs of frames with score >= THETA; runs separated by < 2s are merged.
    Alarms starting inside ignored frames are discarded.
    """
    runs = []
    in_run = False
    run_start = run_end = 0.0
    for t_sec, score in risk_pairs:
        t = float(t_sec)
        s = float(score)
        if s >= THETA:
            if not in_run:
                in_run = True
                run_start = t
            run_end = t
        else:
            if in_run:
                runs.append((run_start, run_end))
                in_run = False
    if in_run:
        runs.append((run_start, run_end))

    if not runs:
        return []

    merged = [runs[0]]
    for s, e in runs[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e < ALARM_MERGE_GAP:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))

    def is_ignored(t: float) -> bool:
        return any(a <= t <= b for a, b in ignored_intervals)

    return [(s, e) for s, e in merged if not is_ignored(s)]


def evaluate_part_b(gt_videos: dict, pred_videos: dict) -> tuple[float, dict]:
    all_labels = []
    all_scores = []

    total_alarms = 0
    matched_alarms = 0
    total_accidents = 0
    ttas = []

    for vname, gobj in sorted(gt_videos.items()):
        events = gobj.get("events", [])
        accidents = sorted((float(e[0]), float(e[1])) for e in events if e[2] == "accident")
        near_misses = [(float(e[0]), float(e[1])) for e in events if e[2] == "near_miss"]

        ignored = []
        for s, e in accidents:
            ignored.append((s, e))
        for s, e in near_misses:
            ignored.append((max(0.0, s - H_HORIZON), e))

        risk_pairs = pred_videos.get(vname, {}).get("risk", [])
        if not risk_pairs:
            fps = float(gobj.get("fps", 25.0))
            dur = float(gobj.get("duration", 0.0))
            n_frames = int(round(dur * fps))
            risk_pairs = [[i / fps, 0.0] for i in range(n_frames)]

        for t_sec, score in risk_pairs:
            t = float(t_sec)
            if any(a <= t <= b for a, b in ignored):
                continue
            is_pos = any((s - H_HORIZON) <= t < s for s, _ in accidents)
            all_labels.append(1 if is_pos else 0)
            all_scores.append(float(score))

        alarms = extract_alarms(risk_pairs, ignored)
        total_alarms += len(alarms)
        total_accidents += len(accidents)

        used_accidents = set()
        for a_start, _ in alarms:
            for idx_acc, (acc_s, _) in enumerate(accidents):
                if idx_acc in used_accidents:
                    continue
                if (acc_s - W_WINDOW) <= a_start < acc_s:
                    used_accidents.add(idx_acc)
                    matched_alarms += 1
                    ttas.append(acc_s - a_start)
                    break

    unmatched_accidents = total_accidents - len(ttas)
    ttas.extend([0.0] * unmatched_accidents)

    ap = compute_average_precision(np.asarray(all_labels, dtype=np.int32), np.asarray(all_scores, dtype=np.float64))
    prec_alarm = matched_alarms / total_alarms if total_alarms > 0 else 0.0
    rec_alarm = len(used_accidents) / total_accidents if total_accidents > 0 else 0.0
    f1_alarm = (
        2 * prec_alarm * rec_alarm / (prec_alarm + rec_alarm)
        if (prec_alarm + rec_alarm) > 0
        else 0.0
    )
    mtta = float(np.mean(ttas)) if ttas else 0.0
    score_b = 0.4 * ap + 0.4 * f1_alarm + 0.2 * (mtta / W_WINDOW)

    return score_b, {
        "AP": ap,
        "F1_alarm": f1_alarm,
        "precision_alarm": prec_alarm,
        "recall_alarm": rec_alarm,
        "mTTA": mtta,
        "total_alarms": total_alarms,
        "total_accidents": total_accidents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and evaluate traffic event predictions.")
    parser.add_argument("--pred", required=True, help="Path to predictions.json")
    parser.add_argument("--gt", help="Path to ground_truth.json")
    parser.add_argument("--validate-only", action="store_true", help="Only validate predictions format")
    args = parser.parse_args()

    with open(args.pred, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    errors = validate_predictions(pred_data)
    if errors:
        print("VALIDATION FAILED:")
        for err in errors[:25]:
            print(f"  - {err}")
        sys.exit(1)

    print(f"[OK] Format validation passed for {args.pred} ({len(pred_data.get('videos', {}))} videos).")
    if args.validate_only or not args.gt:
        return

    with open(args.gt, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)
    gt_videos = gt_raw.get("videos", gt_raw)
    pred_videos = pred_data["videos"]

    missing = [v for v in gt_videos if v not in pred_videos]
    if missing:
        print(f"WARNING: Missing videos in predictions (scored as empty): {missing}")

    score_a, diag_a = evaluate_part_a(gt_videos, pred_videos)
    score_b, diag_b = evaluate_part_b(gt_videos, pred_videos)
    model_score = 0.7 * score_a + 0.3 * score_b

    print("\n================ PART A: EVENT DETECTION ================")
    print(f"{'Class':22s} | {'F1@0.3':>7s} | {'F1@0.5':>7s} | {'F1@0.7':>7s} | {'Mean F1':>7s}")
    print("-" * 58)
    for cls, sc in diag_a["per_class"].items():
        print(
            f"{cls:22s} | {sc['f1_0.3']:7.3f} | {sc['f1_0.5']:7.3f} | "
            f"{sc['f1_0.7']:7.3f} | {sc['mean_f1']:7.3f}"
        )
    print("-" * 58)
    print(f"Score A (Macro F1)    : {score_a:.4f}")
    print(f"Micro F1 (diagnostic) : {diag_a['micro_f1']:.4f}")
    print(f"Agnostic F1 (diag)    : {diag_a['agnostic_f1']:.4f}")

    print("\n============= PART B: ACCIDENT ANTICIPATION =============")
    print(f"Chance-norm AP        : {diag_b['AP']:.4f}")
    print(f"F1_alarm              : {diag_b['F1_alarm']:.4f} (P={diag_b['precision_alarm']:.3f}, R={diag_b['recall_alarm']:.3f})")
    print(f"mTTA                  : {diag_b['mTTA']:.2f}s / {W_WINDOW:.1f}s")
    print(f"Score B               : {score_b:.4f}")

    print("\n===================== OVERALL SCORE =====================")
    print(f"Model Score (M)       : {model_score:.4f}  (0.7 * {score_a:.4f} + 0.3 * {score_b:.4f})")


if __name__ == "__main__":
    main()
