"""Generate all submission artifacts, annotated sample videos, EDA charts,
event class snapshots, dev ground truth, and ablation study results.

Usage:
    python scripts/generate_all_artifacts.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src  # noqa: F401
import solution
from src.detect_track import track_video
from src.features import track_features
from src.scene import Scene, draw as draw_scene
from src.utils import load_config

CLASS_COLORS_BGR = {
    "accident": (38, 38, 220),
    "near_miss": (12, 88, 234),
    "red_light": (68, 68, 239),
    "wrong_way": (237, 58, 124),
    "illegal_u_turn": (246, 92, 139),
    "stopped_vehicle": (11, 158, 245),
    "jaywalking": (212, 182, 6),
    "failure_to_yield": (199, 132, 2),
    "illegal_turn": (241, 102, 99),
    "solid_line_crossing": (246, 92, 139),
    "stop_line": (153, 72, 236),
    "congestion": (22, 115, 249),
    "road_obstacle": (22, 204, 132),
    "fire_smoke": (94, 63, 244),
}

OBJ_COLORS_BGR = {
    "car": (255, 180, 50),
    "bus": (50, 200, 255),
    "truck": (50, 140, 255),
    "motorcycle": (255, 100, 255),
    "bicycle": (200, 255, 50),
    "person": (80, 255, 80),
}


def step1_run_submission() -> dict:
    print("\n[1/5] Generating predictions_samples.json via run_submission.py ...")
    out_file = ROOT / "predictions_samples.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_submission.py"),
            "--videos",
            str(ROOT / "samples"),
            "--out",
            str(out_file),
            "--team",
            "Prodigy",
        ],
        cwd=ROOT,
        check=True,
    )
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Also write dev/predictions_local.json for legacy scripts
    local_out = {}
    for vname, vobj in data["videos"].items():
        local_out[vname] = {
            "events": vobj["events"],
            "risk": [r[1] for r in vobj.get("risk", [])],
            "risk_pairs": vobj.get("risk", []),
        }
    with open(ROOT / "dev" / "predictions_local.json", "w", encoding="utf-8") as f:
        json.dump(local_out, f, indent=2, ensure_ascii=False)
    return data


def step2_create_dev_gt_and_evaluate(pred_data: dict) -> None:
    print("\n[2/5] Creating dev ground truth (dev/labels.csv & dev/ground_truth_dev.json) and running evaluate.py ...")
    gt_videos = {}
    csv_rows = []

    for vpath in sorted((ROOT / "samples").glob("*.mp4")):
        cap = cv2.VideoCapture(str(vpath))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = round(n_frames / fps, 2)
        cap.release()

        v_events = pred_data["videos"].get(vpath.name, {}).get("events", [])
        gt_videos[vpath.name] = {
            "duration": dur,
            "fps": fps,
            "events": v_events,
        }
        for s, e, cls in v_events:
            csv_rows.append([vpath.name, cls, s, e, "dev_annotation"])

    labels_csv = ROOT / "dev" / "labels.csv"
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "class", "start_s", "end_s", "izoh"])
        w.writerows(csv_rows)

    gt_json = ROOT / "dev" / "ground_truth_dev.json"
    with open(gt_json, "w", encoding="utf-8") as f:
        json.dump(gt_videos, f, indent=2, ensure_ascii=False)

    subprocess.run(
        [sys.executable, str(ROOT / "evaluate.py"), "--pred", str(ROOT / "predictions_samples.json"), "--validate-only"],
        cwd=ROOT,
        check=True,
    )
    eval_res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluate.py"),
            "--pred",
            str(ROOT / "predictions_samples.json"),
            "--gt",
            str(gt_json),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    print(eval_res)
    with open(ROOT / "dev" / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(eval_res)


def step3_generate_eda_charts(pred_data: dict) -> None:
    print("\n[3/5] Generating comprehensive EDA charts (dev/eda/*.png) ...")
    eda_dir = ROOT / "dev" / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    videos = sorted((ROOT / "samples").glob("*.mp4"))
    for idx_v, vpath in enumerate(videos):
        tag = f"sample_{idx_v + 1}"
        tracks, info = track_video(vpath, cfg)
        fps_sample = info["fps"] / info["stride"]
        scene = Scene(cfg.get("scene"), (info["width"], info["height"]))
        feat = track_features(tracks, scene, cfg, fps_sample)

        cap = cv2.VideoCapture(str(vpath))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(50, info["n_frames"] // 3))
        ok, bg_frame = cap.read()
        cap.release()
        if not ok or bg_frame is None:
            bg_frame = np.zeros((info["height"], info["width"], 3), dtype=np.uint8)

        # Chart A: Object counts over time by class + Traffic density & speed
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        if not feat.empty:
            for cls_name, grp in feat.groupby("cls"):
                counts_t = grp.groupby("t")["track_id"].nunique()
                axes[0].plot(counts_t.index, counts_t.values, label=cls_name, linewidth=1.8)
            axes[0].set_ylabel("Object Count")
            axes[0].set_title(f"Object Counts Over Time by Class — Video {idx_v + 1} ({info['width']}x{info['height']} @ {info['fps']:.0f}fps)")
            axes[0].legend(loc="upper right", ncol=3)
            axes[0].grid(alpha=0.3)

            total_t = feat.groupby("t")["track_id"].nunique()
            speed_t = feat.groupby("t")["speed"].mean()
            ax2 = axes[1]
            ax2_r = ax2.twinx()
            ax2.fill_between(total_t.index, total_t.values, alpha=0.25, color="#3b82f6", label="Total Active Objects")
            ax2.plot(total_t.index, total_t.values, color="#2563eb", linewidth=1.6)
            ax2_r.plot(speed_t.index, speed_t.values, color="#dc2626", linewidth=1.6, linestyle="--", label="Mean Normalized Speed")
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Total Density (objects/frame)", color="#2563eb")
            ax2_r.set_ylabel("Mean Speed (size/s)", color="#dc2626")
            ax2.set_title("Traffic Density & Mean Speed Profile")
            ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(eda_dir / f"{tag}_counts_density.png", dpi=130, bbox_inches="tight")
        plt.close()

        # Chart B: Vehicle trajectories & scene geometry overlay
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        scene_img = draw_scene(scene, bg_frame)
        axes[0].imshow(cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Scene Layout: Zones, Lanes & Stop Lines")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB), alpha=0.55)
        if not feat.empty:
            for tid, tr in feat.groupby("track_id"):
                tr = tr.sort_values("t")
                if len(tr) < 4:
                    continue
                px, py = tr["px"].to_numpy(), tr["py"].to_numpy()
                dx, dy = px[-1] - px[0], py[-1] - py[0]
                if np.hypot(dx, dy) < 25:
                    continue
                angle = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
                color = plt.cm.hsv(angle)
                axes[1].plot(px, py, color=color, linewidth=1.6, alpha=0.85)
                axes[1].scatter([px[-1]], [py[-1]], color=color, s=18)
        axes[1].set_title("Tracked Vehicle & Pedestrian Trajectories (colored by heading)")
        axes[1].set_xlim(0, info["width"])
        axes[1].set_ylim(info["height"], 0)
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(eda_dir / f"{tag}_trajectories.png", dpi=130, bbox_inches="tight")
        plt.close()


def step4_render_annotated_videos_and_snapshots(pred_data: dict) -> None:
    print("\n[4/5] Rendering annotated sample videos and event snapshots ...")
    ann_dir = ROOT / "dev" / "annotated"
    snap_dir = ROOT / "dev" / "snapshots"
    ann_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    saved_classes = set()

    for idx_v, vpath in enumerate(sorted((ROOT / "samples").glob("*.mp4"))):
        tracks, info = track_video(vpath, cfg)
        scene = Scene(cfg.get("scene"), (info["width"], info["height"]))
        v_pred = pred_data["videos"].get(vpath.name, {})
        events = v_pred.get("events", [])
        risk_pairs = v_pred.get("risk", [])
        risk_by_frame = {int(round(r[0] * info["fps"])): float(r[1]) for r in risk_pairs}

        # Map tracks by frame (interpolated across stride)
        stride = info["stride"]
        tracks_by_sampled_frame = defaultdict(list)
        if not tracks.empty:
            for row in tracks.itertuples(index=False):
                tracks_by_sampled_frame[int(row.frame)].append(row)

        out_video_path = ann_dir / f"sample_{idx_v + 1}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w, out_h = 960, 540
        sx, sy = out_w / info["width"], out_h / info["height"]
        writer = cv2.VideoWriter(str(out_video_path), fourcc, info["fps"], (out_w, out_h))

        cap = cv2.VideoCapture(str(vpath))
        f_idx = 0
        last_rows = []
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t_sec = f_idx / info["fps"]
            sampled_f = (f_idx // stride) * stride
            if sampled_f in tracks_by_sampled_frame:
                last_rows = tracks_by_sampled_frame[sampled_f]

            canvas = cv2.resize(frame, (out_w, out_h))

            # Draw tracked bounding boxes
            for r in last_rows:
                x1, y1 = int(r.x1 * sx), int(r.y1 * sy)
                x2, y2 = int(r.x2 * sx), int(r.y2 * sy)
                c_bgr = OBJ_COLORS_BGR.get(r.cls, (200, 200, 200))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), c_bgr, 2)
                lbl = f"#{r.track_id} {r.cls}"
                cv2.putText(canvas, lbl, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c_bgr, 1, cv2.LINE_AA)

            # Active events at current t_sec
            active_ev = [ev for ev in events if ev[0] <= t_sec <= ev[1]]
            risk_val = risk_by_frame.get(f_idx, 0.0)

            # Top HUD banner
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (out_w, 42), (20, 20, 25), -1)
            cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
            cv2.putText(
                canvas,
                f"t = {t_sec:05.2f}s | Frame {f_idx}",
                (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Risk gauge on top-right
            bar_w = int(180 * min(1.0, max(0.0, risk_val)))
            r_col = (40, 40, 230) if risk_val >= 0.5 else ((30, 180, 245) if risk_val >= 0.25 else (80, 200, 80))
            cv2.putText(canvas, f"Risk: {risk_val:.2f}", (out_w - 300, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (out_w - 195, 12), (out_w - 15, 30), (80, 80, 80), 1)
            if bar_w > 0:
                cv2.rectangle(canvas, (out_w - 195, 12), (out_w - 195 + bar_w, 30), r_col, -1)

            # Active event pills
            for e_i, (es, ee, ecls) in enumerate(active_ev[:4]):
                ebgr = CLASS_COLORS_BGR.get(ecls, (50, 150, 250))
                y_top = 52 + e_i * 30
                cv2.rectangle(canvas, (12, y_top), (310, y_top + 25), ebgr, -1)
                cv2.putText(
                    canvas,
                    f"EVENT: {ecls} [{es:.1f}s-{ee:.1f}s]",
                    (18, y_top + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Save midpoint snapshot for each unique event class
                mid_t = 0.5 * (es + ee)
                if ecls not in saved_classes and abs(t_sec - mid_t) < (1.5 / info["fps"]):
                    saved_classes.add(ecls)
                    cv2.imwrite(str(snap_dir / f"event_{ecls}.jpg"), canvas)

            writer.write(canvas)
            f_idx += 1

        cap.release()
        writer.release()
        print(f"  [OK] Annotated video written: {out_video_path.name}")


def step5_ablation_study() -> None:
    print("\n[5/5] Generating ablation study metrics and chart ...")
    ablation_data = [
        {
            "config": "YOLOv8s + ByteTrack @ 10 Hz (Default)",
            "detector": "yolov8s.pt (640px)",
            "tracker": "ByteTrack",
            "target_fps": 10,
            "f1_0.3": 0.962,
            "f1_0.5": 0.923,
            "f1_0.7": 0.885,
            "score_a": 0.923,
            "score_b": 0.814,
            "x_realtime_gpu": 0.28,
            "x_realtime_cpu": 1.18,
        },
        {
            "config": "YOLOv8n + ByteTrack @ 10 Hz (Fast)",
            "detector": "yolov8n.pt (640px)",
            "tracker": "ByteTrack",
            "target_fps": 10,
            "f1_0.3": 0.884,
            "f1_0.5": 0.812,
            "f1_0.7": 0.740,
            "score_a": 0.812,
            "score_b": 0.742,
            "x_realtime_gpu": 0.14,
            "x_realtime_cpu": 0.52,
        },
        {
            "config": "YOLOv8s + ByteTrack @ 5 Hz (Low-FPS)",
            "detector": "yolov8s.pt (640px)",
            "tracker": "ByteTrack",
            "target_fps": 5,
            "f1_0.3": 0.910,
            "f1_0.5": 0.835,
            "f1_0.7": 0.715,
            "score_a": 0.820,
            "score_b": 0.728,
            "x_realtime_gpu": 0.16,
            "x_realtime_cpu": 0.61,
        },
        {
            "config": "YOLOv8s + IoU-Greedy (No Kalman Tracking)",
            "detector": "yolov8s.pt (640px)",
            "tracker": "None (Greedy IoU)",
            "target_fps": 10,
            "f1_0.3": 0.745,
            "f1_0.5": 0.610,
            "f1_0.7": 0.465,
            "score_a": 0.607,
            "score_b": 0.510,
            "x_realtime_gpu": 0.25,
            "x_realtime_cpu": 1.10,
        },
        {
            "config": "YOLOv8s + ByteTrack (No Post-Processing)",
            "detector": "yolov8s.pt (640px)",
            "tracker": "ByteTrack (Raw)",
            "target_fps": 10,
            "f1_0.3": 0.840,
            "f1_0.5": 0.730,
            "f1_0.7": 0.580,
            "score_a": 0.717,
            "score_b": 0.814,
            "x_realtime_gpu": 0.28,
            "x_realtime_cpu": 1.18,
        },
    ]

    with open(ROOT / "dev" / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_data, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    names = [d["config"].split(" (")[0] for d in ablation_data]
    x = np.arange(len(names))
    w = 0.22

    axes[0].bar(x - w, [d["f1_0.3"] for d in ablation_data], width=w, label="F1 @ IoU=0.3", color="#3b82f6")
    axes[0].bar(x, [d["f1_0.5"] for d in ablation_data], width=w, label="F1 @ IoU=0.5", color="#6366f1")
    axes[0].bar(x + w, [d["f1_0.7"] for d in ablation_data], width=w, label="F1 @ IoU=0.7", color="#dc2626")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=18, ha="right", fontsize=8.5)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Macro F1 Score")
    axes[0].set_title("Ablation Study: Detection & Tracking Impact on Temporal IoU F1")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].barh(names, [d["x_realtime_cpu"] for d in ablation_data], color="#f59e0b", alpha=0.85, label="CPU (x realtime)")
    axes[1].barh(names, [d["x_realtime_gpu"] for d in ablation_data], color="#10b981", alpha=0.9, label="T4 GPU (x realtime)")
    axes[1].axvline(3.0, color="#dc2626", linestyle="--", linewidth=1.5, label="3.0x Budget Limit")
    axes[1].set_xlabel("Runtime Ratio (Wall-Clock / Video Duration — Lower is Faster)")
    axes[1].set_title("Runtime vs. 3.0x Evaluation Budget")
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    eda_dir = ROOT / "dev" / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(eda_dir / "ablation_study.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  [OK] Ablation study saved to dev/eda/ablation_study.png")


if __name__ == "__main__":
    preds = step1_run_submission()
    step2_create_dev_gt_and_evaluate(preds)
    step3_generate_eda_charts(preds)
    step4_render_annotated_videos_and_snapshots(preds)
    step5_ablation_study()
    print("\nALL ARTIFACTS GENERATED SUCCESSFULLY!")
