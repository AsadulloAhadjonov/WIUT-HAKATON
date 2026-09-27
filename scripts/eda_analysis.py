"""Sample videolar boyicha EDA (Exploratory Data Analysis).

Har bir video uchun:
  - Harakat issiqlik xaritasi (motion heatmap)
  - Trek yollari
  - Intensivlik profili

    python scripts/eda_analysis.py samples/

Natijalar: dev/eda_<video_name>.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows UTF-8 chiqishi
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import _common  # noqa: F401
import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib kerak: pip install matplotlib")


def motion_heatmap(video_path: str, max_frames: int = 300) -> np.ndarray:
    """Video boyicha harakat issiqlik xaritasi (frame differencing)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(fps / 5))  # 5 fps da olchash

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return None

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    heatmap = np.zeros((h, w), np.float32)

    i = 0
    n = 0
    while n < max_frames:
        for _ in range(stride - 1):
            cap.grab()
            i += 1
        ret, frame = cap.read()
        i += 1
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray).astype(np.float32)
        heatmap += diff
        prev_gray = gray
        n += 1

    cap.release()
    return heatmap / (heatmap.max() + 1e-9)


def analyze_video(video_path: Path, out_dir: Path):
    """Video boyicha toliq EDA."""
    print(f"\n[EDA] {video_path.name} tahlil qilinmoqda...")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps
    cap.release()

    print(f"  {w}x{h}, {fps:.1f} fps, {duration:.1f}s ({n_frames} kadr)")

    if not HAS_MATPLOTLIB:
        return

    cap = cv2.VideoCapture(str(video_path))
    ret, first_frame = cap.read()
    # O'rta kadr
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames // 2)
    ret2, mid_frame = cap.read()
    cap.release()

    if not ret:
        print(f"  [XATO] Video o'qib bo'lmadi")
        return

    heatmap = motion_heatmap(str(video_path))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    video_name_safe = video_path.name.encode('ascii', 'replace').decode('ascii')
    fig.suptitle(
        f"EDA: {video_name_safe}\n{w}x{h} | {fps:.0f} fps | {duration:.1f}s | {n_frames} kadr",
        fontsize=12, fontweight="bold"
    )

    # 1. Birinchi kadr
    axes[0].imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Birinchi kadr")
    axes[0].axis("off")

    # 2. Harakat issiqlik xaritasi
    if heatmap is not None:
        axes[1].imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB), alpha=0.4)
        im = axes[1].imshow(heatmap, cmap="jet", alpha=0.6)
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title("Harakat issiqlik xaritasi")
        axes[1].axis("off")

        # Eng faol zona
        peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        axes[1].plot(peak_x, peak_y, 'w*', markersize=15, label=f"Eng faol ({peak_x},{peak_y})")
        axes[1].legend(loc="lower right", fontsize=8)
        print(f"  Eng faol mintaqa: x={peak_x}, y={peak_y}")
    else:
        axes[1].text(0.5, 0.5, "Heatmap mavjud emas", ha="center", va="center")
        axes[1].axis("off")

    # 3. Intensivlik profili
    if heatmap is not None:
        col_mean = heatmap.mean(axis=0)  # gorizontal
        row_mean = heatmap.mean(axis=1)  # vertikal

        ax3 = axes[2]
        ax3_twin = ax3.twinx()

        x_cols = np.arange(len(col_mean))
        y_rows = np.arange(len(row_mean))

        ax3.plot(x_cols, col_mean, color="#3b82f6", linewidth=1.5, label="Gorizontal (x)")
        ax3_twin.plot(y_rows, row_mean, color="#ef4444", linewidth=1.5, label="Vertikal (y)", alpha=0.8)

        ax3.set_xlabel("Piksel koordinata")
        ax3.set_ylabel("Harakat intensivligi (x yo'nalish)", color="#3b82f6")
        ax3_twin.set_ylabel("Harakat intensivligi (y yo'nalish)", color="#ef4444")
        ax3.set_title("Harakat intensivlik profili")
        ax3.grid(alpha=0.3)

        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "Profil mavjud emas", ha="center", va="center")
        axes[2].axis("off")

    plt.tight_layout()

    # Fayl nomini xavfsiz qilish
    safe_stem = "".join(c if c.isalnum() or c in "_-" else "_" for c in video_path.stem)
    out_file = out_dir / f"eda_{safe_stem}.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saqlandi: {out_file}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Sample videolar boyicha EDA")
    ap.add_argument("videos", nargs="?", default="samples/", help="Video papkasi yoki fayl")
    ap.add_argument("--out", default="dev/", help="Natijalar papkasi")
    args = ap.parse_args()

    videos = _common.list_videos(args.videos)
    if not videos:
        print(f"Video topilmadi: {args.videos}")
        sys.exit(1)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EDA] {len(videos)} ta video tahlil qilinmoqda...")
    print(f"Natijalar: {out_dir}")

    for v in videos:
        analyze_video(v, out_dir)

    print("\n[OK] EDA tugadi!")


if __name__ == "__main__":
    main()
