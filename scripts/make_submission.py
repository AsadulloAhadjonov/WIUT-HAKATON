"""Topshirish paketini tayyorlash skripti.

Bu skript:
1. sample videolarda detect_events() ni ishlatadi
2. predictions_samples.json hosil qiladi (rasmiy format)
3. run_submission.py va evaluate.py mavjudligini tekshiradi

    python scripts/make_submission.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Repo ildizini import yo'liga qo'shish
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import _common  # noqa: F401
import cv2
import solution


def main():
    print("=" * 60)
    print("Topshirish paketi tayyorlanmoqda...")
    print("=" * 60)
    
    # 1. run_submission.py va evaluate.py tekshiruvi
    for fname in ["run_submission.py", "evaluate.py"]:
        fpath = ROOT / fname
        if not fpath.exists():
            print(f"⚠️  {fname} topilmadi — starter kitdan ko'chiring!")
        else:
            print(f"✅  {fname} mavjud")
    
    # 2. Sample videolarni tahlil qilish
    samples_dir = ROOT / "samples"
    videos = _common.list_videos(str(samples_dir))
    if not videos:
        print(f"❌  samples/ da video topilmadi: {samples_dir}")
        sys.exit(1)
    
    print(f"\n{len(videos)} ta video topildi:")
    for v in videos:
        print(f"  - {v.name}")
    
    print("\nPart A ishlamoqda...")
    results = {}
    
    for v in videos:
        print(f"\n  {v.name}:", end=" ", flush=True)
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = n_frames / fps
        
        t0 = time.time()
        events = solution.detect_events(str(v))
        ta = time.time() - t0
        
        print(f"{len(events)} hodisa, {ta:.1f}s (video {duration:.1f}s)")
        for e in events:
            if isinstance(e, (list, tuple)) and len(e) >= 3:
                print(f"    [{e[0]:.1f}s-{e[1]:.1f}s] {e[2]}")
        
        results[v.name] = {"events": events}
    
    # 3. predictions_samples.json yozish
    out_path = ROOT / "predictions_samples.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅  predictions_samples.json yaratildi: {out_path}")
    print("\nTopshirish uchun tekshiruvlar:")
    print("  python scripts/check_leak.py samples/<video>.mp4")
    print("  python scripts/check_determinism.py samples/")
    print("  python -m pytest -q tests/")
    print("\nBarchasi o'tsa — reponi topshiring!")


if __name__ == "__main__":
    main()
