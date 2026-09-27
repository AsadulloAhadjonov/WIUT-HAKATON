"""Detektor (YOLO, lokal og'irliklar) + treker (ByteTrack) → trayektoriyalar jadvali.

Ikki xil foydalanish:
  * `track_video(path, cfg)`   — Part A: butun videoni o'qiydi, batch bilan tez ishlaydi, keshlaydi.
  * `OnlineTracker.update(frame)` — Part B: bitta kadr keladi, bitta kadr qayta ishlanadi (kauzal).

Trayektoriya jadvali ustunlari (pandas.DataFrame):
    frame, t, track_id, cls, conf, x1, y1, x2, y2
`cls` — matn: person / bicycle / car / motorcycle / bus / truck.
"""
from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import utils

COLUMNS = ["frame", "t", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"]


# --------------------------------------------------------------------------- detektor
class Detector:
    """ultralytics YOLO o'rami. Og'irliklar faqat lokal fayldan yuklanadi — internet ishlatilmaydi."""

    def __init__(self, cfg: dict):
        dcfg = cfg["detector"]
        weights = utils.resolve(dcfg["weights"])
        if not weights.exists():
            raise FileNotFoundError(
                f"Og'irlik fayli topilmadi: {weights}. U repoda bo'lishi shart (internetdan yuklanmaydi)."
            )
        from ultralytics import YOLO  # src/__init__.py oflayn rejimni allaqachon yoqqan

        self.model = YOLO(str(weights))
        self.names_keep = {int(k): v for k, v in dcfg["classes"].items()}  # COCO id -> nom
        self.imgsz = int(dcfg.get("imgsz", 640))
        self.conf = float(dcfg.get("conf", 0.2))
        self.iou = float(dcfg.get("iou", 0.5))
        self.half = bool(dcfg.get("half", False))
        self.batch = int(dcfg.get("batch", 8))
        self.device = self._pick_device(dcfg.get("device", "auto"))
        if self.device == "cpu":
            self.half = False
        utils.log.info("Detektor: %s, qurilma=%s, imgsz=%d", weights.name, self.device, self.imgsz)

    @staticmethod
    def _pick_device(dev: str) -> str:
        if dev != "auto":
            return dev
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """frames (BGR) → har kadr uchun massiv [N, 6]: x1, y1, x2, y2, conf, coco_cls."""
        out = []
        for i in range(0, len(frames), self.batch):
            res = self.model.predict(
                frames[i:i + self.batch],
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                classes=list(self.names_keep),
                device=self.device,
                half=self.half,
                verbose=False,
            )
            for r in res:
                b = r.boxes
                if b is None or len(b) == 0:
                    out.append(np.zeros((0, 6), np.float32))
                    continue
                arr = np.concatenate(
                    [b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], axis=1
                ).astype(np.float32)
                out.append(arr)
        return out


_DETECTORS: dict = {}


def get_detector(cfg: dict) -> Detector:
    """Og'irliklarni bir marta yuklaydi. Model holatsiz — videolar orasida natija almashmaydi."""
    key = repr(sorted(cfg["detector"].items()))
    if key not in _DETECTORS:
        _DETECTORS[key] = Detector(cfg)
    return _DETECTORS[key]


# --------------------------------------------------------------------------- treker
class _Dets:
    """BYTETracker kutadigan "Results-ga o'xshash" obyekt (conf, xywh, cls, boolean indekslash)."""

    def __init__(self, arr: np.ndarray):
        self.arr = arr
        xyxy = arr[:, :4]
        self.conf = arr[:, 4]
        self.cls = arr[:, 5]
        self.xywh = np.stack(
            [(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
             xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], axis=1
        ) if len(arr) else np.zeros((0, 4), np.float32)

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        return _Dets(self.arr[idx])


def make_tracker(cfg: dict, updates_per_second: float):
    """ByteTrack (ultralytics ichidagi). Parametrlar params.yaml → tracker bo'limidan."""
    from ultralytics.trackers.byte_tracker import BYTETracker

    tcfg = dict(cfg["tracker"])
    buffer_s = float(tcfg.pop("track_buffer_s", 1.5))
    args = Namespace(
        tracker_type="bytetrack",
        track_buffer=max(1, int(round(buffer_s * updates_per_second))),
        fuse_score=True,
        **tcfg,
    )
    return BYTETracker(args)


def _tracks_to_rows(tracks: np.ndarray, frame_idx: int, t: float, names: dict) -> list[list]:
    rows = []
    for x1, y1, x2, y2, tid, score, cls, _idx in tracks:
        name = names.get(int(cls))
        if name is None:
            continue
        rows.append([frame_idx, t, int(tid), name, float(score), float(x1), float(y1), float(x2), float(y2)])
    return rows


def stride_for(fps: float, cfg: dict) -> int:
    """Detektor har necha kadrda ishlaydi. target_fps=10 va video 25 fps → stride 2 (12.5 Hz)."""
    target = float(cfg["sampling"].get("target_fps", 10))
    return max(1, int(round(fps / target))) if target > 0 else 1


# --------------------------------------------------------------------------- Part A: butun video
def video_info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Videoni ochib bo'lmadi: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    info = {
        "fps": fps if 1.0 <= fps <= 240.0 else 25.0,  # buzilgan metadata bo'lsa 25 deb olinadi
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def _cache_path(video: Path, cfg: dict) -> Path:
    st = video.stat()
    key = json.dumps(
        [video.name, st.st_size, int(st.st_mtime), cfg["detector"], cfg["tracker"], cfg["sampling"]],
        sort_keys=True, default=str,
    )
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return utils.resolve(cfg.get("cache_dir", "cache")) / f"{video.stem}_{h}.csv.gz"


def track_video(path: str | Path, cfg: dict, detector: Detector | None = None) -> tuple[pd.DataFrame, dict]:
    """Butun video → trayektoriyalar jadvali. Faqat Part A (detect_events) uchun."""
    path = Path(path)
    info = video_info(path)
    stride = stride_for(info["fps"], cfg)
    info["stride"] = stride

    cache = _cache_path(path, cfg) if cfg.get("use_cache", True) else None
    if cache is not None and cache.exists():
        utils.log.info("Kesh: %s", cache.name)
        return pd.read_csv(cache), info

    detector = detector or get_detector(cfg)
    tracker = make_tracker(cfg, info["fps"] / stride)
    names = detector.names_keep
    rows: list[list] = []

    cap = cv2.VideoCapture(str(path))
    buf_frames, buf_idx = [], []
    idx = 0

    def flush():
        for fi, dets in zip(buf_idx, detector(buf_frames)):
            tracks = tracker.update(_Dets(dets))
            if len(tracks):
                rows.extend(_tracks_to_rows(tracks, fi, fi / info["fps"], names))
        buf_frames.clear()
        buf_idx.clear()

    while True:
        ok = cap.grab()             # grab() arzon: keraksiz kadrlarni dekodlamaymiz
        if not ok:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                buf_frames.append(frame)
                buf_idx.append(idx)
                if len(buf_frames) >= detector.batch:
                    flush()
        idx += 1
    if buf_frames:
        flush()
    cap.release()
    info["n_frames"] = idx          # metadata noto'g'ri bo'lishi mumkin — haqiqiy son
    info["duration"] = idx / info["fps"]

    df = pd.DataFrame(rows, columns=COLUMNS)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    return df, info


# --------------------------------------------------------------------------- Part B: onlayn
class OnlineTracker:
    """Kadrma-kadr ishlaydi. Faqat o'ziga berilgan kadrlarni ko'radi — kelajak ma'lumoti yo'q."""

    def __init__(self, cfg: dict, fps: float, detector: Detector | None = None):
        self.detector = detector or get_detector(cfg)
        self.stride = stride_for(fps, cfg)
        self.tracker = make_tracker(cfg, fps / self.stride)
        self.names = self.detector.names_keep

    def update(self, frame: np.ndarray, frame_idx: int, t: float) -> list[list] | None:
        """Detektor ishlagan kadrda qatorlar ro'yxatini, aks holda None qaytaradi."""
        if frame_idx % self.stride != 0:
            return None
        dets = self.detector([frame])[0]
        tracks = self.tracker.update(_Dets(dets))
        return _tracks_to_rows(tracks, frame_idx, t, self.names) if len(tracks) else []
