"""Part A pipeline: video → treklar → belgilar → qoidalar → post-processing → segmentlar."""
from __future__ import annotations

from pathlib import Path

from . import utils
from .detect_track import track_video
from .events import run_rules
from .features import track_features
from .postprocess import postprocess
from .scene import Scene


def detect_segments(video_path: str | Path, cfg: dict | None = None) -> tuple[list[dict], dict]:
    """Qaytaradi: (segmentlar, video ma'lumoti). Segment: {cls, start, end, score, tracks}."""
    cfg = cfg or utils.load_config()
    utils.set_seed(int(cfg.get("seed", 42)))
    video_path = Path(video_path)

    tracks, info = track_video(video_path, cfg)          # detektor faqat kesh bo'lmasa yuklanadi
    fps_sample = info["fps"] / info["stride"]
    scene = Scene(cfg.get("scene"), (info["width"], info["height"]))
    feat = track_features(tracks, scene, cfg, fps_sample)

    duration = info.get("duration", info["n_frames"] / info["fps"])
    ctx = {
        "fps": info["fps"],
        "fps_sample": fps_sample,
        "duration": duration,
        "frames": list(range(0, info["n_frames"], info["stride"])),
    }
    raw = run_rules(feat, scene, cfg, ctx)
    segs = postprocess(raw, cfg, duration)
    utils.log.info("%s: %d trek, %d xom → %d segment", video_path.name,
                   tracks["track_id"].nunique() if len(tracks) else 0, len(raw), len(segs))
    return segs, info
