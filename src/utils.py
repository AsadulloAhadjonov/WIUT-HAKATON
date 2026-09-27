"""Umumiy yordamchilar: seed, konfiguratsiya, loglash, vaqt o'lchash."""
from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger("traffic")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(os.environ.get("TRAFFIC_LOG", "INFO"))


def set_seed(seed: int = 42) -> None:
    """Barcha tasodifiy generatorlarni qotiradi. Ikki ishga tushirish bir xil natija berishi uchun."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # torch yo'q bo'lsa (masalan, faqat unit testlar)
        pass


def resolve(path: str | os.PathLike) -> Path:
    """Nisbiy yo'lni repo ildiziga nisbatan ochadi (ishga tushirilgan papkadan qat'i nazar)."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_yaml(path: str | os.PathLike) -> dict:
    with open(resolve(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(params: str = "configs/params.yaml", scene: str | None = None) -> dict:
    """params.yaml ni o'qiydi; sahna fayli params ichidagi `scene_file` dan yoki argumentdan olinadi.

    Muhit o'zgaruvchilari bilan almashtirish mumkin (ablatsiyalar uchun qulay):
        TRAFFIC_PARAMS=configs/params_fast.yaml
        TRAFFIC_SCENE=configs/scene.yaml
    """
    params = os.environ.get("TRAFFIC_PARAMS", params)
    cfg = load_yaml(params)
    scene = os.environ.get("TRAFFIC_SCENE", scene or cfg.get("scene_file", "configs/scene.yaml"))
    cfg["scene"] = load_yaml(scene) if resolve(scene).exists() else {}
    return cfg


@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    log.info("%s: %.2f s", name, time.perf_counter() - t0)
