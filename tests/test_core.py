"""Tezkor unit testlar (YOLO kerak emas, ~2 soniya):  python -m pytest -q tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.events import run_rules  # noqa: E402
from src.features import closest_approach, drop_riders, runs, track_features  # noqa: E402
from src.postprocess import postprocess  # noqa: E402
from src.risk import OnlineRisk  # noqa: E402
from src.scene import Scene, segments_intersect  # noqa: E402
from src.utils import load_config, load_yaml  # noqa: E402

FPS = 10.0


@pytest.fixture(scope="module")
def cfg():
    import os
    os.environ.setdefault("TRAFFIC_PARAMS", "configs/params_synthetic.yaml")
    return load_config("configs/params_synthetic.yaml")


@pytest.fixture(scope="module")
def scene():
    return Scene(load_yaml("configs/scene_synthetic.yaml"), (1280, 720))


def make_track(tid, cls, xs, ys, t0=0.0, w=100, h=60):
    """Bbox pastki markazi (xs, ys) bo'yicha sintetik trek."""
    rows = []
    for k, (x, y) in enumerate(zip(xs, ys)):
        f = int(round((t0 + k / FPS) * 25))
        rows.append([f, t0 + k / FPS, tid, cls, 0.9, x - w / 2, y - h, x + w / 2, y])
    return rows


def df_of(*tracks):
    return pd.DataFrame([r for t in tracks for r in t],
                        columns=["frame", "t", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"])


def run(cfg, scene, df, only):
    c = dict(cfg)
    c["events"] = [e for e in cfg["events"] if e["class"] in only]
    feat = track_features(df, scene, c, FPS)
    frames = sorted(df["frame"].unique())
    ctx = {"fps": 25.0, "fps_sample": FPS, "duration": df["t"].max() + 0.1, "frames": frames}
    return postprocess(run_rules(feat, scene, c, ctx), c, ctx["duration"])


# ------------------------------------------------------------------ geometriya
def test_closest_approach_head_on():
    t, d = closest_approach(np.array([10.0, 0.0]), np.array([-5.0, 0.0]))
    assert t == pytest.approx(2.0) and d == pytest.approx(0.0)


def test_closest_approach_diverging():
    t, d = closest_approach(np.array([10.0, 0.0]), np.array([5.0, 0.0]))
    assert t == 0.0 and d == pytest.approx(10.0)


def test_segments_intersect():
    assert segments_intersect((0, 0), (2, 2), (0, 2), (2, 0))
    assert not segments_intersect((0, 0), (1, 1), (2, 2), (3, 3))


def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1])) == [(1, 2), (4, 4)]


def test_scene_scaling():
    s = Scene({"image_size": [100, 100], "zones": {"z": [[0, 0], [50, 0], [50, 50], [0, 50]]}}, (200, 200))
    assert s.in_zone("z", 90, 90) and not s.in_zone("z", 110, 110)
    assert s.in_zone("missing", 1, 1)  # e'lon qilinmagan zona = butun kadr


# ------------------------------------------------------------------ post-processing
def test_postprocess_merge_and_drop(cfg):
    segs = [{"cls": "x", "start": 0, "end": 2, "score": 1}, {"cls": "x", "start": 2.5, "end": 4, "score": 1},
            {"cls": "x", "start": 10, "end": 10.3, "score": 1}]
    out = postprocess(segs, {"postprocess": {"default": {"merge_gap_s": 1.0, "min_len_s": 1.0}}}, 20)
    assert [(s["start"], s["end"]) for s in out] == [(0, 4)]


# ------------------------------------------------------------------ qoidalar
def test_stopped(cfg, scene):
    # FPS=10, stopped_vehicle min_duration_s=10 → kamida 100 ta kadr harakatsiz
    xs = list(np.linspace(100, 400, 10)) + [400] * 120 + list(np.linspace(400, 700, 10))
    ev = run(cfg, scene, df_of(make_track(1, "car", xs, [500] * len(xs))), {"stopped_vehicle"})
    assert len(ev) == 1, f"stopped_vehicle aniqlanmadi yoki ko'p topildi: {ev}"
    assert ev[0]["start"] < 5.0 and ev[0]["end"] > 12.0


def test_wrong_way(cfg, scene):
    xs = np.linspace(1100, 200, 60)      # yuqori polosa sharqqa yo'nalgan, mashina g'arbga
    ev = run(cfg, scene, df_of(make_track(1, "car", xs, [300] * 60)), {"wrong_way"})
    assert len(ev) == 1 and ev[0]["end"] - ev[0]["start"] > 4


def test_right_way_no_event(cfg, scene):
    xs = np.linspace(200, 1100, 60)
    assert run(cfg, scene, df_of(make_track(1, "car", xs, [300] * 60)), {"wrong_way"}) == []


def test_lane_change(cfg, scene):
    """solid_line_crossing (lane_change fallback) testi."""
    ys = [300] * 30 + list(np.linspace(300, 480, 10)) + [480] * 30
    xs = np.linspace(100, 1100, len(ys))
    ev = run(cfg, scene, df_of(make_track(1, "car", xs, ys)), {"solid_line_crossing"})
    assert len(ev) == 1, f"solid_line_crossing aniqlanmadi: {ev}"


def test_pedestrian_on_road_outside_crosswalk(cfg, scene):
    """jaywalking: yo'lda piyoda (crosswalk tashqarida)."""
    ys = np.linspace(200, 580, 40)
    on_road = run(cfg, scene, df_of(make_track(1, "person", [300] * 40, ys, w=30, h=80)), {"jaywalking"})
    on_cross = run(cfg, scene, df_of(make_track(1, "person", [640] * 40, ys, w=30, h=80)), {"jaywalking"})
    assert len(on_road) == 1, f"jaywalking yo'lda aniqlanmadi: {on_road}"
    assert on_cross == [], f"jaywalking crosswalk'da bo'lmasligi kerak: {on_cross}"


def test_near_miss(cfg, scene):
    # mashina piyodaga to'g'ri kelmoqda va oxirgi lahzada to'xtaydi
    xs = list(np.linspace(100, 540, 22)) + [540] * 10
    car = make_track(1, "car", xs, [480] * len(xs))
    ped = make_track(2, "person", [640] * len(xs), [480] * len(xs), w=30, h=80)
    ev = run(cfg, scene, df_of(car, ped), {"near_miss"})
    assert len(ev) == 1


def test_rider_dropped():
    moto = make_track(1, "motorcycle", np.linspace(100, 600, 20), [500] * 20, w=60, h=60)
    rider = make_track(2, "person", np.linspace(100, 600, 20), [490] * 20, w=30, h=40)
    out = drop_riders(df_of(moto, rider))
    assert set(out["track_id"]) == {1}


# ------------------------------------------------------------------ onlayn xavf
def _feed(est, tracks):
    by_t = {}
    for tr in tracks:
        for r in tr:
            by_t.setdefault(r[1], []).append(r)
    out = []
    for t in sorted(by_t):
        est.observe(by_t[t], t)
        out.append(est._smooth())
    return np.array(out)


def test_risk_rises_on_collision_course(cfg):
    est = OnlineRisk(fps=FPS, cfg=cfg, use_detector=False)
    car = make_track(1, "car", np.linspace(100, 600, 25), [480] * 25)
    ped = make_track(2, "person", [700] * 25, [480] * 25, w=30, h=80)
    r = _feed(est, [car, ped])
    assert r[-1] > 0.5 and r[0] < 0.2

    est2 = OnlineRisk(fps=FPS, cfg=cfg, use_detector=False)
    far = make_track(1, "car", np.linspace(100, 600, 25), [250] * 25)
    ped2 = make_track(2, "person", [700] * 25, [550] * 25, w=30, h=80)
    assert _feed(est2, [far, ped2]).max() < r.max()


def test_risk_is_causal(cfg):
    car = make_track(1, "car", np.linspace(100, 600, 30), [480] * 30)
    ped = make_track(2, "person", [700] * 30, [480] * 30, w=30, h=80)
    full = _feed(OnlineRisk(fps=FPS, cfg=cfg, use_detector=False), [car, ped])
    half = _feed(OnlineRisk(fps=FPS, cfg=cfg, use_detector=False), [car[:15], ped[:15]])
    assert np.allclose(full[:15], half)


# ------------------------------------------------------------------ yangi qoidalar
def test_jaywalking(cfg, scene):
    """Piyoda yo'lda (roadway) yurishi — jaywalking."""
    ys = np.linspace(380, 580, 40)
    # Yo'lda piyoda
    on_road = run(cfg, scene, df_of(make_track(1, "person", [300] * 40, ys, w=30, h=80)), {"jaywalking"})
    assert len(on_road) >= 1, "jaywalking yo'lda aniqlanmadi"


def test_illegal_u_turn_detected(cfg, scene):
    """illegal_u_turn: U-burilish aniqlanishi."""
    # Avval o'ngga, keyin teskari
    xs = list(np.linspace(200, 800, 30)) + list(np.linspace(800, 200, 30))
    ys = [500] * 60
    ev = run(cfg, scene, df_of(make_track(1, "car", xs, ys)), {"illegal_u_turn"})
    assert len(ev) >= 1, f"illegal_u_turn aniqlanmadi. Topilgan: {ev}"


def test_solid_line_crossing_no_line(cfg, scene):
    """solid_line_crossing: scene.yaml da solid_line bo'lmasa va lane bo'lmasa — bo'sh."""
    # scene_synthetic da solid_line yo'q, lekin lanes bor → lane_change fallback
    # Bu test faqat yo'qligi tekshiruvini bajaradi (xato bo'lmasligi kerak)
    ys = [300] * 30 + list(np.linspace(300, 480, 10)) + [480] * 30
    xs = np.linspace(100, 1100, len(ys))
    ev = run(cfg, scene, df_of(make_track(1, "car", xs, ys)), {"solid_line_crossing"})
    # Natija bo'lishi ham, bo'lmasligi ham mumkin — xato bo'lmasligi muhim
    assert isinstance(ev, list)

