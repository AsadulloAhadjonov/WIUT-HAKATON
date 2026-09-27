"""Trayektoriya belgilari: silliqlangan holat, tezlik, yo'nalish, polosa, zonalar; juftliklar uchun TTC.

Birliklar:
  * sahnada gomografiya bo'lsa — metr va m/s;
  * bo'lmasa — "o'lcham birligi": obyekt bbox'ining sqrt(w*h) si. Tezlik = o'lcham/s.
    Shu tufayli uzoqdagi (kichik) va yaqindagi (katta) obyektlar bir xil chegaralar bilan baholanadi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scene import Scene

# Taxminiy obyekt o'lchami (metr), metrik rejimda TTC uchun radius sifatida
CLASS_RADIUS_M = {"person": 0.4, "bicycle": 0.9, "motorcycle": 1.0, "car": 2.2, "bus": 6.0, "truck": 5.0}


def track_features(df: pd.DataFrame, scene: Scene, cfg: dict, fps_sample: float) -> pd.DataFrame:
    """Trayektoriyalar jadvali → belgilar jadvali (har qator: bitta trek, bitta kadr).

    Part A uchun: markazlashgan silliqlash (kelajakdagi kadrlar ham ishlatiladi) — bu Part A da ruxsat.
    Part B (RiskEstimator) buni ISHLATMAYDI.
    """
    fcfg = cfg["features"]
    if df.empty:
        return df.assign(x=[], y=[], size=[], vx=[], vy=[], speed=[], lane=[])
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)

    # Trek sinfi — eng ko'p uchragan sinf (treker ba'zan sinfni almashtiradi)
    df["cls"] = df.groupby("track_id")["cls"].transform(lambda s: s.mode().iloc[0])

    # Haydovchi/yo'lovchi/motosiklchi: person bbox'i transport bbox'i ichida → alohida piyoda EMAS.
    # Aks holda motosiklchi har doim "to'qnashuv", avtobus oynasidagi odam "yo'ldagi piyoda" bo'ladi.
    if fcfg.get("drop_riders", True):
        df = drop_riders(df, float(fcfg.get("rider_overlap", 0.6)))

    # Qisqa treklarni tashlash (shovqin)
    dur = df.groupby("track_id")["t"].transform(lambda s: s.max() - s.min())
    df = df[dur >= float(fcfg.get("min_track_len_s", 0.5))].copy()
    if df.empty:
        return df.assign(x=[], y=[], size=[], vx=[], vy=[], speed=[], lane=[])

    # Yerga tegib turgan nuqta — bbox pastki o'rtasi
    px = ((df.x1 + df.x2) / 2).to_numpy(np.float32)
    py = df.y2.to_numpy(np.float32)
    w = (df.x2 - df.x1).to_numpy(np.float32)
    h = (df.y2 - df.y1).to_numpy(np.float32)
    df["px"], df["py"] = px, py
    df["w"], df["h"] = w, h

    if scene.metric:
        xy = scene.to_world(np.stack([px, py], 1))
        df["x"], df["y"] = xy[:, 0], xy[:, 1]
        df["size"] = df["cls"].map(CLASS_RADIUS_M).fillna(1.0) * 2
    else:
        df["x"], df["y"] = px, py
        df["size"] = np.sqrt(np.maximum(w * h, 1.0))

    win = max(1, int(round(float(fcfg.get("smooth_window_s", 0.6)) * fps_sample)))
    g = df.groupby("track_id", sort=False)
    for c in ("x", "y", "size", "px", "py"):
        df[c] = g[c].transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())

    # Tezlik: vaqt bo'yicha gradient (kadrlar orasi notekis bo'lishi mumkin).
    # (vx, vy) — ishchi birliklarda; (pvx, pvy) — pikselda (polosa yo'nalishi bilan solishtirish uchun).
    vel = {c: np.zeros(len(df), np.float32) for c in ("vx", "vy", "pvx", "pvy")}
    t_all = df["t"].to_numpy()
    for _, idx in df.groupby("track_id", sort=False).indices.items():
        if len(idx) < 2:
            continue
        t = t_all[idx]
        for src, dst in (("x", "vx"), ("y", "vy"), ("px", "pvx"), ("py", "pvy")):
            vel[dst][idx] = np.gradient(df[src].to_numpy()[idx], t)
    for c, v in vel.items():
        df[c] = v
    vx, vy = vel["vx"], vel["vy"]
    df["speed_raw"] = np.hypot(vx, vy)
    # Metrik rejimda m/s; aks holda o'lcham/s (normallashtirilgan)
    df["speed"] = df["speed_raw"] if scene.metric else df["speed_raw"] / df["size"]
    df["speed"] = df.groupby("track_id", sort=False)["speed"].transform(
        lambda s: s.rolling(win, center=True, min_periods=1).median()
    )

    df["lane"] = [scene.lane_of(a, b) for a, b in zip(df.px, df.py)] if scene.lanes else -1
    return df.reset_index(drop=True)


def drop_riders(df: pd.DataFrame, overlap: float = 0.6) -> pd.DataFrame:
    """Kadrlarning ko'pchiligida transport bbox'i ichida turgan person treklarini olib tashlaydi."""
    carriers = {"bicycle", "motorcycle", "car", "bus", "truck"}
    inside_count: dict[int, int] = {}
    total = df[df["cls"] == "person"].groupby("track_id").size().to_dict()
    if not total:
        return df
    for _, fr in df.groupby("frame", sort=False):
        P = fr[fr["cls"] == "person"]
        V = fr[fr["cls"].isin(carriers)]
        if P.empty or V.empty:
            continue
        pb = P[["x1", "y1", "x2", "y2"]].to_numpy()
        vb = V[["x1", "y1", "x2", "y2"]].to_numpy()
        ix = np.clip(np.minimum(pb[:, None, 2], vb[None, :, 2]) - np.maximum(pb[:, None, 0], vb[None, :, 0]), 0, None)
        iy = np.clip(np.minimum(pb[:, None, 3], vb[None, :, 3]) - np.maximum(pb[:, None, 1], vb[None, :, 1]), 0, None)
        area_p = ((pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1]))[:, None] + 1e-9
        contained = ((ix * iy) / area_p).max(1) >= overlap
        for tid in P["track_id"].to_numpy()[contained]:
            inside_count[tid] = inside_count.get(tid, 0) + 1
    riders = {tid for tid, n in inside_count.items() if n >= 0.5 * total[tid]}
    return df[~df["track_id"].isin(riders)]


def closest_approach(p: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """Nisbiy holat p va nisbiy tezlik v (ikkinchi minus birinchi).

    Qaytaradi (t*, d*): eng yaqin yaqinlashish vaqti va masofasi.
    t* <= 0 bo'lsa obyektlar uzoqlashmoqda (t* = 0, d* = hozirgi masofa).
    """
    vv = float(v @ v)
    if vv < 1e-9:
        return 0.0, float(np.linalg.norm(p))
    t = -float(p @ v) / vv
    if t <= 0:
        return 0.0, float(np.linalg.norm(p))
    return t, float(np.linalg.norm(p + v * t))


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Mantiqiy massivdagi ketma-ket True bo'laklar: [(bosh, oxir_inklyuziv), ...]."""
    if len(mask) == 0:
        return []
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))
