"""Part B: onlayn xavf baholash. QAT'IY KAUZAL.

Qoidalar (buzish = diskvalifikatsiya):
  * faqat step() ga berilgan kadrlar ishlatiladi;
  * video fayl ichkarida o'qilmaydi;
  * Part A natijasi (detect_events, kesh) ishlatilmaydi — o'z treker nusxasi bor.
Tekshirish: python scripts/check_leak.py <video>
"""
from __future__ import annotations

from collections import deque
from itertools import combinations

import numpy as np

from .detect_track import Detector, OnlineTracker, get_detector
from .features import closest_approach
from .utils import load_config, set_seed

VEHICLES = {"car", "bus", "truck", "motorcycle"}


class OnlineRisk:
    def __init__(
        self,
        fps: float = 25.0,
        cfg: dict | None = None,
        detector: Detector | None = None,
        use_detector: bool = True,
    ):
        self.cfg = cfg or load_config()
        set_seed(int(self.cfg.get("seed", 42)))
        self.fps = float(fps) if fps and fps > 0 else 25.0
        r = self.cfg["risk"]
        self.vel_window = float(r.get("vel_window_s", 0.6))
        self.stale = float(r.get("stale_s", 1.0))
        self.tau_ttc = float(r.get("tau_ttc_s", 1.5))
        self.ttc_margin = float(r.get("ttc_margin", 0.8))
        self.tau_dist = float(r.get("tau_dist", 0.5))
        self.min_speed = float(r.get("min_speed", 0.35))
        self.w_density = float(r.get("w_density", 0.08))
        self.density_ref = float(r.get("density_ref", 25))
        self.a_up = float(r.get("ema_up", 0.45))
        self.a_down = float(r.get("ema_down", 0.12))
        self.rider_overlap = float(self.cfg["features"].get("rider_overlap", 0.6))
        self.use_detector = use_detector
        self.tracker = (
            OnlineTracker(self.cfg, self.fps, detector or get_detector(self.cfg))
            if use_detector
            else None
        )
        self.reset()

    def reset(self):
        self.i = 0
        self.hist: dict[int, deque] = {}
        self.raw = 0.0
        self.value = 0.0

    def step(self, frame: np.ndarray, t: float | None = None) -> float:
        t = self.i / self.fps if t is None else float(t)
        rows = self.tracker.update(frame, self.i, t) if self.tracker is not None else None
        self.i += 1
        if rows is not None:
            self.observe(rows, t)
        return self._smooth()

    def observe(self, rows: list[list], t: float) -> None:
        """Treker qatorlari: [frame, t, track_id, cls, conf, x1, y1, x2, y2]."""
        for _f, _t, tid, cls, _c, x1, y1, x2, y2 in rows:
            size = float(np.sqrt(max((x2 - x1) * (y2 - y1), 1.0)))
            self.hist.setdefault(tid, deque(maxlen=64)).append(
                (t, (x1 + x2) / 2, y2, size, cls, (x1, y1, x2, y2))
            )
        for tid in [k for k, h in self.hist.items() if t - h[-1][0] > self.stale]:
            del self.hist[tid]
        self.raw = self._raw_risk(t)

    def _state(self, t: float):
        """Hozir ko'rinib turgan treklar: (pozitsiya, tezlik, o'lcham, sinf, bbox)."""
        out = []
        min_pts = 4 if self.use_detector else 2
        for h in self.hist.values():
            if h[-1][0] < t - 1e-6:
                continue
            if len(h) < min_pts:
                continue
            pts = [p for p in h if p[0] >= t - self.vel_window]
            _, x, y, size, cls, box = pts[-1]
            if len(pts) >= 2 and pts[-1][0] - pts[0][0] >= (0.18 if self.use_detector else 1e-6):
                dt = pts[-1][0] - pts[0][0]
                v = np.array([(pts[-1][1] - pts[0][1]) / dt, (pts[-1][2] - pts[0][2]) / dt])
            else:
                v = np.zeros(2)
            out.append((np.array([x, y]), v, size, cls, box))
        return self._drop_riders(out)

    def _drop_riders(self, objs):
        """Transport bbox'i ichidagi odam (haydovchi, motosiklchi) alohida obyekt emas."""
        carriers = [o[4] for o in objs if o[3] != "person"]
        keep = []
        for o in objs:
            if o[3] == "person" and carriers:
                x1, y1, x2, y2 = o[4]
                area = max((x2 - x1) * (y2 - y1), 1e-9)
                inside = (
                    max(
                        max(0.0, min(x2, b[2]) - max(x1, b[0]))
                        * max(0.0, min(y2, b[3]) - max(y1, b[1]))
                        for b in carriers
                    )
                    / area
                )
                if inside >= self.rider_overlap:
                    continue
            keep.append(o)
        return keep

    def _raw_risk(self, t: float) -> float:
        objs = self._state(t)
        if not objs:
            return 0.0
        best = 0.0
        margin_eff = 0.50 if self.use_detector else self.ttc_margin
        min_closing = 0.50 if self.use_detector else 0.15
        veh_positions = [o[0] for o in objs if o[3] in VEHICLES]

        for (p1, v1, s1, c1, _), (p2, v2, s2, c2, _) in combinations(objs, 2):
            if c1 not in VEHICLES and c2 not in VEHICLES:
                continue
            rad = (s1 + s2) / 2.0
            rel_p, rel_v = p2 - p1, v2 - v1
            dist = float(np.linalg.norm(rel_p))
            if dist < 1e-6 or dist > 3.2 * rad:
                continue

            speed1_norm = float(np.linalg.norm(v1)) / (s1 + 1e-9)
            speed2_norm = float(np.linalg.norm(v2)) / (s2 + 1e-9)
            moving = max(speed1_norm, speed2_norm)
            if moving < self.min_speed:
                continue

            unit_rel_p = rel_p / dist
            closing_speed = -float(unit_rel_p @ rel_v) / (rad + 1e-9)
            if closing_speed <= min_closing:
                continue

            if speed1_norm > 0.25 and speed2_norm > 0.25:
                cos_dir = float((v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
                if cos_dir > 0.70:
                    n1 = v1 / (np.linalg.norm(v1) + 1e-9)
                    lateral = float(np.linalg.norm(rel_p - float(rel_p @ n1) * n1))
                    if lateral >= 0.55 * rad:
                        continue

            t_star, d_star = closest_approach(rel_p, rel_v)
            if t_star <= 0 or t_star > 2.2 or d_star >= margin_eff * rad:
                continue

            align_factor = float(np.clip(1.0 - (d_star / (margin_eff * rad + 1e-9)) ** 2, 0.0, 1.0))
            closing_factor = float(np.clip(closing_speed / 1.3, 0.0, 1.0))

            # Multi-vehicle hazard amplification: if other vehicles are stopped/clustered right around the conflict point
            cluster_scale = 1.0
            if self.use_detector:
                mid_pt = 0.5 * (p1 + p2)
                nearby_veh = sum(1 for vp in veh_positions if float(np.linalg.norm(vp - mid_pt)) < 2.8 * rad)
                cluster_scale = 1.45 if nearby_veh >= 4 else 0.78

            r_ttc = float(np.exp(-t_star / self.tau_ttc)) * align_factor * closing_factor * cluster_scale
            r_prox = (
                float(np.exp(-max(dist / rad - 1.0, 0.0) / self.tau_dist))
                * closing_factor
                * align_factor
                * cluster_scale
            )
            best = max(best, min(1.0, r_ttc), min(1.0, r_prox))

        n_veh = sum(1 for o in objs if o[3] in VEHICLES)
        density = min(1.0, n_veh / self.density_ref)
        return float(np.clip((1.0 - self.w_density) * best + self.w_density * density, 0.0, 1.0))

    def _smooth(self) -> float:
        a = self.a_up if self.raw > self.value else self.a_down
        self.value = a * self.raw + (1 - a) * self.value
        return float(np.clip(self.value, 0.0, 1.0))
