"""Sahna tuzilishi (camera.md dan): zonalar, polosalar, chiziqlar, ixtiyoriy gomografiya.

configs/scene.yaml formati:

    image_size: [1920, 1080]        # koordinatalar qaysi o'lchamdagi kadrda olingan
    zones:                          # nom -> poligon [[x, y], ...]
      roadway: [[0, 400], [1920, 400], [1920, 1080], [0, 1080]]
      crosswalk: [...]
    lanes:                          # polosa: poligon + harakat yo'nalishi (strelka boshi -> uchi)
      - id: north_1
        polygon: [[...], ...]
        direction: [[x1, y1], [x2, y2]]
    lines:                          # nom -> kesma [[x1, y1], [x2, y2]]
      stop_line: [[...], [...]]
    homography:                     # ixtiyoriy: 4 ta nuqta rasmda va yerda (metr)
      image: [[..], [..], [..], [..]]
      world: [[..], [..], [..], [..]]

Video o'lchami image_size dan farq qilsa, koordinatalar avtomatik masshtablanadi.
Bo'sh sahna ham ishlaydi: zona so'ralgan qoida butun kadrda ishlaydi.
"""
from __future__ import annotations

import cv2
import numpy as np


class Scene:
    def __init__(self, spec: dict | None, frame_size: tuple[int, int] | None = None):
        spec = spec or {}
        ref = spec.get("image_size")
        sx = sy = 1.0
        if ref and frame_size:
            sx, sy = frame_size[0] / ref[0], frame_size[1] / ref[1]
        self._s = np.array([sx, sy], np.float32)

        def pts(p):
            return np.asarray(p, np.float32).reshape(-1, 2) * self._s

        # Bo'sh (<3 nuqta) poligonlarni o'tkazib yuboring — cv2.pointPolygonTest crash beradi
        self.zones = {k: pts(v) for k, v in (spec.get("zones") or {}).items()
                      if pts(v).shape[0] >= 3}
        self.lanes = []
        for ln in spec.get("lanes") or []:
            d = pts(ln["direction"])
            v = d[1] - d[0]
            self.lanes.append({"id": str(ln["id"]), "poly": pts(ln["polygon"]),
                               "dir": v / (np.linalg.norm(v) + 1e-9)})
        self.lines = {k: pts(v) for k, v in (spec.get("lines") or {}).items()}

        self.H = None
        hg = spec.get("homography")
        if hg and hg.get("image") and hg.get("world"):
            self.H, _ = cv2.findHomography(pts(hg["image"]), np.asarray(hg["world"], np.float32))

    # ------------------------------------------------------------------ geometriya
    @staticmethod
    def _inside(poly: np.ndarray, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def in_zone(self, name: str, x: float, y: float) -> bool:
        """Zona e'lon qilinmagan bo'lsa — True (butun kadr)."""
        poly = self.zones.get(name)
        return True if poly is None else self._inside(poly, x, y)

    def in_any(self, names: list[str] | None, x: float, y: float) -> bool:
        if not names:
            return True
        return any(self.in_zone(n, x, y) for n in names)

    def has_zones(self, names: list[str] | None) -> bool:
        return bool(names) and all(n in self.zones for n in names)

    def lane_of(self, x: float, y: float) -> int:
        """Nuqta qaysi polosada: self.lanes indeksi yoki -1."""
        for i, ln in enumerate(self.lanes):
            if self._inside(ln["poly"], x, y):
                return i
        return -1

    def to_world(self, xy: np.ndarray) -> np.ndarray:
        """[N, 2] piksel → metr (gomografiya bo'lsa), aks holda o'zgarishsiz."""
        if self.H is None or len(xy) == 0:
            return xy
        p = cv2.perspectiveTransform(xy.reshape(-1, 1, 2).astype(np.float32), self.H)
        return p.reshape(-1, 2)

    @property
    def metric(self) -> bool:
        return self.H is not None


def segments_intersect(p1, p2, q1, q2) -> bool:
    """[p1, p2] va [q1, q2] kesmalari kesishadimi (chiziq kesish qoidasi uchun)."""
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)
    return o1 != o2 and o3 != o4 and 0 not in (o1, o2, o3, o4)


def draw(scene: Scene, img: np.ndarray) -> np.ndarray:
    """Sahnani kadr ustiga chizadi (tekshirish uchun: scripts/draw_scene.py)."""
    out = img.copy()
    palette = [(0, 200, 255), (255, 120, 0), (0, 255, 0), (255, 0, 255), (0, 0, 255), (255, 255, 0)]
    for i, (name, poly) in enumerate(scene.zones.items()):
        c = palette[i % len(palette)]
        cv2.polylines(out, [poly.astype(np.int32)], True, c, 2)
        cv2.putText(out, name, tuple(poly[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    for ln in scene.lanes:
        poly = ln["poly"]
        cv2.polylines(out, [poly.astype(np.int32)], True, (255, 255, 255), 1)
        c = poly.mean(0)
        tip = c + ln["dir"] * 60
        cv2.arrowedLine(out, tuple(c.astype(int)), tuple(tip.astype(int)), (255, 255, 255), 3, tipLength=0.3)
        cv2.putText(out, ln["id"], tuple(c.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for name, seg in scene.lines.items():
        cv2.line(out, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (0, 0, 255), 3)
        cv2.putText(out, name, tuple(seg[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out
