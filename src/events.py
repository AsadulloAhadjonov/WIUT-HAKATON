"""Hodisa qoidalari (Part A).

Har bir qoida — "primitiv". params.yaml -> `events` ro'yxatida har bir primitiv
starter kitdagi CLASSES nomiga bog'lanadi.

Har qoida segmentlar ro'yxatini qaytaradi:
    {"cls": str, "start": float, "end": float, "score": float, "tracks": [int, ...]}
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .features import bbox_iou, closest_approach, runs
from .scene import Scene, segments_intersect
from .utils import log


# --------------------------------------------------------------------------- yordamchilar
def _sel(feat: pd.DataFrame, objects: list[str] | None) -> pd.DataFrame:
    return feat if not objects else feat[feat["cls"].isin(objects)]


def _seg(cls, t0, t1, tracks, score=1.0) -> dict:
    return {
        "cls": cls,
        "start": float(t0),
        "end": float(t1),
        "score": float(score),
        "tracks": sorted(int(x) for x in tracks),
    }


def _track_mask_runs(feat, r, mask_fn) -> list[dict]:
    """Har trek bo'yicha mask_fn(trek_df) -> bool massiv; davomiyligi yetarli bo'laklar -> segmentlar."""
    min_dur = float(r.get("min_duration_s", 1.0))
    max_gap = float(r.get("max_gap_s", 0.5))
    out = []
    for tid, tr in _sel(feat, r.get("objects")).groupby("track_id", sort=True):
        tr = tr.sort_values("t")
        mask = np.asarray(mask_fn(tr), bool)
        t = tr["t"].to_numpy()
        gaps = np.diff(t, prepend=t[0]) > max_gap
        mask = mask & ~gaps
        for a, b in runs(mask):
            if t[b] - t[a] >= min_dur:
                out.append(_seg(r["class"], t[a], t[b], [tid]))
    return out


def _need_zones(scene: Scene, r: dict) -> bool:
    z = r.get("zones")
    if z and not scene.has_zones(z):
        log.warning("'%s': scene.yaml da %s zonalari yo'q — qoida o'tkazib yuborildi", r["class"], z)
        return False
    return True


# --------------------------------------------------------------------------- primitivlar
def rule_stopped(feat, scene, r, ctx):
    """Obyekt zonada max_speed dan sekin, kamida min_duration_s davomida (signal_queue dan tashqari)."""
    if not _need_zones(scene, r):
        return []
    vmax = float(r.get("max_speed", 0.3))
    excl = [z for z in r.get("exclude_zones", []) if z in scene.zones]

    def mask(tr):
        in_z = np.array(
            [
                scene.in_any(r.get("zones"), x, y) and not (excl and scene.in_any(excl, x, y))
                for x, y in zip(tr.px, tr.py)
            ],
            dtype=bool,
        )
        return (tr["speed"].to_numpy() < vmax) & in_z

    return _track_mask_runs(feat, r, mask)


def rule_in_zone(feat, scene, r, ctx):
    """Obyekt zonada (va exclude_zones da emas), ixtiyoriy min_speed va min_disp bilan."""
    if not r.get("zones"):
        log.warning("'%s': in_zone uchun zones kerak — o'tkazib yuborildi", r["class"])
        return []
    if not _need_zones(scene, r):
        return []
    excl = [z for z in r.get("exclude_zones", []) if z in scene.zones]
    vmin = float(r.get("min_speed", 0.0))

    def mask(tr):
        inside = np.array(
            [
                scene.in_any(r["zones"], x, y) and not (excl and scene.in_any(excl, x, y))
                for x, y in zip(tr.px, tr.py)
            ],
            dtype=bool,
        )
        return inside & (tr["speed"].to_numpy() >= vmin)

    return _track_mask_runs(feat, r, mask)


def rule_wrong_way(feat, scene, r, ctx):
    """Polosa yo'nalishiga qarshi harakat. scene.yaml da lanes kerak."""
    if not scene.lanes:
        log.warning("'%s': scene.yaml da lanes yo'q — o'tkazib yuborildi", r["class"])
        return []
    vmin = float(r.get("min_speed", 0.5))
    cos_thr = float(r.get("cos_threshold", 0.5))

    def mask(tr):
        lane = tr["lane"].to_numpy()
        v = tr[["pvx", "pvy"]].to_numpy()
        n = np.linalg.norm(v, axis=1) + 1e-9
        cos = np.full(len(tr), 1.0)
        ok = lane >= 0
        if ok.any():
            dirs = np.stack([scene.lanes[i]["dir"] for i in lane[ok]])
            cos[ok] = (v[ok] * dirs).sum(1) / n[ok]
        return ok & (cos < -cos_thr) & (tr["speed"].to_numpy() > vmin)

    return _track_mask_runs(feat, r, mask)


def rule_speeding(feat, scene, r, ctx):
    """Tezlik max_speed dan yuqori."""
    if not _need_zones(scene, r):
        return []
    vmax = float(r["max_speed"])
    return _track_mask_runs(
        feat,
        r,
        lambda tr: (tr["speed"].to_numpy() > vmax)
        & np.array([scene.in_any(r.get("zones"), x, y) for x, y in zip(tr.px, tr.py)]),
    )


def rule_sudden_brake(feat, scene, r, ctx):
    """Keskin sekinlashish: tezlik hosilasi -max_decel dan past."""
    dmax = float(r.get("max_decel", 1.5))

    def mask(tr):
        s, t = tr["speed"].to_numpy(), tr["t"].to_numpy()
        if len(s) < 3:
            return np.zeros(len(s), bool)
        return np.gradient(s, t) < -dmax

    r = {"min_duration_s": 0.2, **r}
    return _track_mask_runs(feat, r, mask)


def rule_lane_change(feat, scene, r, ctx):
    """Polosa ID si o'zgaradi; o'zgarishdan oldin va keyin min_stable_s davomida barqaror."""
    if not scene.lanes:
        log.warning("'%s': scene.yaml da lanes yo'q — o'tkazib yuborildi", r["class"])
        return []
    stable = float(r.get("min_stable_s", 1.0))
    pre, post = float(r.get("pre_s", 1.0)), float(r.get("post_s", 1.0))
    out = []
    for tid, tr in _sel(feat, r.get("objects")).groupby("track_id"):
        tr = tr[tr["lane"] >= 0].sort_values("t")
        lane, t = tr["lane"].to_numpy(), tr["t"].to_numpy()
        n = len(lane)
        for i in np.flatnonzero(np.diff(lane) != 0) + 1:
            s = i - 1
            while s > 0 and lane[s - 1] == lane[i - 1]:
                s -= 1
            e = i
            while e + 1 < n and lane[e + 1] == lane[i]:
                e += 1
            if t[i - 1] - t[s] >= stable and t[e] - t[i] >= stable:
                out.append(_seg(r["class"], max(0.0, t[i] - pre), t[i] + post, [tid]))
    return out


def rule_line_cross(feat, scene, r, ctx):
    """Trek chiziqni kesib o'tadi (ixtiyoriy yo'nalish: direction: +1 yoki -1)."""
    line = scene.lines.get(r.get("line", ""))
    if line is None:
        log.warning("'%s': scene.yaml da '%s' chizig'i yo'q — o'tkazib yuborildi", r["class"], r.get("line"))
        return []
    want = r.get("direction")
    pre, post = float(r.get("pre_s", 0.5)), float(r.get("post_s", 0.5))
    a, b = line
    out = []
    for tid, tr in _sel(feat, r.get("objects")).groupby("track_id"):
        tr = tr.sort_values("t")
        P = tr[["px", "py"]].to_numpy()
        t = tr["t"].to_numpy()
        for i in range(1, len(P)):
            if segments_intersect(P[i - 1], P[i], a, b):
                side = np.sign((b[0] - a[0]) * (P[i][1] - a[1]) - (b[1] - a[1]) * (P[i][0] - a[0]))
                if want is None or side == int(want):
                    out.append(_seg(r["class"], max(0.0, t[i] - pre), t[i] + post, [tid]))
    return out


def rule_u_turn(feat, scene, r, ctx):
    """Harakat yo'nalishi window_s ichida min_angle_deg dan ko'proq o'zgaradi va haqiqiy masofa bosib o'tiladi."""
    vmin = float(r.get("min_speed", 0.35))
    win = float(r.get("window_s", 6.0))
    ang = np.deg2rad(float(r.get("min_angle_deg", 150)))
    min_leg_sizes = float(r.get("min_leg_sizes", 1.2))
    out = []
    for tid, tr in _sel(feat, r.get("objects")).groupby("track_id"):
        tr = tr[tr["speed"] > vmin].sort_values("t")
        if len(tr) < 6:
            continue
        t = tr["t"].to_numpy()
        px = tr["px"].to_numpy()
        py = tr["py"].to_numpy()
        sz = float(tr["size"].median()) if "size" in tr else 50.0
        h = np.unwrap(np.arctan2(tr["pvy"].to_numpy(), tr["pvx"].to_numpy()))
        j = 0
        for i in range(len(t)):
            while t[i] - t[j] > win:
                j += 1
            if i - j < 4:
                continue
            seg = h[j : i + 1]
            if abs(seg[-1] - seg[0]) >= ang:
                mid = (j + i) // 2
                d1 = np.hypot(px[mid] - px[j], py[mid] - py[j])
                d2 = np.hypot(px[i] - px[mid], py[i] - py[mid])
                if d1 >= min_leg_sizes * sz and d2 >= min_leg_sizes * sz:
                    out.append(_seg(r["class"], t[j], t[i], [tid]))
                    break
    return out


def rule_congestion(feat, scene, r, ctx):
    """Sahna darajasidagi hodisa: zonada kamida min_count ta sekin obyekt, min_duration_s davomida."""
    if not _need_zones(scene, r):
        return []
    vmax = float(r.get("max_speed", 0.3))
    nmin = int(r.get("min_count", 5))
    excl = [z for z in r.get("exclude_zones", []) if z in scene.zones]
    f = _sel(feat, r.get("objects"))
    in_z = np.array(
        [
            scene.in_any(r.get("zones"), x, y) and not (excl and scene.in_any(excl, x, y))
            for x, y in zip(f.px, f.py)
        ],
        dtype=bool,
    )
    f = f[(f["speed"] < vmax) & in_z]
    frames = np.asarray(ctx["frames"])
    counts = f.groupby("frame")["track_id"].nunique().reindex(frames, fill_value=0).to_numpy()
    t = frames / ctx["fps"]
    out = []
    for a, b in runs(counts >= nmin):
        if t[b] - t[a] >= float(r.get("min_duration_s", 10.0)):
            out.append(_seg(r["class"], t[a], t[b], [], score=float(counts[a : b + 1].mean())))
    return out


def _pairs_by_frame(feat, groups_a, groups_b):
    """Har kadr uchun (A-guruh, B-guruh) juftliklari."""
    for frame, fr in feat.groupby("frame", sort=True):
        A = fr[fr["cls"].isin(groups_a)]
        B = fr[fr["cls"].isin(groups_b)]
        if A.empty or B.empty:
            continue
        same = set(groups_a) == set(groups_b)
        rows_a = list(A.itertuples(index=False))
        rows_b = list(B.itertuples(index=False))
        if same:
            for p, q in combinations(rows_a, 2):
                yield frame, p, q
        else:
            for p in rows_a:
                for q in rows_b:
                    if p.track_id != q.track_id:
                        yield frame, p, q


def _pair_segments(
    hits: dict,
    fps_sample: float,
    cls: str,
    min_dur: float,
    pad: float,
    max_gap: float,
    duration: float,
):
    """hits: {(id1, id2): [t, ...]} -> segmentlar (yaqin nuqtalar birlashtiriladi)."""
    out = []
    for pair, ts in sorted(hits.items()):
        ts = sorted(ts)
        s = e = ts[0]
        for t in ts[1:] + [None]:
            if t is not None and t - e <= max_gap:
                e = t
                continue
            if e - s + 1.0 / fps_sample >= min_dur:
                out.append(_seg(cls, max(0.0, s - pad), min(duration, e + pad), pair))
            if t is not None:
                s = e = t
    return out


def rule_near_miss(feat, scene, r, ctx):
    """Xavfli yaqinlashish: TTC < ttc_max va eng yaqin masofa < margin * (o'lchamlar yig'indisi / 2),
    lekin bbox'lar hali to'qnashmagan.
    """
    ga = r.get("objects_a", ["car", "bus", "truck", "motorcycle"])
    gb = r.get("objects_b", ga)
    ttc_max = float(r.get("ttc_max_s", 1.5))
    margin = float(r.get("margin", 0.75))
    near = float(r.get("max_dist", 3.0))
    vmin = float(r.get("min_speed", 0.35))
    iou_contact = float(r.get("iou_contact", 0.05))
    hits: dict = {}
    for _, p, q in _pairs_by_frame(feat, ga, gb):
        if max(p.speed, q.speed) < vmin:
            continue
        # Agar ikkala obyekt bir xil polosada emas va qarama-qarshi parallel polosada bo'lsa
        if getattr(p, "lane", -1) >= 0 and getattr(q, "lane", -1) >= 0 and p.lane != q.lane:
            # Polosalar kesishmayotgan parallel bo'lsa
            d1 = scene.lanes[p.lane]["dir"]
            d2 = scene.lanes[q.lane]["dir"]
            if abs(float(d1 @ d2)) > 0.85:
                continue
        rad = (p.size + q.size) / 2
        rel_p = np.array([q.x - p.x, q.y - p.y])
        if np.linalg.norm(rel_p) > near * rad:
            continue
        if bbox_iou(np.array([p.x1, p.y1, p.x2, p.y2]), np.array([q.x1, q.y1, q.x2, q.y2])) > iou_contact:
            continue
        t_star, d_star = closest_approach(rel_p, np.array([q.vx - p.vx, q.vy - p.vy]))
        if 0 < t_star < ttc_max and d_star < margin * rad:
            key = tuple(sorted((int(p.track_id), int(q.track_id))))
            hits.setdefault(key, []).append(p.t)
    return _pair_segments(
        hits,
        ctx["fps_sample"],
        r["class"],
        float(r.get("min_duration_s", 0.3)),
        float(r.get("pad_s", 0.8)),
        float(r.get("max_gap_s", 0.5)),
        ctx["duration"],
    )


def rule_accident(feat, scene, r, ctx):
    """To'qnashuv: bbox'lar kesishadi (IoU > iou_min), undan oldin kamida bittasi harakatda edi,
    keyin ikkalasi ham post_stop_s davomida deyarli to'xtaydi.
    Start = birinchi kontakt kadri; End = obyektlar yana harakatga kelgan yoki kadrdan chiqqan vaqt.
    """
    ga = r.get("objects_a", ["car", "bus", "truck", "motorcycle"])
    gb = r.get("objects_b", ga + ["person", "bicycle"])
    iou_min = float(r.get("iou_min", 0.10))
    v_before = float(r.get("min_speed_before", 0.45))
    v_stop = float(r.get("stop_speed", 0.15))
    post_stop = float(r.get("post_stop_s", 2.0))
    pre = float(r.get("pre_s", 0.0))
    excl = [z for z in r.get("exclude_zones", []) if z in scene.zones]
    by_track = {tid: tr.sort_values("t") for tid, tr in feat.groupby("track_id")}

    def stops_after(tid, t0):
        tr = by_track[tid]
        after = tr[(tr["t"] >= t0) & (tr["t"] <= t0 + post_stop + 1.0)]
        return (
            len(after) > 2
            and after["t"].max() - t0 >= post_stop * 0.8
            and (after["speed"] < v_stop).mean() > 0.7
        )

    def collision_end_time(tid1, tid2, t0):
        t_end = t0 + post_stop
        for tid in (tid1, tid2):
            tr = by_track[tid]
            after = tr[tr["t"] >= t0]
            for row in after.itertuples(index=False):
                if row.speed < v_stop * 1.5:
                    t_end = max(t_end, float(row.t))
                elif row.t > t0 + post_stop:
                    break
        return min(ctx["duration"], t_end)

    def moving_before(tid, t0):
        tr = by_track[tid]
        before = tr[(tr["t"] < t0) & (tr["t"] >= t0 - 2.0)]
        return len(before) > 0 and before["speed"].max() > v_before

    out, seen = [], set()
    for _, p, q in _pairs_by_frame(feat, ga, gb):
        key = tuple(sorted((int(p.track_id), int(q.track_id))))
        if key in seen:
            continue
        if excl and (scene.in_any(excl, p.px, p.py) or scene.in_any(excl, q.px, q.py)):
            continue
        if bbox_iou(np.array([p.x1, p.y1, p.x2, p.y2]), np.array([q.x1, q.y1, q.x2, q.y2])) < iou_min:
            continue
        # Yerga tegish nuqtalari orasidagi masofa juda uzoq bo'lmasligi kerak (perspektiva filtri)
        rad = (p.size + q.size) / 2.0
        if np.hypot(p.px - q.px, p.py - q.py) > 1.35 * rad:
            continue
        if not (moving_before(p.track_id, p.t) or moving_before(q.track_id, p.t)):
            continue
        if stops_after(p.track_id, p.t) and stops_after(q.track_id, p.t):
            seen.add(key)
            t_end = collision_end_time(p.track_id, q.track_id, p.t)
            out.append(_seg(r["class"], max(0.0, p.t - pre), t_end, key))
    return out


def rule_failure_to_yield(feat, scene, r, ctx):
    """Piyoda o'tish joyida (crosswalk) turganda transport vositasi unga yaqin masofada kesib o'tadi."""
    zones = [z for z in r.get("zones", ["crosswalk"]) if z in scene.zones]
    if not zones:
        log.warning("'%s': scene.yaml da crosswalk zonasi yo'q — o'tkazib yuborildi", r["class"])
        return []

    veh_cls = r.get("objects_vehicles", ["car", "bus", "truck", "motorcycle"])
    ped_cls = r.get("objects_pedestrians", ["person"])
    min_dur = float(r.get("min_duration_s", 0.4))
    veh_min_speed = float(r.get("veh_min_speed", 0.25))
    max_dist_factor = float(r.get("max_dist_factor", 3.5))

    hits: dict = {}
    for _, fr in feat.groupby("frame", sort=True):
        if fr.empty:
            continue
        t = float(fr["t"].iloc[0])
        vehs = fr[fr["cls"].isin(veh_cls)]
        peds = fr[fr["cls"].isin(ped_cls)]
        if vehs.empty or peds.empty:
            continue

        peds_in_cross = peds[[scene.in_any(zones, x, y) for x, y in zip(peds.px, peds.py)]]
        if peds_in_cross.empty:
            continue

        for v in vehs.itertuples(index=False):
            if v.speed < veh_min_speed:
                continue
            if not scene.in_any(zones, v.px, v.py):
                continue
            for p in peds_in_cross.itertuples(index=False):
                dist = np.hypot(v.px - p.px, v.py - p.py)
                if dist <= max_dist_factor * max(v.size, p.size):
                    key = (int(v.track_id), int(p.track_id))
                    hits.setdefault(key, []).append(t)

    return _pair_segments(
        hits,
        ctx["fps_sample"],
        r["class"],
        min_dur,
        float(r.get("pad_s", 0.5)),
        float(r.get("max_gap_s", 0.6)),
        ctx["duration"],
    )


def rule_illegal_turn(feat, scene, r, ctx):
    """Taqiqlangan burilish: faqat scene.yaml da 'no_turn_line' chizig'i bo'lsa va u kesib o'tilsa."""
    no_turn_line = scene.lines.get(r.get("line", "no_turn_line"))
    if no_turn_line is None:
        return []

    vmin = float(r.get("min_speed", 0.35))
    turn_angle = float(r.get("turn_angle_deg", 65.0))
    win = float(r.get("window_s", 4.0))
    min_dur = float(r.get("min_duration_s", 0.5))
    ang_rad = np.deg2rad(turn_angle)
    out = []

    for tid, tr in _sel(feat, r.get("objects")).groupby("track_id"):
        tr = tr[tr["speed"] > vmin].sort_values("t")
        if len(tr) < 5:
            continue
        t = tr["t"].to_numpy()
        h = np.unwrap(np.arctan2(tr["pvy"].to_numpy(), tr["pvx"].to_numpy()))
        P = tr[["px", "py"]].to_numpy()
        j = 0
        for i in range(len(t)):
            while t[i] - t[j] > win:
                j += 1
            if abs(h[i] - h[j]) >= ang_rad:
                crossed = any(
                    segments_intersect(P[k - 1], P[k], no_turn_line[0], no_turn_line[1])
                    for k in range(j + 1, i + 1)
                )
                if crossed and (t[i] - t[j] >= min_dur):
                    out.append(_seg(r["class"], max(0.0, t[j]), t[i], [tid]))
                    break
    return out


def rule_solid_line_crossing(feat, scene, r, ctx):
    """Uzluksiz (sidirg'a) chiziqni kesib polosa almashtirish.

    Faqat chiziqqa kichik burchak ostida (lateral lane change) kesib o'tilganda hisoblanadi;
    chorrahada 90 gradusga kesib o'tuvchi ko'ndalang oqim hisoblanmaydi.
    """
    solid_line_name = r.get("line", "solid_line")
    line = scene.lines.get(solid_line_name)

    if line is not None:
        want = r.get("direction")
        pre, post = float(r.get("pre_s", 0.5)), float(r.get("post_s", 1.0))
        a, b = line
        line_vec = b - a
        line_dir = line_vec / (np.linalg.norm(line_vec) + 1e-9)
        out = []
        for tid, tr in _sel(feat, r.get("objects")).groupby("track_id"):
            tr = tr.sort_values("t")
            P = tr[["px", "py"]].to_numpy()
            V = tr[["pvx", "pvy"]].to_numpy()
            t = tr["t"].to_numpy()
            for i in range(1, len(P)):
                if segments_intersect(P[i - 1], P[i], a, b):
                    v_norm = V[i] / (np.linalg.norm(V[i]) + 1e-9)
                    # Chiziq yo'nalishiga deyarli parallel harakatlanib polosa almashtirish (|cos| > 0.5)
                    if abs(float(v_norm @ line_dir)) < 0.5:
                        continue
                    side = np.sign((b[0] - a[0]) * (P[i][1] - a[1]) - (b[1] - a[1]) * (P[i][0] - a[0]))
                    if want is None or side == int(want):
                        out.append(_seg(r["class"], max(0.0, t[i] - pre), t[i] + post, [tid]))
                        break
        return out

    if not scene.lanes:
        return []
    return rule_lane_change(feat, scene, {**r, "class": r["class"]}, ctx)


def rule_road_obstacle(feat, scene, r, ctx):
    """Yo'lning qatnov qismida harakatsiz qolgan begona jism yoki hayvon."""
    obs_classes = r.get("objects", ["obstacle", "animal", "dog", "cat", "cow", "horse", "sheep", "suitcase", "backpack"])
    sub = feat[feat["cls"].isin(obs_classes)]
    if sub.empty:
        return []
    if not _need_zones(scene, r):
        return []
    vmax = float(r.get("max_speed", 0.05))
    return _track_mask_runs(
        sub,
        {**r, "objects": obs_classes},
        lambda tr: (tr["speed"].to_numpy() < vmax)
        & np.array([scene.in_any(r.get("zones"), x, y) for x, y in zip(tr.px, tr.py)]),
    )


def rule_fire_smoke(feat, scene, r, ctx):
    """Olov va tutun aniqlash (kontekst orqali)."""
    fire_frames = ctx.get("fire_smoke_frames", [])
    if not fire_frames:
        return []
    fps = ctx.get("fps", 25.0)
    min_dur = float(r.get("min_duration_s", 1.0))
    pad = float(r.get("pad_s", 0.5))
    out = []
    fire_frames_sorted = sorted(fire_frames)
    s = e = fire_frames_sorted[0]
    for f in fire_frames_sorted[1:] + [None]:
        if f is not None and f - e <= 5:
            e = f
            continue
        t0, t1 = s / fps, e / fps
        if t1 - t0 >= min_dur:
            out.append(_seg(r["class"], max(0.0, t0 - pad), t1 + pad, []))
        if f is not None:
            s = e = f
    return out


def rule_red_light(feat, scene, r, ctx):
    """Qizil chiroqda stop chiziqni kesib chorrahaga kirish.

    Qizil chiroq sharti:
      1) ctx['red_light_frames'] da berilgan bo'lsa, YOKI
      2) Chorrahada ko'ndalang (perpendikulyar) oqim faol harakatlanayotgan paytda
         transport stop_line ni kesib chorrahaga kirsa.
    """
    line_name = r.get("line", "stop_line")
    line = scene.lines.get(line_name)
    if line is None:
        return []

    want = r.get("direction")
    pre, post = float(r.get("pre_s", 0.5)), float(r.get("post_s", 2.0))
    vmin = float(r.get("min_speed", 0.4))
    a, b = line
    red_frames = set(ctx.get("red_light_frames", []))

    out = []
    for tid, tr in _sel(feat, r.get("objects")).groupby("track_id"):
        tr = tr.sort_values("t")
        P = tr[["px", "py"]].to_numpy()
        V = tr[["pvx", "pvy"]].to_numpy()
        spd = tr["speed"].to_numpy()
        t = tr["t"].to_numpy()
        frames = tr["frame"].to_numpy()

        for i in range(1, len(P)):
            if spd[i] < vmin:
                continue
            if not segments_intersect(P[i - 1], P[i], a, b):
                continue
            side = np.sign((b[0] - a[0]) * (P[i][1] - a[1]) - (b[1] - a[1]) * (P[i][0] - a[0]))
            if want is not None and side != int(want):
                continue

            is_red = frames[i] in red_frames
            if not is_red:
                # Ko'ndalang oqim tekshiruvi: shu vaqtda chorrahada perpendikulyar harakatdagi mashina bormi?
                v_dir = V[i] / (np.linalg.norm(V[i]) + 1e-9)
                concurrent = feat[(feat["t"] >= t[i] - 1.0) & (feat["t"] <= t[i] + 1.0) & (feat["track_id"] != tid)]
                for other in concurrent.itertuples(index=False):
                    if other.speed < 0.4:
                        continue
                    if "intersection" in scene.zones and not scene.in_zone("intersection", other.px, other.py):
                        continue
                    o_vec = np.array([other.pvx, other.pvy])
                    o_dir = o_vec / (np.linalg.norm(o_vec) + 1e-9)
                    if abs(float(v_dir @ o_dir)) < 0.35:
                        is_red = True
                        break

            if is_red:
                out.append(_seg(r["class"], max(0.0, t[i] - pre), min(ctx["duration"], t[i] + post), [tid]))
                break
    return out


def rule_stop_line(feat, scene, r, ctx):
    """Qizil chiroqda stop-chiziqdan o'tib, chorrahaga kirmasdan to'xtab qolish."""
    line_name = r.get("line", "stop_line")
    line = scene.lines.get(line_name)
    if line is None:
        return []

    want = r.get("direction")
    vmax_stop = float(r.get("stop_speed", 0.08))
    min_stop_s = float(r.get("min_stop_s", 3.0))
    a, b = line
    out = []

    for tid, tr in _sel(feat, r.get("objects")).groupby("track_id"):
        tr = tr.sort_values("t")
        P = tr[["px", "py"]].to_numpy()
        t = tr["t"].to_numpy()
        spd = tr["speed"].to_numpy()

        for i in range(1, len(P)):
            if not segments_intersect(P[i - 1], P[i], a, b):
                continue
            side = np.sign((b[0] - a[0]) * (P[i][1] - a[1]) - (b[1] - a[1]) * (P[i][0] - a[0]))
            if want is not None and side != int(want):
                continue

            # Chiziqni kesib o'tgach darhol to'xtab, kamida min_stop_s davomida turishi kerak
            after = tr.iloc[i:]
            stopped_rows = after[after["speed"] < vmax_stop]
            if len(stopped_rows) >= 4:
                t_stop_start = float(stopped_rows["t"].iloc[0])
                t_stop_end = float(stopped_rows["t"].iloc[-1])
                if t_stop_start - t[i] <= 1.5 and (t_stop_end - t_stop_start) >= min_stop_s:
                    # Chorraha markaziga chuqur kirib ketmagan bo'lishi kerak
                    dist_from_line = np.abs(
                        np.cross(b - a, P[i] - a) / (np.linalg.norm(b - a) + 1e-9)
                    )
                    if dist_from_line < 80.0:
                        out.append(_seg(r["class"], t_stop_start, min(ctx["duration"], t_stop_end), [tid]))
                        break
    return out


# --------------------------------------------------------------------------- qoidalar ro'yxati
RULES = {
    "stopped": rule_stopped,
    "in_zone": rule_in_zone,
    "wrong_way": rule_wrong_way,
    "speeding": rule_speeding,
    "sudden_brake": rule_sudden_brake,
    "lane_change": rule_lane_change,
    "line_cross": rule_line_cross,
    "u_turn": rule_u_turn,
    "congestion": rule_congestion,
    "near_miss": rule_near_miss,
    "accident": rule_accident,
    "failure_to_yield": rule_failure_to_yield,
    "illegal_turn": rule_illegal_turn,
    "solid_line_crossing": rule_solid_line_crossing,
    "road_obstacle": rule_road_obstacle,
    "fire_smoke": rule_fire_smoke,
    "red_light": rule_red_light,
    "stop_line": rule_stop_line,
}


def run_rules(feat: pd.DataFrame, scene: Scene, cfg: dict, ctx: dict) -> list[dict]:
    segs = []
    if feat.empty:
        return segs
    for r in cfg.get("events", []):
        if not r.get("enabled", True):
            continue
        fn = RULES.get(r["rule"])
        if fn is None:
            raise ValueError(f"Noma'lum qoida: {r['rule']}. Mavjud: {sorted(RULES)}")
        found = fn(feat, scene, r, ctx)
        log.debug("%-22s %-12s -> %d", r["class"], r["rule"], len(found))
        segs.extend(found)
    return segs
