"""Segmentlarni tozalash: bir xil sinfdagi yaqin bo'laklarni birlashtirish, qisqalarini tashlash,
video chegarasiga qirqish. Shuningdek, 'accident' (kontakt bor) sodir bo'lgan joyda
uning oldidan chiqqan 'near_miss' (kontakt yo'q) dublikatlarini tozalash.
"""
from __future__ import annotations


def _params(cfg: dict, cls: str) -> dict:
    pp = cfg.get("postprocess", {})
    return {**pp.get("default", {}), **(pp.get("per_class", {}) or {}).get(cls, {})}


def postprocess(segments: list[dict], cfg: dict, duration: float | None = None) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for s in segments:
        groups.setdefault(s["cls"], []).append(s)

    out = []
    for cls in sorted(groups):
        p = _params(cfg, cls)
        gap = float(p.get("merge_gap_s", 1.0))
        min_len = float(p.get("min_len_s", 1.0))
        max_len = p.get("max_len_s")
        pad = float(p.get("pad_s", 0.0))

        segs = sorted(groups[cls], key=lambda s: (s["start"], s["end"]))
        merged = [dict(segs[0])]
        for s in segs[1:]:
            cur = merged[-1]
            if s["start"] - cur["end"] <= gap:
                cur["end"] = max(cur["end"], s["end"])
                cur["score"] = max(cur.get("score", 1.0), s.get("score", 1.0))
                cur["tracks"] = sorted(set(cur.get("tracks", [])) | set(s.get("tracks", [])))
            else:
                merged.append(dict(s))

        for s in merged:
            s["start"] = max(0.0, s["start"] - pad)
            s["end"] = s["end"] + pad
            if duration is not None:
                s["end"] = min(s["end"], duration)
            if max_len is not None:
                s["end"] = min(s["end"], s["start"] + float(max_len))
            if s["end"] - s["start"] >= min_len:
                out.append(s)

    # PDF: near_miss = "no contact". Agar accident (collision) bo'lsa, u bilan ustma-ust yoki
    # bevosita undan oldin (10s ichida) chiqqan near_miss segmentlari aslida accident'ning bir qismi.
    accidents = [s for s in out if s["cls"] == "accident"]
    if accidents:
        filtered = []
        for s in out:
            if s["cls"] == "near_miss":
                if any((acc["start"] - 12.5) <= s["end"] and s["start"] <= (acc["end"] + 2.0) for acc in accidents):
                    continue
            filtered.append(s)
        out = filtered

    return sorted(out, key=lambda s: (s["start"], s["cls"], s["end"]))
