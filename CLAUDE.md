# Loyiha konteksti (Claude uchun — har yangi suhbatda shu faylni bering)

Trafik kamerasi videolari: Part A — `detect_events(video)` 14 sinf bo'yicha vaqt segmentlari;
Part B — `RiskEstimator.step(frame)` onlayn xavf 0..1. Baholash: vaqt IoU (0.5 va 0.7).

## Qat'iy qoidalar
- `run_submission.py` va `evaluate.py` ga TEGMA.
- Dastur OFLAYN ishlaydi: inference paytida internet, API (OpenAI/Gemini/Claude) YO'Q.
  Og'irliklar `weights/` da, `src/__init__.py` YOLO_OFFLINE=1 qiladi. Yangi kutubxona qo'shsang — requirements.txt ga versiya bilan.
- `RiskEstimator.step` faqat berilgan kadrlarni ishlatadi; video faylni o'qimaydi; Part A natijasini/keshini ishlatmaydi.
  Har o'zgarishdan keyin: `python scripts/check_leak.py <video>`.
- Determinizm: seed `configs/params.yaml` da; `python scripts/check_determinism.py samples/`.
- Raqamlar kodda emas, `configs/params.yaml` da. Sahna geometriyasi `configs/scene.yaml` da.

## Tuzilish
- `solution.py` — yupqa adapter (CLASSES, detect_events, RiskEstimator)
- `src/detect_track.py` — YOLO + ByteTrack; `track_video` (Part A, kesh), `OnlineTracker` (Part B)
- `src/scene.py` — zonalar/polosalar/chiziqlar; `src/features.py` — tezlik, TTC, riderlarni olib tashlash
- `src/events.py` — qoida primitivlari (RULES); `src/postprocess.py` — birlashtirish/tozalash
- `src/risk.py` — OnlineRisk (kauzal); `src/io_format.py` — chiqish formati (bitta joy)
- `scripts/` — run_local, quick_eval, labels_to_gt, check_leak, check_determinism, draw_scene
- `tests/` — `python -m pytest -q tests/` (tez), `RUN_E2E=1 ...` (to'liq)

## Ish uslubi
- Faqat so'ralgan faylni o'zgartir, to'liq kodini ber. O'lik kod qoldirma.
- O'zgarishdan keyin: pytest + quick_eval dagi F1@.7 ni oldingi bilan solishtir.
