"""To'liq (sekin, ~1 daqiqa CPU'da) test: haqiqiy YOLO + ByteTrack + qoidalar sintetik videoda.

    RUN_E2E=1 python -m pytest -q tests/test_e2e.py
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(os.environ.get("RUN_E2E") != "1", reason="RUN_E2E=1 bilan yoqiladi")


def test_synthetic_video_end_to_end(tmp_path):
    video = tmp_path / "synthetic.mp4"
    subprocess.run([sys.executable, str(ROOT / "tests/make_synthetic_video.py"), "--out", str(video)], check=True)
    env = {
        **os.environ,
        "TRAFFIC_PARAMS": "configs/params_synthetic.yaml",
        "TRAFFIC_SCENE": "configs/scene_synthetic.yaml",
        "TRAFFIC_STRICT": "1",
    }
    code = (
        "import json, solution; solution._cfg()['use_cache'] = False; "
        f"evts = solution.detect_events({str(video)!r}); "
        "print(json.dumps(evts))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True,
                         capture_output=True, text=True).stdout
    import json

    events_raw = json.loads(out.strip().splitlines()[-1])
    # Format: [[start, end, label], ...]
    assert isinstance(events_raw, list), f"Kutilgan ro'yxat, keldi: {type(events_raw)}"
    
    # Dict yoki list formatini qo'llab-quvvatlash
    def get_cls(e):
        if isinstance(e, dict):
            return e.get("class") or e.get("label") or e.get("cls")
        elif isinstance(e, (list, tuple)) and len(e) >= 3:
            return str(e[2])
        return None
    
    def get_start(e):
        if isinstance(e, dict):
            return float(e.get("start", 0))
        elif isinstance(e, (list, tuple)):
            return float(e[0])
        return 0.0
    
    classes = {get_cls(e) for e in events_raw}
    assert "stopped_vehicle" in classes, f"stopped_vehicle topilmadi. Topilgan: {classes}"
    assert "wrong_way" in classes, f"wrong_way topilmadi. Topilgan: {classes}"
    
    stop = next((e for e in events_raw if get_cls(e) == "stopped_vehicle"), None)
    assert stop is not None
    assert abs(get_start(stop) - 3.0) < 2.0, f"stopped_vehicle start xato: {get_start(stop)}"
