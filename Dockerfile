# WIUT Hackathon 2026 — CV Track Offline Evaluation Container
# Build: docker build -t team .
# Run:   docker run --rm --network none -v /data/test:/data/test team python run_submission.py --videos /data/test --out predictions.json
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -m pytest -q tests/test_core.py

CMD ["python", "run_submission.py", "--videos", "samples", "--out", "predictions.json"]
