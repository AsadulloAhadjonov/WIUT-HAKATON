#!/usr/bin/env bash
# weights/download.sh — Ensures model weights are present in weights/ (<= 5 GB).
# Note: yolov8s.pt (22 MB) and yolov8n.pt (6.2 MB) are already shipped directly inside weights/.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR"

if [ -f "$DIR/yolov8s.pt" ] && [ -f "$DIR/yolov8n.pt" ]; then
    echo "[OK] Model weights (yolov8s.pt, yolov8n.pt) are already present in $DIR."
    exit 0
fi

echo "Downloading YOLOv8 weights..."
if [ ! -f "$DIR/yolov8s.pt" ]; then
    curl -L -o "$DIR/yolov8s.pt" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt"
fi
if [ ! -f "$DIR/yolov8n.pt" ]; then
    curl -L -o "$DIR/yolov8n.pt" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
fi
echo "[OK] Weights downloaded to $DIR."
