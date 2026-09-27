"""Trafik hodisalarini aniqlash va onlayn xavf baholash paketi.

Muhim: ultralytics import qilinishidan OLDIN oflayn rejim yoqiladi —
dastur internetga hech qachon murojaat qilmaydi.
"""
import os

os.environ.setdefault("YOLO_OFFLINE", "1")          # ultralytics: onlayn tekshiruvlarni o'chirish
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # CUDA determinizmi uchun
