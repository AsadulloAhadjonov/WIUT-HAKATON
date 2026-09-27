@echo off
REM Veb-saytni ishga tushirish (Traffic Incident Detection Demo)
REM Bu faylni traffic/ papkasidan ishga tushiring

set PYTHONPATH=F:\Telegram\traffic_day1 (2)\traffic_day1\traffic
set PATH=C:\Users\VELO\AppData\Local\Programs\Python\Python311;C:\Users\VELO\AppData\Local\Programs\Python\Python311\Scripts;%PATH%
set YOLO_OFFLINE=1

echo ======================================
echo  TrafficAI - WIUT Hackathon 2026
echo  Veb-saytni ochmoqda...
echo ======================================
echo.
echo Brauzer avtomatik ochiladi: http://localhost:8501
echo To'xtatish uchun: Ctrl+C
echo.

cd /d "F:\Telegram\traffic_day1 (2)\traffic_day1\traffic"
streamlit run webapp/app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
