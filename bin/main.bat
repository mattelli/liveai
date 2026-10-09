@echo off
cd /d "%~dp0"
call ..\.liveai-cpu\Scripts\activate
echo Loading Agent ...
python ..\main.py %*
@echo on
pause