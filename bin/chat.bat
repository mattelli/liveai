@echo off
cd /d "%~dp0"
call ..\.liveai-cpu\Scripts\activate
echo --------------------------------------------------
echo Loading Agent ...
echo --------------------------------------------------
python ..\main-chat.py %*
@echo on
pause