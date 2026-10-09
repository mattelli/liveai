@echo off
cd /d "%~dp0"
call ..\.liveai-cpu\Scripts\activate
echo Loading Agent ...
python ..\main-chat.py %*
@echo on
pause