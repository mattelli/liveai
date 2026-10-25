@echo off
cd /d "%~dp0"
call ..\.liveai-cpu\Scripts\activate
echo Loading Agent ...
python ..\main.py --config ..\config-chat.toml %*
@echo on
pause