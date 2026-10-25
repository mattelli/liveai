@echo off
cd /d "%~dp0"
call ..\.liveai-cpu\Scripts\activate
echo Loading Agent ...
python ..\main.py --config ..\config-chat.toml %*
if errorlevel 1 (
    echo Agent exited with error.
    exit /b 1
)
@echo on
pause