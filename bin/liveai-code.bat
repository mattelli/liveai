@echo off
call %LIVEAI_DIR%\.liveai-cpu\Scripts\activate
echo Loading Agent ...
python %LIVEAI_DIR%\main.py --config %LIVEAI_DIR%\config-code.toml %*
if errorlevel 1 (
    echo Agent exited with error.
    exit /b 1
)
@echo on
pause