@echo off
call %LIVEAI_DIR%\.liveai-cpu\Scripts\activate
echo --------------------------------------------------
echo Local Model Runner
echo Interactive menu: pick options while the terminal
echo stays open. Quit with q or Ctrl+C.
echo --------------------------------------------------
python %LIVEAI_DIR%\samples\local_model_runner.py -i %*
pause
