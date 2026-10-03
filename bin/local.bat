@echo off
cd /d "%~dp0"
call ..\.lms\Scripts\activate
echo --------------------------------------------------
echo Local Model Runner
echo Interactive menu: pick options while the terminal
echo stays open. Quit with q or Ctrl+C.
echo --------------------------------------------------
python ..\samples\local_model_runner.py -i %*
pause
