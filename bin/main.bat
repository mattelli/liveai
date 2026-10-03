@echo off
cd /d "%~dp0"
call ..\.lms\Scripts\activate
echo --------------------------------------------------
echo Loading Agent ...
echo --------------------------------------------------
python ..\main.py %*
@echo on
pause