@echo off
cd /d "%~dp0"
call ..\.lms\Scripts\activate
echo Server status
call lms ps
echo Loading Agent
python ..\chat.py %*
@echo on
pause