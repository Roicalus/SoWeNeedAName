@echo off
rem Double-click: runs the Co-pilot site on http://localhost:8099 and opens the browser.
cd /d "%~dp0"
set PY=
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if not defined PY if exist "%USERPROFILE%\Desktop\SoWeNeedAName-main\.venv\Scripts\python.exe" set PY=%USERPROFILE%\Desktop\SoWeNeedAName-main\.venv\Scripts\python.exe
if not defined PY where py >nul 2>nul && set PY=py -3
if not defined PY set PY=python
%PY% serve.py %*
if errorlevel 1 pause
