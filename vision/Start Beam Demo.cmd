@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" beam_demo.py %*
if errorlevel 1 pause
