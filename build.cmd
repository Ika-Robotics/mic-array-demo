@echo off
setlocal
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /O2 /EHsc /W3 /DUNICODE /D_UNICODE demo.cpp /Fe:MicArrayFocus-Buffered.exe /link mfplat.lib mf.lib mfreadwrite.lib mfuuid.lib ole32.lib user32.lib gdi32.lib msimg32.lib setupapi.lib winmm.lib shell32.lib
exit /b %errorlevel%
