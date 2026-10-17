@echo off
REM Optional developer-only diagnostic; GitHub uploading does not require execution.
setlocal
python run_reproduction.py --mode quick --output-dir outputs\local
if errorlevel 1 (echo Diagnostic run failed.& pause & exit /b 1)
echo Local diagnostic outputs were generated under outputs\local.
pause
