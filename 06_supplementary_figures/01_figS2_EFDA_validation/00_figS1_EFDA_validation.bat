@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM ============================================================
REM Fig. S1 EFDA validation workflow
REM
REM Edit hard-coded EFDA/output paths inside the Python scripts if needed.
REM ============================================================

set "CONFIG_BAT=%~dp0..\..\00_project_config\00_config_paths.bat"
if not exist "%CONFIG_BAT%" (
  echo ERROR: 00_config_paths.bat not found:
  echo %CONFIG_BAT%
  exit /b 1
)
call "%CONFIG_BAT%"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ============================================================
echo Fig. S1 EFDA validation: 06F0 download
echo ============================================================
"%PYTHON_EXE%" "%EFDA_06F0_SCRIPT%" > "%LOG_DIR%\12_figS1_EFDA_06F0_download.log" 2>&1
if errorlevel 1 (
  echo WARNING: 06F0 failed. This may be expected if EFDA files were already downloaded manually.
  type "%LOG_DIR%\12_figS1_EFDA_06F0_download.log"
)

echo ============================================================
echo Fig. S1 EFDA validation: 06F1 inspect
echo ============================================================
"%PYTHON_EXE%" "%EFDA_06F1_SCRIPT%" > "%LOG_DIR%\12_figS1_EFDA_06F1_inspect.log" 2>&1
if errorlevel 1 (
  echo ERROR: 06F1 failed. See log:
  echo %LOG_DIR%\12_figS1_EFDA_06F1_inspect.log
  type "%LOG_DIR%\12_figS1_EFDA_06F1_inspect.log"
  exit /b 1
)

echo ============================================================
echo Fig. S1 EFDA validation: 06F2 aggregate to hex-year
echo ============================================================
"%PYTHON_EXE%" "%EFDA_06F2_SCRIPT%" > "%LOG_DIR%\12_figS1_EFDA_06F2_aggregate.log" 2>&1
if errorlevel 1 (
  echo ERROR: 06F2 failed. See log:
  echo %LOG_DIR%\12_figS1_EFDA_06F2_aggregate.log
  type "%LOG_DIR%\12_figS1_EFDA_06F2_aggregate.log"
  exit /b 1
)

echo ============================================================
echo Fig. S1 EFDA validation: 06F3 compare prediction with EFDA
echo ============================================================
"%PYTHON_EXE%" "%EFDA_06F3_SCRIPT%" > "%LOG_DIR%\12_figS1_EFDA_06F3_compare.log" 2>&1
if errorlevel 1 (
  echo ERROR: 06F3 failed. See log:
  echo %LOG_DIR%\12_figS1_EFDA_06F3_compare.log
  type "%LOG_DIR%\12_figS1_EFDA_06F3_compare.log"
  exit /b 1
)

echo Done: Fig. S1 EFDA validation workflow finished.
exit /b 0
