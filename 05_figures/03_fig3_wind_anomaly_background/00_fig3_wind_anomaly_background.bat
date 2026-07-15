@echo off
setlocal EnableExtensions
set "CONFIG_BAT=%~dp0..\00_project_config\00_config_paths.bat"
if not exist "%CONFIG_BAT%" set "CONFIG_BAT=%~dp0..\..\00_project_config\00_config_paths.bat"
if not exist "%CONFIG_BAT%" set "CONFIG_BAT=%~dp0..\..\..\00_project_config\00_config_paths.bat"
if not exist "%CONFIG_BAT%" (
  echo ERROR: 00_config_paths.bat not found.
  echo Tried parent workflow folders from: %~dp0
  exit /b 1
)
call "%CONFIG_BAT%"
if not exist "%FIG3_OUTDIR%" mkdir "%FIG3_OUTDIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ============================================================
echo Drawing Fig. 3: wind-anomaly background
echo Output: %FIG3_OUTDIR%
echo ============================================================
"%PYTHON_EXE%" "%FIG3_SCRIPT%" > "%LOG_DIR%\07_fig3_wind_anomaly_background.log" 2>&1
if errorlevel 1 (
  echo ERROR: Fig. 3 failed. See %LOG_DIR%\07_fig3_wind_anomaly_background.log
  type "%LOG_DIR%\07_fig3_wind_anomaly_background.log"
  exit /b 1
)
echo Done: Fig. 3 outputs are in %FIG3_OUTDIR%
exit /b 0
