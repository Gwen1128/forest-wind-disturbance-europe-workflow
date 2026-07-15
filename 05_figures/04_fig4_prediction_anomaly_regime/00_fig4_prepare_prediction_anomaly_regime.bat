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

if not exist "%FIG4_PREP_OUTDIR%" mkdir "%FIG4_PREP_OUTDIR%"
if not exist "%FIG4_PREP_OUTDIR%\long_term" mkdir "%FIG4_PREP_OUTDIR%\long_term"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not exist "%FIG1_INDICATOR_CSV%" (
  echo ERROR: Fig.1 final-domain indicator CSV not found:
  echo %FIG1_INDICATOR_CSV%
  echo Run 05_fig1_spatial_indicators.bat first.
  exit /b 1
)

if not exist "%WIND_OVERLAY_LONG_NC%" (
  echo ERROR: long-term wind-anomaly overlay NetCDF not found:
  echo %WIND_OVERLAY_LONG_NC%
  echo Edit WIND_OVERLAY_LONG_NC in 00_config_paths.bat.
  exit /b 1
)

echo ============================================================
echo Preparing Fig. 4 prediction-anomaly regime table
echo Prediction input: %FIG1_INDICATOR_CSV%
echo Wind anomaly NC : %WIND_OVERLAY_LONG_NC%
echo Output CSV      : %FIG4_MERGED_CSV%
echo ============================================================
"%PYTHON_EXE%" "%FIG4_PREP_SCRIPT%" > "%LOG_DIR%\08_fig4_prepare_prediction_anomaly_regime.log" 2>&1
if errorlevel 1 (
  echo ERROR: Fig. 4 preparation failed. See:
  echo %LOG_DIR%\08_fig4_prepare_prediction_anomaly_regime.log
  type "%LOG_DIR%\08_fig4_prepare_prediction_anomaly_regime.log"
  exit /b 1
)

if not exist "%FIG4_MERGED_CSV%" (
  echo ERROR: Fig. 4 preparation finished but expected output was not created:
  echo %FIG4_MERGED_CSV%
  exit /b 1
)

echo Done: Fig. 4 merged CSV:
echo %FIG4_MERGED_CSV%
exit /b 0
