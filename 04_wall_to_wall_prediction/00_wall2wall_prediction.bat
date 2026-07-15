@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM ============================================================
REM WALL-TO-WALL WIND-ONLY PREDICTION
REM Uses the exact successful original parameters and the
REM reproduced locked model.
REM ============================================================

set "CONFIG_BAT=%~dp0..\00_project_config\00_config_paths.bat"
if not exist "%CONFIG_BAT%" set "CONFIG_BAT=%~dp0..\..\00_project_config\00_config_paths.bat"
if not exist "%CONFIG_BAT%" set "CONFIG_BAT=%~dp0..\..\..\00_project_config\00_config_paths.bat"
if not exist "%CONFIG_BAT%" (
  echo ERROR: 00_config_paths.bat not found.
  echo Tried parent workflow folders from: %~dp0
  exit /b 1
)
call "%CONFIG_BAT%"

set "LOG_FILE=%LOG_DIR%\04_wall2wall_prediction.log"

if not exist "%WALL2WALL_OUTDIR%" mkdir "%WALL2WALL_OUTDIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ============================================================
echo Reproducing wall-to-wall wind-only prediction
echo Script       : %WALL2WALL_SCRIPT%
echo Wind NC      : %WIND_NC%
echo Forest mask  : %FOREST_MASK_TIF%
echo Model        : %MODEL_FOR_WALL2WALL%
echo Output       : %WALL2WALL_OUTDIR%
echo Log          : %LOG_FILE%
echo BBOX         : %W2W_BBOX4326%
echo HEX BBOX     : %W2W_HEX_BBOX4326%
echo ============================================================

if not exist "%WALL2WALL_SCRIPT%" (
  echo ERROR: wall-to-wall script not found:
  echo %WALL2WALL_SCRIPT%
  exit /b 1
)

if not exist "%WIND_NC%" (
  echo ERROR: wind input not found:
  echo %WIND_NC%
  exit /b 1
)

if not exist "%FOREST_MASK_TIF%" (
  echo ERROR: forest mask not found:
  echo %FOREST_MASK_TIF%
  exit /b 1
)

if not exist "%MODEL_FOR_WALL2WALL%" (
  echo ERROR: reproduced model not found:
  echo %MODEL_FOR_WALL2WALL%
  echo Run the locked main-model step first.
  exit /b 1
)

"%PYTHON_EXE%" "%WALL2WALL_SCRIPT%" ^
  --wind_nc "%WIND_NC%" ^
  --forest_mask_tif "%FOREST_MASK_TIF%" ^
  --model_joblib "%MODEL_FOR_WALL2WALL%" ^
  --outdir "%WALL2WALL_OUTDIR%" ^
  --wind_var "%WIND_VAR%" ^
  --bbox4326=%W2W_BBOX4326% ^
  --hex_bbox4326=%W2W_HEX_BBOX4326% ^
  --hex_area_km2=%HEX_AREA_KM2% ^
  --forest_frac_min=%FOREST_FRAC_MIN% ^
  --pred_quantile=%PRED_QUANTILE% ^
  --valid_periods_min_per_hex=%VALID_PERIODS_MIN_PER_HEX% ^
  --norm=%W2W_NORM% ^
  --gamma=%W2W_GAMMA% ^
  --qhi=%W2W_QHI% ^
  > "%LOG_FILE%" 2>&1

if errorlevel 1 (
  echo.
  echo ERROR: wall-to-wall prediction failed.
  echo See log:
  echo %LOG_FILE%
  type "%LOG_FILE%"
  exit /b 1
)

echo.
echo ============================================================
echo Done: wall-to-wall prediction completed.
echo Output: %WALL2WALL_OUTDIR%
echo ============================================================
exit /b 0
