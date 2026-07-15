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
if not exist "%ABLA_OUTDIR%" mkdir "%ABLA_OUTDIR%"
if not exist "%MAIN_OUTDIR%" mkdir "%MAIN_OUTDIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ============================================================
echo Building Table 1 from reproduced metrics
echo Also preserving the selected wind-only model for wall-to-wall prediction
echo ============================================================
"%PYTHON_EXE%" "%TABLE1_HELPER%" ^
  --metrics_dir "%ABLA_OUTDIR%" ^
  --out_csv "%ABLA_OUTDIR%\table1_spatial_holdout_from_original_pipeline.csv" ^
  > "%LOG_DIR%\02_make_table1.log" 2>&1
if errorlevel 1 (
  echo ERROR: make Table 1 failed. See %LOG_DIR%\02_make_table1.log
  type "%LOG_DIR%\02_make_table1.log"
  exit /b 1
)

if not exist "%ABLA_OUTDIR%\model_wind.joblib" (
  echo ERROR: wind-only model from Table 1 run was not found:
  echo %ABLA_OUTDIR%\model_wind.joblib
  echo Run 01_table1_model_comparison.bat first.
  exit /b 1
)

copy /Y "%ABLA_OUTDIR%\model_wind.joblib" "%MAIN_MODEL_REPRO%" >nul
if errorlevel 1 (
  echo ERROR: failed to copy wind-only model to:
  echo %MAIN_MODEL_REPRO%
  exit /b 1
)

REM Keep only Table 1 metrics, final Table 1, and the copied final wind-only model.
REM Remove large intermediate predictions and non-final ablation models.
del /q "%ABLA_OUTDIR%\preds_*.csv" 2>nul
del /q "%ABLA_OUTDIR%\model_*.joblib" 2>nul
del /q "%ABLA_OUTDIR%\metrics_*_wind_strata.csv" 2>nul

echo Done: %ABLA_OUTDIR%\table1_spatial_holdout_from_original_pipeline.csv
echo Final wind-only model copied to:
echo %MAIN_MODEL_REPRO%
exit /b 0
