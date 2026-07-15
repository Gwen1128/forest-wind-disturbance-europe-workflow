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
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ============================================================
echo Running Table 1 model-family comparison
rem Output: metrics_*.csv and temporary model_*.joblib in %ABLA_OUTDIR%
echo ============================================================

call :run_one "wind" "wind" "Wind"
if errorlevel 1 exit /b 1
call :run_one "wind,struct" "wind_struct" "Wind+structure"
if errorlevel 1 exit /b 1
call :run_one "wind,cross" "wind_cross" "Wind+interaction"
if errorlevel 1 exit /b 1
call :run_one "wind,struct,cross" "wind_struct_cross" "Wind+structure+interaction"
if errorlevel 1 exit /b 1

echo Done: Table 1 model comparison.
exit /b 0

:run_one
set "FAMILIES=%~1"
set "TAG=%~2"
set "LABEL=%~3"
echo ------------------------------------------------------------
echo Running %LABEL% --families %FAMILIES%
echo ------------------------------------------------------------
"%PYTHON_EXE%" "%ABLATION_SCRIPT%" ^
  --csv "%CLEAN_CSV%" ^
  --holdout_mode bbox4326 ^
  --holdout_arg "%HOLDOUT_BBOX%" ^
  --family_mode grouped ^
  --meta logit ^
  --meta_penalty l2 ^
  --meta_C_grid "%META_C_GRID%" ^
  --n_splits %NSPLITS% ^
  --meta_select_metric prauc ^
  --shift_reweight ^
  --shift_mode gbdt ^
  --shift_clip_q 0.99 ^
  --add_cross ^
  --families "%FAMILIES%" ^
  --disable_residual ^
  --metrics_csv "%ABLA_OUTDIR%\metrics_%TAG%.csv" ^
  --output_preds "%ABLA_OUTDIR%\preds_%TAG%.csv" ^
  --output_model "%ABLA_OUTDIR%\model_%TAG%.joblib" ^
  > "%LOG_DIR%\01_table1_%TAG%.log" 2>&1
if errorlevel 1 (
  echo ERROR: %LABEL% failed. See %LOG_DIR%\01_table1_%TAG%.log
  exit /b 1
)
echo OK: %LABEL% completed.
exit /b 0
