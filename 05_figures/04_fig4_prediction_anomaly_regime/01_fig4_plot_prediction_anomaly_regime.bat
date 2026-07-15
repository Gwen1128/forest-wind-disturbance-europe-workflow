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
if not exist "%FIG4_OUTDIR%" mkdir "%FIG4_OUTDIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not exist "%FIG4_MERGED_CSV%" (
  echo ERROR: Fig. 4 merged CSV not found:
  echo %FIG4_MERGED_CSV%
  echo Run 08_fig4_prepare_prediction_anomaly_regime.bat first.
  exit /b 1
)

echo ============================================================
echo Plotting Fig. 4 prediction-anomaly regime figure
echo Output: %FIG4_OUTDIR%
echo ============================================================
"%PYTHON_EXE%" "%FIG4_PLOT_SCRIPT%" > "%LOG_DIR%\09_fig4_plot_prediction_anomaly_regime.log" 2>&1
if errorlevel 1 (
  echo ERROR: Fig. 4 plot failed. See %LOG_DIR%\09_fig4_plot_prediction_anomaly_regime.log
  type "%LOG_DIR%\09_fig4_plot_prediction_anomaly_regime.log"
  exit /b 1
)

REM Keep final Fig. 4 and panel source tables only; remove full diagnostic hex dump.
del /q "%FIG4_OUTDIR%\hex_with_existing_wind_anomaly_regimes_for_fig5.csv" 2>nul
del /q "%FIG4_OUTDIR%\supp_prediction_mass_concentration_by_regime.csv" 2>nul

echo Done: Fig. 4 outputs are in %FIG4_OUTDIR%
exit /b 0
