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
if not exist "%FIG2_OUTDIR%" mkdir "%FIG2_OUTDIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ============================================================
echo Drawing Fig. 2: peak month and peak-month share
echo Output: %FIG2_OUTDIR%
echo ============================================================
"%PYTHON_EXE%" "%FIG2_SCRIPT%" > "%LOG_DIR%\06_fig2_monthly_concentration.log" 2>&1
if errorlevel 1 (
  echo ERROR: Fig. 2 failed. See %LOG_DIR%\06_fig2_monthly_concentration.log
  type "%LOG_DIR%\06_fig2_monthly_concentration.log"
  exit /b 1
)

REM Keep the final Fig. 2 and the hex-level source table used by Fig. 5.
del /q "%FIG2_OUTDIR%\hex_monthly_likelihood_profiles_long.csv" 2>nul
del /q "%FIG2_OUTDIR%\hex_monthly_likelihood_profiles_wide.csv" 2>nul
del /q "%FIG2_OUTDIR%\hex_monthly_timing_concentration_metrics.csv" 2>nul
del /q "%FIG2_OUTDIR%\hex_monthly_spatiotemporal_metric_summary.csv" 2>nul
del /q "%FIG2_OUTDIR%\hex_peak_month_frequency_*.csv" 2>nul
del /q "%FIG2_OUTDIR%\*.gpkg" 2>nul
del /q "%FIG2_OUTDIR%\FigS_monthly_circular_timing_2x2.*" 2>nul

echo Done: Fig. 2 outputs are in %FIG2_OUTDIR%
exit /b 0
