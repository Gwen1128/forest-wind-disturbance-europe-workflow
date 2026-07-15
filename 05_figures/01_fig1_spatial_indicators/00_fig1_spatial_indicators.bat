@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "CONFIG_BAT=%~dp0..\00_project_config\00_config_paths.bat"
if not exist "%CONFIG_BAT%" set "CONFIG_BAT=%~dp0..\..\00_project_config\00_config_paths.bat"
if not exist "%CONFIG_BAT%" set "CONFIG_BAT=%~dp0..\..\..\00_project_config\00_config_paths.bat"
if not exist "%CONFIG_BAT%" (
  echo ERROR: 00_config_paths.bat not found.
  echo Tried parent workflow folders from: %~dp0
  exit /b 1
)
call "%CONFIG_BAT%"

if not exist "%FIG1_OUTDIR%" mkdir "%FIG1_OUTDIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not exist "%FIG1_SCRIPT%" (
  echo ERROR: Fig. 1 script not found:
  echo %FIG1_SCRIPT%
  exit /b 1
)

if not exist "%HEX_DISTRIBUTION_CSV%" (
  echo ERROR: missing:
  echo %HEX_DISTRIBUTION_CSV%
  echo Run 04_wall2wall_prediction.bat first.
  exit /b 1
)

if not exist "%HEX_PERIOD_CSV%" (
  echo ERROR: missing:
  echo %HEX_PERIOD_CSV%
  echo Run 04_wall2wall_prediction.bat first.
  exit /b 1
)

if not exist "%HEX_GEOMETRY_CSV%" (
  echo ERROR: missing wall-to-wall hex geometry:
  echo %HEX_GEOMETRY_CSV%
  echo Run 04_wall2wall_prediction.bat first.
  exit /b 1
)

echo ============================================================
echo Reproducing Fig. 1 spatial indicators
echo Distribution : %HEX_DISTRIBUTION_CSV%
echo Period       : %HEX_PERIOD_CSV%
echo Geometry     : %HEX_GEOMETRY_CSV%
echo Hex bbox     : %FIG1_HEX_BBOX4326%
echo Plot bbox    : %FIG1_PLOT_BBOX4326%
echo Output       : %FIG1_OUTDIR%
echo ============================================================

"%PYTHON_EXE%" "%FIG1_SCRIPT%" ^
  --hex_distribution_csv "%HEX_DISTRIBUTION_CSV%" ^
  --hex_period_csv "%HEX_PERIOD_CSV%" ^
  --outdir "%FIG1_OUTDIR%" ^
  --countries_src "%COUNTRIES_SRC%" ^
  --hex_bbox4326=%FIG1_HEX_BBOX4326% ^
  --plot_bbox4326=%FIG1_PLOT_BBOX4326% ^
  --hex_geometry_csv "%HEX_GEOMETRY_CSV%" ^
  --hex_area_km2 %HEX_AREA_KM2% ^
  --forest_frac_min 0 ^
  --active_quantile %ACTIVE_QUANTILE% ^
  > "%LOG_DIR%\05_fig1_spatial_indicators.log" 2>&1

if errorlevel 1 (
  echo ERROR: Fig. 1 failed. See:
  echo %LOG_DIR%\05_fig1_spatial_indicators.log
  type "%LOG_DIR%\05_fig1_spatial_indicators.log"
  exit /b 1
)

REM Keep the source table and final 2x2 figure only.
del /q "%FIG1_OUTDIR%\*.gpkg" 2>nul
del /q "%FIG1_OUTDIR%\mean_pred_prob_stylematch_publication.*" 2>nul
del /q "%FIG1_OUTDIR%\p95_pred_prob_stylematch_publication.*" 2>nul
del /q "%FIG1_OUTDIR%\recurrence_index_stylematch_publication.*" 2>nul
del /q "%FIG1_OUTDIR%\conditional_intensity_stylematch_publication.*" 2>nul

echo Done: Fig. 1 outputs are in %FIG1_OUTDIR%
exit /b 0
