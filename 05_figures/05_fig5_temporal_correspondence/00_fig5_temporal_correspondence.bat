@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM ============================================================
REM Fig. 5 final reproduction
REM
REM It rebuilds the final merged CSV, summary tables and figure.
REM It does not reuse old merged_temporal_correspondence CSV.
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

if not defined PYTHON_EXE set "PYTHON_EXE=python"
if not defined FIG5_OUTDIR set "FIG5_OUTDIR=%REPRO_ROOT%\outputs\08_fig5_temporal_correspondence"
if not defined LOG_DIR set "LOG_DIR=%REPRO_ROOT%\logs"
if not defined HEX_AREA_KM2 set "HEX_AREA_KM2=2165"

if not defined FIG1_INDICATOR_CSV set "FIG1_INDICATOR_CSV=%HEX_STYLEMATCH_CSV%"

REM FIG5_SCRIPT is defined in 00_project_config\00_config_paths.bat
set "FIG5_RUN_OUTDIR=%FIG5_OUTDIR%\long_term"
set "LOG_FILE=%LOG_DIR%\10_fig5_temporal_correspondence.log"

if not exist "%FIG5_SCRIPT%" (
  echo ERROR: Fig. 5 script not found:
  echo %FIG5_SCRIPT%
  exit /b 1
)

if not exist "%PRED_TEMP_CSV%" (
  echo ERROR: Fig. 2 prediction temporal metrics not found:
  echo %PRED_TEMP_CSV%
  echo Run 06_fig2_monthly_concentration.bat first.
  exit /b 1
)

if not exist "%FIG1_INDICATOR_CSV%" (
  echo ERROR: Fig. 1 indicator/geometry table not found:
  echo %FIG1_INDICATOR_CSV%
  echo Run 05_fig1_spatial_indicators.bat first.
  exit /b 1
)

if not exist "%COUNTRIES_SRC%" (
  echo ERROR: country boundary file not found:
  echo %COUNTRIES_SRC%
  exit /b 1
)

if not exist "%FIG5_RUN_OUTDIR%" mkdir "%FIG5_RUN_OUTDIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Main Fig.5 settings
set "FIG5_BACKGROUND=long_term"
set "FIG5_WIND_PROFILE_METRIC=frequency"
set "FIG5_FORCE_REBUILD=1"
set "FIG5_REWRITE_TABLES_FROM_EXISTING_CSV=0"
set "FIG5_FORCE_WIND_FROM_NC=1"

REM Locked original Fig.5 area weighting
set "FIG5_HEX_AREA_KM2=%HEX_AREA_KM2%"
set "FIG5_AREA_WEIGHT_MODE=hex"

REM Remove previously generated Fig. 5 outputs before rebuilding.
del /q "%FIG5_RUN_OUTDIR%\merged_temporal_correspondence_long_term.csv" 2>nul
del /q "%FIG5_RUN_OUTDIR%\TableS_temporal_correspondence_summary_long_term.csv" 2>nul
del /q "%FIG5_RUN_OUTDIR%\TableS_peak_month_agreement_classes_long_term.csv" 2>nul
del /q "%FIG5_RUN_OUTDIR%\TableS_temporal_correlations_long_term.csv" 2>nul
del /q "%FIG5_RUN_OUTDIR%\Fig5_peak_month_correspondence_long_term.png" 2>nul
del /q "%FIG5_RUN_OUTDIR%\Fig5_peak_month_correspondence_long_term.pdf" 2>nul

echo ============================================================
echo Reproducing Fig. 5
echo Script              : %FIG5_SCRIPT%
echo Fig.2 input         : %PRED_TEMP_CSV%
echo Fig.1 geometry      : %FIG1_INDICATOR_CSV%
echo Wind input          : %WIND_NC%
echo Output              : %FIG5_RUN_OUTDIR%
echo Wind metric         : frequency
echo Area weighting      : complete hex area, %HEX_AREA_KM2% km2 per hex
echo Force raw WIND_NC   : yes
echo Log                 : %LOG_FILE%
echo ============================================================

"%PYTHON_EXE%" "%FIG5_SCRIPT%" > "%LOG_FILE%" 2>&1

if errorlevel 1 (
  echo.
  echo ERROR: Fig. 5 reproduction failed.
  echo See log:
  echo %LOG_FILE%
  type "%LOG_FILE%"
  exit /b 1
)

type "%LOG_FILE%"

echo.
echo ============================================================
echo Done: Fig. 5 outputs are in:
echo %FIG5_RUN_OUTDIR%
echo ============================================================
exit /b 0
