@echo off
setlocal EnableExtensions
call "%~dp0\00_config_paths.bat"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "ERR=0"

echo ============================================================
echo Checking required scripts and data inputs
echo Project root : %PROJECT_ROOT%
echo Script dir   : %SCRIPT_DIR%
echo Output root  : %OUTPUT_ROOT%
echo ============================================================

call :check_required "%CLEAN_CSV%"
call :check_required "%FOREST_MASK_TIF%"
call :check_required "%WIND_NC%"
call :check_required "%COUNTRIES_SRC%"
call :check_required "%WIND_PRODUCTS_NC%"
call :check_required "%WIND_OVERLAY_LONG_NC%"
call :check_required "%WIND_TEMP_CSV%"

call :check_required "%ABLATION_SCRIPT%"
call :check_required "%TABLE1_HELPER%"
call :check_required "%WALL2WALL_SCRIPT%"
call :check_required "%FIG1_SCRIPT%"
call :check_required "%FIG2_SCRIPT%"
call :check_required "%FIG3_SCRIPT%"
call :check_required "%FIG4_PREP_SCRIPT%"
call :check_required "%FIG4_PLOT_SCRIPT%"
call :check_required "%FIG5_SCRIPT%"

echo.
echo Required configuration values.
call :check_value "W2W_BBOX4326" "%W2W_BBOX4326%"
call :check_value "W2W_HEX_BBOX4326" "%W2W_HEX_BBOX4326%"
call :check_value "W2W_GAMMA" "%W2W_GAMMA%"
call :check_value "W2W_QHI" "%W2W_QHI%"
call :check_value "FIG1_HEX_BBOX4326" "%FIG1_HEX_BBOX4326%"
call :check_value "FIG1_PLOT_BBOX4326" "%FIG1_PLOT_BBOX4326%"
call :check_value "HEX_GEOMETRY_CSV" "%HEX_GEOMETRY_CSV%"
call :check_value "FIG1_INDICATOR_CSV" "%FIG1_INDICATOR_CSV%"
call :check_value "FIG4_MERGED_CSV" "%FIG4_MERGED_CSV%"

echo.
echo Optional inputs/scripts. Missing items here do not block the main workflow.
call :check_optional "%GEE_LAI_SCRIPT%"
call :check_optional "%BFAST_LAI_SCRIPT%"
call :check_optional "%FEATURE_EXTRACTION_SCRIPT%"
call :check_optional "%COMBINE_FEATURE_TILES_SCRIPT%"
call :check_optional "%PREPROCESS_LOCKED_SCRIPT%"
call :check_optional "%BFAST_GLOBAL_CSV%"
call :check_optional "%FEATURES_CSV%"
call :check_optional "%WIND_OVERLAY_MONTH_NC%"
call :check_optional "%WIND_MONTHLY_PROFILE_CSV%"
call :check_optional "%WIND_PERIOD_ANOMALY_CSV%"
call :check_optional "%SUPP_SCRIPT%"

call :check_optional "%ERA5_FG10_SCRIPT%"
call :check_optional "%ERA5_UV_SCRIPT%"
call :check_optional "%ERA5_INTEGRATE_SCRIPT%"
call :check_optional "%EFDA_06F0_SCRIPT%"
call :check_optional "%EFDA_06F1_SCRIPT%"
call :check_optional "%EFDA_06F2_SCRIPT%"
call :check_optional "%EFDA_06F3_SCRIPT%"


if "%ERR%"=="0" (
  echo.
  echo All required inputs and configuration values were found.
  exit /b 0
) else (
  echo.
  echo ERROR: Some required inputs or settings are missing. Edit 00_config_paths.bat.
  exit /b 1
)

:check_required
if "%~1"=="" (
  echo MISSING ^<empty path^>
  set "ERR=1"
) else if exist "%~1" (
  echo OK       %~1
) else (
  echo MISSING  %~1
  set "ERR=1"
)
exit /b 0

:check_optional
if "%~1"=="" (
  echo OPTIONAL MISSING  ^<empty path^>
) else if exist "%~1" (
  echo OPTIONAL OK       %~1
) else (
  echo OPTIONAL MISSING  %~1
)
exit /b 0

:check_value
if "%~2"=="" (
  echo MISSING VALUE  %~1
  set "ERR=1"
) else (
  echo VALUE OK       %~1=%~2
)
exit /b 0
