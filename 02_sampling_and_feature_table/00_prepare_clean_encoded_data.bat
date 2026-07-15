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

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%DATAPREP_OUTDIR%" mkdir "%DATAPREP_OUTDIR%"

echo ============================================================
echo Preparing clean_encoded_data_LOCKED.csv
echo ============================================================
echo GEE step is manual: open %GEE_LAI_SCRIPT% in Earth Engine first.

call :check_file "%FEATURE_EXTRACTION_SCRIPT%" || exit /b 1
call :check_file "%PREPROCESS_LOCKED_SCRIPT%" || exit /b 1

if not exist "%BFAST_GLOBAL_CSV%" (
  echo.
  echo BFAST/global sample table not found:
  echo %BFAST_GLOBAL_CSV%
  echo Trying to run R LAI-departure sampling script.
  call :check_file "%BFAST_LAI_SCRIPT%" || exit /b 1
  call :check_file "%FORWIND_SHP%" || exit /b 1
  call :check_dir "%LAI_MAIN_DIR%" || exit /b 1
  call :check_dir "%LAI_SUPP_DIR%" || exit /b 1

  set "DP_LAI_MAIN_DIR=%LAI_MAIN_DIR%"
  set "DP_LAI_SUPP_DIR=%LAI_SUPP_DIR%"
  set "DP_FORWIND_SHP=%FORWIND_SHP%"
  set "DP_BFAST_OUTPUT_DIR=%DATAPREP_OUTDIR%\bfast_output_global"
  set "DP_BFAST_BATCH_SIZE=%BFAST_BATCH_SIZE%"
  set "DP_NEG_MULTIPLIER=%NEG_MULTIPLIER%"

  "%R_EXE%" "%BFAST_LAI_SCRIPT%" > "%LOG_DIR%\03_prepare_step01_bfast_lai_departure.log" 2>&1
  if errorlevel 1 (
    echo ERROR: R LAI-departure sampling failed. See log:
    echo %LOG_DIR%\03_prepare_step01_bfast_lai_departure.log
    exit /b 1
  )
  if exist "%DATAPREP_OUTDIR%\bfast_output_global\bfast_global.csv" (
    copy /Y "%DATAPREP_OUTDIR%\bfast_output_global\bfast_global.csv" "%BFAST_GLOBAL_CSV%" >nul
  )
)

call :check_file "%BFAST_GLOBAL_CSV%" || exit /b 1

set "DP_INPUT_BFAST_GLOBAL=%BFAST_GLOBAL_CSV%"
set "DP_WORK_DIR=%DATAPREP_OUTDIR%"
set "DP_DEM_FOLDER=%DEM_FOLDER%"
set "DP_DEM_CUTLINE_GEOJSON=%DATAPREP_OUTDIR%\europe_cutline_wgs84.geojson"
set "DP_WIND_NC=%WIND_NC%"
set "DP_MODIS_LC_DIR=%MODIS_LC_DIR%"
set "DP_TREE_HEIGHT_TIF=%TREE_HEIGHT_TIF%"
set "DP_BIOME_TIF=%BIOME_TIF%"
set "DP_ENCODERS_JSON=%DATAPREP_ENCODERS_JSON%"
set "DP_FINAL_FEATURES_CSV=%FEATURES_CSV%"
set "DP_WIND_N_JOBS=%WIND_N_JOBS%"
set "DP_GDAL_WARP_MEMORY_MB=%GDAL_WARP_MEMORY_MB%"
set "DP_DEM_BBOX_WGS84=%DEM_BBOX_WGS84%"

"%PYTHON_EXE%" "%FEATURE_EXTRACTION_SCRIPT%" > "%LOG_DIR%\03_prepare_step02_extract_features.log" 2>&1
if errorlevel 1 (
  echo ERROR: Geospatial feature extraction failed. See log:
  echo %LOG_DIR%\03_prepare_step02_extract_features.log
  exit /b 1
)

"%PYTHON_EXE%" "%PREPROCESS_LOCKED_SCRIPT%" ^
  --input-csv "%FEATURES_CSV%" ^
  --out-csv "%CLEAN_CSV%" ^
  --encoder-json "%ENCODERS_LOCKED_JSON%" ^
  > "%LOG_DIR%\03_prepare_step03_preprocess_locked.log" 2>&1
if errorlevel 1 (
  echo ERROR: Locked preprocessing failed. See log:
  echo %LOG_DIR%\03_prepare_step03_preprocess_locked.log
  exit /b 1
)

echo.
echo Done: %CLEAN_CSV%
exit /b 0

:check_file
if exist "%~1" (
 echo OK       %~1
 exit /b 0
) else (
 echo MISSING  %~1
 exit /b 1
)

:check_dir
if exist "%~1\" (
 echo OK       %~1
 exit /b 0
) else (
 echo MISSING  %~1
 exit /b 1
)
