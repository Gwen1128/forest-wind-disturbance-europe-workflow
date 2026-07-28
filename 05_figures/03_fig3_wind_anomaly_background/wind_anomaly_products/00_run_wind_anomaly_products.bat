@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM ============================================================
REM Build wind-anomaly products used by Fig.3-Fig.5
REM
REM Step 1 creates:
REM   wind_spacetime_products_europe_main_admin_no_russia_no_iceland.nc
REM
REM Step 2 creates:
REM   overlay_result_long_term_background.nc
REM   overlay_result_same_month_background.nc
REM ============================================================

set "CONFIG_BAT=%~dp0..\..\00_project_config\00_config_paths.bat"
if exist "%CONFIG_BAT%" call "%CONFIG_BAT%"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

echo ============================================================
echo Step 1: Build wind spacetime products from WIND_NC
echo ============================================================
echo NOTE: Edit paths inside 01_build_wind_spacetime_products_from_ERA5.py if needed.
"%PYTHON_EXE%" "%~dp0\01_build_wind_spacetime_products_from_ERA5.py"
if errorlevel 1 exit /b 1

echo ============================================================
echo Step 2: Build overlay NetCDF files for Fig.4
echo ============================================================
"%PYTHON_EXE%" "%~dp0\02_build_overlay_netcdf_from_products.py"
if errorlevel 1 exit /b 1

echo Done.
exit /b 0
