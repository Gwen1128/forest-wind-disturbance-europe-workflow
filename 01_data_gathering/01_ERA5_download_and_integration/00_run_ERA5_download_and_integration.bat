@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM ============================================================
REM Optional ERA5 wind data gathering workflow
REM Edit paths at the top of each Python script before running.
REM Requires CDS API credentials.
REM ============================================================

set "CONFIG_BAT=%~dp0..\..\00_project_config\00_config_paths.bat"
if exist "%CONFIG_BAT%" call "%CONFIG_BAT%"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

echo ============================================================
echo Step 1: Download ERA5 fg10 chunks
echo ============================================================
"%PYTHON_EXE%" "%~dp0\01_download_era5_fg10_16day.py"
if errorlevel 1 exit /b 1

echo ============================================================
echo Step 2: Download ERA5 u10/v10 chunks
echo ============================================================
"%PYTHON_EXE%" "%~dp0\02_download_era5_u10v10_16day.py"
if errorlevel 1 exit /b 1

echo ============================================================
echo Step 3: Integrate fg10 and u10/v10 into structural wind features
echo ============================================================
"%PYTHON_EXE%" "%~dp0\03_integrate_fg10_u10v10_to_structural_wind_features.py"
if errorlevel 1 exit /b 1

echo Done.
exit /b 0
