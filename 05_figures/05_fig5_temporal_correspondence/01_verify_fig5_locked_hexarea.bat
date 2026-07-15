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

if not defined FIG5_OUTDIR set "FIG5_OUTDIR=%REPRO_ROOT%\outputs\08_fig5_temporal_correspondence"

echo Checking Fig.5 script...
findstr /C:"LOCKED complete hex-area weights" "%FIG5_SCRIPT%"
if errorlevel 1 (
  echo ERROR: Fig.5 script is NOT the locked hex-area version.
  exit /b 1
)

echo Checking latest Fig.5 merged output total area...
"%PYTHON_EXE%" -c "import pandas as pd; p=r'%FIG5_OUTDIR%\long_term\merged_temporal_correspondence_long_term.csv'; df=pd.read_csv(p); print('rows=',len(df)); print('total_area=',df['area_weight_km2'].sum()); print('unique_area_head=',df['area_weight_km2'].round(6).drop_duplicates().head().tolist())"

exit /b 0
