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

"%PYTHON_EXE%" "%COMBINE_FEATURE_TILES_SCRIPT%" ^
  --folder "%FEATURE_TILE_FOLDER%" ^
  --output "%FEATURES_CSV%" ^
  > "%LOG_DIR%\03A_optional_combine_feature_tiles.log" 2>&1
if errorlevel 1 (
  echo ERROR: optional feature-tile merge failed. See log:
  echo %LOG_DIR%\03A_optional_combine_feature_tiles.log
  exit /b 1
)

echo Done: %FEATURES_CSV%
exit /b 0
