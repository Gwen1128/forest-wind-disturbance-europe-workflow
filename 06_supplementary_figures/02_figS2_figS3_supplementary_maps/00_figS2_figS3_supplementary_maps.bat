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

if not exist "%FIGS_SUPPLEMENTARY_OUTDIR%" mkdir "%FIGS_SUPPLEMENTARY_OUTDIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not exist "%SUPP_SCRIPT%" (
  echo ERROR: supplementary Python script not found:
  echo %SUPP_SCRIPT%
  exit /b 1
)

echo ============================================================
echo Drawing supplementary figures: FigS2 and FigS3
echo Script: %SUPP_SCRIPT%
echo Output: %FIGS_SUPPLEMENTARY_OUTDIR%
echo ============================================================
set "FIGS_OUTDIR=%FIGS_SUPPLEMENTARY_OUTDIR%"
"%PYTHON_EXE%" "%SUPP_SCRIPT%" > "%LOG_DIR%\11_figS2_figS3_supplementary_maps.log" 2>&1
if errorlevel 1 (
  echo ERROR: supplementary figures failed. See:
  echo %LOG_DIR%\11_figS2_figS3_supplementary_maps.log
  type "%LOG_DIR%\11_figS2_figS3_supplementary_maps.log"
  exit /b 1
)

echo Done: supplementary figures are in:
echo %FIGS_SUPPLEMENTARY_OUTDIR%
exit /b 0
