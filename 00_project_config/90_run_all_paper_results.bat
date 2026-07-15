@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM Full runner for the workflow-organized paper-results package.
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"

call :RUN_REQUIRED "Input check" "%PROJECT_ROOT%\00_project_config\00_check_inputs.bat" || exit /b 1
call :RUN_REQUIRED "Table 1 model comparison" "%PROJECT_ROOT%\03_model_training\00_table1_model_comparison.bat" || exit /b 1
call :RUN_REQUIRED "Table 1 formatting and locked model copy" "%PROJECT_ROOT%\03_model_training\01_make_table1_and_copy_locked_model.bat" || exit /b 1
call :RUN_REQUIRED "Wall-to-wall prediction" "%PROJECT_ROOT%\04_wall_to_wall_prediction\00_wall2wall_prediction.bat" || exit /b 1
call :RUN_REQUIRED "Fig. 1 spatial indicators" "%PROJECT_ROOT%\05_figures\01_fig1_spatial_indicators\00_fig1_spatial_indicators.bat" || exit /b 1
call :RUN_REQUIRED "Fig. 2 monthly concentration" "%PROJECT_ROOT%\05_figures\02_fig2_monthly_concentration\00_fig2_monthly_concentration.bat" || exit /b 1
call :RUN_REQUIRED "Fig. 3 wind-anomaly background" "%PROJECT_ROOT%\05_figures\03_fig3_wind_anomaly_background\00_fig3_wind_anomaly_background.bat" || exit /b 1
call :RUN_REQUIRED "Fig. 4 prepare prediction-anomaly regime" "%PROJECT_ROOT%\05_figures\04_fig4_prediction_anomaly_regime\00_fig4_prepare_prediction_anomaly_regime.bat" || exit /b 1
call :RUN_REQUIRED "Fig. 4 plot prediction-anomaly regime" "%PROJECT_ROOT%\05_figures\04_fig4_prediction_anomaly_regime\01_fig4_plot_prediction_anomaly_regime.bat" || exit /b 1
call :RUN_REQUIRED "Fig. 5 temporal correspondence" "%PROJECT_ROOT%\05_figures\05_fig5_temporal_correspondence\00_fig5_temporal_correspondence.bat" || exit /b 1
call :RUN_OPTIONAL "Fig. S1 EFDA validation" "%PROJECT_ROOT%\06_supplementary_figures\01_figS1_EFDA_validation\00_figS1_EFDA_validation.bat"
call :RUN_OPTIONAL "Fig. S2-Fig. S3 supplementary maps" "%PROJECT_ROOT%\06_supplementary_figures\02_figS2_figS3_supplementary_maps\00_figS2_figS3_supplementary_maps.bat"

echo.
echo Done: required paper-result outputs through Fig.5 were reproduced.
exit /b 0

:RUN_REQUIRED
set "_STEP_NAME=%~1"
set "_BAT_FILE=%~2"
if not exist "%_BAT_FILE%" (
  echo.
  echo ERROR: BAT file not found for required step: %_STEP_NAME%
  echo %_BAT_FILE%
  exit /b 1
)
echo.
echo ============================================================
echo Running required step: %_STEP_NAME%
echo BAT               : %_BAT_FILE%
echo ============================================================
call "%_BAT_FILE%"
set "_RC=%ERRORLEVEL%"
if not "%_RC%"=="0" (
  echo.
  echo ERROR: Required step failed: %_STEP_NAME%
  exit /b %_RC%
)
exit /b 0

:RUN_OPTIONAL
set "_STEP_NAME=%~1"
set "_BAT_FILE=%~2"
if not exist "%_BAT_FILE%" (
  echo.
  echo SKIP optional step, BAT not found: %_STEP_NAME%
  echo %_BAT_FILE%
  exit /b 0
)
echo.
echo ============================================================
echo Running optional step: %_STEP_NAME%
echo BAT              : %_BAT_FILE%
echo ============================================================
call "%_BAT_FILE%"
set "_RC=%ERRORLEVEL%"
if not "%_RC%"=="0" (
  echo.
  echo WARNING: Optional step failed and was skipped: %_STEP_NAME%
)
exit /b 0
