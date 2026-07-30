@echo off
REM ============================================================
REM Global configuration for EUfinal reproducible workflow.
REM Edit this file only. All other .bat files read these paths.
REM ============================================================

REM Force UTF-8 console/Python I/O to avoid UnicodeEncodeError on Windows.
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM ------------------------------------------------------------
REM Python executable
REM ------------------------------------------------------------
REM Define the GEO_PYTHON environment variable.
if not defined GEO_PYTHON set "GEO_PYTHON=python"

if exist "%GEO_PYTHON%" (
  set "PYTHON_EXE=%GEO_PYTHON%"
) else if defined CONDA_PREFIX (
  set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
) else (
  set "PYTHON_EXE=python"
)

REM ------------------------------------------------------------
REM Input/output roots
REM ------------------------------------------------------------
REM Original data/source folder. This workflow reads it but does not write results into it.
set "ORIGINAL_BASE=E:\RF B+G"

REM New reproducibility output root. All new outputs go here.
set "REPRO_ROOT=E:\RF_BG_REPRO_from_model_dev"
set "OUTPUT_ROOT=%REPRO_ROOT%\outputs"

REM Project/code root: folder containing this 00_project_config folder.
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
set "SCRIPT_DIR=%PROJECT_ROOT%"

REM ------------------------------------------------------------
REM Core external inputs
REM ------------------------------------------------------------
set "CLEAN_CSV=%ORIGINAL_BASE%\clean_encoded_data_LOCKED.csv"
set "FOREST_MASK_TIF=%ORIGINAL_BASE%\Europe_forest_mask_hansen25.tif"
set "WIND_NC=E:\ERA\ERA5 16 combine\EU_final_structural_wind_features.nc"
set "COUNTRIES_SRC=E:\CNTR_RG_20M_2024_3035\CNTR_RG_20M_2024_3035.shp"

REM ------------------------------------------------------------
REM Data-preparation inputs for rebuilding clean_encoded_data_LOCKED.csv
REM ------------------------------------------------------------
set "DATAPREP_OUTDIR=%OUTPUT_ROOT%\00_data_preparation"
set "BFAST_GLOBAL_CSV=%ORIGINAL_BASE%\bfast_global.csv"
set "FEATURES_CSV=%ORIGINAL_BASE%\bfastglobal_FINAL_ALL_FEATURES.csv"
set "ENCODERS_LOCKED_JSON=%ORIGINAL_BASE%\encoders_LOCKED.json"
set "DATAPREP_ENCODERS_JSON=%DATAPREP_OUTDIR%\encoders_feature_extraction.json"

set "R_EXE=Rscript"
set "LAI_MAIN_DIR=E:\LAI_Export"
set "LAI_SUPP_DIR=E:\LAI_EU_Supple"
set "FORWIND_SHP=E:\FORWIND appendix\FORWIND_v2.shp"
set "BFAST_BATCH_SIZE=5000"
set "NEG_MULTIPLIER=10"

set "DEM_FOLDER=E:\DEM tar"
set "DEM_BBOX_WGS84=-31.5,23.75,41.5,71.2"
set "GDAL_WARP_MEMORY_MB=2048"
set "WIND_N_JOBS=8"
set "MODIS_LC_DIR=E:\MODISLandcover"
set "TREE_HEIGHT_TIF=E:\TREE_HEIGHT\canopy_height_2005_Europe.tif"
set "BIOME_TIF=E:\biome_wwf_ecoregion.tif"
set "FEATURE_TILE_FOLDER=%ORIGINAL_BASE%"

REM ------------------------------------------------------------
REM Existing wind-anomaly products used by Fig. 3-5
REM ------------------------------------------------------------
set "WIND_PRODUCTS_NC=%ORIGINAL_BASE%\wind_spacetime_maps_europe_main_admin\wind_spacetime_products_europe_main_admin_no_russia_no_iceland.nc"
set "WIND_OVERLAY_LONG_NC=%ORIGINAL_BASE%\windmap\overlay_result_long_term_background.nc"
set "WIND_OVERLAY_MONTH_NC=%ORIGINAL_BASE%\windmap\overlay_result_same_month_background.nc"

set "WIND_NC=E:\ERA\ERA5 16 combine\EU_final_structural_wind_features.nc"

REM ------------------------------------------------------------
REM Scripts inside this reproducibility package
REM ------------------------------------------------------------
set "GEE_LAI_SCRIPT=%PROJECT_ROOT%\01_data_gathering\00_GEE_LAI_extract.js"
set "BFAST_LAI_SCRIPT=%PROJECT_ROOT%\02_sampling_and_feature_table\01_bfast_lai_departure_sampling.R"
set "FEATURE_EXTRACTION_SCRIPT=%PROJECT_ROOT%\02_sampling_and_feature_table\02_extract_geospatial_features.py"
set "COMBINE_FEATURE_TILES_SCRIPT=%PROJECT_ROOT%\02_sampling_and_feature_table\03_optional_combine_feature_tiles.py"
set "PREPROCESS_LOCKED_SCRIPT=%PROJECT_ROOT%\02_sampling_and_feature_table\04_preprocess_locked_modeling_table.py"

set "ABLATION_SCRIPT=%PROJECT_ROOT%\03_model_training\train_model_family_comparison.py"
REM Compatibility aliases retained for older BAT variants.
set "MAIN_TRAIN_SCRIPT=%ABLATION_SCRIPT%"
set "SCRIPT_LOCKED_MAIN=%ABLATION_SCRIPT%"
set "SCRIPT_MAIN=%ABLATION_SCRIPT%"
set "TABLE1_HELPER=%PROJECT_ROOT%\03_model_training\make_table1_from_metrics.py"

set "WALL2WALL_SCRIPT=%PROJECT_ROOT%\04_wall_to_wall_prediction\wall2wall_windonly_prediction.py"
set "FIG1_SCRIPT=%PROJECT_ROOT%\05_figures\01_fig1_spatial_indicators\fig1spatial_indicator.py"
set "FIG2_SCRIPT=%PROJECT_ROOT%\05_figures\02_fig2_monthly_concentration\06B_fig2_prediction_monthly_concentration.py"
set "FIG3_SCRIPT=%PROJECT_ROOT%\05_figures\03_fig3_wind_anomaly_background\06C_fig3_wind_anomaly_background.py"
set "FIG4_PREP_SCRIPT=%PROJECT_ROOT%\05_figures\04_fig4_prediction_anomaly_regime\06D_fig4_prediction_anomaly_regime_prepare.py"
set "FIG4_PLOT_SCRIPT=%PROJECT_ROOT%\05_figures\04_fig4_prediction_anomaly_regime\06E_fig4_prediction_anomaly_regime_plot.py"
set "FIG5_SCRIPT=%PROJECT_ROOT%\05_figures\05_fig5_temporal_correspondence\06F_fig5_temporal_correspondence_peak_month.py"
set "SUPP_SCRIPT=%PROJECT_ROOT%\06_supplementary_figures\02_figS2_figS3_supplementary_maps\06A_figS2_figS3_supplementary_maps.py"

REM ------------------------------------------------------------
REM ERA5 data-gathering scripts
REM ------------------------------------------------------------
set "ERA5_FG10_SCRIPT=%PROJECT_ROOT%\01_data_gathering\01_ERA5_download_and_integration\01_download_era5_fg10_16day.py"
set "ERA5_UV_SCRIPT=%PROJECT_ROOT%\01_data_gathering\01_ERA5_download_and_integration\02_download_era5_u10v10_16day.py"
set "ERA5_INTEGRATE_SCRIPT=%PROJECT_ROOT%\01_data_gathering\01_ERA5_download_and_integration\03_integrate_fg10_u10v10_to_structural_wind_features.py"

REM ------------------------------------------------------------
REM Fig. S1 EFDA validation scripts
REM ------------------------------------------------------------
set "EFDA_FIGS1_DIR=%PROJECT_ROOT%\06_supplementary_figures\01_figS1_EFDA_validation"
set "EFDA_06F0_SCRIPT=%EFDA_FIGS1_DIR%\06F0_download_efda_zenodo.py"
set "EFDA_06F1_SCRIPT=%EFDA_FIGS1_DIR%\06F1_inspect_efda_files.py"
set "EFDA_06F2_SCRIPT=%EFDA_FIGS1_DIR%\06F2_aggregate_efda_to_hex_year.py"
set "EFDA_06F3_SCRIPT=%EFDA_FIGS1_DIR%\06F3_compare_prediction_with_efda.py"


REM ------------------------------------------------------------
REM Output folders
REM ------------------------------------------------------------
set "ABLA_OUTDIR=%OUTPUT_ROOT%\01_ablation_original_pipeline"
set "MAIN_OUTDIR=%OUTPUT_ROOT%\02_locked_main_model"
set "WALL2WALL_OUTDIR=%OUTPUT_ROOT%\03_wall2wall_windonly"
set "FIG1_OUTDIR=%OUTPUT_ROOT%\04_fig1_spatial_indicators"
set "FIG2_OUTDIR=%OUTPUT_ROOT%\05_fig2_monthly_concentration"
set "FIG3_OUTDIR=%OUTPUT_ROOT%\06_fig3_wind_anomaly_background"
set "FIG4_PREP_OUTDIR=%OUTPUT_ROOT%\06_fig4_prediction_anomaly_regime_prepare"
set "FIG4_OUTDIR=%OUTPUT_ROOT%\07_fig4_prediction_anomaly_regime_plot"
set "FIG5_OUTDIR=%OUTPUT_ROOT%\08_fig5_temporal_correspondence"
set "FIGS_SUPPLEMENTARY_OUTDIR=%OUTPUT_ROOT%\09_figS2_figS3_supplementary_maps"
set "LOG_DIR=%REPRO_ROOT%\logs"

REM Compatibility output aliases.
set "FIG_OUTDIR=%FIG1_OUTDIR%"
set "SECTION35_REGIME_OUTDIR=%FIG4_PREP_OUTDIR%"
set "FIG5_TEMPORAL_OUTDIR=%FIG5_OUTDIR%\long_term"

set "MAIN_MODEL_REPRO=%MAIN_OUTDIR%\stacked_LOCKED_grouped_logit_main_REPRODUCED.joblib"
set "MODEL_FOR_WALL2WALL=%MAIN_MODEL_REPRO%"

REM ------------------------------------------------------------
REM Shared modeling parameters
REM ------------------------------------------------------------
set "HOLDOUT_BBOX=10.96,55.25,24.17,69.06"
set "META_C_GRID=0.01,0.1,1,3,10"
set "NSPLITS=5"

REM ------------------------------------------------------------
REM Wall-to-wall prediction settings
REM Exact values from the successful original command.
REM ------------------------------------------------------------
set "WIND_VAR=max_wind_speed"
set "W2W_BBOX4326=-11.5,34.0,42.5,72.5"
set "W2W_HEX_BBOX4326=-11.5,34.0,42.5,72.5"
set "HEX_AREA_KM2=2165"
set "FOREST_FRAC_MIN=0.10"
set "PRED_QUANTILE=0.95"
set "VALID_PERIODS_MIN_PER_HEX=10"
set "W2W_NORM=power"
set "W2W_GAMMA=0.55"
set "W2W_QHI=99"

REM Compatibility aliases used by older wall-to-wall BAT variants.
set "BBOX4326=%W2W_BBOX4326%"
set "MAP_NORM=%W2W_NORM%"
set "MAP_GAMMA=%W2W_GAMMA%"
set "MAP_QHI=%W2W_QHI%"

REM ------------------------------------------------------------
REM Fig. 1 settings
REM Fig. 1 retains its original display extent and reads the exact
REM wall-to-wall hex geometry from HEX_GEOMETRY_CSV.
REM ------------------------------------------------------------
set "FIG1_HEX_BBOX4326=-11,34,45,72"
set "FIG1_PLOT_BBOX4326=-11,34,45,72"
set "HEX_BBOX4326=%FIG1_HEX_BBOX4326%"
set "ACTIVE_QUANTILE=0.95"

REM ------------------------------------------------------------
REM Downstream table paths
REM ------------------------------------------------------------
set "HEX_PERIOD_CSV=%WALL2WALL_OUTDIR%\hex_period_summary_wall2wall_windonly_final.csv"
set "HEX_DISTRIBUTION_CSV=%WALL2WALL_OUTDIR%\hex_distribution_wall2wall_windonly_final.csv"
set "HEX_GEOMETRY_CSV=%WALL2WALL_OUTDIR%\hex_grid_geometry_wall2wall_3035.csv"

set "HEX_STYLEMATCH_CSV=%FIG1_OUTDIR%\hex_indicator_summary_stylematch_4326.csv"
set "FIG1_INDICATOR_CSV=%HEX_STYLEMATCH_CSV%"
set "PRED_INDICATOR_CSV=%HEX_STYLEMATCH_CSV%"
set "PRED_TEMP_CSV=%FIG2_OUTDIR%\hex_monthly_spatiotemporal_likelihood_metrics.csv"

REM Fig. 4 main-text setting.
set "FIG4_BACKGROUNDS=long_term"
set "REGIME_BACKGROUNDS=%FIG4_BACKGROUNDS%"
set "FIG4_MERGED_CSV=%FIG4_PREP_OUTDIR%\long_term\merged_prediction_anomaly_long_term.csv"
set "REGIME_MERGED_CSV=%FIG4_MERGED_CSV%"

REM Fig. 5 defaults.
set "FIG5_BACKGROUND=long_term"
set "FIG5_WIND_PROFILE_METRIC=frequency"

REM Fig. 5 locked reproduction settings.
REM The confirmed original Fig. 5 agreement classes use complete hex-area
REM weights of 2165 km2 per analysed hex, not forest_area_km2 weights.
set "FIG5_FORCE_REBUILD=1"
set "FIG5_FORCE_WIND_FROM_NC=1"
set "FIG5_REWRITE_TABLES_FROM_EXISTING_CSV=0"
set "FIG5_AREA_WEIGHT_MODE=hex"
set "FIG5_HEX_AREA_KM2=%HEX_AREA_KM2%"
