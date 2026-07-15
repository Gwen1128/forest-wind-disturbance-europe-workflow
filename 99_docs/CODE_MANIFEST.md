# Code manifest

## BAT entry points

- `00_bat/00_config_paths.bat`: central path and parameter configuration.
- `00_bat/00_check_inputs.bat`: checks required scripts and external inputs.
- `00_bat/01_table1_model_comparison.bat`: trains the Table 1 model-family comparison and saves temporary model artifacts.
- `00_bat/02_make_table1.bat`: builds Table 1 and copies the selected wind-only model for wall-to-wall prediction.
- `00_bat/03_prepare_clean_encoded_data.bat`: optional heavy data-preparation workflow to rebuild `clean_encoded_data_LOCKED.csv`.
- `00_bat/03A_optional_combine_feature_tiles.bat`: optional merge step for tiled feature CSVs.
- `00_bat/04_wall2wall_prediction.bat`: wall-to-wall wind-only prediction using the copied selected model.
- `00_bat/05_fig1_spatial_indicators.bat`: Fig. 1 spatial indicators.
- `00_bat/06_fig2_monthly_concentration.bat`: Fig. 2 monthly concentration.
- `00_bat/07_fig3_wind_anomaly_background.bat`: Fig. 3 wind-anomaly background.
- `00_bat/08_fig4_prepare_prediction_anomaly_regime.bat`: prepares Fig. 4 merged table.
- `00_bat/09_fig4_plot_prediction_anomaly_regime.bat`: plots Fig. 4.
- `00_bat/10_fig5_temporal_correspondence.bat`: plots Fig. 5 using the peak-month correspondence script.
- `00_bat/11_figS1_figS2_supplementary_maps.bat`: optional supplementary figures.
- `00_bat/90_run_all_paper_results.bat`: required end-to-end paper-results runner from `clean_encoded_data_LOCKED.csv`.
- `00_bat/91_run_figures_from_existing_outputs.bat`: figure-only runner from existing wall-to-wall outputs.

## Data-preparation scripts

- `00_data_preparation/scripts/00_GEE_LAI_extract.js`: Google Earth Engine LAI export.
- `00_data_preparation/scripts/01_bfast_lai_departure_sampling.R`: FORWIND sampling and LAI-departure calculation.
- `00_data_preparation/scripts/02_extract_geospatial_features.py`: DEM, ERA5, landcover, tree-height, and biome feature extraction.
- `00_data_preparation/scripts/03_optional_combine_feature_tiles.py`: optional merge of tiled feature CSVs.
- `00_data_preparation/scripts/04_preprocess_locked_modeling_table.py`: locked-schema preprocessing to `clean_encoded_data_LOCKED.csv`.

## Analysis scripts

- `scripts/train_model_family_comparison.py`: locked model-family comparison and saved model artifact generation.
- `01_helpers/make_table1_from_metrics.py`: Table 1 assembly.
- `scripts/18wall2wall_windonly_disturbance_mapping_FINAL.py`: wall-to-wall prediction.
- `scripts/18hex_period_indicator_suite_STYLEMATCH_3035_PUBLICATION_NO_AUTOGEOM.py`: Fig. 1 indicators.
- `scripts/06B_fig2_prediction_monthly_concentration.py`: Fig. 2.
- `scripts/06C_fig3_wind_anomaly_background.py`: Fig. 3.
- `scripts/06D_fig4_prediction_anomaly_regime_prepare.py`: Fig. 4 preparation.
- `scripts/06E_fig4_prediction_anomaly_regime_plot.py`: Fig. 4 plot.
- `scripts/06F_fig5_temporal_correspondence_peak_month.py`: Fig. 5.
- `scripts/06A_figS1_figS2_supplementary_maps.py`: supplementary maps.
- `scripts/06F0_download_efda_zenodo.py`, `06F1_inspect_efda_files.py`, `06F2_aggregate_efda_to_hex_year.py`, `06F3_compare_prediction_with_efda.py`: optional EFDA support workflow.

## Removed from the minimal workflow

The duplicate standalone main-model training/scoring BATs were removed. The retained workflow trains once in `01_table1_model_comparison.bat`, then `02_make_table1.bat` copies the selected wind-only model artifact for wall-to-wall prediction.
