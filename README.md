# Forest wind-disturbance workflow
Code used for the manuscript “Two decades of estimated forest wind disturbance reveal recurrent and episodic patterns across Europe”.

## Workflow
1. Configure paths
   Edit `00_project_config/00_config_paths.bat`.

2. Download and prepare input data
   Run `01_data_gathering/00_GEE_LAI_extract.js` in Google Earth Engine.
   Run `01_data_gathering/01_ERA5_download_and_integration/00_run_ERA5_download_and_integration.bat` to download and integrate ERA5 wind data.

3. Prepare the modeling table
   Run `02_sampling_and_feature_table/00_prepare_clean_encoded_data.bat`.

4. Train and compare models
   Run `03_model_training/00_model_comparison.bat`.
   Run `03_model_training/01_make_table1_and_copy_locked_model.bat` to generate Table 1 and copy the wind-only model used in the following analyses.

5. Generate wall-to-wall predictions
   Run `04_wall_to_wall_prediction/00_wall2wall_prediction.bat`.

6. Generate the main figures
   Run the batch files in `05_figures` in order from Figure 1 to Figure 5. For Figure 4, run the preparation script before the plotting script.

7. Generate the supplementary figures
   Run the batch files in `06_supplementary_figures`.

Input data and intermediate outputs are not included. Local file paths should be adjusted before running the workflow.
