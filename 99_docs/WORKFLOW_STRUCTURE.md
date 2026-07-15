# Workflow structure

This package was reorganized from the earlier technical layout (`00_bat/`, `scripts/`) into a workflow layout:

1. project configuration
2. data gathering
3. sampling and feature-table construction
4. model training and Table 1
5. wall-to-wall prediction
6. main figures Fig. 1–Fig. 5
7. supplementary figures Fig. S1–Fig. S3

The execution logic is still controlled by `00_project_config/00_config_paths.bat`.


## ERA5 wind data

ERA5 downloading and 16-day integration scripts are included under `01_data_gathering/01_ERA5_download_and_integration/`.
