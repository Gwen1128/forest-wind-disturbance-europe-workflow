# 01 Data gathering

This folder contains the Google Earth Engine script used to export the MODIS LAI time series used upstream.

Large external datasets are **not** stored in this repository. Configure their local paths in:

```text
00_project_config/00_config_paths.bat
```


## ERA5 wind data

ERA5 wind downloading and integration scripts are stored in:

```text
01_data_gathering/01_ERA5_download_and_integration/
```

Run order:

```text
01_download_era5_fg10_16day.py
02_download_era5_u10v10_16day.py
03_integrate_fg10_u10v10_to_structural_wind_features.py
```

The final output is:

```text
EU_final_structural_wind_features.nc
```

which is configured as `WIND_NC` in `00_project_config/00_config_paths.bat`.


## Wind-anomaly product generation

Fig. 3 and Fig. 4 use precomputed wind-anomaly products configured in `00_project_config/00_config_paths.bat`. A placeholder/notes folder is included at:

```text
01_data_gathering/02_wind_anomaly_products/
```

