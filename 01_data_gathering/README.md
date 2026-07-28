# 01 Data gathering

This folder contains retrieving wind data from ERA5, and the Google Earth Engine script used to export the MODIS LAI time series used upstream.


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



