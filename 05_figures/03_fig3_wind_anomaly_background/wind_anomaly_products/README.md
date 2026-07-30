# Wind anomaly products workflow

This folder closes the upstream wind-anomaly-product step between the integrated ERA5 wind NetCDF and the Fig.3/Fig.4/Fig.5 analyses.

## Run order

```text
00_run_wind_anomaly_products.bat
```

or manually:

```text
01_build_wind_spacetime_products_from_ERA5.py
02_build_overlay_netcdf_from_products.py
```
