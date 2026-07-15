# Package static check report
Checked items:
- workflow directory structure
- ERA5 download/integration scripts included
- required local script paths in 00_config_paths.bat resolve inside package
- all Python files compile successfully
- BAT files checked for accidental NUL path characters
- Fig.5 locked hex-area Python and verification BAT included
- FigS1 EFDA workflow placed under 06_supplementary_figures/01_figS1_EFDA_validation
- FigS2/FigS3 supplementary-map script relabeled from old FigS1/FigS2 names

Known external inputs still required:
- clean_encoded_data_LOCKED.csv unless rebuilding the modeling table
- forest mask GeoTIFF
- ERA5 structural wind NetCDF or outputs from the included ERA5 workflow
- country boundary shapefile
- wind-anomaly product NetCDFs used by Fig.3 and Fig.4 (`WIND_PRODUCTS_NC`, `WIND_OVERLAY_LONG_NC`, `WIND_OVERLAY_MONTH_NC`)

Important note:
The code that generated the wind-anomaly product NetCDFs for Fig.3/Fig.4 is not included in this package; a placeholder note is provided in `01_data_gathering/02_wind_anomaly_products/`.


## Wind anomaly products

Wind anomaly products workflow was added:

```text
01_data_gathering/02_wind_anomaly_products/
  00_run_wind_anomaly_products.bat
  01_build_wind_spacetime_products_from_ERA5.py
  02_build_overlay_netcdf_from_products.py
```

The recovered products script generates `wind_spacetime_products_europe_main_admin_no_russia_no_iceland.nc`.
The overlay converter derives the Fig.4 overlay NetCDFs from this products file.
