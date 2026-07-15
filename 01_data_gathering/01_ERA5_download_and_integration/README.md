# ERA5 download and integration workflow

This folder contains the ERA5 data gathering and integration code used to build the wind NetCDF input used by the later wall-to-wall prediction step.

## Run order

```text
01_download_era5_fg10_16day.py
02_download_era5_u10v10_16day.py
03_integrate_fg10_u10v10_to_structural_wind_features.py
```

## What each script does

### 01_download_era5_fg10_16day.py

Downloads ERA5 10 m wind gust since previous post-processing (`fg10`) from CDS in 16-day chunks for the Europe domain.

Default period in the script:

```text
2001-01-01 to 2023-12-31
```

Default domain:

```text
North 71.2, West -31.5, South 23.75, East 41.5
```

### 02_download_era5_u10v10_16day.py

Downloads ERA5 10 m u/v wind components (`u10`, `v10`) from CDS in matching 16-day chunks for the same Europe domain and period.

### 03_integrate_fg10_u10v10_to_structural_wind_features.py

Pairs the `fg10` and `u10/v10` chunks by date, computes:

```text
max_wind_speed
wind_direction
```

and writes:

```text
EU_final_structural_wind_features.nc
```

This NetCDF is the `WIND_NC` input used later by:

```text
04_wall_to_wall_prediction/00_wall2wall_prediction.bat
```

## Notes

These ERA5 scripts are upstream data gathering scripts. They require:

```text
cdsapi
xarray
dask
netCDF4 / h5netcdf depending on local environment
```

They also require a configured CDS API key.

The scripts currently retain the original local path style. Adjust the directories at the top of each script before running on a new machine.
