# ERA5 download and integration workflow

This folder contains the ERA5 data gathering and integration code used to build the wind NetCDF input.

## Run order

```text
01_download_era5_fg10_16day.py
02_download_era5_u10v10_16day.py
03_integrate_fg10_u10v10_to_structural_wind_features.py
```


### 01_download_era5_fg10_16day.py

Downloads ERA5 10 m wind gust since previous post-processing (`fg10`) from CDS in 16-day chunks for the Europe domain.

### 02_download_era5_u10v10_16day.py

Downloads ERA5 10 m u/v wind components (`u10`, `v10`) from CDS in matching 16-day chunks.

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

