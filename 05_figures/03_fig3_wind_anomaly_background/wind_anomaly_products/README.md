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

## Step 1

```text
01_build_wind_spacetime_products_from_ERA5.py
```

This is the recovered original-style script previously named:

```text
8extremewindmap_fixed.py
```

It reads:

```text
EU_final_structural_wind_features.nc
```

and creates:

```text
wind_spacetime_products_europe_main_admin_no_russia_no_iceland.nc
```

The products NetCDF includes, among other variables:

```text
max_z_long
freq_long_per_year
score_long
peak_month_long
peak_month_share_long
max_z_month
freq_month_per_year
score_month
peak_month_month
peak_month_share_month
longterm_max
longterm_q95
```

## Step 2

```text
02_build_overlay_netcdf_from_products.py
```

This is a lightweight converter that creates the overlay NetCDF files needed by Fig.4:

```text
overlay_result_long_term_background.nc
overlay_result_same_month_background.nc
```

Each overlay file contains:

```text
z_value
freq_value
```

The mapping is:

```text
long_term:
  z_value    = max_z_long
  freq_value = freq_long_per_year

same_month:
  z_value    = max_z_month
  freq_value = freq_month_per_year
```

## Notes

The exact standalone original script that created the overlay files was not found. The converter is included because the overlay NetCDFs are direct variable subsets/renames of the products NetCDF required by Fig.4.
