# -*- coding: utf-8 -*-
"""
script for European wind spatial + temporal background maps.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm

from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box


# =========================================================
# USER SETTINGS
# =========================================================
INPUT_NC = "E:/ERA/ERA5 16 combine/EU_final_structural_wind_features.nc"
VAR_NAME = "max_wind_speed"

OUTDIR = Path("E:/RF B+G/wind_spacetime_maps_europe_main_admin")
OUTDIR.mkdir(parents=True, exist_ok=True)

# If online access fails on HPC, replace this with your local Natural Earth shapefile/zip.
COUNTRIES_SRC = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"

# Europe main administrative domain
MAIN_LON_MIN, MAIN_LON_MAX = -11.5, 42.5
MAIN_LAT_MIN, MAIN_LAT_MAX = 34.0, 72.5

EXCLUDE_COUNTRIES = {"russia", "russian federation", "iceland"}

# Wind thresholds
Z_THRESHOLD = 2.0       # anomaly event threshold
STD_FLOOR = 1.0         # m/s, avoid unstable z-scores where local std is tiny
Q95 = 0.95              # absolute wind reference

# Timing mask: only interpret peak month when enough high-wind events exist
MIN_HIGHWIND_COUNT_FOR_TIMING = 3

# Figure style
BOUNDARY_COLOR = "black"
BOUNDARY_LW = 0.45
OCEAN_GREY = "#d9d9d9"
DPI = 300


# =========================================================
# BASIC HELPERS
# =========================================================
def normalize_lon_to_180(lon):
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def maybe_fix_dataset_lon(ds, lon_name):
    lon_fixed = normalize_lon_to_180(ds[lon_name].values)
    return ds.assign_coords({lon_name: lon_fixed}).sortby(lon_name)


def detect_country_name_col(gdf):
    candidates = [
        "ADMIN", "admin",
        "NAME", "name",
        "NAME_LONG", "name_long",
        "SOVEREIGNT", "sovereignt"
    ]
    for c in candidates:
        if c in gdf.columns:
            return c
    raise ValueError(f"Could not detect country-name column. Available columns: {list(gdf.columns)}")


def detect_continent_col(gdf):
    for c in ["CONTINENT", "continent"]:
        if c in gdf.columns:
            return c
    return None


def load_europe_admin_boundaries(countries_src, lon_min, lon_max, lat_min, lat_max):
    """
    Load administrative Europe polygons, exclude Russia and Iceland,
    then geometrically clip to the main Europe bbox.
    """
    gdf = gpd.read_file(countries_src)
    if gdf.empty:
        raise ValueError(f"Boundary source is empty: {countries_src}")
    if gdf.crs is None:
        raise ValueError("Boundary source has no CRS information.")

    gdf = gdf.to_crs("EPSG:4326")

    name_col = detect_country_name_col(gdf)
    continent_col = detect_continent_col(gdf)

    if continent_col is not None:
        europe = gdf[gdf[continent_col].astype(str).str.lower() == "europe"].copy()
    else:
        warnings.warn("No continent column found; fallback to all countries before clipping.")
        europe = gdf.copy()

    europe = europe[
        ~europe[name_col].astype(str).str.lower().isin(EXCLUDE_COUNTRIES)
    ].copy()

    clip_geom = box(lon_min, lat_min, lon_max, lat_max)
    europe["geometry"] = europe.geometry.intersection(clip_geom)
    europe = europe[~europe.geometry.is_empty].copy()
    europe = europe.explode(index_parts=False).reset_index(drop=True)

    if europe.empty:
        raise ValueError("No European administrative polygons found after filtering/clipping.")

    return europe


def build_mask_from_polygons(lon, lat, poly_gdf):
    """
    Rasterize polygons to target lon/lat grid.
    Returns 2D bool mask: True = inside selected polygons.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)

    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("lon and lat must be 1D arrays.")
    if len(lon) < 2 or len(lat) < 2:
        raise ValueError("Longitude/latitude arrays must each contain at least 2 values.")

    lon_fixed = normalize_lon_to_180(lon)

    lon_sort_idx = np.argsort(lon_fixed)
    lat_sort_idx = np.argsort(lat)

    lon_sorted = lon_fixed[lon_sort_idx]
    lat_sorted = lat[lat_sort_idx]

    poly_gdf = poly_gdf.to_crs("EPSG:4326")
    poly_gdf = poly_gdf.cx[
        float(lon_sorted.min()):float(lon_sorted.max()),
        float(lat_sorted.min()):float(lat_sorted.max())
    ].copy()

    if poly_gdf.empty:
        raise ValueError("No polygons intersect the target extent.")

    try:
        geom = poly_gdf.geometry.union_all()
    except Exception:
        geom = poly_gdf.unary_union

    dx = float(np.median(np.diff(lon_sorted)))
    dy = float(np.median(np.diff(lat_sorted)))

    west  = float(lon_sorted.min() - dx / 2.0)
    east  = float(lon_sorted.max() + dx / 2.0)
    south = float(lat_sorted.min() - dy / 2.0)
    north = float(lat_sorted.max() + dy / 2.0)

    transform = from_bounds(
        west, south, east, north,
        len(lon_sorted), len(lat_sorted)
    )

    mask_desc = rasterize(
        [(geom, 1)],
        out_shape=(len(lat_sorted), len(lon_sorted)),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False
    ).astype(bool)

    # raster row 0 = north, while lat_sorted is south -> north
    mask_sorted = np.flipud(mask_desc)

    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    mask[np.ix_(lat_sort_idx, lon_sort_idx)] = mask_sorted

    return mask, lon_fixed


def apply_mask_to_dataarray(da, mask_2d, lat_name, lon_name):
    mask_da = xr.DataArray(
        mask_2d,
        coords={lat_name: da[lat_name], lon_name: da[lon_name]},
        dims=(lat_name, lon_name),
        name="europe_admin_mask"
    )
    return da.where(mask_da)


def apply_mask_to_dataset(ds, mask_2d, lat_name, lon_name):
    out = ds.copy()
    for v in out.data_vars:
        dims = out[v].dims
        if lat_name in dims and lon_name in dims:
            out[v] = apply_mask_to_dataarray(out[v], mask_2d, lat_name, lon_name)
    return out


def scalar_quantile(da, q):
    arr = np.asarray(da.values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.quantile(arr, q))


def rank_pct_2d(da):
    """
    Convert a 2D DataArray to percentile rank among valid cells.
    Output range: 0-1.
    """
    arr = np.asarray(da.values, dtype=float)
    valid = np.isfinite(arr)

    out = np.full(arr.shape, np.nan, dtype=np.float32)
    vals = arr[valid]

    if vals.size == 0:
        return xr.full_like(da, np.nan, dtype=np.float32)

    order = np.argsort(vals)
    ranks = np.empty(vals.size, dtype=np.float32)
    ranks[order] = np.arange(vals.size, dtype=np.float32)

    if vals.size > 1:
        pct = ranks / (vals.size - 1)
    else:
        pct = np.zeros_like(ranks, dtype=np.float32)

    out[valid] = pct

    return xr.DataArray(
        out,
        coords=da.coords,
        dims=da.dims,
        name=f"{da.name}_rank_pct" if da.name else "rank_pct"
    )


def calc_effective_months(month_share):
    """
    Effective number of months = 1 / sum(p_m^2).
    Higher values = more temporally diffuse high-wind timing.
    Lower values = stronger monthly concentration.
    """
    return 1.0 / (month_share ** 2).sum(dim="month", skipna=True)


def calc_monthly_timing_metrics(event_bool, lat_name, lon_name, min_count=3):
    """
    event_bool: DataArray(time, lat, lon), bool/int
    Returns:
    - monthly_count: number of high-wind events in each month
    - monthly_share: monthly_count / total_count
    - total_count
    - peak_month
    - peak_month_share
    - effective_months
    """
    monthly_count = event_bool.groupby("time.month").sum(dim="time", skipna=True)
    total_count = monthly_count.sum(dim="month", skipna=True)

    monthly_share = monthly_count / total_count.where(total_count > 0)

    peak_month = monthly_share.idxmax(dim="month", skipna=True)
    peak_month_share = monthly_share.max(dim="month", skipna=True)
    effective_months = calc_effective_months(monthly_share)

    valid_timing = total_count >= min_count

    peak_month = peak_month.where(valid_timing)
    peak_month_share = peak_month_share.where(valid_timing)
    effective_months = effective_months.where(valid_timing)

    return monthly_count, monthly_share, total_count, peak_month, peak_month_share, effective_months


def years_span_from_time(time_values):
    t0 = np.datetime64(time_values[0])
    t1 = np.datetime64(time_values[-1])
    days = (t1 - t0) / np.timedelta64(1, "D")
    years = float(days) / 365.25
    return years if years > 0 else np.nan


# =========================================================
# PLOTTING HELPERS
# =========================================================
def add_boundaries(ax, boundaries_gdf):
    boundaries_gdf.boundary.plot(
        ax=ax,
        color=BOUNDARY_COLOR,
        linewidth=BOUNDARY_LW,
        zorder=5
    )


def format_geo_axis(ax):
    ax.set_xlim(MAIN_LON_MIN, MAIN_LON_MAX)
    ax.set_ylim(MAIN_LAT_MIN, MAIN_LAT_MAX)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def plot_da_panel(
    ax,
    da,
    boundaries_gdf,
    title,
    cmap="magma",
    vmin=None,
    vmax=None,
    cbar_label=None,
    norm=None,
    cbar_ticks=None,
    cbar_ticklabels=None
):
    ax.set_facecolor(OCEAN_GREY)

    cmap_obj = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
    cmap_obj.set_bad(color=OCEAN_GREY)

    im = da.plot(
        ax=ax,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        add_colorbar=False
    )

    add_boundaries(ax, boundaries_gdf)
    format_geo_axis(ax)
    ax.set_title(title, fontsize=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.015)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    if cbar_ticklabels is not None:
        cbar.set_ticklabels(cbar_ticklabels)

    return im


def make_month_cmap():
    """
    Discrete 12-month colormap.
    """
    colors = [
        "#313695", "#4575b4", "#74add1", "#abd9e9",
        "#e0f3f8", "#ffffbf", "#fee090", "#fdae61",
        "#f46d43", "#d73027", "#a50026", "#7f0000"
    ]
    cmap = ListedColormap(colors)
    bounds = np.arange(0.5, 13.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def plot_wind_spacetime_6panel(
    products,
    prefix,
    boundaries_gdf,
    out_png,
    title_prefix
):
    """
    6-panel figure for section 3.4:
    a max z anomaly
    b anomaly frequency
    c score = pct(z) - pct(freq)
    d peak month of anomaly occurrence
    e peak-month share
    f effective number of months
    """
    max_z = products[f"max_z_{prefix}"]
    freq_per_year = products[f"freq_{prefix}_per_year"]
    score = products[f"score_{prefix}"]
    peak_month = products[f"peak_month_{prefix}"]
    peak_share = products[f"peak_month_share_{prefix}"]
    effective_months = products[f"effective_months_{prefix}"]

    z_vmin = scalar_quantile(max_z, 0.02)
    z_vmax = scalar_quantile(max_z, 0.98)

    freq_vmin = 0
    freq_vmax = scalar_quantile(freq_per_year, 0.98)

    share_vmin = scalar_quantile(peak_share, 0.02)
    share_vmax = scalar_quantile(peak_share, 0.98)

    eff_vmin = 1
    eff_vmax = min(12, scalar_quantile(effective_months, 0.98))

    month_cmap, month_norm = make_month_cmap()

    fig, axes = plt.subplots(
        2, 3,
        figsize=(13.5, 8.6),
        facecolor="white",
        constrained_layout=True
    )

    plot_da_panel(
        axes[0, 0],
        max_z,
        boundaries_gdf,
        "(a) Maximum standardized wind anomaly",
        cmap="magma",
        vmin=z_vmin,
        vmax=z_vmax,
        cbar_label="z anomaly"
    )

    plot_da_panel(
        axes[0, 1],
        freq_per_year,
        boundaries_gdf,
        f"(b) Anomaly frequency, z > {Z_THRESHOLD}",
        cmap="magma",
        vmin=freq_vmin,
        vmax=freq_vmax,
        cbar_label="count yr$^{-1}$"
    )

    plot_da_panel(
        axes[0, 2],
        score,
        boundaries_gdf,
        "(c) Intensity-frequency contrast",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
        cbar_label="pct(z) - pct(freq)"
    )

    plot_da_panel(
        axes[1, 0],
        peak_month,
        boundaries_gdf,
        "(d) Peak month of anomalous wind",
        cmap=month_cmap,
        norm=month_norm,
        cbar_label="month",
        cbar_ticks=np.arange(1, 13),
        cbar_ticklabels=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    )

    plot_da_panel(
        axes[1, 1],
        peak_share,
        boundaries_gdf,
        "(e) Peak-month share",
        cmap="viridis",
        vmin=share_vmin,
        vmax=share_vmax,
        cbar_label="share"
    )

    plot_da_panel(
        axes[1, 2],
        effective_months,
        boundaries_gdf,
        "(f) Effective number of months",
        cmap="viridis_r",
        vmin=eff_vmin,
        vmax=eff_vmax,
        cbar_label="months"
    )

    fig.suptitle(title_prefix, fontsize=14, y=1.02)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {out_png}")


def save_single_map(da, boundaries_gdf, out_png, title, cmap="magma", vmin=None, vmax=None, cbar_label=None, norm=None):
    fig, ax = plt.subplots(figsize=(9.5, 8), facecolor="white")
    plot_da_panel(
        ax,
        da,
        boundaries_gdf,
        title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_label=cbar_label,
        norm=norm
    )
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved map: {out_png}")


# =========================================================
# LOAD DATA
# =========================================================
print("Opening dataset...")
ds = xr.open_dataset(INPUT_NC)

if "longitude" in ds.coords:
    lon_name = "longitude"
elif "lon" in ds.coords:
    lon_name = "lon"
else:
    raise ValueError("Cannot find longitude coordinate: expected 'longitude' or 'lon'.")

if "latitude" in ds.coords:
    lat_name = "latitude"
elif "lat" in ds.coords:
    lat_name = "lat"
else:
    raise ValueError("Cannot find latitude coordinate: expected 'latitude' or 'lat'.")

if "time" not in ds.coords:
    raise ValueError("Cannot find time coordinate 'time'.")

if VAR_NAME not in ds.data_vars:
    raise ValueError(f"Cannot find variable '{VAR_NAME}'. Available variables: {list(ds.data_vars)}")

ds = maybe_fix_dataset_lon(ds, lon_name)

print("Cropping raw dataset to Europe main administrative domain...")
lat_desc = np.all(np.diff(ds[lat_name].values) < 0)
ds = ds.sel(
    {
        lon_name: slice(MAIN_LON_MIN, MAIN_LON_MAX),
        lat_name: slice(MAIN_LAT_MAX, MAIN_LAT_MIN) if lat_desc else slice(MAIN_LAT_MIN, MAIN_LAT_MAX)
    }
)

print("Loading European administrative boundaries...")
europe_admin = load_europe_admin_boundaries(
    COUNTRIES_SRC,
    MAIN_LON_MIN,
    MAIN_LON_MAX,
    MAIN_LAT_MIN,
    MAIN_LAT_MAX
)

print("Building administrative Europe mask...")
europe_mask, _ = build_mask_from_polygons(
    ds[lon_name].values,
    ds[lat_name].values,
    europe_admin
)

print("Applying administrative mask...")
ds = apply_mask_to_dataset(ds, europe_mask, lat_name, lon_name)
wind = ds[VAR_NAME]


# =========================================================
# CALCULATE WIND BACKGROUND PRODUCTS
# =========================================================
print("Calculating absolute wind reference layers...")
longterm_max = wind.max(dim="time", skipna=True)
longterm_q95 = wind.quantile(Q95, dim="time", skipna=True)

print("Calculating long-term standardized anomalies...")
mean_long = wind.mean(dim="time", skipna=True)
std_long = wind.std(dim="time", skipna=True)
std_long_safe = xr.where(std_long < STD_FLOOR, STD_FLOOR, std_long)

z_long = (wind - mean_long) / std_long_safe
event_long = z_long > Z_THRESHOLD

max_z_long = z_long.max(dim="time", skipna=True)
freq_long = event_long.sum(dim="time", skipna=True)

print("Calculating same-month standardized anomalies...")
monthly_mean = wind.groupby("time.month").mean(dim="time", skipna=True)
monthly_std = wind.groupby("time.month").std(dim="time", skipna=True)
monthly_std_safe = xr.where(monthly_std < STD_FLOOR, STD_FLOOR, monthly_std)

z_month = (wind.groupby("time.month") - monthly_mean).groupby("time.month") / monthly_std_safe
event_month = z_month > Z_THRESHOLD

max_z_month = z_month.max(dim="time", skipna=True)
freq_month = event_month.sum(dim="time", skipna=True)

print("Annualizing anomaly frequencies...")
years_span = years_span_from_time(ds["time"].values)
freq_long_per_year = freq_long / years_span
freq_month_per_year = freq_month / years_span

print("Calculating intensity-frequency contrast scores...")
score_long = rank_pct_2d(max_z_long) - rank_pct_2d(freq_long)
score_long.name = "score_long"

score_month = rank_pct_2d(max_z_month) - rank_pct_2d(freq_month)
score_month.name = "score_month"

print("Calculating monthly timing metrics for long-term anomaly events...")
(
    monthly_count_long,
    monthly_share_long,
    total_count_long,
    peak_month_long,
    peak_month_share_long,
    effective_months_long
) = calc_monthly_timing_metrics(
    event_long,
    lat_name,
    lon_name,
    min_count=MIN_HIGHWIND_COUNT_FOR_TIMING
)

print("Calculating monthly timing metrics for same-month anomaly events...")
(
    monthly_count_month,
    monthly_share_month,
    total_count_month,
    peak_month_month,
    peak_month_share_month,
    effective_months_month
) = calc_monthly_timing_metrics(
    event_month,
    lat_name,
    lon_name,
    min_count=MIN_HIGHWIND_COUNT_FOR_TIMING
)

# Apply mask again for safety
products = {
    "longterm_max": apply_mask_to_dataarray(longterm_max, europe_mask, lat_name, lon_name),
    "longterm_q95": apply_mask_to_dataarray(longterm_q95, europe_mask, lat_name, lon_name),

    "max_z_long": apply_mask_to_dataarray(max_z_long, europe_mask, lat_name, lon_name),
    "freq_long": apply_mask_to_dataarray(freq_long, europe_mask, lat_name, lon_name),
    "freq_long_per_year": apply_mask_to_dataarray(freq_long_per_year, europe_mask, lat_name, lon_name),
    "score_long": apply_mask_to_dataarray(score_long, europe_mask, lat_name, lon_name),

    "max_z_month": apply_mask_to_dataarray(max_z_month, europe_mask, lat_name, lon_name),
    "freq_month": apply_mask_to_dataarray(freq_month, europe_mask, lat_name, lon_name),
    "freq_month_per_year": apply_mask_to_dataarray(freq_month_per_year, europe_mask, lat_name, lon_name),
    "score_month": apply_mask_to_dataarray(score_month, europe_mask, lat_name, lon_name),

    "total_count_long": apply_mask_to_dataarray(total_count_long, europe_mask, lat_name, lon_name),
    "peak_month_long": apply_mask_to_dataarray(peak_month_long, europe_mask, lat_name, lon_name),
    "peak_month_share_long": apply_mask_to_dataarray(peak_month_share_long, europe_mask, lat_name, lon_name),
    "effective_months_long": apply_mask_to_dataarray(effective_months_long, europe_mask, lat_name, lon_name),

    "total_count_month": apply_mask_to_dataarray(total_count_month, europe_mask, lat_name, lon_name),
    "peak_month_month": apply_mask_to_dataarray(peak_month_month, europe_mask, lat_name, lon_name),
    "peak_month_share_month": apply_mask_to_dataarray(peak_month_share_month, europe_mask, lat_name, lon_name),
    "effective_months_month": apply_mask_to_dataarray(effective_months_month, europe_mask, lat_name, lon_name),
}

# Monthly count/share are 3D: month, lat, lon
monthly_count_long = monthly_count_long.where(monthly_count_long.notnull())
monthly_share_long = monthly_share_long.where(monthly_share_long.notnull())
monthly_count_month = monthly_count_month.where(monthly_count_month.notnull())
monthly_share_month = monthly_share_month.where(monthly_share_month.notnull())


# =========================================================
# SAVE NETCDF
# =========================================================
print("Saving NetCDF...")

out_ds = xr.Dataset(
    {
        **products,
        "monthly_count_long": monthly_count_long,
        "monthly_share_long": monthly_share_long,
        "monthly_count_month": monthly_count_month,
        "monthly_share_month": monthly_share_month,
    }
)

out_nc = OUTDIR / "wind_spacetime_products_europe_main_admin_no_russia_no_iceland.nc"
encoding = {v: {"zlib": True, "complevel": 4} for v in out_ds.data_vars}
out_ds.to_netcdf(out_nc, encoding=encoding)
print(f"Saved NetCDF: {out_nc}")


# =========================================================
# SAVE 6-PANEL FIGURES
# =========================================================
print("Saving 6-panel long-term figure...")
plot_wind_spacetime_6panel(
    products=products,
    prefix="long",
    boundaries_gdf=europe_admin,
    out_png=OUTDIR / "wind_spacetime_longterm_6panel.png",
    title_prefix="Long-term wind-anomaly magnitude, recurrence, and monthly timing"
)

print("Saving 6-panel same-month figure...")
plot_wind_spacetime_6panel(
    products=products,
    prefix="month",
    boundaries_gdf=europe_admin,
    out_png=OUTDIR / "wind_spacetime_samemonth_6panel.png",
    title_prefix="Same-month wind-anomaly magnitude, recurrence, and monthly timing"
)


# =========================================================
# SAVE SINGLE DIAGNOSTIC MAPS
# =========================================================
print("Saving selected single maps...")

save_single_map(
    products["longterm_max"],
    europe_admin,
    OUTDIR / "longterm_max_wind_speed.png",
    "Long-term maximum wind speed",
    cmap="viridis",
    vmin=scalar_quantile(products["longterm_max"], 0.02),
    vmax=scalar_quantile(products["longterm_max"], 0.98),
    cbar_label="wind speed"
)

save_single_map(
    products["longterm_q95"],
    europe_admin,
    OUTDIR / "longterm_q95_wind_speed.png",
    "Long-term 95th percentile wind speed",
    cmap="viridis",
    vmin=scalar_quantile(products["longterm_q95"], 0.02),
    vmax=scalar_quantile(products["longterm_q95"], 0.98),
    cbar_label="wind speed"
)

save_single_map(
    products["freq_long_per_year"],
    europe_admin,
    OUTDIR / "longterm_anomaly_frequency_per_year.png",
    f"Long-term anomaly frequency, z > {Z_THRESHOLD}",
    cmap="magma",
    vmin=0,
    vmax=scalar_quantile(products["freq_long_per_year"], 0.98),
    cbar_label="count yr$^{-1}$"
)

save_single_map(
    products["freq_month_per_year"],
    europe_admin,
    OUTDIR / "samemonth_anomaly_frequency_per_year.png",
    f"Same-month anomaly frequency, z > {Z_THRESHOLD}",
    cmap="magma",
    vmin=0,
    vmax=scalar_quantile(products["freq_month_per_year"], 0.98),
    cbar_label="count yr$^{-1}$"
)

print("All done.")
print(f"Output directory: {OUTDIR}")
