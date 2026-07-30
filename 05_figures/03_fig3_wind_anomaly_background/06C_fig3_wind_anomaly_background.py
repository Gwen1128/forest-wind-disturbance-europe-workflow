# -*- coding: utf-8 -*-
"""
Plot the revised 5-panel European wind-anomaly figure.
"""

from pathlib import Path
import os
import warnings

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Patch
from pyproj import Transformer
from shapely.geometry import box


def open_xarray_dataset_compat(path):
    """Open NetCDF with fallback engines, useful when netCDF4 DLL is broken."""
    last_err = None
    for engine in ["h5netcdf", "scipy", "netcdf4", None]:
        try:
            if engine is None:
                print(f"Opening dataset with xarray default engine: {path}")
                return xr.open_dataset(path)
            print(f"Opening dataset with engine={engine}: {path}")
            return xr.open_dataset(path, engine=engine)
        except Exception as e:
            print(f"  engine {engine or 'default'} failed: {e}")
            last_err = e
    raise last_err

# =========================================================
# USER SETTINGS
# =========================================================
INPUT_PRODUCTS_NC = Path(os.environ.get("WIND_PRODUCTS_NC", r"E:/RF B+G/your_path/wind_spacetime_products_europe_main_admin_no_russia_no_iceland.nc"))

OUTDIR = Path(os.environ.get("FIG3_OUTDIR", r"E:/RF B+G/your_path"))
OUTDIR.mkdir(parents=True, exist_ok=True)

# Use your local country boundary shapefile.
COUNTRIES_SRC = os.environ.get("COUNTRIES_SRC", r"E:/CNTR_RG_20M_2024_3035/CNTR_RG_20M_2024_3035.shp")
# Alternative Natural Earth source if needed:
# COUNTRIES_SRC = r"https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"

# Choose which anomaly background to plot: "long" or "month".
BACKGROUND = "long"

VAR_MAX_Z = f"max_z_{BACKGROUND}"
VAR_FREQ = f"freq_{BACKGROUND}_per_year"
VAR_PEAK_MONTH = f"peak_month_{BACKGROUND}"
VAR_PEAK_SHARE = f"peak_month_share_{BACKGROUND}"

# Bivariate high/low split based on percentile rank among valid cells.
# 0.80 = top 20% high-intensity/high-frequency classes.
BIVAR_HIGH_PERCENTILE = 0.75

# Main Europe extent, defined in lon/lat before projecting.
MAIN_LON_MIN, MAIN_LON_MAX = -11.5, 42.5
MAIN_LAT_MIN, MAIN_LAT_MAX = 34.0, 72.5

PLOT_CRS = "EPSG:3035"       # ETRS89-LAEA Europe
GEOGRAPHIC_CRS = "EPSG:4326"

FIG_DPI = 300
BOUNDARY_COLOR = "black"
BOUNDARY_LW = 0.45
RASTERIZED = True

# Country filtering: this removes extra surrounding country boundaries.
# Includes mainland Europe and nearby European states; excludes Russia and Iceland.
EUROPE_COUNTRY_CODES = {
    "AL", "AD", "AT", "BE", "BA", "BG", "BY", "HR", "CY", "CZ", "DK", "EE",
    "FI", "FR", "DE", "GR", "EL", "HU", "IE", "IT", "XK", "LV", "LI", "LT",
    "LU", "MT", "MD", "MC", "ME", "NL", "MK", "NO", "PL", "PT", "RO", "SM",
    "RS", "SK", "SI", "ES", "SE", "CH", "UA", "GB", "UK", "VA"
}

EUROPE_COUNTRY_NAMES = {
    "albania", "andorra", "austria", "belgium", "bosnia and herz.",
    "bosnia and herzegovina", "bulgaria", "belarus", "croatia", "cyprus",
    "czechia", "czech republic", "denmark", "estonia", "finland", "france",
    "germany", "greece", "hungary", "ireland", "italy", "kosovo", "latvia",
    "liechtenstein", "lithuania", "luxembourg", "malta", "moldova", "monaco",
    "montenegro", "netherlands", "north macedonia", "norway", "poland", "portugal",
    "romania", "san marino", "serbia", "slovakia", "slovenia", "spain", "sweden",
    "switzerland", "ukraine", "united kingdom", "vatican", "vatican city"
}

EXCLUDE_COUNTRIES = {"russia", "russian federation", "iceland"}
EXCLUDE_CODES = {"RU", "RUS", "IS", "ISL"}


# =========================================================
# BASIC HELPERS
# =========================================================
def detect_lon_lat_names(ds):
    lon_candidates = ["longitude", "lon", "x"]
    lat_candidates = ["latitude", "lat", "y"]
    lon_name = next((c for c in lon_candidates if c in ds.coords), None)
    lat_name = next((c for c in lat_candidates if c in ds.coords), None)
    if lon_name is None or lat_name is None:
        raise ValueError(f"Cannot detect lon/lat coordinate names. Available coords: {list(ds.coords)}")
    return lon_name, lat_name


def normalize_lon_to_180(lon):
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def maybe_fix_lon(ds, lon_name):
    lon = ds[lon_name].values
    lon_fixed = normalize_lon_to_180(lon)
    if np.nanmax(np.abs(lon - lon_fixed)) > 1e-6:
        ds = ds.assign_coords({lon_name: lon_fixed}).sortby(lon_name)
    return ds


def check_required_vars(ds, var_names):
    missing = [v for v in var_names if v not in ds.data_vars]
    if missing:
        raise ValueError(
            f"Missing required variable(s): {missing}\n"
            f"Available variables: {list(ds.data_vars)}"
        )


def infer_edges_1d(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Coordinate must be a 1D array with at least two values.")
    mid = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (mid[0] - values[0])
    last = values[-1] + (values[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def transform_grid_edges(lon, lat):
    lon_edges = infer_edges_1d(lon)
    lat_edges = infer_edges_1d(lat)
    lon2, lat2 = np.meshgrid(lon_edges, lat_edges)
    transformer = Transformer.from_crs(GEOGRAPHIC_CRS, PLOT_CRS, always_xy=True)
    x2, y2 = transformer.transform(lon2, lat2)
    return x2, y2


def scalar_quantile(da, q):
    arr = np.asarray(da.values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.quantile(arr, q))


def percentile_rank_2d(da):
    arr = np.asarray(da.values, dtype=float)
    valid = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    vals = arr[valid]
    if vals.size == 0:
        return xr.full_like(da, np.nan, dtype=np.float32)
    order = np.argsort(vals)
    ranks = np.empty(vals.size, dtype=np.float32)
    ranks[order] = np.arange(vals.size, dtype=np.float32)
    out[valid] = ranks / max(vals.size - 1, 1)
    return xr.DataArray(out, coords=da.coords, dims=da.dims)


def build_bivariate_class(max_z, freq):
    """
    Codes:
    0 Background             = low intensity, low frequency
    1 Frequency-dominated              = low intensity, high frequency
    2 Intensity-dominated    = high intensity, low frequency
    3 Compound               = high intensity, high frequency
    """
    intensity_pct = percentile_rank_2d(max_z)
    frequency_pct = percentile_rank_2d(freq)

    valid = np.isfinite(intensity_pct.values) & np.isfinite(frequency_pct.values)
    high_i = intensity_pct.values >= BIVAR_HIGH_PERCENTILE
    high_f = frequency_pct.values >= BIVAR_HIGH_PERCENTILE

    out = np.full(intensity_pct.shape, np.nan, dtype=np.float32)
    out[valid & (~high_i) & (~high_f)] = 0
    out[valid & (~high_i) & high_f] = 1
    out[valid & high_i & (~high_f)] = 2
    out[valid & high_i & high_f] = 3

    return xr.DataArray(out, coords=max_z.coords, dims=max_z.dims, name="bivariate_class")


def detect_country_code_col(gdf):
    candidates = ["CNTR_ID", "ISO_A2", "ISO_A2_EH", "ADM0_A3", "ISO3_CODE", "GU_A3"]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def detect_country_name_col(gdf):
    candidates = ["NAME_ENGL", "NAME", "name", "ADMIN", "admin", "NAME_LONG", "CNTR_NAME"]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def detect_continent_col(gdf):
    for c in ["CONTINENT", "continent"]:
        if c in gdf.columns:
            return c
    return None


def load_europe_admin_boundaries(src):
    """
    Load boundaries, remove non-European neighboring countries, exclude Russia/Iceland,
    clip to the plotting bbox, and project to EPSG:3035.
    """
    gdf = gpd.read_file(src)
    if gdf.empty:
        raise ValueError(f"Boundary source is empty: {src}")
    if gdf.crs is None:
        raise ValueError("Boundary source has no CRS information.")

    gdf = gdf.to_crs(GEOGRAPHIC_CRS)

    continent_col = detect_continent_col(gdf)
    if continent_col is not None:
        gdf = gdf[gdf[continent_col].astype(str).str.lower() == "europe"].copy()

    code_col = detect_country_code_col(gdf)
    name_col = detect_country_name_col(gdf)

    if code_col is not None:
        codes = gdf[code_col].astype(str).str.upper()
        keep_by_code = codes.isin(EUROPE_COUNTRY_CODES) & ~codes.isin(EXCLUDE_CODES)
        gdf = gdf[keep_by_code].copy()
    elif name_col is not None:
        names = gdf[name_col].astype(str).str.lower()
        keep_by_name = names.isin(EUROPE_COUNTRY_NAMES) & ~names.isin(EXCLUDE_COUNTRIES)
        gdf = gdf[keep_by_name].copy()
    else:
        warnings.warn(
            "No country code/name column detected. Falling back to bbox-only clipping; "
            "extra surrounding boundaries may remain."
        )

    if name_col is not None and not gdf.empty:
        names = gdf[name_col].astype(str).str.lower()
        gdf = gdf[~names.isin(EXCLUDE_COUNTRIES)].copy()

    bbox_geom = box(MAIN_LON_MIN, MAIN_LAT_MIN, MAIN_LON_MAX, MAIN_LAT_MAX)
    gdf["geometry"] = gdf.geometry.intersection(bbox_geom)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    if gdf.empty:
        raise ValueError("No boundary polygons remain after filtering and bbox clipping.")

    return gdf.to_crs(PLOT_CRS)


def set_transparent_axis(ax, boundaries_gdf):
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.set_axis_off()
    xmin, ymin, xmax, ymax = boundaries_gdf.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def add_boundaries(ax, boundaries_gdf):
    boundaries_gdf.boundary.plot(ax=ax, color=BOUNDARY_COLOR, linewidth=BOUNDARY_LW, zorder=5)


def transparent_cmap(name):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad((1, 1, 1, 0))
    return cmap


def make_peak_month_cmap():
    # Continuous/cyclic month band: deep purple at both ends.
    colors = [
        "#2c115f", "#3b2f8f", "#4169c1", "#4ca3d9",
        "#7bc8d6", "#c9e7b6", "#f2e88b", "#f7c25a",
        "#ef8a47", "#d24b40", "#9d2b7f", "#5a1e96", "#2c115f"
    ]
    cmap = LinearSegmentedColormap.from_list("month_deeppurple_cyclic", colors, N=256)
    cmap.set_bad((1, 1, 1, 0))
    return cmap


def add_colorbar(fig, im, ax, label=None, ticks=None, ticklabels=None):
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.018, fraction=0.046)
    cbar.ax.set_facecolor("none")
    if label:
        cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    if ticks is not None:
        cbar.set_ticks(ticks)
    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)
    return cbar


def plot_continuous_panel(fig, ax, da, x_edges, y_edges, boundaries_gdf, title, cmap,
                          vmin=None, vmax=None, cbar_label=None,
                          cbar_ticks=None, cbar_ticklabels=None):
    im = ax.pcolormesh(
        x_edges, y_edges, np.asarray(da.values),
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto", rasterized=RASTERIZED, zorder=2
    )
    add_boundaries(ax, boundaries_gdf)
    set_transparent_axis(ax, boundaries_gdf)
    ax.set_title(title, fontsize=10)
    add_colorbar(fig, im, ax, cbar_label, cbar_ticks, cbar_ticklabels)
    return im


def plot_bivariate_panel(ax, da, x_edges, y_edges, boundaries_gdf):
    # Order: Background, Frequency-dominated, Intensity-dominated, Compound.
    class_colors = ["#cfe8f3", "#f4a582", "#f46d43", "#b2182b"]
    cmap = ListedColormap(class_colors)
    cmap.set_bad((1, 1, 1, 0))
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    ax.pcolormesh(
        x_edges, y_edges, np.asarray(da.values),
        cmap=cmap, norm=norm, shading="auto",
        rasterized=RASTERIZED, zorder=2
    )
    add_boundaries(ax, boundaries_gdf)
    set_transparent_axis(ax, boundaries_gdf)
    ax.set_title("(c) Intensity–frequency contrast", fontsize=10)

    legend_elements = [
        Patch(facecolor="#b2182b", edgecolor="none", label="Compound"),
        Patch(facecolor="#f4a582", edgecolor="none", label="Frequency-dominated"),
        Patch(facecolor="#f46d43", edgecolor="none", label="Intensity-dominated"),
        Patch(facecolor="#cfe8f3", edgecolor="none", label="Background"),
    ]
    leg = ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.985),
        frameon=True,
        fontsize=6.4,
        borderaxespad=0.0,
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.25,
        borderpad=0.22,
        columnspacing=0.5,
    )
    leg.get_frame().set_alpha(0.82)
    leg.get_frame().set_linewidth(0.35)


# =========================================================
# MAIN
# =========================================================
print("Opening products NetCDF...")
ds = open_xarray_dataset_compat(INPUT_PRODUCTS_NC)
lon_name, lat_name = detect_lon_lat_names(ds)
ds = maybe_fix_lon(ds, lon_name)
check_required_vars(ds, [VAR_MAX_Z, VAR_FREQ, VAR_PEAK_MONTH, VAR_PEAK_SHARE])

# Crop by lon/lat only for plot stability. Products are already masked in the NetCDF.
lat_desc = np.all(np.diff(ds[lat_name].values) < 0)
ds = ds.sel({
    lon_name: slice(MAIN_LON_MIN, MAIN_LON_MAX),
    lat_name: slice(MAIN_LAT_MAX, MAIN_LAT_MIN) if lat_desc else slice(MAIN_LAT_MIN, MAIN_LAT_MAX),
})

max_z = ds[VAR_MAX_Z]
freq = ds[VAR_FREQ]
peak_month = ds[VAR_PEAK_MONTH]
peak_share = ds[VAR_PEAK_SHARE]

print("Building bivariate classification from existing products...")
bivar = build_bivariate_class(max_z, freq)

print("Preparing projected grid edges in EPSG:3035...")
x_edges, y_edges = transform_grid_edges(ds[lon_name].values, ds[lat_name].values)

print("Loading and filtering Europe boundaries...")
europe_admin = load_europe_admin_boundaries(COUNTRIES_SRC)

z_vmin, z_vmax = scalar_quantile(max_z, 0.02), scalar_quantile(max_z, 0.98)
f_vmin, f_vmax = 0, scalar_quantile(freq, 0.98)
s_vmin, s_vmax = scalar_quantile(peak_share, 0.02), scalar_quantile(peak_share, 0.98)

month_cmap = make_peak_month_cmap()
month_ticks = np.arange(1, 13)
month_ticklabels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

print("Plotting centered 5-panel figure...")
fig = plt.figure(figsize=(14.2, 8.6), facecolor="none")
fig.patch.set_alpha(0.0)

# 2 x 6 layout:
# top row: three panels, each spans two columns;
# bottom row: two panels centered by occupying columns 1:3 and 3:5.
gs = fig.add_gridspec(2, 6)
ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[0, 4:6])
ax_d = fig.add_subplot(gs[1, 1:3])
ax_e = fig.add_subplot(gs[1, 3:5])

plot_continuous_panel(
    fig, ax_a, max_z, x_edges, y_edges, europe_admin,
    "(a) Maximum wind anomaly intensity",
    cmap=transparent_cmap("magma"),
    vmin=z_vmin, vmax=z_vmax,
    cbar_label="z anomaly",
)

plot_continuous_panel(
    fig, ax_b, freq, x_edges, y_edges, europe_admin,
    "(b) Wind anomaly frequency",
    cmap=transparent_cmap("magma"),
    vmin=f_vmin, vmax=f_vmax,
    cbar_label="count yr$^{-1}$",
)

plot_bivariate_panel(ax_c, bivar, x_edges, y_edges, europe_admin)

plot_continuous_panel(
    fig, ax_d, peak_month, x_edges, y_edges, europe_admin,
    "(d) Peak month of wind anomaly",
    cmap=month_cmap,
    vmin=1, vmax=12,
    cbar_label="month",
    cbar_ticks=month_ticks,
    cbar_ticklabels=month_ticklabels,
)

plot_continuous_panel(
    fig, ax_e, peak_share, x_edges, y_edges, europe_admin,
    "(e) Peak-month share",
    cmap=transparent_cmap("viridis"),
    vmin=s_vmin, vmax=s_vmax,
    cbar_label="share",
)

fig.subplots_adjust(
    left=0.02,
    right=0.985,
    bottom=0.04,
    top=0.97,
    wspace=0.36,
    hspace=0.16,
)

out_png = OUTDIR / f"wind_spacetime_{BACKGROUND}_5panel_from_products_nc_EPSG3035_centered_transparent.png"
out_pdf = OUTDIR / f"wind_spacetime_{BACKGROUND}_5panel_from_products_nc_EPSG3035_centered_transparent.pdf"

fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", transparent=True)
fig.savefig(out_pdf, dpi=FIG_DPI, bbox_inches="tight", transparent=True)
plt.close(fig)

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
print("Done.")
