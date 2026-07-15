# -*- coding: utf-8 -*-
"""
Generate two supplementary figures within the minimal reproduction package.

FigS2: Monthly share of predicted wind-disturbance likelihood (12 panels)
FigS3: Absolute wind-speed reference maps combined into one 2-panel figure
        (a) Maximum wind speed
        (b) 95th-percentile wind speed

This script integrates the monthly-share logic and the absolute-wind plotting logic
so that both supplementary figures can be reproduced within the package workflow.
"""

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import geopandas as gpd
from shapely import wkt
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.transform import from_bounds, array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling


# ============================================================
# 0. USER SETTINGS
# ============================================================

# Supplementary figures:
#   FigS2 = monthly share of predicted wind-disturbance likelihood (12 panels)
#   FigS3 = absolute wind-speed reference maps, 2 panels

BASE = Path(os.environ.get("WALL2WALL_OUTDIR", r"E:\RF B+G\wall2wall_windonly_finalmap"))
FIG1_BASE = Path(os.environ.get("FIG1_OUTDIR", str(BASE)))

HEX_PERIOD_CSV = Path(os.environ.get("HEX_PERIOD_CSV", str(BASE / "hex_period_summary_wall2wall_windonly_final.csv")))
HEX_STYLEMATCH_CSV = Path(os.environ.get("HEX_STYLEMATCH_CSV", str(FIG1_BASE / "hex_indicator_summary_stylematch_4326.csv")))

WIND_NC = Path(os.environ.get("WIND_NC", r"E:\ERA\ERA5 16 combine\EU_final_structural_wind_features.nc"))
WIND_VAR = os.environ.get("WIND_VAR", "max_wind_speed")

OUTDIR = Path(os.environ.get("FIGS_OUTDIR", str(BASE / "supplementary_maps")))
OUTDIR.mkdir(parents=True, exist_ok=True)

ADMIN_SHP = Path(os.environ.get("COUNTRIES_SRC", r"E:\CNTR_RG_20M_2024_3035\CNTR_RG_20M_2024_3035.shp"))

DATE_COL = "obs_date"
PRED_COL = "mean_pred_wind"
HEX_ID_COL = "hex_id"

START_YEAR = 2003
END_YEAR = 2023

PROFILE_BASIS = "mean"

PLOT_CRS = "EPSG:3035"
GEOGRAPHIC_CRS = "EPSG:4326"
BBOX4326 = (-12.0, 34.0, 45.0, 72.0)

EUROPE_CNTR_IDS = {
    "AL", "AD", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ", "DE", "DK",
    "EE", "EL", "ES", "FI", "FR", "HU", "HR", "IE", "IT", "LI", "LT", "LU",
    "LV", "MD", "ME", "MK", "MT", "NL", "NO", "PL", "PT", "RO", "RS", "SE",
    "SI", "SK", "SM", "UA", "VA", "XK", "UK", "GB"
}

MONTHS = list(range(1, 13))
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

FIG_DPI = 500
SEA_COLOR = "#eeeeee"
BOUNDARY_COLOR = "0.35"
BOUNDARY_LW_MONTHLY = 0.32
BOUNDARY_LW_WIND = 0.45

MONTHLY_SHARE_VMIN = 0.00
MONTHLY_SHARE_VMAX = 0.30
MONTHLY_CMAP = "YlOrRd"
POINT_SIZE_MONTHLY = 2.8

MAX_WIND_VMIN = None
MAX_WIND_VMAX = None
Q95_WIND_VMIN = None
Q95_WIND_VMAX = None
WIND_CMAP = "viridis"

SAVE_PDF = True

# ============================================================
# 1. GENERAL HELPERS
# ============================================================

def set_publication_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.2,
        "figure.titlesize": 12,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": SEA_COLOR,
    })


def require_file(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def detect_first_existing_column(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Cannot find {label}. Tried {candidates}. Available columns: {list(df.columns)}"
    )


# ============================================================
# 2. ADMIN / GEOMETRY HELPERS
# ============================================================

def load_admin_4326():
    """Load selected European administrative polygons in EPSG:4326."""
    require_file(ADMIN_SHP, "ADMIN_SHP")

    admin = gpd.read_file(ADMIN_SHP)
    if admin.empty:
        raise ValueError(f"Boundary file is empty: {ADMIN_SHP}")
    if admin.crs is None:
        # Your Eurostat file is normally EPSG:3035.
        admin = admin.set_crs(PLOT_CRS)

    if "CNTR_ID" in admin.columns:
        admin = admin[admin["CNTR_ID"].astype(str).isin(EUROPE_CNTR_IDS)].copy()
    else:
        warnings.warn("CNTR_ID not found in admin shapefile. Using bbox clipping only.")

    admin = admin.to_crs(GEOGRAPHIC_CRS)
    admin = admin.explode(index_parts=False).reset_index(drop=True)

    xmin, ymin, xmax, ymax = BBOX4326
    bbox_geom = box(xmin, ymin, xmax, ymax)

    admin = admin[admin.geometry.intersects(bbox_geom)].copy()
    admin["geometry"] = admin.geometry.intersection(bbox_geom)
    admin = admin[admin.geometry.notna() & (~admin.geometry.is_empty)].copy()

    # Remove tiny fragments generated by clipping.
    admin_tmp = admin.to_crs(PLOT_CRS).copy()
    admin["_area_km2"] = admin_tmp.geometry.area / 1e6
    admin = admin[admin["_area_km2"] >= 20.0].drop(columns="_area_km2").copy()

    if admin.empty:
        raise ValueError("No administrative polygons remain after filtering/clipping.")

    return admin


def load_admin_3035():
    return load_admin_4326().to_crs(PLOT_CRS)


def detect_geometry_column(df):
    for c in ["geometry", "geom", "wkt", "WKT"]:
        if c in df.columns:
            return c
    return None


def detect_lon_lat_columns(df):
    lon_candidates = ["lon", "longitude", "centroid_lon", "center_lon", "lon_4326", "longitude_4326"]
    lat_candidates = ["lat", "latitude", "centroid_lat", "center_lat", "lat_4326", "latitude_4326"]
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    return lon_col, lat_col


def detect_projected_xy_columns(df):
    x_candidates = ["x", "X", "centroid_x", "center_x", "x_3035", "X_3035"]
    y_candidates = ["y", "Y", "centroid_y", "center_y", "y_3035", "Y_3035"]
    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)
    return x_col, y_col


def guess_crs_from_bounds(gdf):
    minx, miny, maxx, maxy = gdf.total_bounds
    if (
        -180 <= minx <= 180 and -180 <= maxx <= 180 and
        -90 <= miny <= 90 and -90 <= maxy <= 90
    ):
        return GEOGRAPHIC_CRS
    return PLOT_CRS


def prepare_hex_geometry(hex_csv):
    """
    Read the stylematch/indicator CSV and return a GeoDataFrame in EPSG:3035.
    Supports WKT geometry, lon/lat point columns, or projected x/y point columns.
    """
    require_file(hex_csv, "HEX_STYLEMATCH_CSV")
    hx = pd.read_csv(hex_csv)

    if HEX_ID_COL not in hx.columns:
        raise ValueError(f"{hex_csv} must contain {HEX_ID_COL}.")

    geom_col = detect_geometry_column(hx)
    if geom_col is not None:
        hx["_geometry"] = hx[geom_col].apply(
            lambda z: wkt.loads(z) if isinstance(z, str) and len(z) > 0 else None
        )
        gdf = gpd.GeoDataFrame(hx, geometry="_geometry")
        gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
        gdf = gdf.set_crs(guess_crs_from_bounds(gdf), allow_override=True)
        return gdf.to_crs(PLOT_CRS)

    lon_col, lat_col = detect_lon_lat_columns(hx)
    if lon_col is not None and lat_col is not None:
        hx[lon_col] = pd.to_numeric(hx[lon_col], errors="coerce")
        hx[lat_col] = pd.to_numeric(hx[lat_col], errors="coerce")
        hx = hx[hx[lon_col].between(-180, 180) & hx[lat_col].between(-90, 90)].copy()
        return gpd.GeoDataFrame(
            hx,
            geometry=gpd.points_from_xy(hx[lon_col], hx[lat_col]),
            crs=GEOGRAPHIC_CRS,
        ).to_crs(PLOT_CRS)

    x_col, y_col = detect_projected_xy_columns(hx)
    if x_col is not None and y_col is not None:
        hx[x_col] = pd.to_numeric(hx[x_col], errors="coerce")
        hx[y_col] = pd.to_numeric(hx[y_col], errors="coerce")
        hx = hx[np.isfinite(hx[x_col]) & np.isfinite(hx[y_col])].copy()
        return gpd.GeoDataFrame(
            hx,
            geometry=gpd.points_from_xy(hx[x_col], hx[y_col]),
            crs=PLOT_CRS,
        )

    raise ValueError(
        "No usable geometry, lon/lat, or projected x/y columns detected in HEX_STYLEMATCH_CSV."
    )


def filter_gdf_to_main_europe(gdf, admin_3035):
    """Clip to bbox and keep hexes whose representative point is inside selected admin footprint."""
    xmin, ymin, xmax, ymax = BBOX4326
    bbox_geom = box(xmin, ymin, xmax, ymax)

    gg = gdf.copy()
    if gg.crs is None:
        gg = gg.set_crs(PLOT_CRS, allow_override=True)

    gg = gg.to_crs(GEOGRAPHIC_CRS)
    gg = gg[gg.geometry.notna() & (~gg.geometry.is_empty)].copy()
    gg = gg[gg.geometry.intersects(bbox_geom)].copy()
    gg["geometry"] = gg.geometry.intersection(bbox_geom)
    gg = gg[gg.geometry.notna() & (~gg.geometry.is_empty)].copy()
    gg = gg.to_crs(PLOT_CRS)

    if admin_3035 is not None and len(admin_3035) > 0:
        try:
            admin_union = admin_3035.geometry.union_all()
        except Exception:
            admin_union = admin_3035.unary_union

        geom_types = gg.geometry.geom_type.astype(str)
        rep = gg.geometry if geom_types.isin(["Point", "MultiPoint"]).all() else gg.geometry.representative_point()
        keep = rep.within(admin_union) | rep.touches(admin_union)
        gg = gg[keep].copy()

    return gg


def get_plot_bounds_3035(gdf, pad_ratio=0.025):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    dx = xmax - xmin
    dy = ymax - ymin
    return (xmin - dx * pad_ratio, ymin - dy * pad_ratio,
            xmax + dx * pad_ratio, ymax + dy * pad_ratio)


def style_map_axis(ax, bounds=None):
    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ============================================================
# 3. MONTHLY SHARE CALCULATION AND FIGURE
# ============================================================

def compute_monthly_profiles():
    require_file(HEX_PERIOD_CSV, "HEX_PERIOD_CSV")
    df = pd.read_csv(HEX_PERIOD_CSV)

    # Allow fallbacks if the exact names have changed.
    date_col = DATE_COL if DATE_COL in df.columns else detect_first_existing_column(
        df, ["obs_date", "date", "time", "period", "period_start", "timestamp"], "date column"
    )
    pred_col = PRED_COL if PRED_COL in df.columns else detect_first_existing_column(
        df, ["mean_pred_wind", "mean_pred", "pred", "prediction", "probability", "predicted_probability"],
        "prediction column"
    )
    hex_col = HEX_ID_COL if HEX_ID_COL in df.columns else detect_first_existing_column(
        df, ["hex_id", "id", "hex"], "hex id column"
    )

    df = df[[hex_col, date_col, pred_col]].copy()
    df = df.rename(columns={hex_col: HEX_ID_COL, date_col: "_date", pred_col: "_pred"})
    df["_date"] = pd.to_datetime(df["_date"], errors="coerce")
    df["_pred"] = pd.to_numeric(df["_pred"], errors="coerce")

    df = df[df["_date"].notna() & np.isfinite(df["_pred"])].copy()
    df = df[(df["_date"].dt.year >= START_YEAR) & (df["_date"].dt.year <= END_YEAR)].copy()
    df["month"] = df["_date"].dt.month.astype(int)

    if df.empty:
        raise ValueError("No valid prediction rows remain after date/value filtering.")

    hex_month = (
        df.groupby([HEX_ID_COL, "month"], as_index=False)
          .agg(
              monthly_pred_sum=("_pred", "sum"),
              monthly_pred_mean=("_pred", "mean"),
              n_periods_month=("_pred", "size"),
          )
    )

    all_hex = df[HEX_ID_COL].drop_duplicates().to_numpy()
    full_idx = pd.MultiIndex.from_product([all_hex, MONTHS], names=[HEX_ID_COL, "month"])

    hex_month = (
        hex_month
        .set_index([HEX_ID_COL, "month"])
        .reindex(full_idx)
        .reset_index()
    )

    for col in ["monthly_pred_sum", "monthly_pred_mean", "n_periods_month"]:
        hex_month[col] = hex_month[col].fillna(0)

    hex_month["total_pred_sum_by_months"] = (
        hex_month.groupby(HEX_ID_COL)["monthly_pred_sum"].transform("sum")
    )
    hex_month["total_pred_mean_profile"] = (
        hex_month.groupby(HEX_ID_COL)["monthly_pred_mean"].transform("sum")
    )

    hex_month["monthly_mass_share"] = np.where(
        hex_month["total_pred_sum_by_months"] > 0,
        hex_month["monthly_pred_sum"] / hex_month["total_pred_sum_by_months"],
        np.nan,
    )

    hex_month["monthly_intensity_share"] = np.where(
        hex_month["total_pred_mean_profile"] > 0,
        hex_month["monthly_pred_mean"] / hex_month["total_pred_mean_profile"],
        np.nan,
    )

    if PROFILE_BASIS.lower() == "sum":
        hex_month["monthly_share"] = hex_month["monthly_mass_share"]
    elif PROFILE_BASIS.lower() == "mean":
        hex_month["monthly_share"] = hex_month["monthly_intensity_share"]
    else:
        raise ValueError("PROFILE_BASIS must be either 'mean' or 'sum'.")

    long_csv = OUTDIR / "FigS2_hex_monthly_prediction_profiles_long.csv"
    hex_month.to_csv(long_csv, index=False)

    share_wide = (
        hex_month.pivot(index=HEX_ID_COL, columns="month", values="monthly_share")
                 .reset_index()
                 .rename(columns={m: f"m{m:02d}_share" for m in MONTHS})
    )

    wide_csv = OUTDIR / "FigS2_hex_monthly_prediction_profiles_wide.csv"
    share_wide.to_csv(wide_csv, index=False)

    print(f"Saved monthly long table: {long_csv}")
    print(f"Saved monthly wide table: {wide_csv}")
    print(f"Monthly profile basis: {PROFILE_BASIS}")

    return share_wide


def plot_monthly_share_12panel(gdf, admin_3035):
    cols = [f"m{m:02d}_share" for m in MONTHS]
    for col in cols:
        if col not in gdf.columns:
            raise ValueError(f"Missing monthly share column: {col}")
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    bounds = get_plot_bounds_3035(gdf)
    is_point = gdf.geometry.iloc[0].geom_type in ["Point", "MultiPoint"]

    # Use GridSpec so the colour bar is genuinely outside the 12 map panels.
    fig = plt.figure(figsize=(12.6, 8.6))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.055], wspace=0.03, hspace=0.10)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(3)])
    cax = fig.add_subplot(gs[:, 4])

    cmap = plt.get_cmap(MONTHLY_CMAP).copy()
    cmap.set_bad((1, 1, 1, 0))

    for ax, month, label in zip(axes.ravel(), MONTHS, MONTH_LABELS):
        col = f"m{month:02d}_share"
        ax.set_facecolor(SEA_COLOR)

        gg = gdf[gdf[col].notna()].copy()
        if is_point:
            gg.plot(
                ax=ax,
                column=col,
                cmap=cmap,
                vmin=MONTHLY_SHARE_VMIN,
                vmax=MONTHLY_SHARE_VMAX,
                markersize=POINT_SIZE_MONTHLY,
                linewidth=0,
                alpha=1.0,
            )
        else:
            gg.plot(
                ax=ax,
                column=col,
                cmap=cmap,
                vmin=MONTHLY_SHARE_VMIN,
                vmax=MONTHLY_SHARE_VMAX,
                linewidth=0,
                edgecolor="none",
            )

        if admin_3035 is not None:
            admin_3035.boundary.plot(
                ax=ax,
                linewidth=BOUNDARY_LW_MONTHLY,
                color=BOUNDARY_COLOR,
                alpha=0.75,
                zorder=5,
            )

        ax.set_title(label, fontsize=10, pad=3)
        style_map_axis(ax, bounds=bounds)

    sm = ScalarMappable(
        norm=Normalize(vmin=MONTHLY_SHARE_VMIN, vmax=MONTHLY_SHARE_VMAX),
        cmap=cmap,
    )
    sm.set_array([])

    # Colour bar occupies the separate GridSpec column, outside the map body.
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Monthly share of predicted likelihood")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.965, bottom=0.045)

    out_png = OUTDIR / "FigS2_monthly_likelihood_share_12panel.png"
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {out_png}")

    if SAVE_PDF:
        out_pdf = OUTDIR / "FigS2_monthly_likelihood_share_12panel.pdf"
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
        print(f"Saved figure: {out_pdf}")

    plt.close(fig)


# ============================================================
# 4. WIND-SPEED CALCULATION AND TWO-PANEL FIGURE
# ============================================================

def normalize_lon_to_180(lon):
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def detect_lon_lat_names(ds):
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

    return lon_name, lat_name


def maybe_fix_dataset_lon(ds, lon_name):
    lon_fixed = normalize_lon_to_180(ds[lon_name].values)
    return ds.assign_coords({lon_name: lon_fixed}).sortby(lon_name)


def subset_dataset_to_bbox(ds, lon_name, lat_name):
    xmin, ymin, xmax, ymax = BBOX4326
    ds = ds.sel({lon_name: slice(xmin, xmax)})

    lat_vals = ds[lat_name].values
    if lat_vals[0] < lat_vals[-1]:
        ds = ds.sel({lat_name: slice(ymin, ymax)})
    else:
        ds = ds.sel({lat_name: slice(ymax, ymin)})

    return ds


def build_mask_from_polygons(lon, lat, poly_gdf_4326):
    """Rasterize selected European polygons to the lon/lat grid."""
    lon = normalize_lon_to_180(np.asarray(lon, dtype=np.float64))
    lat = np.asarray(lat, dtype=np.float64)

    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("lon and lat must be 1D arrays.")
    if len(lon) < 2 or len(lat) < 2:
        raise ValueError("Longitude/latitude arrays must each contain at least two values.")

    lon_sort_idx = np.argsort(lon)
    lat_sort_idx = np.argsort(lat)

    lon_sorted = lon[lon_sort_idx]
    lat_sorted = lat[lat_sort_idx]

    pg = poly_gdf_4326.to_crs(GEOGRAPHIC_CRS).copy()
    pg = pg.cx[
        float(lon_sorted.min()):float(lon_sorted.max()),
        float(lat_sorted.min()):float(lat_sorted.max()),
    ].copy()

    if pg.empty:
        raise ValueError("No administrative polygons intersect the wind grid extent.")

    try:
        geom = pg.geometry.union_all()
    except Exception:
        geom = pg.unary_union

    dx = float(np.median(np.diff(lon_sorted)))
    dy = float(np.median(np.diff(lat_sorted)))

    west = float(lon_sorted.min() - dx / 2.0)
    east = float(lon_sorted.max() + dx / 2.0)
    south = float(lat_sorted.min() - dy / 2.0)
    north = float(lat_sorted.max() + dy / 2.0)

    transform = from_bounds(west, south, east, north, len(lon_sorted), len(lat_sorted))

    mask_desc = rasterize(
        [(geom, 1)],
        out_shape=(len(lat_sorted), len(lon_sorted)),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)

    # raster row 0 is north, while lat_sorted is south -> north.
    mask_sorted = np.flipud(mask_desc)

    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    mask[np.ix_(lat_sort_idx, lon_sort_idx)] = mask_sorted
    return mask


def apply_mask_to_da(da, mask_2d, lat_name, lon_name):
    mask_da = xr.DataArray(
        mask_2d,
        coords={lat_name: da[lat_name], lon_name: da[lon_name]},
        dims=(lat_name, lon_name),
    )
    return da.where(mask_da)


def detect_time_dim(da):
    for d in da.dims:
        if d.lower() in ["time", "date"]:
            return d
    for d in da.dims:
        try:
            if np.issubdtype(da[d].dtype, np.datetime64):
                return d
        except Exception:
            pass
    raise ValueError(f"Cannot identify time dimension in {da.dims}.")


def robust_vmin_vmax(da, qlow=0.02, qhigh=0.98):
    arr = np.asarray(da.values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.quantile(arr, qlow)), float(np.quantile(arr, qhigh))



def compute_absolute_wind_products():
    """
    Compute long-term absolute wind-speed products.

    Important: this function does NOT apply the Europe mask before reprojection.
    The mask is applied after reprojection in EPSG:3035. This avoids the visual
    artefacts caused by rasterizing a lon/lat mask and then reprojecting it.
    """
    require_file(WIND_NC, "WIND_NC")
    ds = xr.open_dataset(WIND_NC)

    lon_name, lat_name = detect_lon_lat_names(ds)
    ds = maybe_fix_dataset_lon(ds, lon_name)
    ds = subset_dataset_to_bbox(ds, lon_name, lat_name)

    if WIND_VAR not in ds.data_vars:
        raise ValueError(f"{WIND_VAR} not found. Available variables: {list(ds.data_vars)}")

    wind = ds[WIND_VAR]
    time_dim = detect_time_dim(wind)

    print("Computing long-term maximum wind speed...")
    longterm_max = wind.max(dim=time_dim, skipna=True)

    print("Computing long-term 95th-percentile wind speed...")
    longterm_q95 = wind.quantile(0.95, dim=time_dim, skipna=True)
    if "quantile" in longterm_q95.coords:
        longterm_q95 = longterm_q95.drop_vars("quantile")

    # Save compact products so later checking does not require recomputing.
    out_nc = OUTDIR / "absolute_wind_speed_products_unprojected.nc"
    xr.Dataset({
        "longterm_max_wind_speed": longterm_max,
        "longterm_q95_wind_speed": longterm_q95,
    }).to_netcdf(out_nc)
    print(f"Saved wind products: {out_nc}")

    admin_4326 = load_admin_4326()
    return longterm_max, longterm_q95, admin_4326, lat_name, lon_name


def sort_da_for_rasterio_reprojection(da, lat_name, lon_name):
    """
    Rasterio expects row 0 to be the northernmost row for a north-up raster.
    Therefore longitude is sorted west->east and latitude is sorted north->south.
    This is the key correction relative to the previous version.
    """
    out = da.sortby(lon_name)
    lat_vals = np.asarray(out[lat_name].values, dtype=float)
    if lat_vals[0] < lat_vals[-1]:
        out = out.sortby(lat_name, ascending=False)
    return out.transpose(lat_name, lon_name)


def rasterize_admin_mask_3035(admin_3035, out_shape, transform):
    """Rasterize the selected European land polygons in the projected raster grid."""
    if admin_3035 is None or len(admin_3035) == 0:
        return np.ones(out_shape, dtype=bool)

    geoms = []
    for geom in admin_3035.geometry:
        if geom is not None and (not geom.is_empty):
            geoms.append((geom, 1))
    if not geoms:
        return np.ones(out_shape, dtype=bool)

    return rasterize(
        geoms,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)


def reproject_da_to_3035(da, lat_name, lon_name, mask_gdf_3035=None, resampling=Resampling.bilinear):
    """
    Reproject a 2D lon/lat DataArray to EPSG:3035 for shape-consistent plotting.

    The previous version sorted latitude south->north before calling rasterio.
    That makes the source raster upside down relative to the transform and can
    visibly shift/warp the wind-speed surface. Here the source array is always
    north->south, matching the raster transform.
    """
    da = sort_da_for_rasterio_reprojection(da, lat_name, lon_name)

    lon = np.asarray(da[lon_name].values, dtype=float)
    lat = np.asarray(da[lat_name].values, dtype=float)
    src = np.asarray(da.values, dtype=np.float32)

    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("Wind DataArray must have 1D lon/lat coordinates.")
    if len(lon) < 2 or len(lat) < 2:
        raise ValueError("Need at least two longitude and latitude values for reprojection.")

    dx = float(np.median(np.abs(np.diff(lon))))
    dy = float(np.median(np.abs(np.diff(lat))))

    west = float(lon.min() - dx / 2.0)
    east = float(lon.max() + dx / 2.0)
    south = float(lat.min() - dy / 2.0)
    north = float(lat.max() + dy / 2.0)

    src_transform = from_bounds(west, south, east, north, len(lon), len(lat))

    dst_transform, dst_width, dst_height = calculate_default_transform(
        GEOGRAPHIC_CRS,
        PLOT_CRS,
        len(lon),
        len(lat),
        west,
        south,
        east,
        north,
    )

    dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs=GEOGRAPHIC_CRS,
        dst_transform=dst_transform,
        dst_crs=PLOT_CRS,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=resampling,
    )

    if mask_gdf_3035 is not None:
        mask = rasterize_admin_mask_3035(mask_gdf_3035, dst.shape, dst_transform)
        dst = np.where(mask, dst, np.nan).astype(np.float32)

    left, bottom, right, top = array_bounds(dst_height, dst_width, dst_transform)
    extent = (left, right, bottom, top)
    return dst, extent


def robust_array_vmin_vmax(arr, qlow=0.02, qhigh=0.98):
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.quantile(vals, qlow)), float(np.quantile(vals, qhigh))


def plot_absolute_wind_2panel(longterm_max, longterm_q95, admin_4326, lat_name, lon_name, plot_bounds_3035=None):
    admin_3035 = admin_4326.to_crs(PLOT_CRS)

    # Same map projection and same plotting extent as the monthly 12-panel figure.
    bounds_3035 = plot_bounds_3035 if plot_bounds_3035 is not None else get_plot_bounds_3035(admin_3035)

    arr_3035_max, extent_3035_max = reproject_da_to_3035(
        longterm_max, lat_name, lon_name, mask_gdf_3035=admin_3035
    )
    arr_3035_q95, extent_3035_q95 = reproject_da_to_3035(
        longterm_q95, lat_name, lon_name, mask_gdf_3035=admin_3035
    )

    vmin_max, vmax_max = MAX_WIND_VMIN, MAX_WIND_VMAX
    if vmin_max is None or vmax_max is None:
        auto_vmin, auto_vmax = robust_array_vmin_vmax(arr_3035_max)
        if vmin_max is None:
            vmin_max = auto_vmin
        if vmax_max is None:
            vmax_max = auto_vmax

    vmin_q95, vmax_q95 = Q95_WIND_VMIN, Q95_WIND_VMAX
    if vmin_q95 is None or vmax_q95 is None:
        auto_vmin, auto_vmax = robust_array_vmin_vmax(arr_3035_q95)
        if vmin_q95 is None:
            vmin_q95 = auto_vmin
        if vmax_q95 is None:
            vmax_q95 = auto_vmax

    # Choose the figure height from the real projected extent so the map panels do not look stretched.
    xmin, ymin, xmax, ymax = bounds_3035
    map_aspect = (ymax - ymin) / max((xmax - xmin), 1.0)
    panel_width = 5.2
    panel_height = panel_width * map_aspect
    fig_height = max(4.9, min(6.2, panel_height + 0.55))

    # Add a dedicated spacer column between the left colour bar and panel (b)
    # so that the label of panel (a) never overlaps the main map of panel (b).
    fig = plt.figure(figsize=(12.8, fig_height))
    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[1, 0.05, 0.14, 1, 0.05],
        wspace=0.04
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 3])]
    caxes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 4])]

    for ax in axes:
        ax.set_facecolor(SEA_COLOR)
        style_map_axis(ax, bounds=bounds_3035)

    im0 = axes[0].imshow(
        arr_3035_max,
        extent=extent_3035_max,
        origin="upper",
        cmap=WIND_CMAP,
        vmin=vmin_max,
        vmax=vmax_max,
        interpolation="nearest",
        zorder=1,
    )
    admin_3035.boundary.plot(ax=axes[0], linewidth=BOUNDARY_LW_WIND, color="black", alpha=0.75, zorder=5)
    axes[0].set_title("(a) Maximum wind speed", fontsize=11, pad=5)
    cbar0 = fig.colorbar(im0, cax=caxes[0], orientation="vertical", extend="both")
    cbar0.set_label("Wind speed (m s$^{-1}$)")

    im1 = axes[1].imshow(
        arr_3035_q95,
        extent=extent_3035_q95,
        origin="upper",
        cmap=WIND_CMAP,
        vmin=vmin_q95,
        vmax=vmax_q95,
        interpolation="nearest",
        zorder=1,
    )
    admin_3035.boundary.plot(ax=axes[1], linewidth=BOUNDARY_LW_WIND, color="black", alpha=0.75, zorder=5)
    axes[1].set_title("(b) 95th-percentile wind speed", fontsize=11, pad=5)
    cbar1 = fig.colorbar(im1, cax=caxes[1], orientation="vertical", extend="both")
    cbar1.set_label("Wind speed (m s$^{-1}$)")

    fig.subplots_adjust(left=0.035, right=0.965, top=0.93, bottom=0.055)

    out_png = OUTDIR / "FigS3_absolute_wind_speed_2panel.png"
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {out_png}")

    if SAVE_PDF:
        out_pdf = OUTDIR / "FigS3_absolute_wind_speed_2panel.pdf"
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
        print(f"Saved figure: {out_pdf}")

    plt.close(fig)


# ============================================================
# 5. MAIN
# ============================================================

def main():
    set_publication_style()

    print("\n============================================================")
    print("1. Monthly share 12-panel figure")
    print("============================================================")
    share_wide = compute_monthly_profiles()

    print("Loading hex geometry and administrative boundary...")
    gdf_hex = prepare_hex_geometry(HEX_STYLEMATCH_CSV)
    admin_3035 = load_admin_3035()

    gdf = gdf_hex.merge(share_wide, on=HEX_ID_COL, how="left")
    first_col = "m01_share"
    gdf = gdf[gdf[first_col].notna()].copy()
    gdf = filter_gdf_to_main_europe(gdf, admin_3035)

    print(f"Hexes used in monthly figure: {len(gdf):,}")
    monthly_bounds_3035 = get_plot_bounds_3035(gdf)
    plot_monthly_share_12panel(gdf, admin_3035)

    print("\n============================================================")
    print("2. Absolute wind-speed 2-panel figure")
    print("============================================================")
    longterm_max, longterm_q95, admin_4326, lat_name, lon_name = compute_absolute_wind_products()
    plot_absolute_wind_2panel(longterm_max, longterm_q95, admin_4326, lat_name, lon_name, plot_bounds_3035=monthly_bounds_3035)

    print("\n============================================================")
    print("Done. Outputs saved to:")
    print(OUTDIR)
    print("============================================================")


if __name__ == "__main__":
    main()
