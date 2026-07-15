# -*- coding: utf-8 -*-
"""
Section 3.6 final analysis
Temporal correspondence between wind-anomaly timing and predicted disturbance potential

This script now uses a cached-CSV-first workflow.

If OUTDIR / merged_temporal_correspondence_<background>.csv already exists,
the script reads that CSV and goes directly to plotting. It only rebuilds the
intermediate temporal metrics when the merged CSV is missing, invalid, or when
FIG5_FORCE_REBUILD=1 is set.

Main scientific question
------------------------
Compare wind-anomaly temporal metrics and disturbance-prediction temporal metrics:
1) peak month correspondence
2) peak-month share / temporal concentration correspondence

Default main analysis uses wind-anomaly FREQUENCY timing.
For supplementary anomaly-intensity timing, set WIND_PROFILE_METRIC = "intensity".

Input hierarchy
---------------
Prediction timing:
A) If PRED_TEMP_CSV exists, read prediction peak month/share from it.
B) Otherwise, compute prediction peak month/share from PRED_PERIOD_CSV.

Wind-anomaly timing:
A) If WIND_TEMP_CSV exists, read wind-anomaly peak month/share from it.
B) Else if WIND_MONTHLY_PROFILE_CSV exists, compute peak month/share from monthly profile.
C) Else if WIND_PERIOD_ANOMALY_CSV exists, build monthly profile from existing z anomaly.
D) Else if ALLOW_BUILD_WIND_FROM_NC=True, sample WIND_NC and build z anomaly timing.

Outputs
-------
OUTDIR / merged_temporal_correspondence_<background>.csv
OUTDIR / TableS_temporal_correspondence_summary_<background>.csv
OUTDIR / TableS_peak_month_agreement_classes_<background>.csv
OUTDIR / TableS_temporal_correlations_<background>.csv
OUTDIR / TableS_prediction_temporal_metrics_<background>.csv
OUTDIR / TableS_wind_anomaly_temporal_metrics_<background>.csv
OUTDIR / TableS_wind_monthly_profile_<background>.csv if created/read
OUTDIR / Fig6_temporal_correspondence_<background>.png/pdf
OUTDIR / debug_temporal_correspondence_plot_points_<background>.csv
"""

from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm, ListedColormap

# Optional imports. Maps and NetCDF rebuild need these.
try:
    import geopandas as gpd
    from shapely import wkt
    from shapely.geometry import box
    HAS_GEO = True
except Exception:
    HAS_GEO = False

try:
    import xarray as xr
    HAS_XARRAY = True
except Exception:
    HAS_XARRAY = False

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False




# ============================================================
# NetCDF opening helper with backend fallback
# ============================================================

def open_dataset_fallback(path):
    """Open NetCDF with a backend fallback to avoid netCDF4 DLL issues on Windows."""
    path = str(path)
    errors = []
    for engine in ["h5netcdf", "scipy", "netcdf4", None]:
        try:
            if engine is None:
                print(f"Opening NetCDF with xarray default: {path}")
                return xr.open_dataset(path)
            print(f"Opening NetCDF with engine={engine}: {path}")
            return xr.open_dataset(path, engine=engine)
        except Exception as e:
            errors.append(f"{engine or 'default'}: {repr(e)}")
    raise RuntimeError("Could not open NetCDF file with available engines:\n" + "\n".join(errors))

# ============================================================
# 0. USER CONFIG
# ============================================================

BACKGROUND = os.environ.get("FIG5_BACKGROUND", "long_term")       # "long_term" or "same_month"
WIND_PROFILE_METRIC = os.environ.get("FIG5_WIND_PROFILE_METRIC", "frequency")  # "frequency" for main text, "intensity" for supplementary

BASE = Path(os.environ.get("ORIGINAL_BASE", r"E:\RF B+G"))

# ---- Prediction timing inputs ----
# If this file exists, prediction peak month/share are read directly.
# Otherwise they are derived from PRED_PERIOD_CSV.
PRED_TEMP_CSV = Path(os.environ.get("PRED_TEMP_CSV", r"E:\RF_BG_REPRO_from_model_dev\outputs\05_fig2_monthly_concentration\hex_monthly_spatiotemporal_likelihood_metrics.csv"))

# Period-level wall-to-wall prediction table. Used when PRED_TEMP_CSV is missing or cannot be read.
PRED_PERIOD_CSV = Path(os.environ.get("HEX_PERIOD_CSV", r"E:\RF_BG_REPRO_from_model_dev\outputs\03_wall2wall_windonly\hex_period_summary_wall2wall_windonly_final.csv"))
PRED_DATE_COL = "obs_date"
PRED_VALUE_COL = "mean_pred_wind"

# ---- Wind-anomaly timing inputs ----
# Optional already-aggregated wind-anomaly temporal metrics table.
# If missing, the script tries WIND_MONTHLY_PROFILE_CSV, then WIND_PERIOD_ANOMALY_CSV, then WIND_NC.
WIND_TEMP_CSV = Path(os.environ.get("WIND_TEMP_CSV", str(BASE / r"wind_anomaly_temporal_metrics\hex_wind_anomaly_temporal_metrics_long_term.csv")))

# Monthly anomaly profile table, long format preferred:
# hex_id, month, anomaly_frequency, anomaly_intensity
WIND_MONTHLY_PROFILE_CSV = Path(os.environ.get("WIND_MONTHLY_PROFILE_CSV", str(BASE / r"section_3_6_existing_wind_anomaly_timing\hex_monthly_wind_anomaly_profiles_long_term.csv")))

# Existing period-level z-anomaly table:
# hex_id, obs_date, z_value / z_anomaly
WIND_PERIOD_ANOMALY_CSV = Path(os.environ.get("WIND_PERIOD_ANOMALY_CSV", str(BASE / r"section_3_6_existing_wind_anomaly_timing\hex_period_wind_anomaly_long_term.csv")))
ANOM_DATE_COL = "obs_date"
ANOM_Z_COL = None
Z_EVENT_THRESHOLD = 2.0

# If no existing wind-anomaly timing files exist, optionally rebuild from wind NetCDF.
# This can be slow. Set False if you only want to use already-generated anomaly tables.
ALLOW_BUILD_WIND_FROM_NC = True
WIND_NC = Path(os.environ.get("WIND_NC", r"E:\ERA\ERA5 16 combine\EU_final_structural_wind_features.nc"))
WIND_VAR = "max_wind_speed"

# Hex geometry / centroid table, used for area and maps; also used if rebuilding wind anomaly from NetCDF.
HEX_GEOM_CSV = Path(os.environ.get("FIG1_INDICATOR_CSV", r"E:\RF_BG_REPRO_from_model_dev\outputs\04_fig1_spatial_indicators\hex_indicator_summary_stylematch_4326.csv"))
ADMIN_SHP = Path(os.environ.get("COUNTRIES_SRC", r"E:\CNTR_RG_20M_2024_3035\CNTR_RG_20M_2024_3035.shp"))

# ---- Output ----
OUTDIR_ROOT = Path(os.environ.get("FIG5_OUTDIR", r"E:\RF_BG_REPRO_from_model_dev\outputs\08_fig5_temporal_correspondence"))
OUTDIR = OUTDIR_ROOT / BACKGROUND
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- Locked Fig. 5 area weighting ----
# Confirmed original Fig. 5 used complete hex-area weights, not forest-area weights.
LOCKED_FIG5_HEX_AREA_KM2 = float(os.environ.get("FIG5_HEX_AREA_KM2", os.environ.get("HEX_AREA_KM2", "2165")))


# ---- Cached workflow ----
# Default behavior:
#   1) If merged_temporal_correspondence_<BACKGROUND>.csv exists in OUTDIR, read it.
#   2) Skip prediction/wind temporal rebuilding and skip summary-table rewriting.
#   3) Draw Fig. 5 directly from the existing CSV.
#
# Set FIG5_FORCE_REBUILD=1 only when you intentionally want to regenerate all
# intermediate CSVs from upstream inputs.
FORCE_REBUILD = os.environ.get("FIG5_FORCE_REBUILD", "0").strip().lower() in {"1", "true", "yes", "y"}
REWRITE_TABLES_FROM_EXISTING_CSV = os.environ.get("FIG5_REWRITE_TABLES_FROM_EXISTING_CSV", "0").strip().lower() in {"1", "true", "yes", "y"}
FORCE_WIND_FROM_NC = os.environ.get("FIG5_FORCE_WIND_FROM_NC", "0").strip().lower() in {"1", "true", "yes", "y"}

# ---- Map settings ----
BBOX4326 = (-12, 34, 45, 72)
PLOT_CRS = "EPSG:3035"
DPI = 500
POINT_SIZE = 14
ALPHA = 0.92

# Publication-map style, matched to 18hex_period_indicator_suite_STYLEMATCH_3035_PUBLICATION_NO_AUTOGEOM.py
ADMIN_LW = 0.28
ADMIN_COLOR = "0.35"
MIN_ADMIN_PART_AREA_KM2 = 20

# Keep only main European countries; Russia and Iceland excluded.
EUROPE_CNTR_IDS = {
    "AL", "AD", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ", "DE", "DK",
    "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LI", "LT", "LU",
    "LV", "MD", "ME", "MK", "MT", "NL", "NO", "PL", "PT", "RO", "RS", "SE",
    "SI", "SK", "SM", "UA", "VA", "XK", "UK", "GB"
}

MONTHS = list(range(1, 13))
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# Default month colors. Same scale for wind and prediction maps.
MONTH_COLORS = [
    "#440154", "#482878", "#3E4989", "#31688E", "#26828E", "#1F9E89",
    "#35B779", "#6DCD59", "#B4DE2C", "#FDE725", "#F46D43", "#A50026"
]
MONTH_CMAP = ListedColormap(MONTH_COLORS)
MONTH_NORM = BoundaryNorm(np.arange(0.5, 13.5, 1), MONTH_CMAP.N)


# ============================================================
# 1. STYLE
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
        "figure.titlesize": 13,
        "axes.linewidth": 0.9,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def detect_col(df, candidates, required=True, role="column"):
    if isinstance(candidates, str):
        candidates = [candidates]
    lower_map = {str(c).lower(): c for c in df.columns}

    for cand in candidates:
        if cand is None:
            continue
        if str(cand).lower() in lower_map:
            return lower_map[str(cand).lower()]

    # softer substring matching
    for c in df.columns:
        cl = str(c).lower()
        for cand in candidates:
            if cand is not None and str(cand).lower() in cl:
                return c

    if required:
        raise KeyError(f"Cannot detect {role}. Tried {candidates}. Available columns: {list(df.columns)}")
    return None


def standardize_hex_id(df):
    out = df.copy()
    col = detect_col(out, ["hex_id", "hexid", "id"], required=True, role="hex_id")
    if col != "hex_id":
        out = out.rename(columns={col: "hex_id"})
    out["hex_id"] = pd.to_numeric(out["hex_id"], errors="coerce")
    out = out[out["hex_id"].notna()].copy()
    if len(out) > 0 and np.allclose(out["hex_id"], np.round(out["hex_id"]), equal_nan=False):
        out["hex_id"] = out["hex_id"].round().astype("int64")
    return out


def normalize_profile(values):
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.where(values < 0, 0.0, values)
    total = values.sum()
    if total <= 0:
        return np.full(len(values), np.nan, dtype=float)
    return values / total


def peak_month_and_share(values):
    prof = normalize_profile(values)
    if np.any(~np.isfinite(prof)):
        return np.nan, np.nan
    idx = int(np.argmax(prof))
    return idx + 1, float(prof[idx])


def effective_months(values):
    prof = normalize_profile(values)
    if np.any(~np.isfinite(prof)):
        return np.nan
    hhi = float(np.sum(prof ** 2))
    if hhi <= 0:
        return np.nan
    return 1.0 / hhi


def circular_peak_month_difference(m1, m2):
    if pd.isna(m1) or pd.isna(m2):
        return np.nan
    d = abs(int(round(float(m1))) - int(round(float(m2))))
    return min(d, 12 - d)


def signed_circular_month_shift(pred_m, wind_m):
    """predicted peak month minus wind peak month, wrapped to [-6, 6]."""
    if pd.isna(pred_m) or pd.isna(wind_m):
        return np.nan
    d = int(round(float(pred_m))) - int(round(float(wind_m)))
    while d > 6:
        d -= 12
    while d < -6:
        d += 12
    return d


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))


def weighted_quantile(values, q, weights=None):
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(values, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[ok]
    weights = weights[ok]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cw = np.cumsum(weights)
    cutoff = q * cw[-1]
    return float(values[np.searchsorted(cw, cutoff, side="left")])


def weighted_pearson(x, y, w):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return np.nan
    x = x[ok]
    y = y[ok]
    w = w[ok]
    wx = np.sum(w * x) / np.sum(w)
    wy = np.sum(w * y) / np.sum(w)
    cov = np.sum(w * (x - wx) * (y - wy)) / np.sum(w)
    vx = np.sum(w * (x - wx) ** 2) / np.sum(w)
    vy = np.sum(w * (y - wy) ** 2) / np.sum(w)
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(cov / np.sqrt(vx * vy))


def month_label(m):
    if pd.isna(m):
        return np.nan
    return MONTH_LABELS[int(round(float(m))) - 1]


def background_label(bg):
    if bg == "long_term":
        return "long-term background"
    if bg == "same_month":
        return "same-month background"
    return str(bg).replace("_", "-")


# ============================================================
# 3. GEOMETRY / COORDINATE HELPERS
# ============================================================

def detect_geometry_column(df):
    for c in ["geometry", "geom", "wkt", "WKT", "geometry_wkt", "hex_wkt"]:
        if c in df.columns:
            return c
    return None


def guess_crs_from_bounds(gdf):
    minx, miny, maxx, maxy = gdf.total_bounds
    if (-180 <= minx <= 180) and (-180 <= maxx <= 180) and (-90 <= miny <= 90) and (-90 <= maxy <= 90):
        return "EPSG:4326"
    return "EPSG:3035"


def _find_exact_col(df, candidates):
    """Find a column by exact case-insensitive match only."""
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def load_hex_points():
    """
    Return hex_id, lon, lat, x_3035, y_3035, area_from_geom_km2 if possible.

    The plotting CRS for Section 3.6 maps is EPSG:3035, matching the
    previous temporal maps. lon/lat are retained only for bbox filtering.
    """
    if not Path(HEX_GEOM_CSV).exists():
        print(f"HEX_GEOM_CSV not found: {HEX_GEOM_CSV}")
        return None
    if not HAS_GEO:
        print("geopandas/shapely unavailable; maps will be skipped.")
        return None

    hx = pd.read_csv(HEX_GEOM_CSV)
    hx = standardize_hex_id(hx)

    print("Loaded HEX_GEOM_CSV for geometry/points:")
    print(HEX_GEOM_CSV)
    print("Geometry table columns:")
    print(list(hx.columns))

    # --------------------------------------------------------
    # 1) True lon/lat columns, exact-name detection only.
    # --------------------------------------------------------
    lon_col = _find_exact_col(
        hx,
        ["lon", "longitude", "centroid_lon", "center_lon", "lon_4326", "longitude_4326"]
    )
    lat_col = _find_exact_col(
        hx,
        ["lat", "latitude", "centroid_lat", "center_lat", "lat_4326", "latitude_4326"]
    )

    if lon_col is not None and lat_col is not None:
        print(f"Using lon/lat columns: {lon_col}, {lat_col}")
        out = hx.loc[:, ["hex_id", lon_col, lat_col]].copy()
        out.columns = ["hex_id", "lon", "lat"]
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
        out = out[np.isfinite(out["lon"]) & np.isfinite(out["lat"])].copy()
        gdf = gpd.GeoDataFrame(
            out,
            geometry=gpd.points_from_xy(out["lon"], out["lat"]),
            crs="EPSG:4326"
        ).to_crs("EPSG:3035")
        out["x_3035"] = gdf.geometry.x.values
        out["y_3035"] = gdf.geometry.y.values
        out["area_from_geom_km2"] = np.nan
        return out.drop_duplicates("hex_id")

    # --------------------------------------------------------
    # 2) WKT geometry column. Prefer this over x/y because it is
    #    less ambiguous in the stylematch CSV.
    # --------------------------------------------------------
    geom_col = detect_geometry_column(hx)
    if geom_col is not None:
        print(f"Using WKT geometry column: {geom_col}")
        hx["_geometry"] = hx[geom_col].apply(
            lambda z: wkt.loads(z) if isinstance(z, str) and len(z) > 0 else None
        )
        gdf = gpd.GeoDataFrame(hx, geometry="_geometry")
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()
        if len(gdf) > 0:
            guessed = guess_crs_from_bounds(gdf)
            print(f"Guessed geometry CRS: {guessed}")
            gdf = gdf.set_crs(guessed, allow_override=True)

            gdf_3035 = gdf.to_crs("EPSG:3035").copy()
            if gdf_3035.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any():
                area = gdf_3035.geometry.area.values / 1e6
                rep_3035 = gdf_3035.geometry.representative_point()
            else:
                area = np.full(len(gdf_3035), np.nan, dtype=float)
                rep_3035 = gdf_3035.geometry

            rep_4326 = gpd.GeoSeries(rep_3035, crs="EPSG:3035").to_crs("EPSG:4326")

            out = pd.DataFrame({
                "hex_id": gdf_3035["hex_id"].values,
                "lon": rep_4326.x.values,
                "lat": rep_4326.y.values,
                "x_3035": rep_3035.x.values,
                "y_3035": rep_3035.y.values,
                "area_from_geom_km2": area,
            })
            out = out[
                np.isfinite(out["lon"]) & np.isfinite(out["lat"]) &
                np.isfinite(out["x_3035"]) & np.isfinite(out["y_3035"])
            ].copy()
            return out.drop_duplicates("hex_id")

        print("Geometry column exists but no valid geometries were parsed.")

    # --------------------------------------------------------
    # 3) Projected x/y fallback, exact-name detection only.
    # --------------------------------------------------------
    x_col = _find_exact_col(hx, ["x", "X", "centroid_x", "center_x", "x_3035", "X_3035"])
    y_col = _find_exact_col(hx, ["y", "Y", "centroid_y", "center_y", "y_3035", "Y_3035"])

    if x_col is not None and y_col is not None:
        print(f"Using projected x/y columns: {x_col}, {y_col}")
        temp = hx.loc[:, ["hex_id", x_col, y_col]].copy()
        temp.columns = ["hex_id", "x_3035", "y_3035"]
        temp["x_3035"] = pd.to_numeric(temp["x_3035"], errors="coerce")
        temp["y_3035"] = pd.to_numeric(temp["y_3035"], errors="coerce")
        temp = temp[np.isfinite(temp["x_3035"]) & np.isfinite(temp["y_3035"])].copy()
        gdf = gpd.GeoDataFrame(
            temp,
            geometry=gpd.points_from_xy(temp["x_3035"], temp["y_3035"]),
            crs="EPSG:3035"
        ).to_crs("EPSG:4326")
        out = pd.DataFrame({
            "hex_id": temp["hex_id"].values,
            "lon": gdf.geometry.x.values,
            "lat": gdf.geometry.y.values,
            "x_3035": temp["x_3035"].values,
            "y_3035": temp["y_3035"].values,
        })
        out["area_from_geom_km2"] = np.nan
        return out.drop_duplicates("hex_id")

    print("No geometry/lon-lat/x-y columns found in HEX_GEOM_CSV; maps will be skipped.")
    return None


# ============================================================
# 4. PREDICTION TEMPORAL METRICS
# ============================================================

def compute_prediction_temporal_from_period():
    if not Path(PRED_PERIOD_CSV).exists():
        raise FileNotFoundError(f"PRED_PERIOD_CSV not found: {PRED_PERIOD_CSV}")

    df = pd.read_csv(PRED_PERIOD_CSV)
    df = standardize_hex_id(df)
    date_col = detect_col(df, [PRED_DATE_COL, "date", "obs_date", "time", "period_start"], role="prediction date")
    pred_col = detect_col(df, [PRED_VALUE_COL, "mean_pred_wind", "pred", "prediction", "pred_prob", "probability", "mean_prob"], role="period prediction")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df = df[df[date_col].notna() & np.isfinite(df[pred_col])].copy()
    df["month"] = df[date_col].dt.month.astype(int)

    monthly = (
        df.groupby(["hex_id", "month"], as_index=False)
        .agg(pred_month_value=(pred_col, "mean"), n_pred_periods=(pred_col, "count"))
    )

    rows = []
    for hid, sub in monthly.groupby("hex_id"):
        vals = np.zeros(12, dtype=float)
        for _, r in sub.iterrows():
            m = int(r["month"])
            if 1 <= m <= 12:
                vals[m - 1] = float(r["pred_month_value"])
        pm, ps = peak_month_and_share(vals)
        em = effective_months(vals)
        rows.append({
            "hex_id": hid,
            "pred_peak_month": pm,
            "pred_peak_share": ps,
            "pred_effective_months": em,
        })

    pred = pd.DataFrame(rows)
    pred.to_csv(OUTDIR / f"TableS_prediction_temporal_metrics_{BACKGROUND}.csv", index=False, encoding="utf-8-sig")
    print(f"Computed prediction peak month/share from period table. Valid hexes: {pred['pred_peak_month'].notna().sum():,}")
    return pred


def load_prediction_temporal():
    if Path(PRED_TEMP_CSV).exists():
        try:
            df = pd.read_csv(PRED_TEMP_CSV)
            df = standardize_hex_id(df)
            peak_col = detect_col(df, ["pred_peak_month", "prediction_peak_month", "peak_month", "peak_month_num", "peak_month_id"], required=True, role="prediction peak month")
            share_col = detect_col(df, ["pred_peak_share", "prediction_peak_share", "peak_month_share", "pred_peak_month_share", "max_month_share"], required=True, role="prediction peak-month share")
            out = df[["hex_id", peak_col, share_col]].copy()
            out = out.rename(columns={peak_col: "pred_peak_month", share_col: "pred_peak_share"})
            out["pred_peak_month"] = pd.to_numeric(out["pred_peak_month"], errors="coerce")
            out["pred_peak_share"] = pd.to_numeric(out["pred_peak_share"], errors="coerce")
            out = out[out["pred_peak_month"].between(1, 12) & np.isfinite(out["pred_peak_share"])].copy()
            print(f"Loaded prediction temporal metrics from: {PRED_TEMP_CSV}")
            print(f"Prediction temporal rows: {len(out):,}")
            return out.drop_duplicates("hex_id")
        except Exception as e:
            print(f"Could not use PRED_TEMP_CSV because: {e}")
            print("Falling back to PRED_PERIOD_CSV.")

    return compute_prediction_temporal_from_period()


# ============================================================
# 5. WIND-ANOMALY TEMPORAL METRICS
# ============================================================

def temporal_from_monthly_profile(monthly):
    monthly = standardize_hex_id(monthly)

    month_col = detect_col(monthly, ["month", "mon", "month_id"], required=True, role="month")
    monthly[month_col] = pd.to_numeric(monthly[month_col], errors="coerce")
    monthly = monthly[monthly[month_col].between(1, 12)].copy()
    monthly["month"] = monthly[month_col].astype(int)

    if WIND_PROFILE_METRIC.lower().startswith("freq"):
        value_col = detect_col(monthly, ["anomaly_frequency", "anomaly_frequency_month", "anom_freq", "freq", "frequency"], role="monthly anomaly frequency")
        metric_name = "frequency"
    else:
        value_col = detect_col(monthly, ["anomaly_intensity", "anomaly_intensity_month", "anom_intensity", "intensity", "z_positive", "zpos"], role="monthly anomaly intensity")
        metric_name = "intensity"

    monthly[value_col] = pd.to_numeric(monthly[value_col], errors="coerce")
    monthly["wind_profile_value"] = np.where(np.isfinite(monthly[value_col]), monthly[value_col], 0.0)

    rows = []
    for hid, sub in monthly.groupby("hex_id"):
        vals = np.zeros(12, dtype=float)
        for _, r in sub.iterrows():
            m = int(r["month"])
            if 1 <= m <= 12:
                vals[m - 1] = float(r["wind_profile_value"])
        pm, ps = peak_month_and_share(vals)
        em = effective_months(vals)
        rows.append({
            "hex_id": hid,
            "wind_peak_month": pm,
            "wind_peak_share": ps,
            "wind_effective_months": em,
            "wind_profile_metric": metric_name,
        })

    out = pd.DataFrame(rows)
    return out, monthly[["hex_id", "month", "wind_profile_value"]].copy()


def build_monthly_profile_from_period_anomaly(path):
    df = pd.read_csv(path)
    df = standardize_hex_id(df)
    date_col = detect_col(df, [ANOM_DATE_COL, "date", "obs_date", "time", "period_start"], role="anomaly date")
    z_col = detect_col(df, [ANOM_Z_COL, "z_value", "z_anomaly", "standardized_anomaly", "z", "anomaly_z"], role="existing anomaly z")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[z_col] = pd.to_numeric(df[z_col], errors="coerce")
    df = df[df[date_col].notna() & np.isfinite(df[z_col])].copy()
    df["month"] = df[date_col].dt.month.astype(int)
    df["anomaly_frequency"] = (df[z_col] >= Z_EVENT_THRESHOLD).astype(float)
    df["anomaly_intensity"] = np.maximum(df[z_col].values.astype(float), 0.0)

    monthly = (
        df.groupby(["hex_id", "month"], as_index=False)
        .agg(
            anomaly_frequency=("anomaly_frequency", "mean"),
            anomaly_intensity=("anomaly_intensity", "mean"),
            n_anomaly_periods=(z_col, "count"),
        )
    )
    return monthly


def infer_nc_coord_names(ds):
    x_candidates = ["lon", "longitude", "x", "X"]
    y_candidates = ["lat", "latitude", "y", "Y"]
    t_candidates = ["time", "date", "obs_date"]
    x_name = next((c for c in x_candidates if c in ds.coords or c in ds.dims), None)
    y_name = next((c for c in y_candidates if c in ds.coords or c in ds.dims), None)
    t_name = next((c for c in t_candidates if c in ds.coords or c in ds.dims), None)
    if x_name is None or y_name is None or t_name is None:
        raise KeyError(f"Cannot infer NetCDF coordinate names. Dims={list(ds.dims)}, coords={list(ds.coords)}")
    return x_name, y_name, t_name


def build_monthly_profile_from_wind_nc():
    if not ALLOW_BUILD_WIND_FROM_NC:
        raise FileNotFoundError("No wind-anomaly timing input found and ALLOW_BUILD_WIND_FROM_NC=False.")
    if not HAS_XARRAY:
        raise ImportError("xarray is required to rebuild wind anomaly timing from WIND_NC.")
    if not Path(WIND_NC).exists():
        raise FileNotFoundError(f"WIND_NC not found: {WIND_NC}")
    if not Path(PRED_PERIOD_CSV).exists():
        raise FileNotFoundError(f"PRED_PERIOD_CSV not found: {PRED_PERIOD_CSV}")

    points = load_hex_points()
    if points is None or len(points) == 0:
        raise ValueError("Cannot rebuild wind anomaly from NC without hex coordinates/geometry.")

    # Prediction period dates define the time grid.
    period = pd.read_csv(PRED_PERIOD_CSV)
    period = standardize_hex_id(period)
    date_col = detect_col(period, [PRED_DATE_COL, "date", "obs_date", "time", "period_start"], role="prediction period date")
    period[date_col] = pd.to_datetime(period[date_col], errors="coerce")
    period = period[period[date_col].notna()].copy()
    period = period[["hex_id", date_col]].drop_duplicates().copy()
    period = period.merge(points[["hex_id", "lon", "lat"]], on="hex_id", how="inner")
    period["month"] = period[date_col].dt.month.astype(int)

    ds = open_dataset_fallback(WIND_NC)
    if WIND_VAR not in ds.data_vars:
        raise KeyError(f"{WIND_VAR} not found in WIND_NC. Available variables: {list(ds.data_vars)}")
    x_name, y_name, t_name = infer_nc_coord_names(ds)
    da = ds[WIND_VAR]

    xs = period["lon"].to_numpy(dtype=float)
    ys = period["lat"].to_numpy(dtype=float)
    if x_name.lower() in ["lon", "longitude"]:
        nc_x = ds[x_name].values
        if np.nanmin(nc_x) >= 0 and np.nanmax(nc_x) > 180:
            xs = np.where(xs < 0, xs + 360, xs)
    else:
        # If NC uses projected coordinates, convert points to EPSG:3035.
        if not HAS_GEO:
            ds.close()
            raise ImportError("geopandas required to convert lon/lat to projected coordinates for this WIND_NC.")
        gdf = gpd.GeoDataFrame(period, geometry=gpd.points_from_xy(period["lon"], period["lat"]), crs="EPSG:4326").to_crs("EPSG:3035")
        xs = gdf.geometry.x.values
        ys = gdf.geometry.y.values

    print("Sampling wind values from NetCDF. This may take a while...")
    sampled = da.sel(
        {
            x_name: xr.DataArray(xs, dims="points"),
            y_name: xr.DataArray(ys, dims="points"),
            t_name: xr.DataArray(period[date_col].values.astype("datetime64[ns]"), dims="points"),
        },
        method="nearest"
    )
    period["wind_value"] = np.asarray(sampled.values, dtype=float).reshape(-1)
    ds.close()

    period = period[np.isfinite(period["wind_value"])].copy()
    if BACKGROUND == "long_term":
        stats = period.groupby("hex_id")["wind_value"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "mu", "std": "sd"})
        period = period.merge(stats, on="hex_id", how="left")
    elif BACKGROUND == "same_month":
        stats = period.groupby(["hex_id", "month"])["wind_value"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "mu", "std": "sd"})
        period = period.merge(stats, on=["hex_id", "month"], how="left")
    else:
        raise ValueError("BACKGROUND must be long_term or same_month")

    period["z_value"] = (period["wind_value"] - period["mu"]) / period["sd"]
    period.loc[~np.isfinite(period["z_value"]), "z_value"] = np.nan
    period_out = OUTDIR / f"hex_period_wind_anomaly_{BACKGROUND}.csv"
    period[["hex_id", date_col, "month", "wind_value", "z_value"]].to_csv(period_out, index=False, encoding="utf-8-sig")
    print(f"Saved rebuilt period anomaly table: {period_out}")

    period["anomaly_frequency"] = (period["z_value"] >= Z_EVENT_THRESHOLD).astype(float)
    period["anomaly_intensity"] = np.maximum(period["z_value"].values.astype(float), 0.0)
    monthly = (
        period.groupby(["hex_id", "month"], as_index=False)
        .agg(
            anomaly_frequency=("anomaly_frequency", "mean"),
            anomaly_intensity=("anomaly_intensity", "mean"),
            n_anomaly_periods=("z_value", "count"),
        )
    )
    return monthly


def load_wind_temporal():
    # For a true raw-data reproduction, ignore all precomputed wind timing tables
    # and rebuild the anomaly timing directly from WIND_NC.
    if FORCE_WIND_FROM_NC:
        print("FIG5_FORCE_WIND_FROM_NC=1; rebuilding wind-anomaly timing directly from WIND_NC.")
        monthly = build_monthly_profile_from_wind_nc()
        monthly.to_csv(OUTDIR / f"hex_monthly_wind_anomaly_profiles_{BACKGROUND}.csv", index=False, encoding="utf-8-sig")
        wind, monthly_out = temporal_from_monthly_profile(monthly)
        wind.to_csv(OUTDIR / f"TableS_wind_anomaly_temporal_metrics_{BACKGROUND}_{WIND_PROFILE_METRIC}.csv", index=False, encoding="utf-8-sig")
        if monthly_out is not None:
            monthly_out.to_csv(OUTDIR / f"TableS_wind_monthly_profile_{BACKGROUND}_{WIND_PROFILE_METRIC}.csv", index=False, encoding="utf-8-sig")
        print(f"Computed wind-anomaly peak month/share from raw WIND_NC using monthly {WIND_PROFILE_METRIC}. Valid hexes: {wind['wind_peak_month'].notna().sum():,}")
        return wind, monthly_out

    # 1. Already aggregated temporal metrics.
    if Path(WIND_TEMP_CSV).exists():
        df = pd.read_csv(WIND_TEMP_CSV)
        df = standardize_hex_id(df)
        peak_col = detect_col(df, ["wind_peak_month", "wind_anomaly_peak_month", "anomaly_peak_month", "wind_anom_peak_month", "peak_month"], required=True, role="wind-anomaly peak month")
        share_col = detect_col(df, ["wind_peak_share", "wind_peak_month_share", "wind_anomaly_peak_month_share", "anomaly_peak_month_share", "wind_anom_peak_share", "peak_month_share"], required=True, role="wind-anomaly peak-month share")
        out = df[["hex_id", peak_col, share_col]].copy().rename(columns={peak_col: "wind_peak_month", share_col: "wind_peak_share"})
        out["wind_peak_month"] = pd.to_numeric(out["wind_peak_month"], errors="coerce")
        out["wind_peak_share"] = pd.to_numeric(out["wind_peak_share"], errors="coerce")
        out = out[out["wind_peak_month"].between(1, 12) & np.isfinite(out["wind_peak_share"])].copy()
        out["wind_profile_metric"] = "precomputed"
        print(f"Loaded wind temporal metrics from: {WIND_TEMP_CSV}")
        return out.drop_duplicates("hex_id"), None

    print(f"WIND_TEMP_CSV not found: {WIND_TEMP_CSV}")

    monthly = None
    # 2. Existing monthly anomaly profile.
    if Path(WIND_MONTHLY_PROFILE_CSV).exists():
        print(f"Using WIND_MONTHLY_PROFILE_CSV: {WIND_MONTHLY_PROFILE_CSV}")
        monthly = pd.read_csv(WIND_MONTHLY_PROFILE_CSV)
    # 3. Existing period-level anomaly table.
    elif Path(WIND_PERIOD_ANOMALY_CSV).exists():
        print(f"Using WIND_PERIOD_ANOMALY_CSV: {WIND_PERIOD_ANOMALY_CSV}")
        monthly = build_monthly_profile_from_period_anomaly(WIND_PERIOD_ANOMALY_CSV)
        monthly.to_csv(OUTDIR / f"hex_monthly_wind_anomaly_profiles_{BACKGROUND}.csv", index=False, encoding="utf-8-sig")
    # 4. Rebuild from wind NetCDF.
    else:
        print("No existing monthly/period anomaly timing table found. Rebuilding from WIND_NC...")
        monthly = build_monthly_profile_from_wind_nc()
        monthly.to_csv(OUTDIR / f"hex_monthly_wind_anomaly_profiles_{BACKGROUND}.csv", index=False, encoding="utf-8-sig")

    wind, monthly_out = temporal_from_monthly_profile(monthly)
    wind.to_csv(OUTDIR / f"TableS_wind_anomaly_temporal_metrics_{BACKGROUND}_{WIND_PROFILE_METRIC}.csv", index=False, encoding="utf-8-sig")
    if monthly_out is not None:
        monthly_out.to_csv(OUTDIR / f"TableS_wind_monthly_profile_{BACKGROUND}_{WIND_PROFILE_METRIC}.csv", index=False, encoding="utf-8-sig")
    print(f"Computed wind-anomaly peak month/share from monthly {WIND_PROFILE_METRIC} profile. Valid hexes: {wind['wind_peak_month'].notna().sum():,}")
    return wind, monthly_out


# ============================================================
# 6. MERGE + SUMMARY
# ============================================================

def add_area_weight(df):
    """
    Locked Fig. 5 reproduction weighting.

    The confirmed original merged_temporal_correspondence_long_term.csv used
    complete hex-area weights, approximately 2165 km2 for every analysed hex.
    With 3,535 valid hexes this gives a total area weight of 7,653,275 km2.

    Do not replace this with forest_area_km2 for reproducing the published
    Fig. 5 outputs.
    """
    out = df.copy()
    out["area_weight_km2"] = float(LOCKED_FIG5_HEX_AREA_KM2)
    print(
        f"LOCKED complete hex-area weights: "
        f"{LOCKED_FIG5_HEX_AREA_KM2:g} km2 per hex."
    )
    return out


def build_merged_table():
    print("Loading prediction timing...")
    pred = load_prediction_temporal()
    print(f"Prediction temporal rows: {len(pred):,}; valid timing rows: {pred[['pred_peak_month', 'pred_peak_share']].dropna().shape[0]:,}")

    print("Loading wind-anomaly timing...")
    wind, raw_monthly = load_wind_temporal()
    print(f"Wind-anomaly temporal rows: {len(wind):,}; valid timing rows: {wind[['wind_peak_month', 'wind_peak_share']].dropna().shape[0]:,}")

    common = set(pred["hex_id"]).intersection(set(wind["hex_id"]))
    print(f"Common valid hex_id count: {len(common):,}")

    df = pred.merge(wind[["hex_id", "wind_peak_month", "wind_peak_share", "wind_profile_metric"]], on="hex_id", how="inner")
    print(f"Rows after prediction-wind merge: {len(df):,}")

    df = add_area_weight(df)
    before = len(df)
    ok = (
        df["pred_peak_month"].between(1, 12) &
        df["wind_peak_month"].between(1, 12) &
        np.isfinite(df["pred_peak_share"]) &
        np.isfinite(df["wind_peak_share"]) &
        np.isfinite(df["area_weight_km2"]) &
        (df["area_weight_km2"] > 0)
    )
    df = df[ok].copy()
    print(f"Rows before final validity filter: {before:,}; rows after filter: {len(df):,}")

    df["circular_peak_month_difference"] = [
        circular_peak_month_difference(p, w) for p, w in zip(df["pred_peak_month"], df["wind_peak_month"])
    ]
    df["signed_peak_month_shift"] = [
        signed_circular_month_shift(p, w) for p, w in zip(df["pred_peak_month"], df["wind_peak_month"])
    ]
    df["peak_month_share_difference"] = df["pred_peak_share"] - df["wind_peak_share"]
    df["abs_peak_month_share_difference"] = np.abs(df["peak_month_share_difference"])

    out_cols = [
        "hex_id", "pred_peak_month", "pred_peak_share", "area_weight_km2",
        "wind_peak_month", "wind_peak_share", "wind_profile_metric",
        "circular_peak_month_difference", "signed_peak_month_shift",
        "peak_month_share_difference", "abs_peak_month_share_difference"
    ]
    df[out_cols].to_csv(OUTDIR / f"merged_temporal_correspondence_{BACKGROUND}.csv", index=False, encoding="utf-8-sig")
    return df[out_cols].copy()



def validate_existing_merged_table(df, source_path):
    """
    Validate a previously saved merged_temporal_correspondence CSV.

    The cached CSV is expected to contain at least:
      hex_id, pred_peak_month, pred_peak_share,
      wind_peak_month, wind_peak_share

    Missing derived columns are recomputed here, so older cached CSVs can still
    be used for direct plotting.
    """
    df = standardize_hex_id(df)

    rename_map = {}
    required_aliases = {
        "pred_peak_month": ["pred_peak_month", "prediction_peak_month", "peak_month_pred"],
        "pred_peak_share": ["pred_peak_share", "prediction_peak_share", "peak_share_pred"],
        "wind_peak_month": ["wind_peak_month", "wind_anomaly_peak_month", "anomaly_peak_month", "peak_month_wind"],
        "wind_peak_share": ["wind_peak_share", "wind_anomaly_peak_share", "anomaly_peak_share", "peak_share_wind"],
    }
    for target, aliases in required_aliases.items():
        col = detect_col(df, aliases, required=True, role=target)
        if col != target:
            rename_map[col] = target
    if rename_map:
        df = df.rename(columns=rename_map)

    numeric_cols = [
        "pred_peak_month", "pred_peak_share",
        "wind_peak_month", "wind_peak_share",
        "area_weight_km2",
        "circular_peak_month_difference", "signed_peak_month_shift",
        "peak_month_share_difference", "abs_peak_month_share_difference",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "wind_profile_metric" not in df.columns:
        df["wind_profile_metric"] = "cached"

    # Keep existing area weights when present. If absent/invalid, derive them
    # from the geometry CSV or fall back to equal weights.
    if "area_weight_km2" not in df.columns:
        df["area_weight_km2"] = np.nan
    if df["area_weight_km2"].notna().sum() == 0 or (df["area_weight_km2"] > 0).sum() == 0:
        print("Cached merged CSV has no valid area_weight_km2; deriving area weights.")
        df = add_area_weight(df)

    # Recompute derived correspondence columns to avoid using stale columns from
    # older test runs.
    df["circular_peak_month_difference"] = [
        circular_peak_month_difference(p, w)
        for p, w in zip(df["pred_peak_month"], df["wind_peak_month"])
    ]
    df["signed_peak_month_shift"] = [
        signed_circular_month_shift(p, w)
        for p, w in zip(df["pred_peak_month"], df["wind_peak_month"])
    ]
    df["peak_month_share_difference"] = df["pred_peak_share"] - df["wind_peak_share"]
    df["abs_peak_month_share_difference"] = np.abs(df["peak_month_share_difference"])

    before = len(df)
    ok = (
        df["pred_peak_month"].between(1, 12) &
        df["wind_peak_month"].between(1, 12) &
        np.isfinite(df["pred_peak_share"]) &
        np.isfinite(df["wind_peak_share"]) &
        np.isfinite(df["area_weight_km2"]) &
        (df["area_weight_km2"] > 0) &
        np.isfinite(df["circular_peak_month_difference"])
    )
    df = df[ok].copy()

    out_cols = [
        "hex_id", "pred_peak_month", "pred_peak_share", "area_weight_km2",
        "wind_peak_month", "wind_peak_share", "wind_profile_metric",
        "circular_peak_month_difference", "signed_peak_month_shift",
        "peak_month_share_difference", "abs_peak_month_share_difference"
    ]

    print(f"Loaded existing merged CSV: {source_path}")
    print(f"Rows before cached CSV validity filter: {before:,}; rows after filter: {len(df):,}")

    if len(df) == 0:
        raise ValueError("Existing merged CSV was found, but no valid rows remained after filtering.")

    return df[out_cols].copy()


def load_existing_merged_table():
    """
    Return a cached merged table if it exists and FORCE_REBUILD is not enabled.
    Return None when the full rebuild path should be used.
    """
    merged_path = OUTDIR / f"merged_temporal_correspondence_{BACKGROUND}.csv"

    if FORCE_REBUILD:
        print("FIG5_FORCE_REBUILD=1; ignoring cached merged CSV and rebuilding all temporal inputs.")
        return None

    if not merged_path.exists():
        print(f"No cached merged CSV found: {merged_path}")
        return None

    try:
        cached = pd.read_csv(merged_path)
        return validate_existing_merged_table(cached, merged_path)
    except Exception as e:
        print(f"Cached merged CSV exists but could not be used: {e}")
        print("Falling back to full temporal rebuild.")
        return None


def load_or_build_merged_table():
    """
    Cached-CSV-first entry point used by main().

    Returns
    -------
    df : pandas.DataFrame
        Valid temporal correspondence table.
    used_existing_csv : bool
        True when df came from merged_temporal_correspondence_<BACKGROUND>.csv.
    """
    cached = load_existing_merged_table()
    if cached is not None:
        return cached, True

    return build_merged_table(), False


def create_summary_tables(df):
    w = df["area_weight_km2"].values

    summary_specs = [
        ("Peak-month difference", "circular_peak_month_difference"),
        ("Signed peak-month shift", "signed_peak_month_shift"),
        ("Prediction peak-month share", "pred_peak_share"),
        ("Wind-anomaly peak-month share", "wind_peak_share"),
        ("Peak-month share difference", "peak_month_share_difference"),
        ("Absolute peak-month share difference", "abs_peak_month_share_difference"),
    ]
    rows = []
    for label, col in summary_specs:
        rows.append({
            "metric": label,
            "weighted_mean": weighted_mean(df[col], w),
            "weighted_median": weighted_quantile(df[col], 0.50, w),
            "weighted_p25": weighted_quantile(df[col], 0.25, w),
            "weighted_p75": weighted_quantile(df[col], 0.75, w),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTDIR / f"TableS_temporal_correspondence_summary_{BACKGROUND}.csv", index=False, encoding="utf-8-sig")

    total_area = df["area_weight_km2"].sum()
    class_specs = [
        ("Same peak month", df["circular_peak_month_difference"] == 0, "Delta month = 0"),
        ("One-month offset", df["circular_peak_month_difference"] == 1, "Delta month = 1"),
        ("Two-month offset", df["circular_peak_month_difference"] == 2, "Delta month = 2"),
        ("Mismatched", df["circular_peak_month_difference"] >= 3, "Delta month >= 3"),
        ("Exact alignment", df["circular_peak_month_difference"] == 0, "Delta month = 0"),
        ("Near alignment", df["circular_peak_month_difference"] <= 1, "Delta month <= 1"),
        ("Broad alignment", df["circular_peak_month_difference"] <= 2, "Delta month <= 2"),
        ("Timing mismatch", df["circular_peak_month_difference"] >= 3, "Delta month >= 3"),
    ]
    agreement_rows = []
    for name, mask, definition in class_specs:
        area = df.loc[mask, "area_weight_km2"].sum()
        agreement_rows.append({
            "agreement_class": name,
            "definition": definition,
            "forest_area_km2": area,
            "forest_area_share_percent": 100 * area / total_area if total_area > 0 else np.nan,
        })
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(OUTDIR / f"TableS_peak_month_agreement_classes_{BACKGROUND}.csv", index=False, encoding="utf-8-sig")

    # Correlations
    corr_rows = []
    pearson_share = weighted_pearson(df["pred_peak_share"], df["wind_peak_share"], w)
    if HAS_SCIPY:
        rho_share, p_share = spearmanr(df["pred_peak_share"], df["wind_peak_share"], nan_policy="omit")
    else:
        rho_share, p_share = np.nan, np.nan
    corr_rows.append({
        "comparison": "Prediction peak-month share vs wind-anomaly peak-month share",
        "weighted_pearson": pearson_share,
        "spearman_rho": rho_share,
        "spearman_p": p_share,
    })

    # circular components for peak month
    pred_ang = 2 * np.pi * (df["pred_peak_month"].values.astype(float) - 1) / 12.0
    wind_ang = 2 * np.pi * (df["wind_peak_month"].values.astype(float) - 1) / 12.0
    for comp_name, pred_comp, wind_comp in [
        ("Peak-month cosine component", np.cos(pred_ang), np.cos(wind_ang)),
        ("Peak-month sine component", np.sin(pred_ang), np.sin(wind_ang)),
    ]:
        pearson = weighted_pearson(pred_comp, wind_comp, w)
        if HAS_SCIPY:
            rho, pval = spearmanr(pred_comp, wind_comp, nan_policy="omit")
        else:
            rho, pval = np.nan, np.nan
        corr_rows.append({
            "comparison": comp_name,
            "weighted_pearson": pearson,
            "spearman_rho": rho,
            "spearman_p": pval,
        })
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUTDIR / f"TableS_temporal_correlations_{BACKGROUND}.csv", index=False, encoding="utf-8-sig")

    print("\nSummary:")
    print(summary)
    print("\nAgreement classes:")
    print(agreement)
    print("\nCorrelations:")
    print(corr)
    return summary, agreement, corr


# ============================================================
# 7. PLOTTING -- publication style matched to indicator-suite maps
# ============================================================


def load_europe_admin_publication(countries_src, bbox4326):
    """
    Load and filter admin boundaries using the same logic as the style-matched
    EPSG:3035 publication maps.
    """
    if not HAS_GEO:
        return None
    if countries_src is None or not Path(countries_src).exists():
        print("Admin boundary path is missing. Maps will be drawn without admin borders.")
        return None

    admin = gpd.read_file(countries_src)
    if admin.crs is None:
        admin = admin.set_crs(PLOT_CRS)

    if "CNTR_ID" in admin.columns:
        admin = admin[admin["CNTR_ID"].astype(str).isin(EUROPE_CNTR_IDS)].copy()

    admin = admin.to_crs("EPSG:4326")
    admin = admin.explode(index_parts=False).reset_index(drop=True)

    xmin, ymin, xmax, ymax = bbox4326
    bbox_geom = box(xmin, ymin, xmax, ymax)
    admin = admin[admin.geometry.intersects(bbox_geom)].copy()
    admin["geometry"] = admin.geometry.intersection(bbox_geom)
    admin = admin[~admin.geometry.is_empty].copy()
    admin = admin[admin.geometry.notna()].copy()

    # Remove tiny fragments after clipping, matching the style script.
    admin_tmp = admin.to_crs(PLOT_CRS).copy()
    admin["area_km2_tmp"] = admin_tmp.geometry.area / 1e6
    admin = admin[admin["area_km2_tmp"] >= MIN_ADMIN_PART_AREA_KM2].copy()
    admin = admin.drop(columns="area_km2_tmp")

    admin = admin.to_crs(PLOT_CRS)
    print(f"Admin features kept after main-Europe filtering: {len(admin):,}")
    return admin


def load_hex_geometry_gdf():
    """
    Load hex geometry for publication-style maps.

    This follows the style-matched plotting logic: use WKT geometry whenever
    available, keep hex_id, infer CRS from bounds, then project to EPSG:3035.
    """
    if not HAS_GEO:
        print("geopandas/shapely unavailable; maps will be skipped.")
        return None
    if not Path(HEX_GEOM_CSV).exists():
        print(f"HEX_GEOM_CSV not found: {HEX_GEOM_CSV}")
        return None

    hx = pd.read_csv(HEX_GEOM_CSV, low_memory=False)
    hx = standardize_hex_id(hx)

    geom_col = detect_geometry_column(hx)
    if geom_col is not None:
        print(f"Using geometry column from HEX_GEOM_CSV: {geom_col}")
        gg = hx[["hex_id", geom_col]].copy()
        gg = gg.rename(columns={geom_col: "geometry"})
        gg = gg[gg["geometry"].notna()].copy()
        gg["geometry"] = gg["geometry"].apply(
            lambda z: wkt.loads(z) if isinstance(z, str) and len(z) > 0 else None
        )
        gdf = gpd.GeoDataFrame(gg, geometry="geometry")
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()
        if len(gdf) == 0:
            return None
        guessed = guess_crs_from_bounds(gdf)
        print(f"Guessed hex geometry CRS: {guessed}")
        gdf = gdf.set_crs(guessed, allow_override=True)
        return gdf[["hex_id", "geometry"]].drop_duplicates("hex_id")

    # fallback: use exact lon/lat or x/y coordinates as points
    pts = load_hex_points()
    if pts is None or len(pts) == 0:
        return None
    gdf = gpd.GeoDataFrame(
        pts[["hex_id", "x_3035", "y_3035"]].copy(),
        geometry=gpd.points_from_xy(pts["x_3035"], pts["y_3035"]),
        crs=PLOT_CRS,
    )
    return gdf[["hex_id", "geometry"]].drop_duplicates("hex_id")


def filter_main_europe_geometries(gdf, bbox4326):
    """Clip/filter geometries to main Europe in lon/lat, then keep EPSG:4326 geometries."""
    xmin, ymin, xmax, ymax = bbox4326
    bbox_geom = box(xmin, ymin, xmax, ymax)

    if gdf.crs is None:
        gdf = gdf.set_crs(PLOT_CRS)

    gg = gdf.to_crs("EPSG:4326").copy()
    gg = gg[gg.geometry.notna()].copy()
    gg = gg[~gg.geometry.is_empty].copy()
    gg = gg[gg.geometry.intersects(bbox_geom)].copy()
    if gg.empty:
        return gg

    # For polygons this clips to the bbox. For points, intersection is harmless.
    gg["geometry"] = gg.geometry.intersection(bbox_geom)
    gg = gg[~gg.geometry.is_empty].copy()
    gg = gg[gg.geometry.notna()].copy()
    return gg


def project_to_plot_crs(gdf):
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(PLOT_CRS)


def filter_hex_to_admin_footprint(gdf, admin_gdf):
    """Keep hexes whose representative point lies inside the selected admin footprint."""
    if admin_gdf is None or len(admin_gdf) == 0 or gdf is None or len(gdf) == 0:
        return gdf

    gg = gdf.copy()
    try:
        admin_union = admin_gdf.geometry.union_all()
    except Exception:
        admin_union = admin_gdf.unary_union

    geom_types = gg.geometry.geom_type.astype(str)
    if geom_types.isin(["Point", "MultiPoint"]).all():
        rep = gg.geometry
    else:
        rep = gg.geometry.representative_point()

    keep = rep.within(admin_union) | rep.touches(admin_union)
    return gg[keep].copy()


def prepare_plot_gdf(df, admin_plot, bbox4326):
    """Merge temporal-correspondence metrics with hex geometry and apply publication-style map filters."""
    geom = load_hex_geometry_gdf()
    if geom is None or len(geom) == 0:
        print("No usable hex geometry found. Map panels will be skipped.")
        return None

    # Merge first, so only analysed hexes are plotted.
    gdf = geom.merge(df, on="hex_id", how="inner")
    print(f"Hex geometries matched to temporal table: {len(gdf):,}")
    if len(gdf) == 0:
        return None

    gdf = filter_main_europe_geometries(gdf, bbox4326)
    if len(gdf) == 0:
        print("Geometry became empty after main-Europe bbox filtering.")
        return None

    gdf = project_to_plot_crs(gdf)
    gdf = filter_hex_to_admin_footprint(gdf, admin_plot)
    if len(gdf) == 0:
        print("Geometry became empty after admin-footprint filtering.")
        return None

    return gdf


def get_plot_bounds(hex_gdf, pad_ratio=0.025):
    xmin, ymin, xmax, ymax = hex_gdf.total_bounds
    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= 0 or dy <= 0:
        return xmin - 1, xmax + 1, ymin - 1, ymax + 1
    return xmin - dx * pad_ratio, xmax + dx * pad_ratio, ymin - dy * pad_ratio, ymax + dy * pad_ratio


def style_map_ax(ax, bounds):
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def draw_admin(ax, admin_gdf):
    if admin_gdf is not None and len(admin_gdf) > 0:
        admin_gdf.boundary.plot(ax=ax, linewidth=ADMIN_LW, color=ADMIN_COLOR, zorder=4)


def add_panel_label(ax, label):
    ax.text(
        -0.02, 1.03, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=12,
        fontweight="bold",
        zorder=20,
        clip_on=False,
    )


def plot_temporal_map_panel(ax, hex_gdf, value_col, title, cbar_label, admin_plot, bounds,
                            cmap, norm=None, vmin=None, vmax=None, missing_color="#efefef"):
    """Publication-style map panel using hex geometry in EPSG:3035."""
    vals = pd.to_numeric(hex_gdf[value_col], errors="coerce")
    bg = hex_gdf.copy()
    fg = hex_gdf[np.isfinite(vals)].copy()

    if len(bg) > 0:
        bg.plot(ax=ax, color=missing_color, edgecolor="none", linewidth=0.0, zorder=1)

    if len(fg) > 0:
        fg.plot(
            ax=ax,
            column=value_col,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.98,
            zorder=2,
            missing_kwds={"color": "#f5f5f5"},
        )

    draw_admin(ax, admin_plot)
    style_map_ax(ax, bounds)
    ax.set_title(title, pad=8)

    if len(fg) > 0:
        if norm is None:
            norm = mpl.colors.Normalize(vmin=vmin if vmin is not None else vals.min(),
                                        vmax=vmax if vmax is not None else vals.max())
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = ax.figure.colorbar(sm, ax=ax, shrink=0.72)
        cb.set_label(cbar_label, fontsize=10)
        cb.ax.tick_params(labelsize=9, length=3)
        cb.outline.set_linewidth(0.7)


def plot_temporal_correspondence(df):
    """
    Peak-month correspondence figure only.

    Panels:
      (a) Peak-month difference map
      (b) Area-weighted peak-month correspondence

    This version intentionally removes all peak-month share / concentration panels.
    """
    admin_plot = load_europe_admin_publication(ADMIN_SHP, BBOX4326)
    plot_gdf = prepare_plot_gdf(df, admin_plot, BBOX4326)
    if plot_gdf is None or len(plot_gdf) == 0:
        print("No geometry available for map panels. Tables were still saved.")
        return

    plot_gdf.to_csv(
        OUTDIR / f"debug_peak_month_correspondence_plot_attributes_{BACKGROUND}.csv",
        index=False,
        encoding="utf-8-sig"
    )
    try:
        plot_gdf.to_file(
            OUTDIR / f"debug_peak_month_correspondence_plot_footprint_{BACKGROUND}.gpkg",
            driver="GPKG"
        )
    except Exception as e:
        print(f"Could not save debug GPKG: {e}")

    # Use analysed hexes with valid peak-month difference for tight map bounds.
    extent_mask = np.isfinite(pd.to_numeric(plot_gdf["circular_peak_month_difference"], errors="coerce"))
    extent_gdf = plot_gdf.loc[extent_mask].copy()
    if extent_gdf.empty:
        extent_gdf = plot_gdf.copy()
    bounds = get_plot_bounds(extent_gdf, pad_ratio=0.025)

    print(f"Hexes kept for peak-month maps: {len(plot_gdf):,}")
    print(f"Plot CRS: {plot_gdf.crs}")
    print(f"Plot bounds: {bounds}")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12.0, 4.8),
        facecolor="white",
        gridspec_kw={"width_ratios": [1.05, 1.25]}
    )
    ax1, ax2 = axes.ravel()

    # --------------------------------------------------------
    # (a) Peak-month difference map
    # --------------------------------------------------------
    diff_cmap = mpl.cm.get_cmap("YlGnBu")
    diff_norm = mpl.colors.Normalize(vmin=0, vmax=6)
    plot_temporal_map_panel(
        ax=ax1,
        hex_gdf=plot_gdf,
        value_col="circular_peak_month_difference",
        title="Peak-month difference",
        cbar_label="Month difference",
        admin_plot=admin_plot,
        bounds=bounds,
        cmap=diff_cmap,
        norm=diff_norm,
    )
    add_panel_label(ax1, "(a)")

    # --------------------------------------------------------
    # (b) Area-weighted peak-month correspondence
    # Mutually exclusive classes: 0, 1, 2, and >=3 months.
    # --------------------------------------------------------
    total_area = plot_gdf["area_weight_km2"].sum()
    class_specs = [
        ("Exact\n0 months", plot_gdf["circular_peak_month_difference"] == 0, "#2b6cb0"),
        ("1-month\ndifference", plot_gdf["circular_peak_month_difference"] == 1, "#67a9cf"),
        ("2-month\ndifference", plot_gdf["circular_peak_month_difference"] == 2, "#d1e5f0"),
        ("Mismatch\n≥3 months", plot_gdf["circular_peak_month_difference"] >= 3, "#b2182b"),
    ]

    labels, vals, colors = [], [], []
    for lab, mask, col in class_specs:
        area = plot_gdf.loc[mask, "area_weight_km2"].sum()
        share = 100 * area / total_area if total_area > 0 else np.nan
        labels.append(lab)
        vals.append(share)
        colors.append(col)

    ax2.bar(labels, vals, color=colors, edgecolor="none", alpha=0.92)
    ymax = max(40, np.nanmax(vals) * 1.18 if np.isfinite(vals).any() else 40)
    ax2.set_ylim(0, ymax)
    ax2.set_ylabel("Forest-area share (%)")
    ax2.set_title("Area-weighted peak-month correspondence", pad=8)
    ax2.grid(axis="y", alpha=0.22)
    ax2.set_axisbelow(True)

    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax2.text(i, v + ymax * 0.025, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    add_panel_label(ax2, "(b)")

    plt.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.88,
        bottom=0.12,
        wspace=0.18,
    )

    out_png = OUTDIR / f"Fig5_peak_month_correspondence_{BACKGROUND}.png"
    out_pdf = OUTDIR / f"Fig5_peak_month_correspondence_{BACKGROUND}.pdf"
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {out_png}")

# ============================================================
# 8. MAIN
# ============================================================

def main():
    set_publication_style()
    print("=" * 80)
    print("Section 3.6: temporal correspondence between prediction and wind anomaly")
    print("=" * 80)
    print(f"Background: {BACKGROUND}")
    print(f"Wind profile metric: {WIND_PROFILE_METRIC}")
    print(f"LOCKED complete hex-area weighting is enabled; hex area = {LOCKED_FIG5_HEX_AREA_KM2:g} km2")
    print(f"Force rebuild: {FORCE_REBUILD}")
    print(f"Force wind rebuild from NC: {FORCE_WIND_FROM_NC}")

    print("\nLoading temporal correspondence table...")
    df, used_existing_csv = load_or_build_merged_table()
    print(f"Merged valid hexes: {len(df):,}")
    print(f"Total area weight: {df['area_weight_km2'].sum():,.2f}")

    if used_existing_csv and not REWRITE_TABLES_FROM_EXISTING_CSV:
        print("\nExisting merged CSV was used; skipping temporal rebuild and summary-table rewriting.")
        print("Set FIG5_REWRITE_TABLES_FROM_EXISTING_CSV=1 if you want to refresh summary tables from the cached CSV.")
    else:
        print("\nCreating summary tables...")
        create_summary_tables(df)

    print("\nPlotting main figure...")
    plot_temporal_correspondence(df)

    print("\nDone.")
    print(f"Outputs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
