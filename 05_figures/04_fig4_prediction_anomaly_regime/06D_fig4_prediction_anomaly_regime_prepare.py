# -*- coding: utf-8 -*-
"""
Section 3.5 final analysis: prediction × wind-anomaly frequency–intensity regimes

Why this script
---------------
This script replaces the earlier prediction-defined hotspot typology with a more
transparent two-step design:

1. Define wind-anomaly regimes independently from prediction outputs, using only
   anomaly frequency and standardized anomaly intensity.
2. Quantify whether mean and p95 wind-only predictions are enriched within those
   independent wind-anomaly regimes.

Main output for each background in BACKGROUNDS
---------------------------------------------
1. merged_prediction_anomaly_<background>.csv

This is the only preparation output required by
06E_fig4_prediction_anomaly_regime_plot.py. Diagnostic tables and preliminary
maps from the development version are intentionally not written in this minimal
paper-results package.

Author: adapted from your 19winddisturboverlap.py and 19spatialoverlaplongterm.py
"""

from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely import wkt
from shapely.geometry import Point, box

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression




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

# Prediction table used to draw mean and p95 maps.
# It should contain geometry/WKT or lon/lat, plus mean and p95 probability columns.
PRED_CSV = Path(os.environ.get(
    "FIG1_INDICATOR_CSV",
    r"E:\RF_BG_REPRO_from_model_dev\outputs\04_fig1_spatial_indicators\hex_indicator_summary_stylematch_4326.csv"
))

# Wind anomaly maps. Each NetCDF should contain:
#   z_value   = standardized anomaly intensity
#   freq_value = anomaly frequency
ANOM_NC_BY_BG = {
    "long_term": Path(os.environ.get("WIND_OVERLAY_LONG_NC", r"E:\RF B+G\windmap\overlay_result_long_term_background.nc")),
    "same_month": Path(os.environ.get("WIND_OVERLAY_MONTH_NC", r"E:\RF B+G\windmap\overlay_result_same_month_background.nc")),
}

# Main analysis output folder
OUTDIR = Path(os.environ.get("FIG4_PREP_OUTDIR", r"E:\RF_BG_REPRO_from_model_dev\outputs\06_fig4_prediction_anomaly_regime_prepare"))

# Run long_term as the main analysis. Keep same_month as sensitivity if needed.
# Recommended for the paper main text: BACKGROUNDS = ["long_term"]
# Recommended for supplementary check: BACKGROUNDS = [x.strip() for x in os.environ.get("FIG4_BACKGROUNDS", "long_term").split(",") if x.strip()]
BACKGROUNDS = [x.strip() for x in os.environ.get("FIG4_BACKGROUNDS", "long_term").split(",") if x.strip()]

# Variable overrides if auto-detection fails
VAR_OVERRIDES = {
    "same_month": {"z": "z_value", "freq": "freq_value"},
    "long_term": {"z": "z_value", "freq": "freq_value"},
}

# Administrative boundary shapefile, EPSG:3035
ADMIN_SHP = Path(os.environ.get("COUNTRIES_SRC", r"E:\CNTR_RG_20M_2024_3035\CNTR_RG_20M_2024_3035.shp"))

# Main Europe extent in lon/lat; Russia and Iceland are excluded below.
BBOX_LONLAT = (-12, 34, 45, 72)
PLOT_CRS = "EPSG:3035"

# Keep only main European countries. Russia and Iceland are intentionally excluded.
EUROPE_CNTR_IDS = {
    "AL", "AD", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ", "DE", "DK",
    "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LI", "LT", "LU",
    "LV", "MD", "ME", "MK", "MT", "NL", "NO", "PL", "PT", "RO", "RS", "SE",
    "SI", "SK", "SM", "UA", "VA", "XK", "UK", "GB"
}
MIN_ADMIN_PART_AREA_KM2 = 20

# Prediction thresholds used for visualization / high-prediction context.
# 0.80 = upper quintile = top 20% by forest-area-weighted prediction value.
Q_HIGH_PRED_BASELINE = 0.80
TOP_PRED_QS_FOR_SENSITIVITY = [0.70, 0.80, 0.90]  # top 30%, 20%, 10%

# Wind regime thresholds.
# 0.75 is the baseline for high frequency and high intensity.
WIND_Q_BASELINE = 0.75
WIND_QS_FOR_SENSITIVITY = [0.70, 0.75, 0.80]

# Use forest-area-weighted quantiles for all thresholds.
USE_WEIGHTED_QUANTILE_FOR_THRESHOLDS = True

# Plot settings
FIG_DPI = 500
FIG_W_2X2 = 12.4
FIG_H_2X2 = 10.8
FIG_W_MAP = 9.2
FIG_H_MAP = 8.6
FIG_W_BAR = 12.0
FIG_H_BAR = 5.2

# Color stretch for map backgrounds
MAP_Q_LOW = 0.02
MAP_Q_HIGH = 0.98

# Boundary / overlay styles
ADMIN_LW = 0.28
ADMIN_COLOR = "0.35"
TOP_OUTLINE_COLOR = "black"
TOP_OUTLINE_LW = 0.95

# Wind-regime colors
REGIME_ORDER = [
    "Compound recurrent-intense",
    "Recurrent anomaly",
    "Intensity-dominated anomaly",
    "Background / mixed",
]
REGIME_COLORS = {
    "Compound recurrent-intense": "#7b3294",
    "Recurrent anomaly": "#2c7fb8",
    "Intensity-dominated anomaly": "#d7191c",
    "Background / mixed": "#e6e6e6",
}

SAVE_FIGURES = True


# ============================================================
# 1. STYLE AND UTILITIES
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
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def detect_col(df, candidates, required=True, label="column"):
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for col in df.columns:
        cl = col.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return col

    if required:
        raise KeyError(
            f"Cannot detect {label}. Tried: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )
    return None


def weighted_quantile(values, quantile, weights=None):
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    values = values[mask]

    if weights is None:
        return float(np.nanquantile(values, quantile))

    weights = np.asarray(weights, dtype=float)[mask]
    ok = np.isfinite(weights) & (weights > 0)
    values = values[ok]
    weights = weights[ok]

    if len(values) == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    return float(np.interp(quantile, cdf, values))


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def weighted_sum(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.sum(values[mask] * weights[mask]))


def percentile_rank(series):
    return pd.Series(series).rank(method="average", pct=True)


def get_analysis_valid_mask(df):
    required = [
        "mean_prob",
        "p95_prob",
        "z_anomaly",
        "anomaly_frequency",
        "area_weight_km2",
    ]

    valid = np.ones(len(df), dtype=bool)
    for c in required:
        valid &= np.isfinite(pd.to_numeric(df[c], errors="coerce"))
    valid &= pd.to_numeric(df["area_weight_km2"], errors="coerce") > 0
    return valid


# ============================================================
# 2. NETCDF SAMPLING
# ============================================================

def infer_lat_lon_names(ds):
    lon_candidates = ["lon", "longitude", "x"]
    lat_candidates = ["lat", "latitude", "y"]

    lon_name = None
    lat_name = None

    for c in lon_candidates:
        if c in ds.coords or c in ds.dims:
            lon_name = c
            break

    for c in lat_candidates:
        if c in ds.coords or c in ds.dims:
            lat_name = c
            break

    if lon_name is None or lat_name is None:
        raise KeyError(
            f"Cannot infer lon/lat coordinate names. "
            f"Dims: {ds.dims}, Coords: {list(ds.coords)}"
        )

    return lon_name, lat_name


def find_var(ds, background, kind, override=None):
    if override is not None:
        if override not in ds.data_vars:
            raise KeyError(
                f"Override variable '{override}' not found. "
                f"Available variables: {list(ds.data_vars)}"
            )
        return override

    variables = list(ds.data_vars)

    if kind == "z":
        preferred = ["z_value", "z_anomaly", "z", "anomaly_z", "standardized_anomaly"]
        for p in preferred:
            if p in variables:
                return p
        candidates = []
        for v in variables:
            vl = v.lower()
            if (("z" in vl or "anom" in vl or "anomaly" in vl)
                    and "freq" not in vl and "score" not in vl
                    and "class" not in vl and "mask" not in vl):
                candidates.append(v)

    elif kind == "freq":
        preferred = ["freq_value", "frequency", "anomaly_frequency", "freq"]
        for p in preferred:
            if p in variables:
                return p
        candidates = []
        for v in variables:
            vl = v.lower()
            if (("freq" in vl or "frequency" in vl or "count" in vl)
                    and "score" not in vl and "class" not in vl
                    and "mask" not in vl and "target" not in vl):
                candidates.append(v)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    if len(candidates) != 1:
        print("\nAvailable NetCDF variables:")
        for v in variables:
            print("  -", v)
        raise KeyError(
            f"Cannot uniquely detect variable for background={background}, kind={kind}. "
            f"Candidates: {candidates}. Please set VAR_OVERRIDES manually."
        )
    return candidates[0]


def sample_nc_to_points(ds, varname, lons, lats):
    lon_name, lat_name = infer_lat_lon_names(ds)
    da = ds[varname]

    # If there are unexpected extra dimensions, take max across them.
    spatial_dims = {lon_name, lat_name}
    extra_dims = [d for d in da.dims if d not in spatial_dims]
    if len(extra_dims) > 0:
        print(f"Variable {varname} has extra dims {extra_dims}; using max over them.")
        da = da.max(dim=extra_dims, skipna=True)

    lon_values = ds[lon_name].values
    lons_sample = np.asarray(lons, dtype=float).copy()
    lats_sample = np.asarray(lats, dtype=float).copy()

    # Convert lon from -180/180 to 0/360 if needed.
    if np.nanmin(lon_values) >= 0 and np.nanmax(lon_values) > 180:
        lons_sample = np.where(lons_sample < 0, lons_sample + 360, lons_sample)

    sampled = da.sel(
        {
            lon_name: xr.DataArray(lons_sample, dims="points"),
            lat_name: xr.DataArray(lats_sample, dims="points"),
        },
        method="nearest",
    )

    return np.asarray(sampled.values, dtype=float).reshape(-1)


# ============================================================
# 3. DATA LOADING
# ============================================================

def load_prediction_table(pred_csv: Path):
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    df = pd.read_csv(pred_csv)

    print("=" * 80)
    print("Loaded prediction table")
    print("=" * 80)
    print(pred_csv)
    print(f"Rows: {len(df):,}")
    print("Columns:")
    print(list(df.columns))

    mean_col = detect_col(
        df,
        [
            "mean_pred_prob", "mean_probability", "mean_prob",
            "pred_prob_mean", "prob_mean", "mean_predicted_probability",
            "mean_predicted_prob", "mean_pred_wind", "mean",
        ],
        label="mean predicted probability column",
    )

    p95_col = detect_col(
        df,
        [
            "p95_pred_prob", "p95_probability", "p95_prob",
            "pred_prob_p95", "prob_p95", "q95_pred_prob",
            "q95_probability", "p95_predicted_probability",
            "p95_predicted_prob", "p95_pred_wind", "p95",
        ],
        label="p95 predicted probability column",
    )

    area_col = detect_col(
        df,
        [
            "forest_area_km2", "forest_km2", "area_forest_km2",
            "hex_forest_area_km2", "area_weight_km2", "area_km2", "hex_area_km2",
        ],
        required=False,
        label="area/forest area column",
    )

    geom_col = detect_col(
        df,
        ["geometry", "geom", "wkt", "geometry_wkt", "hex_wkt"],
        required=False,
        label="geometry/WKT column",
    )

    lon_col = detect_col(
        df,
        ["lon", "longitude", "centroid_lon", "center_lon", "lon_4326", "x_lon"],
        required=False,
        label="longitude column",
    )

    lat_col = detect_col(
        df,
        ["lat", "latitude", "centroid_lat", "center_lat", "lat_4326", "y_lat"],
        required=False,
        label="latitude column",
    )

    if geom_col is not None:
        geom = df[geom_col].apply(lambda z: wkt.loads(z) if isinstance(z, str) else z)
        gdf = gpd.GeoDataFrame(df.copy(), geometry=geom)

        bounds = gdf.total_bounds
        xmin, ymin, xmax, ymax = bounds
        if (-180 <= xmin <= 180) and (-180 <= xmax <= 180) and (-90 <= ymin <= 90) and (-90 <= ymax <= 90):
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        else:
            gdf = gdf.set_crs("EPSG:3035", allow_override=True).to_crs("EPSG:4326")

        # Use representative point instead of centroid in geographic CRS.
        rep = gdf.to_crs("EPSG:3035").geometry.representative_point()
        rep4326 = gpd.GeoSeries(rep, crs="EPSG:3035").to_crs("EPSG:4326")
        gdf["lon_centroid"] = rep4326.x.values
        gdf["lat_centroid"] = rep4326.y.values

    elif lon_col is not None and lat_col is not None:
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")
        gdf["lon_centroid"] = pd.to_numeric(df[lon_col], errors="coerce")
        gdf["lat_centroid"] = pd.to_numeric(df[lat_col], errors="coerce")

    else:
        raise KeyError("Cannot find geometry/WKT or lon/lat columns.")

    gdf["mean_prob"] = pd.to_numeric(gdf[mean_col], errors="coerce")
    gdf["p95_prob"] = pd.to_numeric(gdf[p95_col], errors="coerce")

    if area_col is not None:
        gdf["area_weight_km2"] = pd.to_numeric(gdf[area_col], errors="coerce")
    else:
        if gdf.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]:
            area_gdf = gdf.to_crs("EPSG:3035")
            gdf["area_weight_km2"] = area_gdf.geometry.area / 1e6
        else:
            gdf["area_weight_km2"] = 1.0

    gdf.loc[gdf["area_weight_km2"] <= 0, "area_weight_km2"] = np.nan

    print(f"Detected mean probability column: {mean_col}")
    print(f"Detected p95 probability column: {p95_col}")
    print(f"Detected area column: {area_col}")

    return gdf


def add_anomaly_metrics(gdf, anom_nc: Path, background: str):
    if not anom_nc.exists():
        raise FileNotFoundError(f"Anomaly NetCDF not found: {anom_nc}")

    ds = open_dataset_fallback(anom_nc)

    z_override = VAR_OVERRIDES.get(background, {}).get("z", None)
    freq_override = VAR_OVERRIDES.get(background, {}).get("freq", None)

    z_var = find_var(ds, background, "z", z_override)
    freq_var = find_var(ds, background, "freq", freq_override)

    print("\n" + "=" * 80)
    print(f"Sampling anomaly metrics: {background}")
    print("=" * 80)
    print(f"z/intensity variable: {z_var}")
    print(f"frequency variable: {freq_var}")

    out = gdf.copy()

    out["z_anomaly"] = sample_nc_to_points(
        ds, z_var, out["lon_centroid"].values, out["lat_centroid"].values
    )
    out["anomaly_intensity"] = out["z_anomaly"]

    out["anomaly_frequency"] = sample_nc_to_points(
        ds, freq_var, out["lon_centroid"].values, out["lat_centroid"].values
    )

    ds.close()

    out["pct_z"] = percentile_rank(out["z_anomaly"])
    out["pct_freq"] = percentile_rank(out["anomaly_frequency"])
    out["score_z_minus_freq"] = out["pct_z"] - out["pct_freq"]

    return out, z_var, freq_var


# ============================================================
# 4. FLAGS AND INDEPENDENT WIND REGIME TYPOLOGY
# ============================================================

def add_prediction_flags(df, q_pred=Q_HIGH_PRED_BASELINE):
    out = df.copy()
    valid = get_analysis_valid_mask(out)
    d = out.loc[valid].copy()
    weights = d["area_weight_km2"].values if USE_WEIGHTED_QUANTILE_FOR_THRESHOLDS else None

    q_mean = weighted_quantile(d["mean_prob"], q_pred, weights)
    q_p95 = weighted_quantile(d["p95_prob"], q_pred, weights)

    out["analysis_valid"] = valid
    out["high_mean"] = False
    out["high_p95"] = False
    out.loc[valid, "high_mean"] = out.loc[valid, "mean_prob"] >= q_mean
    out.loc[valid, "high_p95"] = out.loc[valid, "p95_prob"] >= q_p95

    return out, {"q_mean_prob_top20": q_mean, "q_p95_prob_top20": q_p95}


def add_frequency_intensity_regime(df, wind_q=WIND_Q_BASELINE):
    """
    Independent typology based only on frequency and intensity.
    Prediction variables are not used here.
    """
    out = df.copy()
    valid = get_analysis_valid_mask(out)
    d = out.loc[valid].copy()
    weights = d["area_weight_km2"].values if USE_WEIGHTED_QUANTILE_FOR_THRESHOLDS else None

    q_freq = weighted_quantile(d["anomaly_frequency"], wind_q, weights)
    q_intensity = weighted_quantile(d["anomaly_intensity"], wind_q, weights)

    out["high_frequency"] = False
    out["high_intensity"] = False
    out.loc[valid, "high_frequency"] = out.loc[valid, "anomaly_frequency"] >= q_freq
    out.loc[valid, "high_intensity"] = out.loc[valid, "anomaly_intensity"] >= q_intensity

    out["wind_regime"] = "Invalid / outside analysis mask"
    out.loc[valid, "wind_regime"] = "Background / mixed"

    out.loc[
        valid & out["high_frequency"] & out["high_intensity"],
        "wind_regime"
    ] = "Compound recurrent-intense"

    out.loc[
        valid & out["high_frequency"] & (~out["high_intensity"]),
        "wind_regime"
    ] = "Recurrent anomaly"

    out.loc[
        valid & (~out["high_frequency"]) & out["high_intensity"],
        "wind_regime"
    ] = "Intensity-dominated anomaly"

    thresholds = {
        "wind_q": wind_q,
        "q_frequency_high": q_freq,
        "q_intensity_high": q_intensity,
    }
    return out, thresholds


def add_all_flags(df):
    out, pred_thr = add_prediction_flags(df, q_pred=Q_HIGH_PRED_BASELINE)
    out, wind_thr = add_frequency_intensity_regime(out, wind_q=WIND_Q_BASELINE)
    thresholds = {**pred_thr, **wind_thr}

    print("\nBaseline thresholds:")
    for k, v in thresholds.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"  {k}: {v:.6g}")
        else:
            print(f"  {k}: {v}")

    valid = out["analysis_valid"] == True
    print(f"Analysis-valid cells: {valid.sum():,} / {len(out):,}")
    print(f"Analysis-valid area: {out.loc[valid, 'area_weight_km2'].sum():,.2f} km²")

    return out, thresholds


# ============================================================
# 5. QUANTIFICATION TABLES
# ============================================================

def spearman_table(df):
    pred_vars = ["mean_prob", "p95_prob"]
    anomaly_vars = ["anomaly_intensity", "anomaly_frequency", "score_z_minus_freq"]
    rows = []
    for px in pred_vars:
        for ay in anomaly_vars:
            d = df[[px, ay]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) < 10:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(d[px], d[ay])
            rows.append({
                "prediction_metric": px,
                "anomaly_metric": ay,
                "spearman_rho": rho,
                "p_value": pval,
                "n": len(d),
            })
    return pd.DataFrame(rows)


def partial_spearman_table(df):
    tests = [
        ("mean_prob", "anomaly_intensity", "anomaly_frequency"),
        ("mean_prob", "anomaly_frequency", "anomaly_intensity"),
        ("p95_prob", "anomaly_intensity", "anomaly_frequency"),
        ("p95_prob", "anomaly_frequency", "anomaly_intensity"),
    ]
    rows = []
    for x, y, control in tests:
        d = df[[x, y, control]].replace([np.inf, -np.inf], np.nan).dropna()
        n = len(d)
        if n < 20:
            rows.append({"x": x, "y": y, "control": control,
                         "partial_spearman_rho": np.nan, "p_value": np.nan, "n": n})
            continue
        rx = d[x].rank(method="average").values.reshape(-1, 1)
        ry = d[y].rank(method="average").values.reshape(-1, 1)
        rc = d[control].rank(method="average").values.reshape(-1, 1)
        model_x = LinearRegression().fit(rc, rx)
        model_y = LinearRegression().fit(rc, ry)
        res_x = rx.ravel() - model_x.predict(rc).ravel()
        res_y = ry.ravel() - model_y.predict(rc).ravel()
        rho, pval = spearmanr(res_x, res_y)
        rows.append({"x": x, "y": y, "control": control,
                     "partial_spearman_rho": rho, "p_value": pval, "n": n})
    return pd.DataFrame(rows)


def top_mask_by_weighted_quantile(d, pred_col, q_pred):
    weights = d["area_weight_km2"].values if USE_WEIGHTED_QUANTILE_FOR_THRESHOLDS else None
    q = weighted_quantile(d[pred_col], q_pred, weights)
    return d[pred_col].values >= q, q


def table_s1_overlay_enrichment_sensitivity(df):
    """
    For top 30/20/10% prediction areas, quantify enrichment in:
    high frequency, high intensity, compound high frequency-high intensity.
    """
    valid = get_analysis_valid_mask(df)
    base = df.loc[valid].copy()
    total_area = base["area_weight_km2"].sum()
    rows = []

    # Baseline wind flags are based on WIND_Q_BASELINE.
    wind_contexts = {
        "High anomaly frequency": base["high_frequency"].values,
        "High anomaly intensity": base["high_intensity"].values,
        "High frequency + high intensity": (base["high_frequency"].values & base["high_intensity"].values),
        "Low frequency + high intensity": ((~base["high_frequency"].values) & base["high_intensity"].values),
    }
    pred_metrics = {
        "Mean probability": "mean_prob",
        "P95 probability": "p95_prob",
    }

    for pred_label, pred_col in pred_metrics.items():
        for q_pred in TOP_PRED_QS_FOR_SENSITIVITY:
            top_mask, threshold = top_mask_by_weighted_quantile(base, pred_col, q_pred)
            top_area = base.loc[top_mask, "area_weight_km2"].sum()
            top_pct = 100 * (1 - q_pred)

            for context, c_mask in wind_contexts.items():
                bg_area = base.loc[c_mask, "area_weight_km2"].sum()
                top_context_area = base.loc[top_mask & c_mask, "area_weight_km2"].sum()
                bg_share = bg_area / total_area if total_area > 0 else np.nan
                top_share = top_context_area / top_area if top_area > 0 else np.nan
                rows.append({
                    "prediction_metric": pred_label,
                    "prediction_column": pred_col,
                    "top_prediction_quantile": q_pred,
                    "top_prediction_area_percent_nominal": top_pct,
                    "prediction_threshold": threshold,
                    "wind_context": context,
                    "background_area_share_percent": 100 * bg_share,
                    "top_prediction_area_share_percent": 100 * top_share,
                    "enrichment_ratio": top_share / bg_share if bg_share > 0 else np.nan,
                    "overlap_area_km2": top_context_area,
                })

    return pd.DataFrame(rows)


def table_s2_wind_regime_prediction_mass(df):
    valid = get_analysis_valid_mask(df)
    d = df.loc[valid].copy()
    total_area = d["area_weight_km2"].sum()
    total_mean_mass = weighted_sum(d["mean_prob"], d["area_weight_km2"])
    total_p95_mass = weighted_sum(d["p95_prob"], d["area_weight_km2"])

    rows = []
    for regime in REGIME_ORDER:
        g = d[d["wind_regime"] == regime].copy()
        area = g["area_weight_km2"].sum()
        area_share = area / total_area if total_area > 0 else np.nan
        mean_mass = weighted_sum(g["mean_prob"], g["area_weight_km2"])
        p95_mass = weighted_sum(g["p95_prob"], g["area_weight_km2"])
        mean_mass_share = mean_mass / total_mean_mass if total_mean_mass > 0 else np.nan
        p95_mass_share = p95_mass / total_p95_mass if total_p95_mass > 0 else np.nan
        rows.append({
            "wind_regime": regime,
            "n_cells": len(g),
            "area_km2": area,
            "area_share_percent": 100 * area_share,
            "mean_prob_area_weighted_mean": weighted_mean(g["mean_prob"], g["area_weight_km2"]),
            "p95_prob_area_weighted_mean": weighted_mean(g["p95_prob"], g["area_weight_km2"]),
            "anomaly_frequency_area_weighted_mean": weighted_mean(g["anomaly_frequency"], g["area_weight_km2"]),
            "anomaly_intensity_area_weighted_mean": weighted_mean(g["anomaly_intensity"], g["area_weight_km2"]),
            "mean_prediction_mass_share_percent": 100 * mean_mass_share,
            "mean_mass_to_area_ratio": mean_mass_share / area_share if area_share > 0 else np.nan,
            "p95_prediction_mass_share_percent": 100 * p95_mass_share,
            "p95_mass_to_area_ratio": p95_mass_share / area_share if area_share > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def table_s3_regime_enrichment_top_prediction(df, q_pred=Q_HIGH_PRED_BASELINE):
    valid = get_analysis_valid_mask(df)
    d = df.loc[valid].copy()
    total_area = d["area_weight_km2"].sum()
    pred_metrics = {"Mean probability": "mean_prob", "P95 probability": "p95_prob"}
    rows = []

    for pred_label, pred_col in pred_metrics.items():
        top_mask, threshold = top_mask_by_weighted_quantile(d, pred_col, q_pred)
        top_area = d.loc[top_mask, "area_weight_km2"].sum()
        for regime in REGIME_ORDER:
            r_mask = d["wind_regime"].values == regime
            bg_area = d.loc[r_mask, "area_weight_km2"].sum()
            top_regime_area = d.loc[top_mask & r_mask, "area_weight_km2"].sum()
            bg_share = bg_area / total_area if total_area > 0 else np.nan
            top_share = top_regime_area / top_area if top_area > 0 else np.nan
            rows.append({
                "prediction_metric": pred_label,
                "prediction_column": pred_col,
                "top_prediction_quantile": q_pred,
                "prediction_threshold": threshold,
                "wind_regime": regime,
                "background_area_share_percent": 100 * bg_share,
                "top_prediction_area_share_percent": 100 * top_share,
                "enrichment_ratio": top_share / bg_share if bg_share > 0 else np.nan,
                "top_prediction_regime_area_km2": top_regime_area,
            })
    return pd.DataFrame(rows)


def table_s4_threshold_sensitivity(df):
    rows = []
    valid = get_analysis_valid_mask(df)
    base = df.loc[valid].copy()

    for wind_q in WIND_QS_FOR_SENSITIVITY:
        tmp, thr = add_frequency_intensity_regime(base, wind_q=wind_q)
        mass = table_s2_wind_regime_prediction_mass(tmp)
        enrich = table_s3_regime_enrichment_top_prediction(tmp, q_pred=Q_HIGH_PRED_BASELINE)

        for pred_label in ["Mean probability", "P95 probability"]:
            ee = enrich[enrich["prediction_metric"] == pred_label].copy()
            if len(ee) == 0:
                best_regime = np.nan
                best_enrich = np.nan
            else:
                ee = ee.sort_values("enrichment_ratio", ascending=False)
                best_regime = ee.iloc[0]["wind_regime"]
                best_enrich = ee.iloc[0]["enrichment_ratio"]

            for _, r in mass.iterrows():
                rows.append({
                    "wind_quantile": wind_q,
                    "frequency_threshold": thr["q_frequency_high"],
                    "intensity_threshold": thr["q_intensity_high"],
                    "prediction_metric_for_best_enrichment": pred_label,
                    "best_enriched_regime": best_regime,
                    "best_enrichment_ratio": best_enrich,
                    "wind_regime": r["wind_regime"],
                    "area_share_percent": r["area_share_percent"],
                    "mean_mass_to_area_ratio": r["mean_mass_to_area_ratio"],
                    "p95_mass_to_area_ratio": r["p95_mass_to_area_ratio"],
                })

    return pd.DataFrame(rows)


# ============================================================
# 6. ADMIN AND MAP GEOMETRY HELPERS
# ============================================================

def load_admin(admin_path):
    if not admin_path.exists():
        print(f"Admin shapefile not found: {admin_path}. Maps will be drawn without boundaries.")
        return None

    admin = gpd.read_file(admin_path)
    if admin.crs is None:
        admin = admin.set_crs("EPSG:3035")

    if "CNTR_ID" not in admin.columns:
        raise KeyError("CNTR_ID not found in admin shapefile. Cannot filter Europe cleanly.")

    admin = admin[admin["CNTR_ID"].isin(EUROPE_CNTR_IDS)].copy()
    admin = admin.to_crs("EPSG:4326")
    admin = admin.explode(index_parts=False).reset_index(drop=True)

    xmin, ymin, xmax, ymax = BBOX_LONLAT
    bbox_geom = box(xmin, ymin, xmax, ymax)
    admin = admin[admin.geometry.intersects(bbox_geom)].copy()
    admin["geometry"] = admin.geometry.intersection(bbox_geom)
    admin = admin[~admin.geometry.is_empty].copy()
    admin = admin[admin.geometry.notna()].copy()

    admin_tmp = admin.to_crs(PLOT_CRS).copy()
    admin["area_km2_tmp"] = admin_tmp.geometry.area / 1e6
    admin = admin[admin["area_km2_tmp"] >= MIN_ADMIN_PART_AREA_KM2].copy()
    admin = admin.drop(columns="area_km2_tmp")

    admin = admin.to_crs(PLOT_CRS)
    print(f"Admin features kept after main-Europe filtering: {len(admin):,}")
    return admin


def filter_main_europe_hexes(gdf):
    xmin, ymin, xmax, ymax = BBOX_LONLAT
    bbox_geom = box(xmin, ymin, xmax, ymax)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf_4326 = gdf.to_crs("EPSG:4326").copy()
    gdf_4326 = gdf_4326[gdf_4326.geometry.intersects(bbox_geom)].copy()
    gdf_4326["geometry"] = gdf_4326.geometry.intersection(bbox_geom)
    gdf_4326 = gdf_4326[~gdf_4326.geometry.is_empty].copy()
    gdf_4326 = gdf_4326[gdf_4326.geometry.notna()].copy()
    return gdf_4326


def filter_hex_to_admin_footprint(gdf, admin_gdf):
    if admin_gdf is None or len(admin_gdf) == 0:
        return gdf
    gg = gdf.copy()
    gg = gg.to_crs(PLOT_CRS)
    try:
        admin_union = admin_gdf.geometry.union_all()
    except Exception:
        admin_union = admin_gdf.unary_union
    rep = gg.geometry.representative_point()
    keep = rep.within(admin_union) | rep.touches(admin_union)
    return gg.loc[keep].copy()


def prepare_gdf_for_mapping(gdf, admin_gdf=None):
    gg = filter_main_europe_hexes(gdf)
    gg = gg.to_crs(PLOT_CRS)
    gg = filter_hex_to_admin_footprint(gg, admin_gdf)
    return gg


def get_plot_bounds(hex_gdf, pad_ratio=0.035):
    xmin, ymin, xmax, ymax = hex_gdf.total_bounds
    dx = xmax - xmin
    dy = ymax - ymin
    return (
        xmin - dx * pad_ratio,
        ymin - dy * pad_ratio,
        xmax + dx * pad_ratio,
        ymax + dy * pad_ratio,
    )


def style_ax(ax, bounds):
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_admin(ax, admin_gdf):
    if admin_gdf is not None and len(admin_gdf) > 0:
        admin_gdf.boundary.plot(ax=ax, linewidth=ADMIN_LW, color=ADMIN_COLOR, zorder=8)


def add_panel_label(ax, label):
    ax.text(
        0.015, 0.975, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=2.0),
        zorder=20,
    )


def robust_limits(values, q_low=MAP_Q_LOW, q_high=MAP_Q_HIGH):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().values
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 1.0
    vmin = np.nanquantile(vals, q_low)
    vmax = np.nanquantile(vals, q_high)
    if np.isclose(vmin, vmax):
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
    return float(vmin), float(vmax)


def overlay_dissolved_boundary(ax, gdf_sub, edgecolor=TOP_OUTLINE_COLOR, linewidth=TOP_OUTLINE_LW):
    if gdf_sub is None or len(gdf_sub) == 0:
        return
    try:
        geom = gdf_sub.geometry.union_all()
    except Exception:
        geom = gdf_sub.unary_union
    dissolved = gpd.GeoDataFrame(geometry=[geom], crs=gdf_sub.crs)
    dissolved.boundary.plot(ax=ax, color=edgecolor, linewidth=linewidth, zorder=10)


def add_overlay_legend(ax, label):
    handles = [Patch(facecolor="none", edgecolor=TOP_OUTLINE_COLOR, label=label)]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        borderpad=0.55,
        handlelength=1.4,
        fontsize=8.2,
    )
    leg.get_frame().set_linewidth(0.6)


# ============================================================
# 7. FIGURES
# ============================================================

def add_horizontal_colorbar(fig, ax_left, ax_right, cmap, vmin, vmax, label, y_offset=0.045):
    pos1 = ax_left.get_position()
    pos2 = ax_right.get_position()
    left = min(pos1.x0, pos2.x0)
    right = max(pos1.x1, pos2.x1)
    bottom = min(pos1.y0, pos2.y0) - y_offset
    width = right - left
    height = 0.014
    cax = fig.add_axes([left, bottom, width, height])
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5, length=2.5)
    cb.outline.set_linewidth(0.7)


def plot_2x2_overlay(gdf_map, admin_gdf, background, outdir):
    freq_vmin, freq_vmax = robust_limits(gdf_map["anomaly_frequency"])
    int_vmin, int_vmax = robust_limits(gdf_map["anomaly_intensity"])
    bounds = get_plot_bounds(gdf_map)

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_2X2, FIG_H_2X2))

    panels = [
        (axes[0, 0], "anomaly_frequency", "YlGnBu", freq_vmin, freq_vmax, "high_mean",
         "Mean prediction × anomaly frequency", "Top 20% mean probability", "(a)"),
        (axes[0, 1], "anomaly_intensity", "YlOrRd", int_vmin, int_vmax, "high_mean",
         "Mean prediction × anomaly intensity", "Top 20% mean probability", "(b)"),
        (axes[1, 0], "anomaly_frequency", "YlGnBu", freq_vmin, freq_vmax, "high_p95",
         "P95 prediction × anomaly frequency", "Top 20% p95 probability", "(c)"),
        (axes[1, 1], "anomaly_intensity", "YlOrRd", int_vmin, int_vmax, "high_p95",
         "P95 prediction × anomaly intensity", "Top 20% p95 probability", "(d)"),
    ]

    for ax, col, cmap, vmin, vmax, high_col, title, legend_label, panel_label in panels:
        gdf_map.plot(
            ax=ax,
            column=col,
            cmap=cmap,
            linewidth=0.0,
            vmin=vmin,
            vmax=vmax,
            legend=False,
            zorder=1,
            missing_kwds={"color": "#f5f5f5"},
        )
        top = gdf_map[gdf_map[high_col] == True].copy()
        overlay_dissolved_boundary(ax, top)
        draw_admin(ax, admin_gdf)
        style_ax(ax, bounds)
        add_panel_label(ax, panel_label)
        ax.set_title(title, pad=7)
        add_overlay_legend(ax, legend_label)

    plt.subplots_adjust(left=0.025, right=0.995, top=0.935, bottom=0.095, wspace=0.035, hspace=0.090)
    add_horizontal_colorbar(fig, axes[1, 0], axes[1, 0], "YlGnBu", freq_vmin, freq_vmax, "Anomaly frequency", y_offset=0.043)
    add_horizontal_colorbar(fig, axes[1, 1], axes[1, 1], "YlOrRd", int_vmin, int_vmax, "Standardized anomaly intensity", y_offset=0.043)

    out_png = outdir / f"Fig4_2x2_prediction_vs_frequency_intensity_{background}.png"
    out_pdf = outdir / f"Fig4_2x2_prediction_vs_frequency_intensity_{background}.pdf"
    fig.savefig(out_png, dpi=FIG_DPI)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved 2x2 overlay figure:\n{out_png}")


def plot_typology_map(gdf_map, admin_gdf, background, outdir):
    bounds = get_plot_bounds(gdf_map)
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W_MAP, FIG_H_MAP))

    for regime in REGIME_ORDER[::-1]:
        sub = gdf_map[gdf_map["wind_regime"] == regime].copy()
        if len(sub) == 0:
            continue
        sub.plot(ax=ax, color=REGIME_COLORS[regime], linewidth=0.0, zorder=1)

    draw_admin(ax, admin_gdf)
    style_ax(ax, bounds)
    add_panel_label(ax, "(a)")
    ax.set_title("Frequency-intensity wind-anomaly typology", pad=8)

    handles = [Patch(facecolor=REGIME_COLORS[r], edgecolor="none", label=r) for r in REGIME_ORDER]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        borderpad=0.55,
        handlelength=1.2,
        labelspacing=0.40,
        fontsize=8.3,
        title="Wind-anomaly regime",
        title_fontsize=8.8,
    )
    leg.get_frame().set_linewidth(0.6)

    plt.tight_layout()
    out_png = outdir / f"Fig5_frequency_intensity_typology_{background}.png"
    out_pdf = outdir / f"Fig5_frequency_intensity_typology_{background}.pdf"
    fig.savefig(out_png, dpi=FIG_DPI)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved typology map:\n{out_png}")


def plot_enrichment_and_mass(table_s3, table_s2, background, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_BAR, FIG_H_BAR))
    x = np.arange(len(REGIME_ORDER))
    width = 0.36

    # Panel a: enrichment in top 20% areas
    ax = axes[0]
    vals_mean = []
    vals_p95 = []
    for regime in REGIME_ORDER:
        m = table_s3[(table_s3["prediction_metric"] == "Mean probability") & (table_s3["wind_regime"] == regime)]
        p = table_s3[(table_s3["prediction_metric"] == "P95 probability") & (table_s3["wind_regime"] == regime)]
        vals_mean.append(float(m["enrichment_ratio"].iloc[0]) if len(m) else np.nan)
        vals_p95.append(float(p["enrichment_ratio"].iloc[0]) if len(p) else np.nan)

    ax.bar(x - width / 2, vals_mean, width, label="Top 20% mean prediction")
    ax.bar(x + width / 2, vals_p95, width, label="Top 20% p95 prediction")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_ORDER, rotation=25, ha="right", fontsize=8.3)
    ax.set_ylabel("Enrichment ratio")
    ax.set_title("(a) Regime enrichment in high-prediction areas", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.3)

    # Panel b: prediction mass-to-area ratio
    ax = axes[1]
    table_s2 = table_s2.set_index("wind_regime").reindex(REGIME_ORDER).reset_index()
    ax.bar(x - width / 2, table_s2["mean_mass_to_area_ratio"], width, label="Mean prediction mass")
    ax.bar(x + width / 2, table_s2["p95_mass_to_area_ratio"], width, label="P95 prediction mass")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_ORDER, rotation=25, ha="right", fontsize=8.3)
    ax.set_ylabel("Prediction mass-to-area ratio")
    ax.set_title("(b) Prediction mass concentration by regime", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.3)

    plt.tight_layout()
    out_png = outdir / f"Fig6_regime_enrichment_and_mass_{background}.png"
    out_pdf = outdir / f"Fig6_regime_enrichment_and_mass_{background}.pdf"
    fig.savefig(out_png, dpi=FIG_DPI)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved bar summary figure:\n{out_png}")


def make_all_figures(df, background, outdir, admin_gdf):
    gdf_map = prepare_gdf_for_mapping(df, admin_gdf=admin_gdf)
    print(f"Map hexes after Europe/admin filtering: {len(gdf_map):,}")
    if len(gdf_map) == 0:
        print("No map hexes left after filtering. Skipping figures.")
        return

    table_s2 = table_s2_wind_regime_prediction_mass(df)
    table_s3 = table_s3_regime_enrichment_top_prediction(df, q_pred=Q_HIGH_PRED_BASELINE)
    plot_2x2_overlay(gdf_map, admin_gdf, background, outdir)
    plot_typology_map(gdf_map, admin_gdf, background, outdir)
    plot_enrichment_and_mass(table_s3, table_s2, background, outdir)


# ============================================================
# 8. MAIN WORKFLOW
# ============================================================

def run_background(base_gdf, background):
    """Prepare the single merged table required by the Fig. 4 plotting script.

    The original development script also wrote diagnostic tables and preliminary
    maps. Those outputs are intentionally omitted from the paper-results package
    to keep the reproduction folder minimal. All Fig. 4 source-data tables are
    generated downstream by 06E_fig4_prediction_anomaly_regime_plot.py.
    """
    print("\n" + "#" * 80)
    print(f"Running background: {background}")
    print("#" * 80)

    bg_outdir = OUTDIR / background
    ensure_outdir(bg_outdir)

    anom_nc = ANOM_NC_BY_BG[background]
    df, z_var, freq_var = add_anomaly_metrics(base_gdf, anom_nc, background)
    df, thresholds = add_all_flags(df)

    save_df = pd.DataFrame(df.drop(columns="geometry"))
    save_df["geometry_wkt"] = df.geometry.apply(lambda g: g.wkt)
    merged_csv = bg_outdir / f"merged_prediction_anomaly_{background}.csv"
    save_df.to_csv(merged_csv, index=False, encoding="utf-8-sig")

    print("\nSaved Fig. 4 preparation table:")
    print(merged_csv)
    print("Anomaly variables used:")
    print(f"  intensity: {z_var}")
    print(f"  frequency: {freq_var}")
    print("Thresholds used for high-prediction and wind-regime flags:")
    for key, value in thresholds.items():
        print(f"  {key}: {value}")

    return df


def main():
    ensure_outdir(OUTDIR)

    print("Loading prediction table...")
    base_gdf = load_prediction_table(PRED_CSV)

    results = {}
    for bg in BACKGROUNDS:
        results[bg] = run_background(base_gdf, bg)

    print("\nDone.")
    print(f"Fig. 4 preparation outputs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
