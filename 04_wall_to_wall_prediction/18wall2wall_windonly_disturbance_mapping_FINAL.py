
# -*- coding: utf-8 -*-
"""
Wall-to-wall wind-only full inference from raw wind nc + forest mask + trained artifact.
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import shapely.geometry as sg
import joblib
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from pyproj import Transformer, Geod
from scipy.stats import rankdata

COUNTRIES_SRC_DEFAULT = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
EXCLUDE_COUNTRIES = {"russia", "russian federation", "iceland"}
HEX_AREA_TARGET_M2 = 2_165_000_000.0
CMAP_RED = None


def make_lighter_cmap(name="Reds", factor=0.90):
    base = mpl.colormaps.get_cmap(name)
    x = np.linspace(0, 1, 256)
    colors = base(x)
    white = np.ones_like(colors)
    colors[:, :3] = colors[:, :3] * factor + white[:, :3] * (1 - factor)
    return mpl.colors.LinearSegmentedColormap.from_list(f"{name}_light", colors)

CMAP_RED = make_lighter_cmap("Reds", factor=0.90)


def open_dataset_safely(path):
    """Open NetCDF robustly on Windows. Try engines that do not rely on the
    netCDF4 DLL first, then fall back to netcdf4/default.
    """
    errors = []
    for engine in ["h5netcdf", "scipy", "netcdf4", None]:
        try:
            if engine is None:
                print("Opening dataset with xarray default engine...")
                return xr.open_dataset(path)
            print(f"Opening dataset with xarray engine={engine}...")
            return xr.open_dataset(path, engine=engine)
        except Exception as e:
            errors.append(f"{engine or 'default'}: {type(e).__name__}: {e}")
    raise RuntimeError("Could not open NetCDF file with any xarray engine:\n" + "\n".join(errors))


def detect_country_name_col(gdf):
    # Supports both Natural Earth and Eurostat/GISCO CNTR boundary files.
    candidates = [
        "NAME_ENGL", "CNTR_NAME", "CNTR_ID",
        "ADMIN", "admin", "NAME", "name", "NAME_LONG", "name_long",
        "SOVEREIGNT", "sovereignt", "ISO3_CODE"
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


def load_europe_admin_boundaries(countries_src, lon_min, lat_min, lon_max, lat_max):
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
        europe = gdf.copy()
    europe = europe[~europe[name_col].astype(str).str.lower().isin(EXCLUDE_COUNTRIES)].copy()

    # Do not geometrically intersect with bbox for plotting; otherwise bbox cut-lines
    # appear as fake country boundaries. Keep only countries intersecting the bbox,
    # and rely on axis limits to crop the final figure.
    clip_geom = sg.box(lon_min, lat_min, lon_max, lat_max)
    europe = europe[europe.geometry.intersects(clip_geom)].copy()
    europe = europe.explode(index_parts=False).reset_index(drop=True)
    if europe.empty:
        raise ValueError("No European administrative polygons found after filtering/clipping.")
    return europe


def parse_bbox4326_str(bbox_str):
    vals = [float(x) for x in str(bbox_str).split(",")]
    if len(vals) != 4:
        raise ValueError(f"Invalid bbox string: {bbox_str}")
    return vals[0], vals[1], vals[2], vals[3]


def normalize_lon_to_180(lon):
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def infer_cell_edges(coord_1d):
    coord_1d = np.asarray(coord_1d, dtype=float)
    if coord_1d.ndim != 1 or coord_1d.size < 2:
        raise ValueError("Coordinate array must be 1D with at least 2 values.")
    mids = (coord_1d[:-1] + coord_1d[1:]) / 2.0
    first = coord_1d[0] - (mids[0] - coord_1d[0])
    last = coord_1d[-1] + (coord_1d[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])

def compute_lonlat_cell_area_km2(lon_1d, lat_1d):
    """
    Approximate geodesic area for each lon-lat grid cell in km^2.
    Returns array shape (lat, lon).
    """
    lon = normalize_lon_to_180(np.asarray(lon_1d, dtype=float))
    lat = np.asarray(lat_1d, dtype=float)
    lon_edges = infer_cell_edges(lon)
    lat_edges = infer_cell_edges(lat)
    geod = Geod(ellps="WGS84")
    area = np.zeros((lat.size, lon.size), dtype=np.float64)
    for i in range(lat.size):
        lat0 = lat_edges[i]
        lat1 = lat_edges[i + 1]
        for j in range(lon.size):
            lon0 = lon_edges[j]
            lon1 = lon_edges[j + 1]
            poly_lon = [lon0, lon1, lon1, lon0, lon0]
            poly_lat = [lat0, lat0, lat1, lat1, lat0]
            a, _ = geod.polygon_area_perimeter(poly_lon, poly_lat)
            area[i, j] = abs(a) / 1e6
    return area

def maybe_fix_dataset_lon(ds, lon_name):
    lon_fixed = normalize_lon_to_180(ds[lon_name].values)
    ds = ds.assign_coords({lon_name: lon_fixed}).sortby(lon_name)
    return ds


def hex_side_from_area(a):
    return float(np.sqrt(2 * a / (3 * np.sqrt(3))))


def build_hex_grid(bbox4326, hex_area_km2):
    side = hex_side_from_area(max(1.0, hex_area_km2) * 1e6)
    bb = parse_bbox4326_str(bbox4326)
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    xx, yy = tf.transform([bb[0], bb[2], bb[0], bb[2]], [bb[1], bb[1], bb[3], bb[3]])
    xmin, xmax, ymin, ymax = min(xx), max(xx), min(yy), max(yy)
    cx0, cy0 = (xmin + xmax) / 2, (ymin + ymax) / 2
    dq = int(np.ceil((xmax - xmin) / (1.5 * side))) + 4
    dr = int(np.ceil((ymax - ymin) / (np.sqrt(3) * side))) + 4
    hexes, ids, k = [], [], 0
    ang = np.deg2rad([0, 60, 120, 180, 240, 300])
    for q in range(-dq, dq + 1):
        for r in range(-dr, dr + 1):
            cx = cx0 + side * (1.5 * q)
            cy = cy0 + side * (np.sqrt(3) * (r + q / 2))
            if (xmin - 2 * side) <= cx <= (xmax + 2 * side) and (ymin - 2 * side) <= cy <= (ymax + 2 * side):
                hexes.append(sg.Polygon(np.c_[cx + side * np.cos(ang), cy + side * np.sin(ang)]))
                ids.append(k)
                k += 1
    return gpd.GeoDataFrame({"hex_id": ids}, geometry=hexes, crs=3035)


def sjoin_within(left, right):
    try:
        return gpd.sjoin(left, right, how="inner", predicate="within")
    except TypeError:
        return gpd.sjoin(left, right, how="inner", op="within")


def build_norm(vals, mode="power", gamma=0.45, qhi=99.0):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    hi = np.nanpercentile(vals, qhi)
    lo = np.nanmin(vals)
    if not np.isfinite(hi):
        hi = np.nanmax(vals)
    if not np.isfinite(lo):
        lo = 0.0
    if hi <= lo:
        hi = lo + 1e-6
    if mode == "linear":
        return mpl.colors.Normalize(vmin=lo, vmax=hi)
    if mode == "power":
        return mpl.colors.PowerNorm(gamma=gamma, vmin=lo, vmax=hi)
    def forward(x):
        x = np.asarray(x, dtype=float)
        x = np.clip(x, lo, hi)
        return np.log1p(x - lo) / np.log1p(hi - lo)
    def inverse(y):
        y = np.asarray(y, dtype=float)
        y = np.clip(y, 0.0, 1.0)
        return np.expm1(y * np.log1p(hi - lo)) + lo
    return mpl.colors.FuncNorm((forward, inverse), vmin=lo, vmax=hi)


def plot_hex_map_style(hex_gdf, value_col, title, out_png, europe_admin, lon_min, lon_max, lat_min, lat_max,
                       norm_mode="power", gamma=0.45, qhi=99.0, cbar_label=None, forest_mask_col="mask_forest",
                       noevent_mask_col=None):
    plot_gdf = hex_gdf.to_crs(4326)
    vals = plot_gdf[value_col].to_numpy(dtype=float)
    data = plot_gdf[np.isfinite(vals)].copy()
    if data.empty:
        print(f"[skip] no finite values for {value_col}")
        return
    norm = build_norm(data[value_col].to_numpy(dtype=float), mode=norm_mode, gamma=gamma, qhi=qhi)
    forest_mask = plot_gdf[forest_mask_col].astype("boolean").fillna(False).to_numpy()
    bg = plot_gdf.loc[forest_mask].copy()
    if noevent_mask_col is not None and noevent_mask_col in plot_gdf.columns:
        noevent_mask = plot_gdf[noevent_mask_col].astype("boolean").fillna(False).to_numpy()
        bg_noevent = plot_gdf.loc[noevent_mask].copy()
    else:
        bg_noevent = plot_gdf.iloc[0:0].copy()

    fig, ax = plt.subplots(figsize=(10, 9.5), facecolor="white")
    ax.set_facecolor("#d9d9d9")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # Correct the visual flattening that occurs when plotting lon/lat directly in EPSG:4326.
    # This keeps geographic axes while approximating metric shape preservation around Europe.
    mean_lat = (lat_min + lat_max) / 2.0
    ax.set_aspect(1 / np.cos(np.deg2rad(mean_lat)))

    if not bg.empty:
        bg.plot(ax=ax, color=(1,1,1,0.12), edgecolor=(1,1,1,0.0), linewidth=0.0, zorder=1.0)
    if not bg_noevent.empty:
        bg_noevent.plot(ax=ax, color="#c9ced3", edgecolor="#c9ced3", linewidth=0.08, zorder=1.6)
    data.plot(ax=ax, column=value_col, cmap=CMAP_RED, norm=norm, linewidth=0.10,
              edgecolor=(1,1,1,0.18), alpha=0.98, zorder=2.5)
    if europe_admin is not None:
        europe_admin.boundary.plot(ax=ax, color="black", linewidth=0.35, zorder=3.0)

    sm = mpl.cm.ScalarMappable(cmap=CMAP_RED, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.80, pad=0.02)
    cbar.set_label(cbar_label or title, fontsize=11)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved image: saved:", out_png)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wind_nc", required=True)
    ap.add_argument("--forest_mask_tif", required=True)
    ap.add_argument("--model_joblib", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--wind_var", default="max_wind_speed")
    ap.add_argument("--bbox4326", default="-11.5,34.0,42.5,72.5")
    ap.add_argument("--hex_bbox4326", default="-11,34,45,72")
    ap.add_argument("--hex_area_km2", type=float, default=2165.0)
    ap.add_argument("--forest_frac_min", type=float, default=0.10)
    ap.add_argument("--pred_quantile", type=float, default=0.95)
    ap.add_argument("--hex_active_cover_min", type=float, default=0.005, help="Deprecated in HEXMEAN version; retained for command compatibility and ignored.")
    ap.add_argument("--valid_periods_min_per_hex", type=int, default=10)
    ap.add_argument("--countries_src", default=COUNTRIES_SRC_DEFAULT)
    ap.add_argument("--norm", choices=["linear", "power", "log1p"], default="power")
    ap.add_argument("--gamma", type=float, default=0.45)
    ap.add_argument("--qhi", type=float, default=99.0)
    # optional severity
    ap.add_argument("--lai_nc", default="", help="Optional wall-to-wall LAI departure nc matching wind grid/time")
    ap.add_argument("--lai_var", default="LAI_departure")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading model artifact:", args.model_joblib)
    artifact = joblib.load(args.model_joblib)
    feature_groups = artifact.get("feature_groups", {})
    base_models = artifact.get("base_models", {})
    if "wind" not in base_models:
        raise KeyError("Artifact has no wind base model.")
    entry = base_models["wind"]
    if isinstance(entry, dict):
        wind_model = entry.get("model")
        wind_cols = entry.get("cols", feature_groups.get("wind", ["gust_peak_speed", "gust_peak_percentile"]))
    else:
        wind_model = entry
        wind_cols = feature_groups.get("wind", ["gust_peak_speed", "gust_peak_percentile"])
    print("Wind feature columns in artifact:", wind_cols)
    if set(wind_cols) != {"gust_peak_speed", "gust_peak_percentile"}:
        print("Warning: Wind feature set differs from expected 2-feature setup. Script will still try to proceed.")

    print("Opening wind nc:", args.wind_nc)
    ds = open_dataset_safely(args.wind_nc)
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    if lon_name is None or lat_name is None:
        raise KeyError("Cannot find lon/lat coords in wind nc.")
    if "time" not in ds.coords:
        raise KeyError("wind nc must contain time coord.")
    if args.wind_var not in ds.data_vars:
        raise KeyError(f"wind var '{args.wind_var}' not found.")

    ds = maybe_fix_dataset_lon(ds, lon_name)
    lon_min, lat_min, lon_max, lat_max = parse_bbox4326_str(args.bbox4326)
    lat_desc = np.all(np.diff(ds[lat_name].values) < 0)
    ds = ds.sel({
        lon_name: slice(lon_min, lon_max),
        lat_name: slice(lat_max, lat_min) if lat_desc else slice(lat_min, lat_max),
    })
    wind = ds[args.wind_var].astype("float32")
    lons = ds[lon_name].values.astype(float)
    lats = ds[lat_name].values.astype(float)
    times = pd.to_datetime(ds["time"].values)
    nt, ny, nx = wind.shape
    print(f"Wind grid after crop: time={nt}, lat={ny}, lon={nx}")

    # reproject forest mask to wind grid as forest fraction
    print("Reprojecting forest mask to wind grid:", args.forest_mask_tif)
    with rasterio.open(args.forest_mask_tif) as src:
        src_arr = src.read(1).astype("float32")
        src_transform = src.transform
        src_crs = src.crs
        dx = float(np.median(np.diff(lons))) if len(lons) > 1 else 0.1
        dy = float(np.median(np.diff(np.sort(lats)))) if len(lats) > 1 else 0.1
        west = float(lons.min() - dx / 2.0)
        east = float(lons.max() + dx / 2.0)
        south = float(lats.min() - dy / 2.0)
        north = float(lats.max() + dy / 2.0)
        dst_transform = from_bounds(west, south, east, north, nx, ny)
        dst = np.zeros((ny, nx), dtype="float32")
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.average,
        )
    forest_frac = np.clip(dst, 0, 1)
    forest_mask = np.isfinite(forest_frac) & (forest_frac >= args.forest_frac_min)
    print("Forest cells kept:", int(forest_mask.sum()), "out of", int(forest_mask.size))
    if forest_mask.sum() == 0:
        raise RuntimeError("No forest cells survived forest_frac_min. Check mask/grid alignment.")

    # optional lai nc
    lai_flat = None
    if args.lai_nc:
        print("Opening optional LAI departure nc:", args.lai_nc)
        lai_ds = open_dataset_safely(args.lai_nc)
        if args.lai_var not in lai_ds.data_vars:
            raise KeyError(f"LAI var '{args.lai_var}' not found in lai nc.")
        lai_lon = "longitude" if "longitude" in lai_ds.coords else ("lon" if "lon" in lai_ds.coords else None)
        lai_lat = "latitude" if "latitude" in lai_ds.coords else ("lat" if "lat" in lai_ds.coords else None)
        if lai_lon is None or lai_lat is None or "time" not in lai_ds.coords:
            raise KeyError("LAI nc must have lon/lat/time coordinates.")
        lai_ds = maybe_fix_dataset_lon(lai_ds, lai_lon)
        lat_desc2 = np.all(np.diff(lai_ds[lai_lat].values) < 0)
        lai_ds = lai_ds.sel({
            lai_lon: slice(lon_min, lon_max),
            lai_lat: slice(lat_max, lat_min) if lat_desc2 else slice(lat_min, lat_max),
        })
        lai = lai_ds[args.lai_var]
        if lai.shape != wind.shape:
            raise ValueError("Optional LAI nc must match wind nc shape after crop.")
        lai_flat = lai.values[:, forest_mask].astype("float32")
    else:
        print("Info: No wall-to-wall LAI departure nc provided; severity map will be skipped.")

    # flatten forest cells and assign to hex once
    forest_rows, forest_cols = np.where(forest_mask)
    forest_lons = lons[forest_cols]
    forest_lats = lats[forest_rows]
    cell_area_km2_grid = compute_lonlat_cell_area_km2(ds[lon_name].values, ds[lat_name].values).astype("float32")
    forest_weights = forest_frac[forest_mask].astype("float32")
    forest_area_cell_km2 = (cell_area_km2_grid[forest_mask] * forest_weights).astype("float32")

    pts = gpd.GeoDataFrame(
        {
            "grid_row": forest_rows.astype(int),
            "grid_col": forest_cols.astype(int),
            "forest_frac": forest_weights.astype(float),
            "forest_area_cell_km2": forest_area_cell_km2.astype(float),
            "lon": forest_lons.astype(float),
            "lat": forest_lats.astype(float),
        },
        geometry=gpd.points_from_xy(forest_lons, forest_lats),
        crs=4326,
    ).to_crs(3035)

    hexgrid = build_hex_grid(args.hex_bbox4326, args.hex_area_km2)
    join = sjoin_within(pts[["grid_row", "grid_col", "forest_frac", "forest_area_cell_km2", "geometry"]], hexgrid[["hex_id", "geometry"]])
    join = join[["grid_row", "grid_col", "forest_frac", "forest_area_cell_km2", "hex_id"]].drop_duplicates().copy()
    pts = pts.merge(join, on=["grid_row", "grid_col", "forest_frac", "forest_area_cell_km2"], how="inner")
    print("Forest cells inside hex domain:", len(pts))
    if len(pts) == 0:
        raise RuntimeError("No forest cells joined to hex grid.")

    # compressed cell arrays aligned to pts order
    # map each pts row to column index in flattened forest arrays
    forest_index_map = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(forest_rows.tolist(), forest_cols.tolist()))}
    cell_idx = np.array([forest_index_map[(int(r), int(c))] for r, c in zip(pts["grid_row"], pts["grid_col"])], dtype=int)

    wind_flat_all = wind.values[:, forest_mask].astype("float32")
    wind_flat = wind_flat_all[:, cell_idx]
    weight_cell = pts["forest_area_cell_km2"].to_numpy(dtype=float)
    hex_ids = pts["hex_id"].to_numpy(dtype=int)

    # percentile rank per cell across time
    print("Computing within-cell wind percentiles across time...")
    pct = rankdata(wind_flat, axis=0, method='average') / float(nt) * 100.0
    pct = pct.astype("float32")

    # score wind-only model for all cell-period pairs
    print("Scoring wall-to-wall pred_wind on forest cells...")
    n_pairs = nt * wind_flat.shape[1]
    pred_flat = np.empty(n_pairs, dtype="float32")
    chunk = 500_000
    gust = wind_flat.reshape(-1)
    gpct = pct.reshape(-1)
    # Build features in the exact order expected by wind model
    X_all = {
        "gust_peak_speed": gust,
        "gust_peak_percentile": gpct,
    }
    cols = wind_cols
    for start in range(0, n_pairs, chunk):
        end = min(start + chunk, n_pairs)
        X = pd.DataFrame({c: X_all[c][start:end] for c in cols})
        X = X.fillna(X.median(numeric_only=True))
        pred_flat[start:end] = wind_model.predict_proba(X)[:, 1].astype("float32")
    pred = pred_flat.reshape(nt, wind_flat.shape[1])

    # ------------------------------------------------------------------
    unique_hex, hex_code = np.unique(hex_ids, return_inverse=True)
    denom = np.bincount(hex_code, weights=weight_cell, minlength=len(unique_hex)).astype(float)
    T = nt
    mean_pred_arr = np.full((T, len(unique_hex)), np.nan, dtype="float32")
    severity_arr = np.full((T, len(unique_hex)), np.nan, dtype="float32") if lai_flat is not None else None

    for t in range(T):
        pred_num = np.bincount(hex_code, weights=weight_cell * pred[t], minlength=len(unique_hex)).astype(float)
        mean_pred = np.divide(pred_num, denom, out=np.full_like(pred_num, np.nan), where=denom > 0)
        mean_pred_arr[t] = mean_pred.astype("float32")

        if severity_arr is not None:
            sev = np.clip(-lai_flat[t], 0, None)
            sev_num = np.bincount(hex_code, weights=weight_cell * sev, minlength=len(unique_hex)).astype(float)
            sev_mean = np.divide(sev_num, denom, out=np.full_like(sev_num, np.nan), where=denom > 0)
            severity_arr[t] = sev_mean.astype("float32")

    # Final map:
    mean_pred_all = np.nanmean(mean_pred_arr, axis=0)
    pred_p95 = np.nanquantile(mean_pred_arr, 0.95, axis=0)

    pred_thr = float(np.quantile(mean_pred_arr[np.isfinite(mean_pred_arr)], args.pred_quantile))
    print(f"Global HEX-PERIOD mean pred_wind active threshold Q{int(args.pred_quantile*100)} = {pred_thr:.6f}")
    active_hex_period = mean_pred_arr >= pred_thr
    valid_periods = np.repeat(denom[None, :] > 0, T, axis=0)
    active_share = np.divide(
        active_hex_period.sum(axis=0),
        valid_periods.sum(axis=0),
        out=np.full(len(unique_hex), np.nan),
        where=valid_periods.sum(axis=0) > 0,
    )
    if severity_arr is not None:
        severity = np.array([
            np.nanmean(severity_arr[:, i][active_hex_period[:, i]]) if active_hex_period[:, i].any() else np.nan
            for i in range(len(unique_hex))
        ], dtype=float)
    else:
        severity = np.full(len(unique_hex), np.nan, dtype=float)

    hex_df = pd.DataFrame({
        "hex_id": unique_hex,
        "forest_area_km2": denom,
        "mean_pred_disturbance": mean_pred_all,
        "pred_p95": pred_p95,
        "active_share_q": active_share,
        "severity_cond": severity,
        "valid_periods": valid_periods.sum(axis=0).astype(int),
        "active_periods": active_hex_period.sum(axis=0).astype(int),
    })
    hex_df["mask_forest"] = hex_df["forest_area_km2"] > 0
    hex_df["mask_valid_periods"] = hex_df["valid_periods"] >= args.valid_periods_min_per_hex
    hex_df.loc[~hex_df["mask_valid_periods"], ["mean_pred_disturbance", "pred_p95", "active_share_q", "severity_cond"]] = np.nan
    hex_df["mask_sev_zero"] = hex_df["mask_forest"] & hex_df["severity_cond"].isna()

    hexgrid = hexgrid.merge(hex_df, on="hex_id", how="left")
    outcsv = os.path.join(args.outdir, "hex_distribution_wall2wall_windonly_final.csv")
    hex_df.to_csv(outcsv, index=False)
    print("saved: saved:", outcsv)

    # also save period summaries if useful
    hp_out = []
    for t, dt in enumerate(times):
        tmp = pd.DataFrame({
            "hex_id": unique_hex,
            "obs_date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
            "mean_pred_wind": mean_pred_arr[t],
            "active_period": active_hex_period[t].astype(int),
        })
        hp_out.append(tmp)
    pd.concat(hp_out, ignore_index=True).to_csv(os.path.join(args.outdir, "hex_period_summary_wall2wall_windonly_final.csv"), index=False)

    # plot
    europe_admin = load_europe_admin_boundaries(args.countries_src, *parse_bbox4326_str(args.hex_bbox4326))
    lon_min2, lat_min2, lon_max2, lat_max2 = parse_bbox4326_str(args.hex_bbox4326)
    plot_hex_map_style(
        hexgrid, "mean_pred_disturbance",
        f"Mean predicted wind-disturbance probability (wind-only model, {args.start_year if hasattr(args, 'start_year') else ''}2003–2023)",
        os.path.join(args.outdir, "map_mean_predicted_disturbance_windonly.png"),
        europe_admin, lon_min2, lon_max2, lat_min2, lat_max2,
        norm_mode=args.norm, gamma=args.gamma, qhi=args.qhi,
        cbar_label="Mean predicted disturbance probability",
        forest_mask_col="mask_forest",
    )
    plot_hex_map_style(
        hexgrid, "pred_p95",
        "95th percentile of predicted wind-disturbance probability (wind-only model, 2003–2023)",
        os.path.join(args.outdir, "map_p95_predicted_disturbance_windonly.png"),
        europe_admin, lon_min2, lon_max2, lat_min2, lat_max2,
        norm_mode=args.norm, gamma=args.gamma, qhi=args.qhi,
        cbar_label="95th percentile predicted probability",
        forest_mask_col="mask_forest",
    )
    if args.lai_nc:
        plot_hex_map_style(
            hexgrid, "severity_cond",
            f"Conditional severity of predicted wind disturbance (active periods above global Q{int(args.pred_quantile*100)}, 2003–2023)",
            os.path.join(args.outdir, "map_conditional_severity_windonly.png"),
            europe_admin, lon_min2, lon_max2, lat_min2, lat_max2,
            norm_mode=args.norm, gamma=args.gamma, qhi=args.qhi,
            cbar_label="Conditional severity",
            forest_mask_col="mask_forest",
            noevent_mask_col="mask_sev_zero",
        )

    print("Final main map = mean_pred_disturbance")
    # save compressed score cube optionally summarized only
    ds_out = xr.Dataset(
        {
            "forest_fraction": (("lat", "lon"), forest_frac.astype("float32")),
        },
        coords={"lat": lats, "lon": lons}
    )
    ds_out.to_netcdf(os.path.join(args.outdir, "forest_fraction_on_wind_grid.nc"))

    print("Hexes with valid periods:", int(np.isfinite(hex_df['mean_pred_disturbance']).sum()))
    print("Done: Wall-to-wall wind-only pipeline finished.")


if __name__ == "__main__":
    main()
