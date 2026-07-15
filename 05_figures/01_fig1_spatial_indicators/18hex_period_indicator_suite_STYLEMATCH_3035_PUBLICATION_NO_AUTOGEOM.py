# -*- coding: utf-8 -*-
"""
Hex-period indicator suite with publication-style EPSG:3035 maps.

This version keeps the original indicator calculations and the original red
color palette, but replaces the old lon/lat plotting layout with the same
publication map style used in the spatial-overlap and temporal-overlap figures:

  - main-Europe bbox cleaning in lon/lat
  - final plotting in EPSG:3035
  - equal aspect with adjustable='box'
  - no projected coordinate labels
  - filtered European administrative footprint
  - optional individual maps and one combined 2x2 figure

Inputs are identical to the previous script:
  --hex_distribution_csv
  --hex_period_csv
  --outdir
  --countries_src
"""

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl

from shapely import wkt
from shapely.geometry import box
from pyproj import Transformer
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 0. Style / constants
# ============================================================

CMAP_RED = LinearSegmentedColormap.from_list(
    "soft_red",
    ["#f7f2ee", "#f3d3c7", "#f29b7b", "#ef5a47", "#d92523", "#8f1020"]
)

# Main Europe extent in lon/lat, matched to the spatial-overlap figures.
BBOX_LONLAT_DEFAULT = "-11,34,45,72"
PLOT_CRS = "EPSG:3035"

FIG_DPI = 500
ADMIN_LW = 0.28
ADMIN_COLOR = "0.35"
MIN_ADMIN_PART_AREA_KM2 = 20

# Keep only main European countries; Russia and Iceland are excluded.
EUROPE_CNTR_IDS = {
    "AL", "AD", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ", "DE", "DK",
    "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LI", "LT", "LU",
    "LV", "MD", "ME", "MK", "MT", "NL", "NO", "PL", "PT", "RO", "RS", "SE",
    "SI", "SK", "SM", "UA", "VA", "XK", "UK", "GB"
}

EXCLUDE_COUNTRIES = {"russia", "russian federation", "iceland"}


# ============================================================
# 1. General helpers
# ============================================================

def parse_bbox4326(s):
    vals = [float(x) for x in str(s).split(",")]
    if len(vals) != 4:
        raise ValueError("bbox must be minlon,minlat,maxlon,maxlat")
    return vals[0], vals[1], vals[2], vals[3]


def first_existing(cols, candidates):
    cols_set = set(cols)
    for c in candidates:
        if c in cols_set:
            return c
    return None


def infer_pred_col(df):
    candidates = [
        "hex_period_mean_pred", "mean_pred_wind", "mean_pred_disturbance",
        "mean_pred", "pred_prob_mean", "pred_wind", "pred", "score"
    ]
    c = first_existing(df.columns, candidates)
    if c is not None:
        return c
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["pred", "score", "prob"]) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        cl = c.lower()
        if pd.api.types.is_numeric_dtype(df[c]) and not any(
            k in cl for k in ["id", "year", "month", "day", "period", "season", "hex"]
        ):
            return c
    raise KeyError("Could not infer prediction column from hex_period_summary CSV.")


def build_norm(values, mode="power", gamma=0.60, qhi=99.0):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    vmin = np.nanmin(arr)
    vmax = np.nanpercentile(arr, qhi)
    if not np.isfinite(vmax):
        vmax = np.nanmax(arr)
    if not np.isfinite(vmin):
        vmin = 0.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9
    if mode == "power":
        return mpl.colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)


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


# ============================================================
# 2. Hex grid / input loading
# ============================================================

def hex_side_from_area(area_m2):
    return float(np.sqrt(2.0 * area_m2 / (3.0 * np.sqrt(3.0))))


def sg_hex_wkt(cx, cy, side, ang):
    pts = np.c_[cx + side * np.cos(ang), cy + side * np.sin(ang)]
    pts = list(map(tuple, pts))
    pts.append(pts[0])
    coords = ", ".join(f"{x} {y}" for x, y in pts)
    return f"POLYGON (({coords}))"


def build_hex_grid_from_bbox(bbox4326, hex_area_km2):
    lon_min, lat_min, lon_max, lat_max = parse_bbox4326(bbox4326)
    side = hex_side_from_area(max(1.0, hex_area_km2) * 1e6)
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    xx, yy = tf.transform(
        [lon_min, lon_max, lon_min, lon_max],
        [lat_min, lat_min, lat_max, lat_max]
    )
    xmin, xmax, ymin, ymax = min(xx), max(xx), min(yy), max(yy)
    cx0, cy0 = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    dq = int(np.ceil((xmax - xmin) / (1.5 * side))) + 4
    dr = int(np.ceil((ymax - ymin) / (np.sqrt(3) * side))) + 4
    hexes, ids = [], []
    ang = np.deg2rad([0, 60, 120, 180, 240, 300])
    k = 0
    for q in range(-dq, dq + 1):
        for r in range(-dr, dr + 1):
            cx = cx0 + side * (1.5 * q)
            cy = cy0 + side * (np.sqrt(3) * (r + q / 2.0))
            if (xmin - 2 * side) <= cx <= (xmax + 2 * side) and (ymin - 2 * side) <= cy <= (ymax + 2 * side):
                poly = wkt.loads(sg_hex_wkt(cx, cy, side, ang))
                hexes.append(poly)
                ids.append(k)
                k += 1
    return gpd.GeoDataFrame({"hex_id": ids}, geometry=hexes, crs="EPSG:3035")


def detect_geometry_column(df):
    return first_existing(df.columns, [
        "geometry", "geom", "wkt", "geom_wkt", "geometry_wkt", "hex_wkt"
    ])


def load_geometry_table(geometry_path, geometry_crs="EPSG:3035"):
    geometry_path = str(geometry_path)
    if geometry_path.lower().endswith('.gpkg'):
        gdf = gpd.read_file(geometry_path)
        id_col = first_existing(gdf.columns, ["hex_id", "hexid", "grid_id", "cell_id", "id"])
        if id_col is None:
            raise KeyError("Cannot find hex ID column in geometry file.")
        if id_col != 'hex_id':
            gdf = gdf.rename(columns={id_col: 'hex_id'})
        if gdf.crs is None:
            gdf = gdf.set_crs(geometry_crs)
        return gdf[["hex_id", "geometry"]].copy()

    df = pd.read_csv(geometry_path, low_memory=False)
    id_col = first_existing(df.columns, ["hex_id", "hexid", "grid_id", "cell_id", "id"])
    if id_col is None:
        raise KeyError("Cannot find hex ID column in geometry CSV.")
    geom_col = detect_geometry_column(df)
    if geom_col is None:
        raise KeyError("Cannot find geometry/WKT column in geometry CSV.")

    gg = df[[id_col, geom_col]].copy()
    gg = gg.rename(columns={id_col: 'hex_id', geom_col: 'geometry'})
    gg = gg[gg['geometry'].notna()].copy()
    gg['geometry'] = gg['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(gg, geometry='geometry', crs=geometry_crs)
    return gdf[["hex_id", "geometry"]].copy()


def guess_geometry_source(hex_distribution_csv, outdir=None):
    candidates = []
    base_dir = Path(hex_distribution_csv).resolve().parent
    candidates.extend([
        base_dir / 'hex_indicator_summary_stylematch_publication.gpkg',
        base_dir / 'hex_indicator_summary_stylematch_4326.csv',
    ])
    if outdir is not None:
        out_dir = Path(outdir).resolve()
        candidates.extend([
            out_dir / 'hex_indicator_summary_stylematch_publication.gpkg',
            out_dir / 'hex_indicator_summary_stylematch_4326.csv',
        ])
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def load_hex_distribution(csv_path, bbox4326, hex_area_km2, forest_frac_min=0.0,
                          geometry_path=None, geometry_crs="EPSG:3035", outdir=None):
    df = pd.read_csv(csv_path, low_memory=False)
    id_col = first_existing(df.columns, ["hex_id", "hexid", "grid_id", "cell_id", "id"])
    if id_col is None:
        raise KeyError("Cannot find hex ID column in hex_distribution CSV.")

    forest_area_col = first_existing(df.columns, ["forest_area_km2", "forest_area_cell_km2", "forest_km2", "forest_area"])
    forest_frac_col = first_existing(df.columns, ["forest_frac", "forest_fraction", "forest_cover_frac", "forest_cover"])

    mask = np.ones(len(df), dtype=bool)
    if forest_area_col is not None:
        fa = pd.to_numeric(df[forest_area_col], errors="coerce").fillna(0.0)
        mask &= (fa > 0)
    if forest_frac_col is not None:
        ff = pd.to_numeric(df[forest_frac_col], errors="coerce").fillna(0.0)
        mask &= (ff >= forest_frac_min)

    df = df.copy()
    df["mask_forest"] = mask

    if id_col != "hex_id":
        df = df.rename(columns={id_col: "hex_id"})
        id_col = "hex_id"

    # Priority 1: explicit external geometry file.
    # Priority 2: geometry column already present in hex_distribution CSV.
    # Priority 3: auto-detect a companion geometry file.
    # Priority 4: fallback to rebuilding the grid from bbox (requires original bbox to match).
    geom_col = detect_geometry_column(df)
    geometry_source = None

    if geometry_path:
        print(f"[INFO] Using external geometry source: {geometry_path}")
        hexgrid = load_geometry_table(geometry_path, geometry_crs=geometry_crs)
        geometry_source = geometry_path
    elif geom_col is not None:
        print(f"[INFO] Using geometry column from hex_distribution CSV: {geom_col}")
        gg = df[["hex_id", geom_col]].copy()
        gg = gg.rename(columns={geom_col: "geometry"})
        gg = gg[gg["geometry"].notna()].copy()
        gg["geometry"] = gg["geometry"].apply(wkt.loads)
        hexgrid = gpd.GeoDataFrame(gg, geometry="geometry", crs=geometry_crs)
        geometry_source = f"{csv_path}:{geom_col}"
    else:
        print("[INFO] No explicit geometry source provided. Rebuilding hex grid from --hex_bbox4326. "
              "This is correct only when --hex_bbox4326 exactly matches the original wall-to-wall run.")
        hexgrid = build_hex_grid_from_bbox(bbox4326, hex_area_km2)
        geometry_source = "rebuild_from_bbox"

    hexgrid = hexgrid.drop_duplicates(subset=["hex_id"]).copy()
    gdf = hexgrid.merge(df.drop(columns=[c for c in [geom_col] if c is not None], errors='ignore'), on="hex_id", how="left")
    if "mask_forest" not in gdf.columns:
        gdf["mask_forest"] = False
    gdf["mask_forest"] = gdf["mask_forest"].fillna(False)
    gdf.attrs["geometry_source"] = geometry_source
    return gdf, id_col


def load_hex_period_summary(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    id_col = first_existing(df.columns, ["hex_id", "hexid", "grid_id", "cell_id", "id"])
    if id_col is None:
        raise KeyError("Cannot find hex ID column in hex_period_summary CSV.")
    pred_col = infer_pred_col(df)
    return df, id_col, pred_col


def compute_indicator_suite(period_df, id_col, pred_col, active_quantile=0.95):
    date_col = first_existing(period_df.columns, ["obs_date", "date", "event_date", "time"])
    use_cols = [id_col, pred_col] + ([date_col] if date_col else [])
    df = period_df[use_cols].copy()
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df = df[np.isfinite(df[pred_col])].copy()
    if df.empty:
        raise ValueError("No valid prediction values found in period summary table.")

    if date_col:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        df["_month"] = dt.dt.month
        df["_year"] = dt.dt.year
        df["_is_winter"] = df["_month"].isin([12, 1, 2])
    else:
        df["_year"] = np.nan
        df["_is_winter"] = False

    thr = df[pred_col].quantile(active_quantile)
    grp = df.groupby(id_col)[pred_col]

    out = pd.DataFrame({
        id_col: grp.mean().index,
        "n_periods": grp.size().values,
        "mean_pred_prob": grp.mean().values,
        "p95_pred_prob": grp.quantile(0.95).values,
        "p99_pred_prob": grp.quantile(0.99).values,
        "max_pred_prob": grp.max().values,
    })

    act_df = df[df[pred_col] >= thr].copy()
    grp_act = act_df.groupby(id_col)[pred_col]

    active_n = grp_act.size().reindex(out[id_col], fill_value=0)
    conditional_mean = grp_act.mean().reindex(out[id_col], fill_value=0.0)
    conditional_p95 = grp_act.quantile(0.95).reindex(out[id_col], fill_value=0.0)

    out["active_n"] = active_n.to_numpy()
    out["recurrence_index"] = out["active_n"] / out["n_periods"].clip(lower=1)
    out["conditional_intensity"] = conditional_mean.to_numpy()
    out["conditional_p95"] = conditional_p95.to_numpy()
    out["active_threshold_used"] = thr

    if date_col:
        active_year_prop = (
            act_df.groupby([id_col, "_year"]).size().reset_index()
            .groupby(id_col)["_year"].nunique()
            .reindex(out[id_col], fill_value=0)
        )
        total_years = (
            df.groupby(id_col)["_year"].nunique()
            .reindex(out[id_col], fill_value=0)
            .replace(0, np.nan)
        )
        out["active_year_prop_q95"] = (active_year_prop / total_years).fillna(0.0).to_numpy()
        winter_share = act_df.groupby(id_col)["_is_winter"].mean().reindex(out[id_col], fill_value=0.0)
        out["winter_share_q95"] = winter_share.to_numpy()
    else:
        out["active_year_prop_q95"] = 0.0
        out["winter_share_q95"] = 0.0

    return out, thr


# ============================================================
# 3. Publication map helpers
# ============================================================

def load_europe_admin_publication(countries_src, bbox4326):
    """
    Load and filter admin boundaries in the same way as the spatial-overlap
    figures. If a Eurostat CNTR shapefile is supplied, CNTR_ID is used to keep
    the main European footprint and exclude Russia/Iceland.
    """
    if not countries_src or not os.path.exists(countries_src):
        print("[WARN] Warning: admin boundary path is missing. Maps will be drawn without admin borders.")
        return None

    admin = gpd.read_file(countries_src)

    if admin.crs is None:
        admin = admin.set_crs("EPSG:3035")

    # Eurostat country boundary shapefile.
    if "CNTR_ID" in admin.columns:
        admin = admin[admin["CNTR_ID"].isin(EUROPE_CNTR_IDS)].copy()
    else:
        # Natural Earth / generic fallback.
        if admin.crs.to_epsg() != 4326:
            admin = admin.to_crs("EPSG:4326")
        name_col = first_existing(admin.columns, ["name", "NAME", "ADMIN", "admin", "SOVEREIGNT"])
        continent_col = first_existing(admin.columns, ["continent", "CONTINENT"])
        if continent_col is not None:
            europe = admin[admin[continent_col].astype(str).str.lower() == "europe"].copy()
            if name_col is not None:
                extras = admin[admin[name_col].astype(str).isin(["Turkey", "Cyprus"])].copy()
                europe = pd.concat([europe, extras], ignore_index=True)
            admin = europe
        if name_col is not None:
            admin = admin[~admin[name_col].astype(str).str.lower().isin(EXCLUDE_COUNTRIES)].copy()

    # Clip to main-Europe bbox in lon/lat.
    admin = admin.to_crs("EPSG:4326")
    admin = admin.explode(index_parts=False).reset_index(drop=True)

    xmin, ymin, xmax, ymax = bbox4326
    bbox_geom = box(xmin, ymin, xmax, ymax)

    admin = admin[admin.geometry.intersects(bbox_geom)].copy()
    admin["geometry"] = admin.geometry.intersection(bbox_geom)
    admin = admin[~admin.geometry.is_empty].copy()
    admin = admin[admin.geometry.notna()].copy()

    # Remove tiny fragments / remote islands after clipping.
    admin_tmp = admin.to_crs(PLOT_CRS).copy()
    admin["area_km2_tmp"] = admin_tmp.geometry.area / 1e6
    admin = admin[admin["area_km2_tmp"] >= MIN_ADMIN_PART_AREA_KM2].copy()
    admin = admin.drop(columns="area_km2_tmp")

    admin = admin.to_crs(PLOT_CRS)

    print(f"Admin features kept after main-Europe filtering: {len(admin):,}")
    return admin


def filter_main_europe_geometries(gdf, bbox4326):
    """Broad bbox clip in lon/lat, then return lon/lat geometries."""
    xmin, ymin, xmax, ymax = bbox4326
    bbox_geom = box(xmin, ymin, xmax, ymax)

    if gdf.crs is None:
        gdf = gdf.set_crs(PLOT_CRS)

    gdf_4326 = gdf.to_crs("EPSG:4326").copy()
    gdf_4326 = gdf_4326[gdf_4326.geometry.intersects(bbox_geom)].copy()
    gdf_4326["geometry"] = gdf_4326.geometry.intersection(bbox_geom)
    gdf_4326 = gdf_4326[~gdf_4326.geometry.is_empty].copy()
    gdf_4326 = gdf_4326[gdf_4326.geometry.notna()].copy()

    return gdf_4326


def project_to_plot_crs(gdf):
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(PLOT_CRS)


def filter_hex_to_admin_footprint(gdf, admin_gdf):
    """Keep only hexes whose representative point lies inside allowed admin footprint."""
    if admin_gdf is None or len(admin_gdf) == 0:
        return gdf

    gg = gdf.copy()
    gg = gg[gg.geometry.notna()].copy()
    gg = gg[~gg.geometry.is_empty].copy()

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


def get_plot_bounds(hex_gdf, pad_ratio=0.035):
    xmin, ymin, xmax, ymax = hex_gdf.total_bounds
    dx = xmax - xmin
    dy = ymax - ymin
    pad_x = dx * pad_ratio
    pad_y = dy * pad_ratio
    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def style_ax(ax, bounds):
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
        admin_gdf.boundary.plot(
            ax=ax,
            linewidth=ADMIN_LW,
            color=ADMIN_COLOR,
            zorder=4,
        )


def add_panel_label(ax, label):
    ax.text(
        -0.02,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        zorder=20,
        clip_on=False,
    )


def prepare_plot_gdf(hexgrid, admin_plot, bbox4326):
    """Apply the same Europe clipping/projection/footprint logic as the overlap figures."""
    gdf = filter_main_europe_geometries(hexgrid, bbox4326)
    gdf = project_to_plot_crs(gdf)
    gdf = filter_hex_to_admin_footprint(gdf, admin_plot)
    return gdf


def get_forest_mask(gdf, forest_mask_col="mask_forest"):
    if forest_mask_col in gdf.columns:
        return gdf[forest_mask_col].astype("boolean").fillna(False).to_numpy()
    return np.ones(len(gdf), dtype=bool)


def plot_indicator_panel(
    ax,
    hex_gdf,
    value_col,
    title,
    cbar_label,
    admin_plot,
    bounds,
    cmap=CMAP_RED,
    norm_mode="power",
    gamma=0.60,
    qhi=99.0,
    forest_mask_col="mask_forest",
    add_cbar=True,
):
    vals = pd.to_numeric(hex_gdf[value_col], errors="coerce").to_numpy(dtype=float)
    forest_mask = get_forest_mask(hex_gdf, forest_mask_col)

    bg = hex_gdf.loc[forest_mask].copy()
    fg = hex_gdf.loc[forest_mask & np.isfinite(vals)].copy()

    if not bg.empty:
        bg.plot(
            ax=ax,
            color="#efefef",
            edgecolor="none",
            linewidth=0.0,
            zorder=1,
        )

    norm = None
    if not fg.empty:
        norm = build_norm(fg[value_col].to_numpy(dtype=float), mode=norm_mode, gamma=gamma, qhi=qhi)
        fg.plot(
            ax=ax,
            column=value_col,
            cmap=cmap,
            norm=norm,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.98,
            zorder=2,
            missing_kwds={"color": "#f5f5f5"},
        )

    draw_admin(ax, admin_plot)
    style_ax(ax, bounds)
    ax.set_title(title, pad=8)

    if add_cbar and norm is not None:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = ax.figure.colorbar(sm, ax=ax, shrink=0.72)
        cb.set_label(cbar_label, fontsize=10)
        cb.ax.tick_params(labelsize=9, length=3)
        cb.outline.set_linewidth(0.7)


def plot_individual_indicator_map(
    hex_gdf,
    value_col,
    title,
    out_path,
    admin_plot,
    bounds,
    cbar_label,
    cmap=CMAP_RED,
    norm_mode="power",
    gamma=0.60,
    qhi=99.0,
):
    fig, ax = plt.subplots(figsize=(7.2, 7.6), facecolor="white")
    plot_indicator_panel(
        ax=ax,
        hex_gdf=hex_gdf,
        value_col=value_col,
        title=title,
        cbar_label=cbar_label,
        admin_plot=admin_plot,
        bounds=bounds,
        cmap=cmap,
        norm_mode=norm_mode,
        gamma=gamma,
        qhi=qhi,
        add_cbar=True,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[FIG] saved:", out_path)


def plot_indicator_suite_2x2(
    hex_gdf,
    metrics,
    title_map,
    cbar_map,
    out_png,
    admin_plot,
    bounds,
    cmap=CMAP_RED,
    norm_mode="power",
    gamma=0.60,
    qhi=99.0,
):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.2), facecolor="white")
    axes = axes.ravel()
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, m, lab in zip(axes, metrics, labels):
        plot_indicator_panel(
            ax=ax,
            hex_gdf=hex_gdf,
            value_col=m,
            title=title_map.get(m, m),
            cbar_label=cbar_map.get(m, m),
            admin_plot=admin_plot,
            bounds=bounds,
            cmap=cmap,
            norm_mode=norm_mode,
            gamma=gamma,
            qhi=qhi,
            add_cbar=True,
        )
        add_panel_label(ax, lab)

    plt.subplots_adjust(
        left=0.025,
        right=0.995,
        top=0.925,
        bottom=0.045,
        wspace=0.035,
        hspace=0.115,
    )

    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_png).replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[FIG] saved:", out_png)


# ============================================================
# 4. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hex_distribution_csv", required=True, type=str)
    parser.add_argument("--hex_period_csv", required=True, type=str)
    parser.add_argument("--outdir", required=True, type=str)
    parser.add_argument("--countries_src", default=None, type=str)
    parser.add_argument("--hex_bbox4326", default=BBOX_LONLAT_DEFAULT, type=str,
                        help="IMPORTANT: must match the hex_bbox4326 used in the original wall-to-wall prediction run.")
    parser.add_argument("--plot_bbox4326", default="-12,34,45,72", type=str,
                        help="Only used for map clipping/plot extent. Does not rebuild or renumber hexes.")
    parser.add_argument("--hex_area_km2", default=2165.0, type=float)
    parser.add_argument("--forest_frac_min", default=0.0, type=float)
    parser.add_argument("--hex_geometry_csv", default=None, type=str,
                        help="Optional geometry source (CSV with WKT geometry or GPKG). If provided, hex geometries are read from this file instead of rebuilt from bbox.")
    parser.add_argument("--geometry_crs", default="EPSG:3035", type=str,
                        help="CRS of WKT geometries when reading geometry from CSV. Default: EPSG:3035")
    parser.add_argument("--active_quantile", default=0.95, type=float)
    parser.add_argument("--norm", choices=["linear", "power"], default="power")
    parser.add_argument("--gamma", default=0.60, type=float)
    parser.add_argument("--qhi", default=99.0, type=float)
    parser.add_argument("--skip_individual", action="store_true", help="Only save the combined 2x2 figure and CSV.")
    args = parser.parse_args()

    set_publication_style()
    os.makedirs(args.outdir, exist_ok=True)

    grid_bbox4326 = parse_bbox4326(args.hex_bbox4326)
    plot_bbox4326 = parse_bbox4326(args.plot_bbox4326)

    print("[LOAD] Loading hex distribution...")
    hex_gdf, hex_id_dist = load_hex_distribution(
        args.hex_distribution_csv,
        bbox4326=args.hex_bbox4326,
        hex_area_km2=args.hex_area_km2,
        forest_frac_min=args.forest_frac_min,
        geometry_path=args.hex_geometry_csv,
        geometry_crs=args.geometry_crs,
        outdir=args.outdir,
    )
    print(f"[INFO] Geometry source used: {hex_gdf.attrs.get('geometry_source', 'unknown')}")

    print("[LOAD] Loading hex period summary...")
    period_df, hex_id_period, pred_col = load_hex_period_summary(args.hex_period_csv)
    print(f"[INFO] inferred prediction column: {pred_col}")

    if hex_id_period != hex_id_dist:
        period_df = period_df.rename(columns={hex_id_period: hex_id_dist})

    print("[COMPUTE] Computing indicator suite...")
    ind_df, thr = compute_indicator_suite(
        period_df,
        id_col=hex_id_dist,
        pred_col=pred_col,
        active_quantile=args.active_quantile,
    )
    print(f"[INFO] active threshold ({args.active_quantile:.2f} quantile) = {thr:.6f}")

    hexgrid = hex_gdf.merge(ind_df, on=hex_id_dist, how="left")
    matched_ids = hexgrid["mean_pred_prob"].notna().sum() if "mean_pred_prob" in hexgrid.columns else 0
    print(f"[INFO] Hex rows after merging indicators: {len(hexgrid):,}; matched indicator rows: {matched_ids:,}")
    print(f"[INFO] Grid bbox used for rebuilding hexes: {args.hex_bbox4326}")
    print(f"[INFO] Plot bbox used only for clipping: {args.plot_bbox4326}")


    fill_zero_cols = [
        "n_periods", "mean_pred_prob", "p95_pred_prob", "p99_pred_prob", "max_pred_prob",
        "active_n", "recurrence_index", "conditional_intensity", "conditional_p95",
        "active_year_prop_q95", "winter_share_q95"
    ]
    for c in fill_zero_cols:
        if c in hexgrid.columns:
            hexgrid[c] = pd.to_numeric(hexgrid[c], errors="coerce").fillna(0.0)

    # Keep this legacy output name because downstream scripts may already use it.
    out_csv = os.path.join(args.outdir, "hex_indicator_summary_stylematch_4326.csv")
    save_df = pd.DataFrame(hexgrid.drop(columns="geometry")).copy()
    save_df["geometry"] = hexgrid.geometry.to_wkt()
    save_df.to_csv(out_csv, index=False)
    print("[SAVE] saved:", out_csv)

    print("[MAP] Loading Europe admin boundaries...")
    europe_admin_plot = load_europe_admin_publication(args.countries_src, plot_bbox4326)

    print("[MAP] Applying publication-style Europe clipping and projection...")
    hexgrid_plot = prepare_plot_gdf(hexgrid, europe_admin_plot, plot_bbox4326)

    # Tighten plot bounds using only forested hexes with valid indicator values,
    # but never rebuild the grid extent itself.
    extent_mask = hexgrid_plot["mask_forest"].astype("boolean").fillna(False)
    valid_cols = [c for c in ["mean_pred_prob", "p95_pred_prob", "recurrence_index", "conditional_intensity"] if c in hexgrid_plot.columns]
    if valid_cols:
        valid_any = np.zeros(len(hexgrid_plot), dtype=bool)
        for c in valid_cols:
            vv = pd.to_numeric(hexgrid_plot[c], errors="coerce").to_numpy(dtype=float)
            valid_any |= np.isfinite(vv) & (vv > 0)
        extent_gdf = hexgrid_plot.loc[extent_mask & valid_any].copy()
    else:
        extent_gdf = hexgrid_plot.loc[extent_mask].copy()
    if extent_gdf.empty:
        extent_gdf = hexgrid_plot.copy()

    bounds = get_plot_bounds(extent_gdf, pad_ratio=0.025)
    print(f"Hexes kept for maps: {len(hexgrid_plot):,}")
    print(f"Hexes used for plot extent: {len(extent_gdf):,}")
    print(f"Plot CRS: {hexgrid_plot.crs}")
    print(f"Plot bounds: {bounds}")

    # Save plotted footprint too, useful for checking consistency.
    try:
        gpkg_path = os.path.join(args.outdir, "hex_indicator_summary_stylematch_publication.gpkg")
        hexgrid_plot.to_file(gpkg_path, driver="GPKG")
        print("[SAVE] saved:", gpkg_path)
    except Exception as e:
        print(f"[WARN] Could not save GPKG: {e}")

    qlabel = f"Q{int(args.active_quantile * 100)}"
    title_map = {
        "recurrence_index": "High-probability recurrence",
        "conditional_intensity": "Active-period mean probability",
        "mean_pred_prob": "Mean predicted probability",
        "p95_pred_prob": "95th percentile predicted probability",
    }
    cbar_map = {
        "recurrence_index": "High-probability recurrence",
        "conditional_intensity": "Active-period mean probability",
        "mean_pred_prob": "Mean predicted probability",
        "p95_pred_prob": "95th percentile predicted probability",
    }
    metrics = ["mean_pred_prob", "p95_pred_prob", "recurrence_index", "conditional_intensity"]

    print("[FIG] Plotting publication-style maps...")
    if not args.skip_individual:
        for m in metrics:
            out_png = os.path.join(args.outdir, f"{m}_stylematch_publication.png")
            plot_individual_indicator_map(
                hex_gdf=hexgrid_plot,
                value_col=m,
                title=title_map.get(m, m),
                out_path=out_png,
                admin_plot=europe_admin_plot,
                bounds=bounds,
                cbar_label=cbar_map.get(m, m),
                cmap=CMAP_RED,
                norm_mode=args.norm,
                gamma=args.gamma,
                qhi=args.qhi,
            )

    out_2x2 = os.path.join(args.outdir, "Fig_hex_indicator_suite_stylematch_publication_2x2.png")
    plot_indicator_suite_2x2(
        hex_gdf=hexgrid_plot,
        metrics=metrics,
        title_map=title_map,
        cbar_map=cbar_map,
        out_png=out_2x2,
        admin_plot=europe_admin_plot,
        bounds=bounds,
        cmap=CMAP_RED,
        norm_mode=args.norm,
        gamma=args.gamma,
        qhi=args.qhi,
    )

    print("[DONE] Done.")


if __name__ == "__main__":
    main()
