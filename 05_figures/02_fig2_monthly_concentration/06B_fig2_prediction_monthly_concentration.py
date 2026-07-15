# ============================================================
# Monthly peak timing and concentration of wall-to-wall predicted
# wind-disturbance likelihood
#
# Revised version:
#   1. Uses monthly rather than seasonal metrics
#   2. Does not mask peak-month values
#   3. Fixes 12-month panel color scale to 0–0.30
#   4. Produces publication-style maps without projected axis labels
#   5. Uses the same deep-purple cyclic month colour band as the wind-anomaly figure
#   6. Outputs only peak month and peak-month share maps
#
# Inputs:
#   E:\RF B+G\wall2wall_windonly_finalmap
#     ├─ hex_period_summary_wall2wall_windonly_final.csv
#     └─ hex_indicator_summary_stylematch_4326.csv
#
# Main output figure:
#   Fig_peak_month_and_peak_month_share_1x2
# ============================================================

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

try:
    import geopandas as gpd
    from shapely import wkt
    from shapely.geometry import box
    HAS_GEO = True
except Exception:
    HAS_GEO = False
    warnings.warn(
        "geopandas/shapely not available. CSV outputs will be saved, but maps will be skipped."
    )


# ============================================================
# 0. User settings
# ============================================================

BASE = Path(os.environ.get("WALL2WALL_OUTDIR", r"E:\RF B+G\wall2wall_windonly_finalmap"))
FIG1_BASE = Path(os.environ.get("FIG1_OUTDIR", str(BASE)))

HEX_PERIOD_CSV = Path(os.environ.get("HEX_PERIOD_CSV", str(BASE / "hex_period_summary_wall2wall_windonly_final.csv")))
HEX_STYLEMATCH_CSV = Path(os.environ.get("HEX_STYLEMATCH_CSV", str(FIG1_BASE / "hex_indicator_summary_stylematch_4326.csv")))

OUTDIR = Path(os.environ.get("FIG2_OUTDIR", str(BASE / "monthly_spatiotemporal_concentration_wall2wall_FIXED")))
OUTDIR.mkdir(parents=True, exist_ok=True)

ADMIN_SHP = Path(os.environ.get("COUNTRIES_SRC", r"E:\CNTR_RG_20M_2024_3035\CNTR_RG_20M_2024_3035.shp"))

DATE_COL = "obs_date"
PRED_COL = "mean_pred_wind"

START_YEAR = 2003
END_YEAR = 2023

PLOT_CRS = "EPSG:3035"

# Europe main extent in lon/lat; converted to EPSG:3035 for plotting.
BBOX4326 = (-12, 34, 45, 72)

# Keep the same main-Europe administrative footprint used in the
# spatial-overlap and temporal-overlap figures. Russia and Iceland are excluded.
EUROPE_CNTR_IDS = {
    "AL", "AD", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ", "DE", "DK",
    "EE", "EL", "ES", "FI", "FR", "HU", "HR", "IE", "IT", "LI", "LT", "LU",
    "LV", "MD", "ME", "MK", "MT", "NL", "NO", "PL", "PT", "RO", "RS", "SE",
    "SI", "SK", "SM", "UA", "VA", "XK", "UK", "GB"
}
MIN_ADMIN_PART_AREA_KM2 = 20

FIG_DPI = 500

MONTHS = list(range(1, 13))
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# Recommended:
# "mean" = average predicted likelihood by month, then normalize across 12 months.
# This avoids bias from unequal numbers of 16-day observations per month.
PROFILE_BASIS = "mean"

# Peak-month mask:
# Peak month is only interpreted where the monthly profile is sufficiently concentrated.
PEAK_MONTH_SHARE_MIN = 0.20
EFFECTIVE_MONTHS_MAX = 6.0

# "or"  = show peak month if either condition is satisfied
# "and" = show peak month only if both conditions are satisfied
PEAK_MASK_RULE = "or"

# 12-month panel color scale
MONTHLY_SHARE_VMIN = 0.00
MONTHLY_SHARE_VMAX = 0.30

POINT_SIZE_MAIN = 4.0
POINT_SIZE_SMALL = 2.8

DIFFUSE_COLOR = "#d9d9d9"
SEA_COLOR = "white"
BOUNDARY_COLOR = "0.35"

# Set after geometry filtering. Stored as (xmin, ymin, xmax, ymax) in EPSG:3035.
MAP_BOUNDS_3035 = None



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
        "axes.facecolor": "white",
    })


def make_peak_month_cmap():
    """
    Month colour band used consistently with the wind-anomaly timing map.
    Jan and Dec both grade into deep purple, giving a cyclic month scale.
    """
    colors = [
        "#2c115f",  # Jan: deep purple
        "#3b2f8f",
        "#4764c2",
        "#4f9ddf",
        "#7bc8d6",
        "#c9e7b6",
        "#f2e88b",
        "#f7c25a",
        "#ef8a47",
        "#d24b40",
        "#9d2b7f",
        "#5a1e96",
        "#2c115f",  # closes the cycle back to deep purple
    ]
    return LinearSegmentedColormap.from_list(
        "month_deeppurple_cyclic",
        colors,
        N=256
    )


# ============================================================
# 1. Monthly metric helpers
# ============================================================

def effective_months_from_shares(shares):
    shares = np.asarray(shares, dtype=float)
    shares = np.where(np.isfinite(shares), shares, 0.0)

    total = shares.sum()
    if total <= 0:
        return np.nan, np.nan

    shares = shares / total
    hhi = np.sum(shares ** 2)

    if hhi <= 0:
        return np.nan, np.nan

    effective_months = 1.0 / hhi
    return hhi, effective_months


def circular_mean_month_from_shares(shares):
    """
    Circular mean timing of monthly likelihood distribution.

    Jan = 1, Feb = 2, ..., Dec = 12.
    circular_concentration ranges from 0 to 1.
    """
    shares = np.asarray(shares, dtype=float)
    shares = np.where(np.isfinite(shares), shares, 0.0)

    total = shares.sum()
    if total <= 0:
        return np.nan, np.nan

    shares = shares / total

    angles = 2.0 * np.pi * np.arange(12) / 12.0

    x = np.sum(shares * np.cos(angles))
    y = np.sum(shares * np.sin(angles))
    r = np.sqrt(x ** 2 + y ** 2)

    if r <= 0:
        return np.nan, 0.0

    angle = np.arctan2(y, x)
    if angle < 0:
        angle += 2.0 * np.pi

    mean_month = angle / (2.0 * np.pi) * 12.0 + 1.0

    return mean_month, r


def month_name(m):
    if pd.isna(m):
        return np.nan
    m_int = int(round(float(m)))
    m_int = max(1, min(12, m_int))
    return MONTH_LABELS[m_int - 1]


def circular_month_difference(m1, m2):
    """
    Absolute distance between two months on circular 12-month scale.
    Returns 0 to 6.
    """
    if pd.isna(m1) or pd.isna(m2):
        return np.nan

    d = abs(int(round(float(m1))) - int(round(float(m2))))
    return min(d, 12 - d)


# ============================================================
# 2. Geometry and CRS helpers
# ============================================================

def get_projected_bbox(bbox4326=BBOX4326, target_crs=PLOT_CRS):
    global MAP_BOUNDS_3035

    # Once maps are filtered, all plotting functions use the same data-driven
    # EPSG:3035 bounds. This keeps the layout identical across all panels and
    # avoids the left-side blank space caused by broad lon/lat bboxes.
    if target_crs == PLOT_CRS and MAP_BOUNDS_3035 is not None:
        return MAP_BOUNDS_3035

    xmin, ymin, xmax, ymax = bbox4326

    bbox_gs = gpd.GeoSeries(
        [box(xmin, ymin, xmax, ymax)],
        crs="EPSG:4326"
    ).to_crs(target_crs)

    return bbox_gs.total_bounds


def detect_geometry_column(df):
    for c in ["geometry", "geom", "wkt", "WKT"]:
        if c in df.columns:
            return c
    return None


def detect_true_lon_lat_columns(df):
    lon_candidates = [
        "lon", "longitude", "centroid_lon", "center_lon",
        "lon_4326", "longitude_4326"
    ]
    lat_candidates = [
        "lat", "latitude", "centroid_lat", "center_lat",
        "lat_4326", "latitude_4326"
    ]

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

    if not np.isfinite([minx, miny, maxx, maxy]).all():
        return "EPSG:3035"

    if (
        -180 <= minx <= 180 and
        -180 <= maxx <= 180 and
        -90 <= miny <= 90 and
        -90 <= maxy <= 90
    ):
        return "EPSG:4326"

    return "EPSG:3035"


def prepare_hex_geometry(hex_csv):
    """
    Read hex-level CSV and return GeoDataFrame in EPSG:3035.

    Handles:
      1. WKT geometry column
      2. true lon/lat columns
      3. projected x/y columns
    """
    if not HAS_GEO:
        return None

    hx = pd.read_csv(hex_csv)

    print("\n============================================================")
    print("Loaded hex geometry/stylematch file")
    print("============================================================")
    print(hex_csv)
    print(f"Rows: {len(hx):,}")
    print("Columns:")
    for c in hx.columns:
        print("  -", c)

    if "hex_id" not in hx.columns:
        raise ValueError("hex_indicator_summary_stylematch_4326.csv must contain hex_id.")

    geom_col = detect_geometry_column(hx)

    if geom_col is not None:
        print(f"\nDetected geometry column: {geom_col}")

        hx["_geometry"] = hx[geom_col].apply(
            lambda z: wkt.loads(z) if isinstance(z, str) and len(z) > 0 else None
        )

        gdf = gpd.GeoDataFrame(hx, geometry="_geometry")
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()

        guessed_crs = guess_crs_from_bounds(gdf)
        print(f"Guessed geometry CRS from bounds: {guessed_crs}")

        gdf = gdf.set_crs(guessed_crs, allow_override=True)
        gdf = gdf.to_crs(PLOT_CRS)

        return gdf

    lon_col, lat_col = detect_true_lon_lat_columns(hx)

    if lon_col is not None and lat_col is not None:
        print(f"\nDetected true lon/lat columns: {lon_col}, {lat_col}")

        hx[lon_col] = pd.to_numeric(hx[lon_col], errors="coerce")
        hx[lat_col] = pd.to_numeric(hx[lat_col], errors="coerce")

        hx = hx[
            hx[lon_col].between(-180, 180) &
            hx[lat_col].between(-90, 90)
        ].copy()

        gdf = gpd.GeoDataFrame(
            hx,
            geometry=gpd.points_from_xy(hx[lon_col], hx[lat_col]),
            crs="EPSG:4326"
        )

        gdf = gdf.to_crs(PLOT_CRS)
        return gdf

    x_col, y_col = detect_projected_xy_columns(hx)

    if x_col is not None and y_col is not None:
        print(f"\nDetected projected x/y columns: {x_col}, {y_col}")
        print("Treating x/y as EPSG:3035 coordinates.")

        hx[x_col] = pd.to_numeric(hx[x_col], errors="coerce")
        hx[y_col] = pd.to_numeric(hx[y_col], errors="coerce")

        hx = hx[np.isfinite(hx[x_col]) & np.isfinite(hx[y_col])].copy()

        gdf = gpd.GeoDataFrame(
            hx,
            geometry=gpd.points_from_xy(hx[x_col], hx[y_col]),
            crs="EPSG:3035"
        )

        gdf = gdf.to_crs(PLOT_CRS)
        return gdf

    print("\nNo usable geometry, lon/lat, or x/y columns detected. Maps will be skipped.")
    return None


def load_admin_3035():
    if not HAS_GEO:
        return None

    if not ADMIN_SHP.exists():
        print("\nAdmin shapefile not found. Maps will be drawn without country boundaries.")
        return None

    admin = gpd.read_file(ADMIN_SHP)

    if admin.crs is None:
        admin = admin.set_crs("EPSG:3035")

    if "CNTR_ID" in admin.columns:
        admin = admin[admin["CNTR_ID"].astype(str).isin(EUROPE_CNTR_IDS)].copy()
    else:
        print("Warning: CNTR_ID not found in admin shapefile. Using bbox filtering only.")

    # Clip admin footprint in lon/lat first, then project to EPSG:3035.
    admin = admin.to_crs("EPSG:4326")
    admin = admin.explode(index_parts=False).reset_index(drop=True)

    xmin, ymin, xmax, ymax = BBOX4326
    bbox_geom = box(xmin, ymin, xmax, ymax)

    admin = admin[admin.geometry.intersects(bbox_geom)].copy()
    admin["geometry"] = admin.geometry.intersection(bbox_geom)
    admin = admin[~admin.geometry.is_empty].copy()
    admin = admin[admin.geometry.notna()].copy()

    # Remove tiny fragments generated by clipping.
    admin_tmp = admin.to_crs(PLOT_CRS).copy()
    admin["area_km2_tmp"] = admin_tmp.geometry.area / 1e6
    admin = admin[admin["area_km2_tmp"] >= MIN_ADMIN_PART_AREA_KM2].copy()
    admin = admin.drop(columns="area_km2_tmp")

    admin = admin.to_crs(PLOT_CRS)
    print(f"Admin features kept after main-Europe filtering: {len(admin):,}")
    return admin


def filter_main_europe_geometries(gdf):
    """
    Clip hex geometries to the main Europe bbox in lon/lat, then return EPSG:3035.
    """
    xmin, ymin, xmax, ymax = BBOX4326
    bbox_geom = box(xmin, ymin, xmax, ymax)

    gg = gdf.copy()
    if gg.crs is None:
        gg = gg.set_crs(PLOT_CRS, allow_override=True)

    gg = gg.to_crs("EPSG:4326")
    gg = gg[gg.geometry.notna()].copy()
    gg = gg[~gg.geometry.is_empty].copy()
    gg = gg[gg.geometry.intersects(bbox_geom)].copy()
    gg["geometry"] = gg.geometry.intersection(bbox_geom)
    gg = gg[~gg.geometry.is_empty].copy()
    gg = gg[gg.geometry.notna()].copy()

    return gg.to_crs(PLOT_CRS)


def filter_hex_to_admin_footprint(gdf, admin_gdf):
    """
    Keep only hexes whose representative point lies inside the selected
    European administrative footprint. This removes non-target eastern cells.
    """
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
    return (
        xmin - dx * pad_ratio,
        ymin - dy * pad_ratio,
        xmax + dx * pad_ratio,
        ymax + dy * pad_ratio,
    )


def set_global_plot_bounds(gdf):
    global MAP_BOUNDS_3035

    # Use valid prediction hexes for a tight map extent, not the broad bbox.
    extent_gdf = gdf.copy()
    if "mean_pred_longterm" in extent_gdf.columns:
        vals = pd.to_numeric(extent_gdf["mean_pred_longterm"], errors="coerce")
        extent_gdf = extent_gdf[np.isfinite(vals.to_numpy(dtype=float)) & (vals.to_numpy(dtype=float) > 0)].copy()

    if extent_gdf.empty:
        extent_gdf = gdf.copy()

    MAP_BOUNDS_3035 = get_plot_bounds(extent_gdf, pad_ratio=0.025)
    return MAP_BOUNDS_3035


def apply_publication_axis_style(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def clean_gdf_for_plot(gdf, column):
    gg = gdf.copy()

    gg = gg[gg.geometry.notna()].copy()
    gg = gg[~gg.geometry.is_empty].copy()

    gg[column] = pd.to_numeric(gg[column], errors="coerce")
    gg = gg[np.isfinite(gg[column])].copy()

    return gg


def set_europe_extent(ax):
    bxmin, bymin, bxmax, bymax = get_projected_bbox()
    ax.set_xlim(bxmin, bxmax)
    ax.set_ylim(bymin, bymax)
    apply_publication_axis_style(ax)


def add_panel_label(ax, label):
    ax.text(
        -0.02,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        zorder=20,
        clip_on=False,
    )

def get_is_point(gdf):
    if len(gdf) == 0:
        return False
    return gdf.geometry.iloc[0].geom_type in ["Point", "MultiPoint"]


# ============================================================
# 3. Plot functions
# ============================================================

def draw_admin(ax, admin):
    if admin is not None:
        admin.boundary.plot(
            ax=ax,
            linewidth=0.32,
            color=BOUNDARY_COLOR,
            alpha=0.75
        )


def add_colorbar(fig, ax, cmap, vmin, vmax, label=None, month_ticks=False):
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, shrink=0.72)

    if month_ticks:
        cbar.set_ticks(MONTHS)
        cbar.set_ticklabels(MONTH_LABELS)

    if label is not None:
        cbar.set_label(label)

    return cbar


def plot_continuous_map(
    gdf,
    column,
    title,
    outname,
    cmap="viridis",
    vmin=None,
    vmax=None,
    legend_label=None,
    admin=None,
    point_size=POINT_SIZE_MAIN,
    month_colorbar=False,
):
    gg = clean_gdf_for_plot(gdf, column)

    if len(gg) == 0:
        print(f"Skipping {outname}: no valid rows for {column}")
        return

    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.set_facecolor(SEA_COLOR)

    draw_admin(ax, admin)

    is_point = get_is_point(gg)

    if vmin is None:
        vmin = np.nanpercentile(gg[column], 2)
    if vmax is None:
        vmax = np.nanpercentile(gg[column], 98)

    gg.plot(
        column=column,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        markersize=point_size if is_point else None,
        linewidth=0,
        legend=False,
    )

    add_colorbar(
        fig,
        ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        label=legend_label,
        month_ticks=month_colorbar
    )

    set_europe_extent(ax)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

    fig.tight_layout()

    fig.savefig(OUTDIR / f"{outname}.png", dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(OUTDIR / f"{outname}.pdf", bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {OUTDIR / f'{outname}.png'}")


def plot_masked_peak_month_map(
    gdf,
    title,
    outname,
    admin=None,
    point_size=POINT_SIZE_MAIN,
):
    """
    Plot peak month only where monthly profile is sufficiently concentrated.
    Diffuse hexes are shown in grey.
    """

    gg_all = gdf.copy()
    gg_all = gg_all[gg_all.geometry.notna()].copy()
    gg_all = gg_all[~gg_all.geometry.is_empty].copy()

    if len(gg_all) == 0:
        print(f"Skipping {outname}: no valid geometry")
        return

    gg_masked = clean_gdf_for_plot(gg_all, "peak_month_masked")

    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.set_facecolor(SEA_COLOR)

    draw_admin(ax, admin)

    is_point_all = get_is_point(gg_all)

    # Background: diffuse / weakly concentrated profiles
    gg_all.plot(
        ax=ax,
        color=DIFFUSE_COLOR,
        markersize=point_size if is_point_all else None,
        linewidth=0,
        alpha=0.70,
    )

    # Overlay: reliable peak month
    if len(gg_masked) > 0:
        is_point = get_is_point(gg_masked)

        gg_masked.plot(
            column="peak_month_masked",
            ax=ax,
            cmap=make_peak_month_cmap(),
            vmin=1,
            vmax=12,
            markersize=point_size if is_point else None,
            linewidth=0,
            legend=False,
        )

        add_colorbar(
            fig,
            ax,
            cmap=make_peak_month_cmap(),
            vmin=1,
            vmax=12,
            label="Peak month",
            month_ticks=True
        )

    set_europe_extent(ax)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

    # Add a small note
    note = (
        f"Grey: diffuse profile\n"
        f"shown if peak share ≥ {PEAK_MONTH_SHARE_MIN:.2f} "
        f"{PEAK_MASK_RULE.upper()} effective months ≤ {EFFECTIVE_MONTHS_MAX:g}"
    )

    ax.text(
        0.02,
        0.035,
        note,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2)
    )

    fig.tight_layout()

    fig.savefig(OUTDIR / f"{outname}.png", dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(OUTDIR / f"{outname}.pdf", bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {OUTDIR / f'{outname}.png'}")


def plot_monthly_share_grid(
    gdf,
    share_prefix,
    outname,
    title,
    admin=None,
):
    month_cols = [f"{share_prefix}{m:02d}_share" for m in MONTHS]
    existing_cols = [c for c in month_cols if c in gdf.columns]

    if len(existing_cols) == 0:
        print(f"Skipping {outname}: no monthly share columns found.")
        return

    vmin = MONTHLY_SHARE_VMIN
    vmax = MONTHLY_SHARE_VMAX

    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    axes = axes.ravel()

    bxmin, bymin, bxmax, bymax = get_projected_bbox()

    for i, m in enumerate(MONTHS):
        ax = axes[i]
        ax.set_facecolor(SEA_COLOR)

        draw_admin(ax, admin)

        col = f"{share_prefix}{m:02d}_share"
        gg = clean_gdf_for_plot(gdf, col)

        if len(gg) > 0:
            is_point = get_is_point(gg)

            gg.plot(
                column=col,
                ax=ax,
                cmap="YlOrRd",
                vmin=vmin,
                vmax=vmax,
                markersize=POINT_SIZE_SMALL if is_point else None,
                linewidth=0,
                legend=False,
            )

        ax.set_xlim(bxmin, bxmax)
        ax.set_ylim(bymin, bymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(MONTH_LABELS[m - 1], fontsize=10)
        ax.grid(False)

    for _ax in axes:
        apply_publication_axis_style(_ax)

    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="YlOrRd")
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes.tolist(),
        shrink=0.62,
        pad=0.02,
        fraction=0.035
    )
    cbar.set_label("Monthly share of predicted likelihood")

    fig.suptitle(title, fontsize=14, y=0.98)

    fig.savefig(OUTDIR / f"{outname}.png", dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(OUTDIR / f"{outname}.pdf", bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {OUTDIR / f'{outname}.png'}")


def plot_combined_monthly_2x2(gdf, admin=None):
    """
    Main figure with only two panels:
      (a) Peak month
      (b) Peak-month share
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.4))
    axes = axes.ravel()

    bxmin, bymin, bxmax, bymax = get_projected_bbox()

    # --------------------------------------------------------
    # Panel a: peak month, unmasked
    # --------------------------------------------------------
    ax = axes[0]
    ax.set_facecolor(SEA_COLOR)
    draw_admin(ax, admin)

    gg = clean_gdf_for_plot(gdf, "peak_month")

    if len(gg) > 0:
        is_point = get_is_point(gg)

        gg.plot(
            column="peak_month",
            ax=ax,
            cmap=make_peak_month_cmap(),
            vmin=1,
            vmax=12,
            markersize=POINT_SIZE_MAIN if is_point else None,
            linewidth=0,
            legend=False,
        )

        add_colorbar(
            fig,
            ax,
            cmap=make_peak_month_cmap(),
            vmin=1,
            vmax=12,
            label="Peak month",
            month_ticks=True
        )

    ax.set_xlim(bxmin, bxmax)
    ax.set_ylim(bymin, bymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Peak month", fontsize=11)
    ax.grid(False)
    add_panel_label(ax, "(a)")

    # --------------------------------------------------------
    # Panel b: peak-month share
    # --------------------------------------------------------
    ax = axes[1]
    ax.set_facecolor(SEA_COLOR)
    draw_admin(ax, admin)

    gg = clean_gdf_for_plot(gdf, "peak_month_share")
    if len(gg) > 0:
        is_point = get_is_point(gg)

        gg.plot(
            column="peak_month_share",
            ax=ax,
            cmap="inferno",
            vmin=0,
            vmax=0.60,
            markersize=POINT_SIZE_MAIN if is_point else None,
            linewidth=0,
            legend=False,
        )

        add_colorbar(
            fig,
            ax,
            cmap="inferno",
            vmin=0,
            vmax=0.60,
            label="Peak-month share",
            month_ticks=False
        )

    ax.set_xlim(bxmin, bxmax)
    ax.set_ylim(bymin, bymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Peak-month share", fontsize=11)
    ax.grid(False)
    add_panel_label(ax, "(b)")

    for _ax in axes:
        apply_publication_axis_style(_ax)

    fig.tight_layout()

    fig.savefig(
        OUTDIR / "Fig_peak_month_and_peak_month_share_1x2.png",
        dpi=FIG_DPI,
        bbox_inches="tight"
    )
    fig.savefig(
        OUTDIR / "Fig_peak_month_and_peak_month_share_1x2.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {OUTDIR / 'Fig_peak_month_and_peak_month_share_1x2.png'}")


def plot_combined_circular_2x2(gdf, admin=None):
    """
    Supplementary circular timing figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()

    bxmin, bymin, bxmax, bymax = get_projected_bbox()

    panels = [
        {
            "col": "circular_mean_month",
            "title": "Circular mean month",
            "cmap": make_peak_month_cmap(),
            "vmin": 1,
            "vmax": 12,
            "label": "Circular mean month",
            "month": True
        },
        {
            "col": "circular_concentration",
            "title": "Circular concentration",
            "cmap": "plasma",
            "vmin": 0,
            "vmax": 1,
            "label": "Circular concentration",
            "month": False
        },
        {
            "col": "peak_month",
            "title": "Unpeak month",
            "cmap": make_peak_month_cmap(),
            "vmin": 1,
            "vmax": 12,
            "label": "Peak month",
            "month": True
        },
        {
            "col": "peak_vs_circular_month_diff",
            "title": "Peak vs circular mean difference",
            "cmap": "magma_r",
            "vmin": 0,
            "vmax": 6,
            "label": "Month difference",
            "month": False
        },
    ]

    for ax, panel, lab in zip(axes, panels, ["(a)", "(b)", "(c)", "(d)"]):
        ax.set_facecolor(SEA_COLOR)
        draw_admin(ax, admin)

        col = panel["col"]
        gg = clean_gdf_for_plot(gdf, col)

        if len(gg) > 0:
            is_point = get_is_point(gg)

            gg.plot(
                column=col,
                ax=ax,
                cmap=panel["cmap"],
                vmin=panel["vmin"],
                vmax=panel["vmax"],
                markersize=POINT_SIZE_MAIN if is_point else None,
                linewidth=0,
                legend=False,
            )

            add_colorbar(
                fig,
                ax,
                cmap=panel["cmap"],
                vmin=panel["vmin"],
                vmax=panel["vmax"],
                label=panel["label"],
                month_ticks=panel["month"]
            )

        ax.set_xlim(bxmin, bxmax)
        ax.set_ylim(bymin, bymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(panel["title"], fontsize=11)
        ax.grid(False)
        add_panel_label(ax, lab)

    for _ax in axes:
        apply_publication_axis_style(_ax)

    fig.tight_layout()

    fig.savefig(
        OUTDIR / "FigS_monthly_circular_timing_2x2.png",
        dpi=FIG_DPI,
        bbox_inches="tight"
    )
    fig.savefig(
        OUTDIR / "FigS_monthly_circular_timing_2x2.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {OUTDIR / 'FigS_monthly_circular_timing_2x2.png'}")


# ============================================================
# 4. Load hex-period prediction data
# ============================================================

set_publication_style()

df = pd.read_csv(HEX_PERIOD_CSV)

print("\n============================================================")
print("Loaded hex-period prediction file")
print("============================================================")
print(HEX_PERIOD_CSV)
print(f"Rows: {len(df):,}")
print("Columns:")
for c in df.columns:
    print("  -", c)

if "hex_id" not in df.columns:
    raise ValueError("hex_period_summary_wall2wall_windonly_final.csv must contain hex_id.")

if DATE_COL not in df.columns:
    raise ValueError(f"DATE_COL={DATE_COL} not found.")

if PRED_COL not in df.columns:
    raise ValueError(f"PRED_COL={PRED_COL} not found.")

df["_date"] = pd.to_datetime(df[DATE_COL], errors="coerce")
df["_pred"] = pd.to_numeric(df[PRED_COL], errors="coerce")

df = df[df["_date"].notna() & df["_pred"].notna()].copy()

n_neg = (df["_pred"] < 0).sum()
n_gt1 = (df["_pred"] > 1).sum()

if n_neg > 0:
    warnings.warn(f"{n_neg:,} rows have negative predictions. They will be clipped to 0.")
if n_gt1 > 0:
    warnings.warn(f"{n_gt1:,} rows have predictions > 1. They will be clipped to 1.")

df["_pred"] = df["_pred"].clip(lower=0, upper=1)

df["year"] = df["_date"].dt.year
df["month"] = df["_date"].dt.month

df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

print("\nAfter filtering:")
print(f"Rows: {len(df):,}")
print(f"Hexes: {df['hex_id'].nunique():,}")
print(f"Years: {df['year'].min()} - {df['year'].max()}")
print(f"Mean predicted likelihood: {df['_pred'].mean():.6f}")


# ============================================================
# 5. Monthly likelihood profiles by hex
# ============================================================

hex_total = (
    df.groupby("hex_id", as_index=False)
      .agg(
          total_pred_mass=("_pred", "sum"),
          mean_pred_longterm=("_pred", "mean"),
          median_pred_longterm=("_pred", "median"),
          max_pred=("_pred", "max"),
          n_periods=("_pred", "size"),
      )
)

if "active_period" in df.columns:
    active_summary = (
        df.groupby("hex_id", as_index=False)
          .agg(
              n_active_periods=("active_period", "sum"),
              active_fraction=("active_period", "mean")
          )
    )
    hex_total = hex_total.merge(active_summary, on="hex_id", how="left")
else:
    hex_total["n_active_periods"] = hex_total["n_periods"]
    hex_total["active_fraction"] = 1.0


hex_month = (
    df.groupby(["hex_id", "month"], as_index=False)
      .agg(
          monthly_pred_sum=("_pred", "sum"),
          monthly_pred_mean=("_pred", "mean"),
          n_periods_month=("_pred", "size")
      )
)

all_hex = hex_total["hex_id"].unique()
full_idx = pd.MultiIndex.from_product(
    [all_hex, MONTHS],
    names=["hex_id", "month"]
)

hex_month = (
    hex_month
    .set_index(["hex_id", "month"])
    .reindex(full_idx)
    .reset_index()
)

hex_month["monthly_pred_sum"] = hex_month["monthly_pred_sum"].fillna(0)
hex_month["monthly_pred_mean"] = hex_month["monthly_pred_mean"].fillna(0)
hex_month["n_periods_month"] = hex_month["n_periods_month"].fillna(0)

hex_month["total_pred_sum_by_months"] = (
    hex_month.groupby("hex_id")["monthly_pred_sum"].transform("sum")
)

hex_month["total_pred_mean_profile"] = (
    hex_month.groupby("hex_id")["monthly_pred_mean"].transform("sum")
)

hex_month["monthly_mass_share"] = np.where(
    hex_month["total_pred_sum_by_months"] > 0,
    hex_month["monthly_pred_sum"] / hex_month["total_pred_sum_by_months"],
    np.nan
)

hex_month["monthly_intensity_share"] = np.where(
    hex_month["total_pred_mean_profile"] > 0,
    hex_month["monthly_pred_mean"] / hex_month["total_pred_mean_profile"],
    np.nan
)

if PROFILE_BASIS.lower() == "sum":
    hex_month["monthly_share"] = hex_month["monthly_mass_share"]
    SHARE_DESC = "monthly mass share"
elif PROFILE_BASIS.lower() == "mean":
    hex_month["monthly_share"] = hex_month["monthly_intensity_share"]
    SHARE_DESC = "monthly intensity share based on monthly mean likelihood"
else:
    raise ValueError("PROFILE_BASIS must be either 'mean' or 'sum'.")

hex_month.to_csv(OUTDIR / "hex_monthly_likelihood_profiles_long.csv", index=False)

print("\nMonthly profile basis:")
print(SHARE_DESC)


# ============================================================
# 6. Wide monthly table
# ============================================================

share_wide = (
    hex_month.pivot(index="hex_id", columns="month", values="monthly_share")
             .reset_index()
)

mean_wide = (
    hex_month.pivot(index="hex_id", columns="month", values="monthly_pred_mean")
             .reset_index()
)

sum_wide = (
    hex_month.pivot(index="hex_id", columns="month", values="monthly_pred_sum")
             .reset_index()
)

share_wide = share_wide.rename(
    columns={m: f"m{m:02d}_share" for m in MONTHS}
)

mean_wide = mean_wide.rename(
    columns={m: f"m{m:02d}_mean_pred" for m in MONTHS}
)

sum_wide = sum_wide.rename(
    columns={m: f"m{m:02d}_sum_pred" for m in MONTHS}
)

hex_month_wide = (
    share_wide
    .merge(mean_wide, on="hex_id", how="left")
    .merge(sum_wide, on="hex_id", how="left")
)

hex_month_wide.to_csv(OUTDIR / "hex_monthly_likelihood_profiles_wide.csv", index=False)


# ============================================================
# 7. Monthly timing and concentration metrics
# ============================================================

metric_rows = []

for hid, sub in hex_month.groupby("hex_id"):
    sub = sub.sort_values("month")

    shares = sub["monthly_share"].to_numpy(dtype=float)
    shares = np.where(np.isfinite(shares), shares, 0.0)

    if shares.sum() <= 0:
        peak_month = np.nan
        peak_month_share = np.nan
        monthly_hhi = np.nan
        effective_months = np.nan
        circular_mean_month = np.nan
        circular_concentration = np.nan
    else:
        shares = shares / shares.sum()

        peak_month = int(np.argmax(shares) + 1)
        peak_month_share = float(np.max(shares))

        monthly_hhi, effective_months = effective_months_from_shares(shares)
        circular_mean_month, circular_concentration = circular_mean_month_from_shares(shares)

    peak_vs_circular_diff = circular_month_difference(peak_month, circular_mean_month)

    metric_rows.append({
        "hex_id": hid,
        "profile_basis": PROFILE_BASIS,
        "peak_month": peak_month,
        "peak_month_name": month_name(peak_month),
        "peak_month_share": peak_month_share,
        "monthly_hhi": monthly_hhi,
        "effective_months": effective_months,
        "circular_mean_month": circular_mean_month,
        "circular_mean_month_name": month_name(circular_mean_month),
        "circular_concentration": circular_concentration,
        "peak_vs_circular_month_diff": peak_vs_circular_diff,
    })

hex_month_metrics = pd.DataFrame(metric_rows)

hex_month_metrics.to_csv(
    OUTDIR / "hex_monthly_timing_concentration_metrics.csv",
    index=False
)


# ============================================================
# 8. Combine all hex-level metrics
# ============================================================

hex_out = (
    hex_total
    .merge(hex_month_wide, on="hex_id", how="left")
    .merge(hex_month_metrics, on="hex_id", how="left")
)

# No peak-month mask.
# Keep these compatibility columns for older downstream scripts, but do not remove
# any hexes from the peak-month map.
mask_peak_reliable = hex_out["peak_month"].notna()
hex_out["peak_month_reliable"] = mask_peak_reliable.astype(int)
hex_out["peak_month_masked"] = hex_out["peak_month"]

hex_out.to_csv(
    OUTDIR / "hex_monthly_spatiotemporal_likelihood_metrics.csv",
    index=False
)

print("\n============================================================")
print("Saved hex-level monthly spatio-temporal metrics")
print("============================================================")
print(OUTDIR / "hex_monthly_spatiotemporal_likelihood_metrics.csv")
print(f"Rows: {len(hex_out):,}")

print("\nPeak-month valid-data summary:")
print(hex_out["peak_month_reliable"].value_counts(dropna=False))


# ============================================================
# 9. Summary tables
# ============================================================

summary_cols = [
    "total_pred_mass",
    "mean_pred_longterm",
    "peak_month",
    "peak_month_share",
    "effective_months",
    "circular_mean_month",
    "circular_concentration",
    "peak_vs_circular_month_diff",
    "active_fraction",
    "peak_month_reliable",
]

summary_rows = []

for c in summary_cols:
    if c not in hex_out.columns:
        continue

    vals = pd.to_numeric(hex_out[c], errors="coerce").dropna()

    if len(vals) == 0:
        continue

    summary_rows.append({
        "metric": c,
        "n": len(vals),
        "mean": vals.mean(),
        "sd": vals.std(),
        "p05": np.percentile(vals, 5),
        "p25": np.percentile(vals, 25),
        "median": np.percentile(vals, 50),
        "p75": np.percentile(vals, 75),
        "p95": np.percentile(vals, 95),
        "min": vals.min(),
        "max": vals.max(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    OUTDIR / "hex_monthly_spatiotemporal_metric_summary.csv",
    index=False
)

peak_month_counts = (
    hex_out["peak_month"]
    .dropna()
    .astype(int)
    .value_counts()
    .reindex(MONTHS, fill_value=0)
    .rename_axis("peak_month")
    .reset_index(name="n_hex")
)

peak_month_counts["month_name"] = peak_month_counts["peak_month"].apply(
    lambda m: MONTH_LABELS[m - 1]
)

peak_month_counts["share_hex"] = (
    peak_month_counts["n_hex"] / peak_month_counts["n_hex"].sum()
)

peak_month_counts.to_csv(
    OUTDIR / "hex_peak_month_frequency_unmasked.csv",
    index=False
)

peak_month_counts_masked = (
    hex_out.loc[hex_out["peak_month_reliable"] == 1, "peak_month"]
    .dropna()
    .astype(int)
    .value_counts()
    .reindex(MONTHS, fill_value=0)
    .rename_axis("peak_month")
    .reset_index(name="n_hex")
)

peak_month_counts_masked["month_name"] = peak_month_counts_masked["peak_month"].apply(
    lambda m: MONTH_LABELS[m - 1]
)

peak_month_counts_masked["share_hex"] = (
    peak_month_counts_masked["n_hex"] / peak_month_counts_masked["n_hex"].sum()
    if peak_month_counts_masked["n_hex"].sum() > 0 else np.nan
)

peak_month_counts_masked.to_csv(
    OUTDIR / "hex_peak_month_frequency_unmasked_copy.csv",
    index=False
)

print("\nMetric summary:")
print(summary_df)

print("\nUnmasked peak-month frequency:")
print(peak_month_counts)

print("\nPeak-month frequency copy for compatibility:")
print(peak_month_counts_masked)


# ============================================================
# 10. Merge with geometry and make maps
# ============================================================

if HAS_GEO:
    gdf_hex = prepare_hex_geometry(HEX_STYLEMATCH_CSV)

    if gdf_hex is not None:
        gdf = gdf_hex.merge(hex_out, on="hex_id", how="left")
        gdf = gdf[gdf["mean_pred_longterm"].notna()].copy()

        print("\n============================================================")
        print("Merged geometry with monthly metrics")
        print("============================================================")
        print(f"Rows: {len(gdf):,}")
        print(f"CRS: {gdf.crs}")
        print(f"Bounds: {gdf.total_bounds}")

        gpkg_path = OUTDIR / "hex_monthly_spatiotemporal_likelihood_metrics.gpkg"

        try:
            gdf.to_file(gpkg_path, driver="GPKG")
            print(f"Saved geopackage: {gpkg_path}")
        except Exception as e:
            print(f"Could not save GPKG: {e}")

        admin = load_admin_3035()

        # Match the publication-style footprint used in the spatial and temporal
        # overlap figures: bbox clip in lon/lat, administrative footprint filter,
        # and data-driven EPSG:3035 bounds.
        gdf = filter_main_europe_geometries(gdf)
        gdf = filter_hex_to_admin_footprint(gdf, admin)
        bounds = set_global_plot_bounds(gdf)

        print("\nPublication-style filtered map extent:")
        print(f"Rows after Europe/admin filtering: {len(gdf):,}")
        print(f"Bounds used for all maps: {bounds}")

        # Main two-panel figure only:
        #   (a) Peak month
        #   (b) Peak-month share
        plot_combined_monthly_2x2(gdf, admin=admin)

    else:
        print("\nNo geometry available. CSV metrics were saved, but maps were not generated.")

else:
    print("\nGeopandas/shapely unavailable. CSV metrics were saved, but maps were not generated.")


# ============================================================
# 11. Done
# ============================================================

print("\n============================================================")
print("All outputs saved to:")
print(OUTDIR)
print("============================================================")