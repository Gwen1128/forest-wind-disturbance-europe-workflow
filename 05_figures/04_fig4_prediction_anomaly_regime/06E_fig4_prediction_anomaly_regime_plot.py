# -*- coding: utf-8 -*-
"""
START-FROM-SCRATCH combined Fig.4 + coarse spatial-quadrant threshold plot

Centre: Prague, lon = 14.4378, lat = 50.0755
Northwest: lon < centre_lon and lat >= centre_lat
Northeast: lon >= centre_lon and lat >= centre_lat
Southwest: lon < centre_lon and lat < centre_lat
Southeast: lon >= centre_lon and lat < centre_lat
"""

from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ============================================================
# PATHS
# ============================================================


MERGED_CSV = Path(os.environ.get(
    "FIG4_MERGED_CSV",
    r"E:\RF_BG_REPRO_from_model_dev\your_path\merged_prediction_anomaly_long_term.csv"
))

OUTDIR = Path(os.environ.get(
    "FIG4_QUADRANT_OUTDIR",
    os.environ.get(
        "FIG4_OUTDIR",
        r"E:\RF_BG_REPRO_from_model_dev\your_path\07_combined_abc_with_quadrant_smallmultiples"
    )
))

# ============================================================
#  SETTINGS
# ============================================================

TOP_PCTS = [5, 10, 20, 30]
N_BINS = 5
MIN_BIN_N = 10
DPI = 500

PLOT_CRS = "EPSG:3035"
LONLAT_CRS = "EPSG:4326"

CENTER_LON = 14.4378
CENTER_LAT = 50.0755

MEAN_COL = "mean_prob"
P95_COL = "p95_prob"
Z_COL = "z_anomaly"
FREQ_COL = "anomaly_frequency"
AREA_COL = "area_weight_km2"
CLASS_COL = "wind_regime"

REGION_COL = "spatial_quadrant"
REGION_ORDER = ["Northwest", "Northeast", "Southwest", "Southeast"]
REGION_TITLES = {r: r for r in REGION_ORDER}

CLASS_ORDER_ALL = [
    "Compound recurrent-intense",
    "Recurrent anomaly",
    "Intensity-dominated anomaly",
    "Background / mixed",
]
CLASS_ORDER_PLOTTED = [
    "Compound recurrent-intense",
    "Recurrent anomaly",
    "Intensity-dominated anomaly",
]
CLASS_LABELS = {
    "Compound recurrent-intense": "Compound",
    "Recurrent anomaly": "Frequency-dominated",
    "Intensity-dominated anomaly": "Intensity-dominated",
    "Background / mixed": "Background",
}
CLASS_COLORS = {
    "Compound recurrent-intense": "#7b3294",
    "Recurrent anomaly": "#2b8cbe",
    "Intensity-dominated anomaly": "#d7191c",
    "Background / mixed": "#7f7f7f",
}
CLASS_MARKERS = {
    "Compound recurrent-intense": "o",
    "Recurrent anomaly": "s",
    "Intensity-dominated anomaly": "^",
    "Background / mixed": "D",
}
METRIC_MAP = {
    "mean_prob": "Mean prediction",
    "p95_prob": "P95 prediction",
    "mean_prob_use": "Mean prediction",
    "p95_prob_use": "P95 prediction",
    "Mean probability": "Mean prediction",
    "P95 probability": "P95 prediction",
}

COLUMN_CANDIDATES = {
    "mean": ["mean_prob", "mean_prediction", "mean_pred", "pred_mean", "mean_probability"],
    "p95": ["p95_prob", "p95_prediction", "p95_pred", "pred_p95", "p95_probability"],
    "z": ["z_anomaly", "z_value", "max_z", "z_max", "standardized_anomaly_intensity", "anomaly_intensity"],
    "freq": ["anomaly_frequency", "frequency", "freq", "z_gt2_count", "frequency_z2", "anomaly_freq"],
    "area": ["area_weight_km2", "forest_area_km2", "hex_forest_area_km2", "area_km2", "forest_km2"],
    "class": ["wind_class", "wind_anomaly_class", "wind_regime", "wind_anomaly_regime", "regime"],
    "geometry": ["geometry", "geom", "wkt", "geometry_wkt", "hex_wkt"],
    "lon": ["lon", "longitude", "centroid_lon", "center_lon", "lon_4326", "x_lon"],
    "lat": ["lat", "latitude", "centroid_lat", "center_lat", "lat_4326", "y_lat"],
}


def pick_col(df, user_col, candidates, label, required=True):
    if user_col is not None:
        if user_col in df.columns:
            return user_col
        if required:
            raise ValueError(f"Specified {label} column not found: {user_col}\nAvailable columns:\n{list(df.columns)}")
        return None
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"Cannot detect {label} column. Tried: {candidates}\nAvailable columns:\n{list(df.columns)}")
    return None


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.sum(values[m] * weights[m]) / np.sum(weights[m]))


def safe_quantile(values, q):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.nanquantile(values, q))


def add_quantile_bins(df, col, n_bins, new_col):
    x = pd.to_numeric(df[col], errors="coerce")
    valid = np.isfinite(x)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if valid.sum() < n_bins:
        df[new_col] = out
        return df
    ranked = x[valid].rank(method="first")
    out.loc[valid] = pd.qcut(ranked, q=n_bins, labels=np.arange(1, n_bins + 1)).astype(float)
    df[new_col] = out
    return df


def normalize_class_labels(series):
    s = series.astype(str).str.strip()
    return s.replace({
        "Compound": "Compound recurrent-intense",
        "Compound recurrent–intense": "Compound recurrent-intense",
        "Compound recurrent-intense": "Compound recurrent-intense",
        "Recurrent": "Recurrent anomaly",
        "Recurrent anomaly": "Recurrent anomaly",
        "Intensity-dominated": "Intensity-dominated anomaly",
        "Intensity dominated": "Intensity-dominated anomaly",
        "Intensity-dominated anomaly": "Intensity-dominated anomaly",
        "Background": "Background / mixed",
        "Background/mixed": "Background / mixed",
        "Background / Mixed": "Background / mixed",
        "Background / mixed": "Background / mixed",
    })


def shared_twoslope_norm(values_a, values_b, center=1.0, min_dev=0.05):
    values = np.concatenate([np.asarray(values_a, dtype=float).ravel(), np.asarray(values_b, dtype=float).ravel()])
    values = values[np.isfinite(values)]
    max_dev = np.nanpercentile(np.abs(values - center), 95)
    max_dev = max(float(max_dev), min_dev)
    return TwoSlopeNorm(vmin=max(0, center - max_dev), vcenter=center, vmax=center + max_dev)


def load_merged_table():
    if not MERGED_CSV.exists():
        raise FileNotFoundError(f"Merged CSV not found: {MERGED_CSV}")
    df = pd.read_csv(MERGED_CSV).copy()

    mean_col = pick_col(df, MEAN_COL, COLUMN_CANDIDATES["mean"], "mean prediction")
    p95_col = pick_col(df, P95_COL, COLUMN_CANDIDATES["p95"], "p95 prediction")
    z_col = pick_col(df, Z_COL, COLUMN_CANDIDATES["z"], "standardized anomaly intensity")
    freq_col = pick_col(df, FREQ_COL, COLUMN_CANDIDATES["freq"], "anomaly frequency")
    area_col = pick_col(df, AREA_COL, COLUMN_CANDIDATES["area"], "area weight")
    class_col = pick_col(df, CLASS_COL, COLUMN_CANDIDATES["class"], "wind-anomaly class")

    geom_col = pick_col(df, None, COLUMN_CANDIDATES["geometry"], "geometry", required=False)
    lon_col = pick_col(df, None, COLUMN_CANDIDATES["lon"], "longitude", required=False)
    lat_col = pick_col(df, None, COLUMN_CANDIDATES["lat"], "latitude", required=False)

    for c in [mean_col, p95_col, z_col, freq_col, area_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["wind_anomaly_class"] = normalize_class_labels(df[class_col])

    valid = (
        np.isfinite(df[mean_col]) & np.isfinite(df[p95_col]) &
        np.isfinite(df[z_col]) & np.isfinite(df[freq_col]) &
        np.isfinite(df[area_col]) & (df[area_col] > 0) &
        df["wind_anomaly_class"].isin(CLASS_ORDER_ALL)
    )
    df = df.loc[valid].copy()

    if geom_col is not None:
        geom = df[geom_col].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
        gdf = gpd.GeoDataFrame(df, geometry=geom)
        xmin, ymin, xmax, ymax = gdf.total_bounds
        if (-180 <= xmin <= 180) and (-180 <= xmax <= 180) and (-90 <= ymin <= 90) and (-90 <= ymax <= 90):
            gdf = gdf.set_crs(LONLAT_CRS, allow_override=True)
        else:
            gdf = gdf.set_crs(PLOT_CRS, allow_override=True)
    elif lon_col is not None and lat_col is not None:
        gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])], crs=LONLAT_CRS)
    else:
        raise ValueError("Need geometry/WKT or lon/lat columns for spatial-quadrant assignment.")

    cols = {"mean": mean_col, "p95": p95_col, "z": z_col, "freq": freq_col, "area": area_col, "class": "wind_anomaly_class"}
    print("========== Loaded merged table ==========")
    print(MERGED_CSV)
    print(f"Valid rows: {len(gdf):,}")
    print("Columns used:")
    print(cols)
    return gdf, cols


def assign_spatial_quadrants(gdf):
    g = gdf.to_crs(LONLAT_CRS).copy()
    rep = g.geometry.representative_point()
    lon = rep.x
    lat = rep.y
    conditions = [
        (lon < CENTER_LON) & (lat >= CENTER_LAT),
        (lon >= CENTER_LON) & (lat >= CENTER_LAT),
        (lon < CENTER_LON) & (lat < CENTER_LAT),
        (lon >= CENTER_LON) & (lat < CENTER_LAT),
    ]
    choices = ["Northwest", "Northeast", "Southwest", "Southeast"]
    g[REGION_COL] = np.select(conditions, choices, default="Unassigned")
    g["rep_lon"] = lon
    g["rep_lat"] = lat
    g = g[g[REGION_COL].isin(REGION_ORDER)].copy()
    print("\n========== Spatial quadrant counts ==========")
    print(f"Centre lon/lat: {CENTER_LON}, {CENTER_LAT}")
    print(g[REGION_COL].value_counts().reindex(REGION_ORDER))
    return g


def response_surface(df, pred_col, cols):
    d = df.copy()
    d = add_quantile_bins(d, cols["freq"], N_BINS, "freq_bin")
    d = add_quantile_bins(d, cols["z"], N_BINS, "z_bin")
    global_mean = weighted_mean(d[pred_col], d[cols["area"]])
    rows = []
    for zb in range(1, N_BINS + 1):
        for fb in range(1, N_BINS + 1):
            m = (d["z_bin"] == zb) & (d["freq_bin"] == fb)
            n = int(m.sum())
            if n < MIN_BIN_N:
                lift = np.nan
            else:
                pred_mean = weighted_mean(d.loc[m, pred_col], d.loc[m, cols["area"]])
                lift = pred_mean / global_mean if np.isfinite(global_mean) and global_mean != 0 else np.nan
            rows.append({"z_bin": zb, "freq_bin": fb, "n": n, "prediction_lift": lift})
    out = pd.DataFrame(rows)
    mat = out.pivot(index="z_bin", columns="freq_bin", values="prediction_lift").sort_index(ascending=True)
    return out, mat


def class_enrichment(df, pred_col, cols, top_pct):
    threshold = safe_quantile(df[pred_col], 1 - top_pct / 100.0)
    top = df[pred_col] >= threshold
    total_w = df[cols["area"]].sum()
    top_w = df.loc[top, cols["area"]].sum()
    rows = []
    for cls in CLASS_ORDER_ALL:
        cls_mask = df[cols["class"]] == cls
        bg_share = df.loc[cls_mask, cols["area"]].sum() / total_w if total_w > 0 else np.nan
        top_share = df.loc[top & cls_mask, cols["area"]].sum() / top_w if top_w > 0 else np.nan
        rows.append({
            "prediction_metric": pred_col,
            "metric_label": METRIC_MAP.get(pred_col, pred_col),
            "top_pct": top_pct,
            "threshold": threshold,
            "class": cls,
            "background_area_share": bg_share,
            "top_area_share": top_share,
            "enrichment_ratio": top_share / bg_share if bg_share > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def calculate_all_metrics(df, cols):
    mean_surface, mean_mat = response_surface(df, cols["mean"], cols)
    p95_surface, p95_mat = response_surface(df, cols["p95"], cols)
    cont_parts = []
    for pct in TOP_PCTS:
        cont_parts.append(class_enrichment(df, cols["mean"], cols, pct))
        cont_parts.append(class_enrichment(df, cols["p95"], cols, pct))
    continental_enrichment = pd.concat(cont_parts, ignore_index=True)
    regional_parts = []
    for region in REGION_ORDER:
        d = df[df[REGION_COL] == region].copy()
        if len(d) == 0:
            continue
        for pct in TOP_PCTS:
            regional_parts.append(class_enrichment(d, cols["mean"], cols, pct).assign(**{REGION_COL: region}))
            regional_parts.append(class_enrichment(d, cols["p95"], cols, pct).assign(**{REGION_COL: region}))
    regional_enrichment = pd.concat(regional_parts, ignore_index=True)
    return mean_surface, p95_surface, mean_mat, p95_mat, continental_enrichment, regional_enrichment


def plotted_enrichment(df):
    d = df.copy()
    d["metric_label"] = d["metric_label"].replace(METRIC_MAP)
    return d[d["metric_label"].isin(["Mean prediction", "P95 prediction"]) & d["class"].isin(CLASS_ORDER_PLOTTED) & d["top_pct"].isin(TOP_PCTS)].copy()


def get_regional_ylim(df):
    d = plotted_enrichment(df)
    vals = d["enrichment_ratio"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 2.0
    return min(0.0, np.nanmin(vals) - 0.15), np.nanmax(vals) + 0.25


def plot_lift_surface(ax, mat, title, norm):
    im = ax.imshow(mat.values, origin="lower", aspect="auto", cmap="RdBu_r", norm=norm)
    n_bins = mat.shape[0]
    ax.set_xticks(np.arange(n_bins))
    ax.set_yticks(np.arange(n_bins))
    ax.set_xticklabels([f"Q{i}" for i in range(1, n_bins + 1)])
    ax.set_yticklabels([f"Q{i}" for i in range(1, n_bins + 1)])
    ax.set_xlabel("Anomaly frequency")
    ax.set_ylabel("Standardized anomaly intensity")
    ax.set_title(title, fontsize=12.2, pad=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    return im


def plot_continental_lines(ax, enrichment):
    d = plotted_enrichment(enrichment)
    for cls in CLASS_ORDER_PLOTTED:
        for metric, ls in [("Mean prediction", "-"), ("P95 prediction", "--")]:
            sub = d[(d["class"] == cls) & (d["metric_label"] == metric)].sort_values("top_pct")
            if len(sub) == 0:
                continue
            ax.plot(sub["top_pct"], sub["enrichment_ratio"], color=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls], linestyle=ls, linewidth=2.0, markersize=5.8)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.9)
    ax.set_xticks(TOP_PCTS)
    ax.set_xlabel("High-prediction threshold (%)")
    ax.set_ylabel("Enrichment ratio")
    ax.set_title("(c) Europe-wide enrichment of wind-anomaly classes", fontsize=12.2, pad=8)
    ax.set_ylim(0.35, 2.70)
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def add_combined_legend(ax):
    ax.axis("off")
    class_handles = [Line2D([0], [0], color=CLASS_COLORS[c], marker=CLASS_MARKERS[c], linewidth=2.0, markersize=6.0, label=CLASS_LABELS[c]) for c in CLASS_ORDER_PLOTTED]
    line_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="Mean prediction"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="P95 prediction"),
    ]
    handles = class_handles + [Line2D([], [], linestyle="none", label="")] + line_handles
    leg = ax.legend(
        handles=handles,
        loc="center left",
        frameon=True,
        title="Legend",
        title_fontsize=11.0,
        fontsize=10.2,
        handlelength=2.1,
        labelspacing=0.6,
        borderaxespad=0.0,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(1.0)
    leg.get_frame().set_edgecolor("white")
    leg.get_frame().set_linewidth(0.0)


def plot_regional_small_multiple(ax, df, metric, region, y_lim, show_ylabel=False, show_xlabel=False, title_text=None, dashed=False):
    d = plotted_enrichment(df)
    d = d[(d["metric_label"] == metric) & (d[REGION_COL] == region)].copy()
    ls = "--" if dashed else "-"
    for cls in CLASS_ORDER_PLOTTED:
        sub = d[d["class"] == cls].sort_values("top_pct")
        if len(sub) == 0:
            continue
        ax.plot(sub["top_pct"], sub["enrichment_ratio"], color=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls], linestyle=ls, linewidth=1.8, markersize=5.0)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(TOP_PCTS)
    ax.set_ylim(*y_lim)
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    if title_text is not None:
        ax.set_title(title_text, pad=6, fontsize=11.4)
    ax.set_ylabel(f"{metric}\nEnrichment ratio" if show_ylabel else "")
    ax.set_xlabel("High-prediction threshold (%)" if show_xlabel else "")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def make_figure(mean_mat, p95_mat, continental_enrichment, regional_enrichment, outdir):
    lift_norm = shared_twoslope_norm(mean_mat.values, p95_mat.values)
    y_lim = get_regional_ylim(regional_enrichment)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10.8,
        "axes.titlesize": 12.2,
        "axes.labelsize": 11.2,
        "xtick.labelsize": 10.2,
        "ytick.labelsize": 10.2,
        "legend.fontsize": 10.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(17.6, 10.4), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.08, 1.45], hspace=0.08)
    top = outer[0].subgridspec(1, 4, width_ratios=[1.18, 1.18, 1.20, 0.88], wspace=0.25)
    bottom = outer[1].subgridspec(3, 4, height_ratios=[0.12, 1.0, 1.0], hspace=0.0, wspace=0.0)

    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(top[0, 2])
    ax_leg = fig.add_subplot(top[0, 3])

    ax_dtitle = fig.add_subplot(bottom[0, :])
    ax_dtitle.axis("off")
    ax_dtitle.text(0.5, 0.45, "(d) Regional enrichment of wind-anomaly classes", ha="center", va="center", fontsize=12.6)

    axes_mid, axes_bot = [], []
    first_mid = fig.add_subplot(bottom[1, 0])
    first_bot = fig.add_subplot(bottom[2, 0], sharex=first_mid, sharey=first_mid)
    axes_mid.append(first_mid)
    axes_bot.append(first_bot)
    for i in range(1, 4):
        axes_mid.append(fig.add_subplot(bottom[1, i], sharex=first_mid, sharey=first_mid))
        axes_bot.append(fig.add_subplot(bottom[2, i], sharex=first_mid, sharey=first_mid))

    im_a = plot_lift_surface(ax_a, mean_mat, "(a) Europe-wide mean-prediction lift", lift_norm)
    im_b = plot_lift_surface(ax_b, p95_mat, "(b) Europe-wide p95-prediction lift", lift_norm)
    plot_continental_lines(ax_c, continental_enrichment)
    add_combined_legend(ax_leg)

    cax = inset_axes(ax_b, width="3.1%", height="82%", loc="center right", bbox_to_anchor=(0.06, 0.0, 1, 1), bbox_transform=ax_b.transAxes, borderpad=0)
    cb = fig.colorbar(im_b, cax=cax, orientation="vertical")
    cb.set_label("Prediction lift")

    for j, region in enumerate(REGION_ORDER):
        plot_regional_small_multiple(axes_mid[j], regional_enrichment, "Mean prediction", region, y_lim, show_ylabel=(j == 0), show_xlabel=False, title_text=REGION_TITLES[region], dashed=False)
        plot_regional_small_multiple(axes_bot[j], regional_enrichment, "P95 prediction", region, y_lim, show_ylabel=(j == 0), show_xlabel=(j == 1), title_text=None, dashed=True)

    for j, ax in enumerate(axes_mid):
        if j > 0:
            ax.tick_params(axis="y", labelleft=False)
            ax.set_ylabel("")
        ax.tick_params(axis="x", labelbottom=False)
    for j, ax in enumerate(axes_bot):
        if j > 0:
            ax.tick_params(axis="y", labelleft=False)
            ax.set_ylabel("")
        if j != 1:
            ax.set_xlabel("")

    fig.suptitle("Association between predicted wind disturbance probability and wind-anomaly classes. ", fontsize=14.6)

    out_png = outdir / "Fig_combined_abc_with_quadrant_smallmultiples.png"
    out_pdf = outdir / "Fig_combined_abc_with_quadrant_smallmultiples.pdf"
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("\nSaved figure:")
    print(out_png)
    print(out_pdf)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    gdf, cols = load_merged_table()
    gdf = assign_spatial_quadrants(gdf)
    gdf_out = gdf.copy()
    gdf_out["geometry"] = gdf_out.geometry.to_wkt()
    gdf_out.to_csv(OUTDIR / "fig4_merged_with_spatial_quadrants.csv", index=False)
    mean_surface, p95_surface, mean_mat, p95_mat, continental_enrichment, regional_enrichment = calculate_all_metrics(gdf, cols)
    mean_surface.to_csv(OUTDIR / "continental_mean_prediction_lift_surface.csv", index=False)
    p95_surface.to_csv(OUTDIR / "continental_p95_prediction_lift_surface.csv", index=False)
    continental_enrichment.to_csv(OUTDIR / "continental_class_enrichment_across_thresholds.csv", index=False)
    regional_enrichment.to_csv(OUTDIR / "quadrant_class_enrichment_across_thresholds.csv", index=False)
    plotted_enrichment(regional_enrichment).to_csv(OUTDIR / "plotted_quadrant_class_enrichment_subset.csv", index=False)
    make_figure(mean_mat, p95_mat, continental_enrichment, regional_enrichment, OUTDIR)
    print("\n========== Done ==========")
    print(f"Output folder: {OUTDIR}")


if __name__ == "__main__":
    main()
