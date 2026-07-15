# -*- coding: utf-8 -*-
"""
06F3_compare_prediction_with_efda_DECILE.py

Long-term spatial correspondence between modelled wind-disturbance probability
and EFDA annual wind/bark-beetle complex disturbance.

This version is intentionally simple and visual:
    Do higher long-term predicted-probability hexagons also show higher
    cumulative EFDA wind/bark-beetle disturbance rate?

It uses only long-term prediction deciles and EFDA wind/bark disturbance rate.

Main idea
---------
1. Aggregate 16-day modelled probabilities to hex-level long-term metrics:
   - pred_mean_long: mean prediction across 2003-2023
   - pred_p95_long : 95th-percentile prediction across 2003-2023

2. Aggregate EFDA annual wind/bark-beetle disturbance to hex-level cumulative
   disturbance rate:
   - efda_wind_bark_mean_annual_rate =
       sum(EFDA wind/bark disturbed area, 2003-2023) / hex forest area

3. Group hexagons into deciles of pred_mean_long and pred_p95_long.
4. Plot area-weighted EFDA wind/bark cumulative disturbance rate by prediction decile.

Outputs
-------
prediction_efda_wind_bark_longterm_hex_joined.csv
prediction_efda_wind_bark_decile_summary.csv
prediction_efda_wind_bark_decile_key_stats.csv
prediction_efda_wind_bark_deciles_pred_mean_long.png
prediction_efda_wind_bark_deciles_pred_p95_long.png
prediction_efda_wind_bark_deciles_combined.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# USER SETTINGS
# ============================================================

PRED_PERIOD_CSV = Path(
    r"E:/RF_BG_REPRO_from_model_dev/outputs/03_wall2wall_windonly/"
    r"hex_period_summary_wall2wall_windonly_final.csv"
)

EFDA_HEX_YEAR_CSV = Path(
    r"E:/RF_BG_REPRO_from_model_dev/outputs/06F_efda_external_disturbance/"
    r"efda_hex_year_disturbance.csv"
)

OUTDIR = Path(
    r"E:/RF_BG_REPRO_from_model_dev/outputs/06F_efda_external_disturbance/comparison_decile"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2003
END_YEAR = 2023

# Set manually if automatic detection fails.
PRED_HEX_COL = None
PRED_DATE_COL = None
PRED_PROB_COL = None

EFDA_HEX_COL = None
EFDA_YEAR_COL = None
EFDA_RATE_COL = None
EFDA_AREA_COL = None
EFDA_PIXEL_COL = None
EFDA_VALID_PIXEL_COL = None
EFDA_WEIGHT_COL = None

N_DECILES = 10
FIG_DPI = 300


# ============================================================
# HELPERS
# ============================================================

def find_col(df, candidates, required=True):
    """Find a column by exact or partial case-insensitive match."""
    lower = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]

    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c

    if required:
        raise ValueError(
            f"Cannot find column among {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def weighted_mean(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)

    if ok.sum() == 0:
        return np.nan

    return float(np.average(v[ok], weights=w[ok]))


def safe_ratio(a, b):
    if b is None or not np.isfinite(b) or b == 0:
        return np.nan
    return float(a / b)


def spearman_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)

    if ok.sum() < 3:
        return np.nan, np.nan

    r, p = spearmanr(x[ok], y[ok])
    return float(r), float(p)


# ============================================================
# PREDICTION
# ============================================================

def detect_prediction_columns(df):
    hex_col = PRED_HEX_COL or find_col(
        df, ["hex_id", "hex", "id", "grid_id", "h3"]
    )

    date_col = PRED_DATE_COL or find_col(
        df, ["date", "time", "period", "period_start", "start_date"]
    )

    prob_col = PRED_PROB_COL or find_col(
        df,
        [
            "pred_prob",
            "predicted_probability",
            "probability",
            "prob",
            "prediction",
            "pred",
            "p_mean",
            "mean_pred",
            "p",
        ],
    )

    return hex_col, date_col, prob_col


def aggregate_prediction_to_hex(path):
    df = pd.read_csv(path)
    hex_col, date_col, prob_col = detect_prediction_columns(df)

    print(f"Prediction hex column : {hex_col}")
    print(f"Prediction date column: {date_col}")
    print(f"Prediction prob column: {prob_col}")

    df = df[[hex_col, date_col, prob_col]].copy()
    df = df.rename(
        columns={
            hex_col: "hex_id",
            date_col: "date",
            prob_col: "pred_prob",
        }
    )

    df["hex_id"] = df["hex_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()

    df["year"] = df["date"].dt.year
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

    out = (
        df.groupby("hex_id", as_index=False)
        .agg(
            pred_mean_long=("pred_prob", "mean"),
            pred_p95_long=("pred_prob", lambda x: np.nanquantile(x, 0.95)),
            pred_max_long=("pred_prob", "max"),
            n_periods=("pred_prob", "count"),
        )
    )

    return out


# ============================================================
# EFDA WIND/BARK TARGET
# ============================================================

def detect_efda_columns(df):
    hex_col = EFDA_HEX_COL or find_col(
        df, ["hex_id", "hex", "id", "grid_id"]
    )

    year_col = EFDA_YEAR_COL or find_col(
        df, ["year"]
    )

    rate_col = EFDA_RATE_COL or find_col(
        df,
        [
            "disturbance_rate_agent_wind_bark",
            "rate_agent_wind_bark",
            "wind_bark_rate",
            "disturbance_rate_wind_bark",
        ],
        required=False,
    )

    area_col = EFDA_AREA_COL or find_col(
        df,
        [
            "disturbed_area_km2_agent_wind_bark",
            "area_km2_agent_wind_bark",
            "wind_bark_area_km2",
            "disturbed_area_wind_bark",
        ],
        required=False,
    )

    pixel_col = EFDA_PIXEL_COL or find_col(
        df,
        [
            "disturbed_px_agent_wind_bark",
            "px_agent_wind_bark",
            "wind_bark_px",
            "disturbed_pixels_agent_wind_bark",
        ],
        required=False,
    )

    valid_pixel_col = EFDA_VALID_PIXEL_COL or find_col(
        df,
        ["valid_px", "valid_pixels", "forest_px", "forest_pixels"],
        required=False,
    )

    weight_col = EFDA_WEIGHT_COL or find_col(
        df,
        ["forest_area_km2", "valid_area_km2", "area_km2"],
        required=False,
    )

    return hex_col, year_col, rate_col, area_col, pixel_col, valid_pixel_col, weight_col


def prepare_efda_hex_year(path):
    df = pd.read_csv(path)
    hex_col, year_col, rate_col, area_col, pixel_col, valid_pixel_col, weight_col = detect_efda_columns(df)

    print(f"EFDA hex column       : {hex_col}")
    print(f"EFDA year column      : {year_col}")
    print(f"EFDA wind/bark rate   : {rate_col}")
    print(f"EFDA wind/bark area   : {area_col}")
    print(f"EFDA wind/bark pixels : {pixel_col}")
    print(f"EFDA valid pixels     : {valid_pixel_col}")
    print(f"EFDA forest area      : {weight_col}")

    df = df.rename(columns={hex_col: "hex_id", year_col: "year"}).copy()
    df["hex_id"] = df["hex_id"].astype(str)
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

    # Forest area per hex-year.
    if weight_col is None:
        if valid_pixel_col is None:
            raise ValueError(
                "Cannot determine forest area. Set EFDA_WEIGHT_COL manually, "
                "or provide forest_area_km2 / valid_px."
            )
        df["forest_area_km2"] = pd.to_numeric(df[valid_pixel_col], errors="coerce") * 0.0009
    elif weight_col != "forest_area_km2":
        df["forest_area_km2"] = pd.to_numeric(df[weight_col], errors="coerce")
    else:
        df["forest_area_km2"] = pd.to_numeric(df["forest_area_km2"], errors="coerce")

    # Annual wind/bark area.
    if area_col is not None:
        df["efda_wind_bark_area_km2"] = pd.to_numeric(df[area_col], errors="coerce")
    elif pixel_col is not None:
        df["efda_wind_bark_area_km2"] = pd.to_numeric(df[pixel_col], errors="coerce") * 0.0009
    elif rate_col is not None:
        df["efda_wind_bark_area_km2"] = (
            pd.to_numeric(df[rate_col], errors="coerce") * df["forest_area_km2"]
        )
    else:
        raise ValueError(
            "Cannot determine EFDA wind/bark area. Set EFDA_AREA_COL manually, "
            "or provide disturbed_area_km2_agent_wind_bark / disturbed_px_agent_wind_bark / "
            "disturbance_rate_agent_wind_bark."
        )

    # Annual wind/bark rate, mainly for diagnostics.
    if rate_col is not None:
        df["efda_wind_bark_rate_annual"] = pd.to_numeric(df[rate_col], errors="coerce")
    else:
        df["efda_wind_bark_rate_annual"] = np.where(
            df["forest_area_km2"] > 0,
            df["efda_wind_bark_area_km2"] / df["forest_area_km2"],
            np.nan,
        )

    out = df[
        [
            "hex_id",
            "year",
            "forest_area_km2",
            "efda_wind_bark_area_km2",
            "efda_wind_bark_rate_annual",
        ]
    ].copy()

    return out


def aggregate_efda_to_hex(path):
    df = prepare_efda_hex_year(path)

    # Use median forest area because the same hex forest area is repeated each year.
    out = (
        df.groupby("hex_id", as_index=False)
        .agg(
            forest_area_km2=("forest_area_km2", "median"),
            forest_area_km2_years=("forest_area_km2", "sum"),
            efda_wind_bark_area_km2_total=("efda_wind_bark_area_km2", "sum"),
            efda_wind_bark_rate_annual_mean=("efda_wind_bark_rate_annual", "mean"),
            n_years=("year", "nunique"),
        )
    )

    # Mean annual disturbance rate across 2003–2023.
    out["efda_wind_bark_mean_annual_rate"] = np.where(
        out["forest_area_km2_years"] > 0,
        out["efda_wind_bark_area_km2_total"] / out["forest_area_km2_years"],
        np.nan,
    )

    return out


# ============================================================
# DECILE ANALYSIS
# ============================================================

def add_deciles(df, metric, n_deciles=10):
    out = df.copy()
    decile_col = f"{metric}_decile"

    valid = out[metric].replace([np.inf, -np.inf], np.nan).notna()
    out[decile_col] = np.nan

    # qcut can fail if there are many duplicate values; duplicates='drop' is safer.
    dec = pd.qcut(
        out.loc[valid, metric],
        q=n_deciles,
        labels=False,
        duplicates="drop",
    )

    out.loc[valid, decile_col] = dec.astype(float) + 1
    out[decile_col] = out[decile_col].astype("Int64")

    return out, decile_col


def summarize_deciles(df, metric):
    tmp, decile_col = add_deciles(df, metric, N_DECILES)

    records = []

    for decile, g in tmp.dropna(subset=[decile_col]).groupby(decile_col, sort=True):
        forest_area = float(np.nansum(g["forest_area_km2"]))
        efda_area = float(np.nansum(g["efda_wind_bark_area_km2_total"]))

        # Area-weighted cumulative EFDA disturbance rate for this prediction decile.
        efda_rate = safe_ratio(efda_area, forest_area)

        records.append(
            {
                "prediction_metric": metric,
                "decile": int(decile),
                "n_hex": int(len(g)),
                "forest_area_km2": forest_area,
                "mean_prediction": weighted_mean(g[metric], g["forest_area_km2"]),
                "efda_wind_bark_area_km2_total": efda_area,
                "efda_wind_bark_mean_annual_rate": efda_rate,
                "efda_wind_bark_mean_annual_rate_percent": efda_rate * 100 if np.isfinite(efda_rate) else np.nan,
            }
        )

    summary = pd.DataFrame(records)

    return summary


def key_stats(df, decile_summary, metric):
    use = df[[metric, "efda_wind_bark_mean_annual_rate", "forest_area_km2"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    spearman, spearman_p = spearman_safe(
        use[metric],
        use["efda_wind_bark_mean_annual_rate"],
    )

    sub = decile_summary[decile_summary["prediction_metric"] == metric].copy()
    d_low = sub[sub["decile"] == sub["decile"].min()]
    d_high = sub[sub["decile"] == sub["decile"].max()]

    low_rate = float(d_low["efda_wind_bark_mean_annual_rate"].iloc[0]) if len(d_low) else np.nan
    high_rate = float(d_high["efda_wind_bark_mean_annual_rate"].iloc[0]) if len(d_high) else np.nan

    return {
        "prediction_metric": metric,
        "spearman_hex_level": spearman,
        "spearman_p": spearman_p,
        "lowest_decile_efda_rate": low_rate,
        "highest_decile_efda_rate": high_rate,
        "lowest_decile_efda_rate_percent": low_rate * 100 if np.isfinite(low_rate) else np.nan,
        "highest_decile_efda_rate_percent": high_rate * 100 if np.isfinite(high_rate) else np.nan,
        "highest_vs_lowest_decile_ratio": safe_ratio(high_rate, low_rate),
        "n_hex": int(len(use)),
    }


# ============================================================
# PLOTS
# ============================================================

def plot_single_metric(decile_summary, metric):
    plot_df = decile_summary[decile_summary["prediction_metric"] == metric].copy()
    plot_df = plot_df.sort_values("decile")

    if plot_df.empty:
        return

    if metric == "pred_mean_long":
        title = "EFDA wind/bark disturbance across long-term mean prediction deciles"
        xlabel = "Long-term mean predicted probability decile"
    elif metric == "pred_p95_long":
        title = "EFDA wind/bark disturbance across long-term p95 prediction deciles"
        xlabel = "Long-term p95 predicted probability decile"
    else:
        title = f"EFDA wind/bark disturbance across {metric} deciles"
        xlabel = f"{metric} decile"

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    ax.plot(
        plot_df["decile"],
        plot_df["efda_wind_bark_mean_annual_rate_percent"],
        marker="o",
        linewidth=1.8,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("EFDA wind/bark cumulative disturbance rate (%)")
    ax.set_title(title)
    ax.set_xticks(plot_df["decile"])

    fig.tight_layout()

    out_png = OUTDIR / f"prediction_efda_wind_bark_deciles_{metric}.png"
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {out_png}")


def plot_combined(decile_summary):
    plot_df = decile_summary[
        decile_summary["prediction_metric"].isin(["pred_mean_long", "pred_p95_long"])
    ].copy()

    if plot_df.empty:
        return

    label_map = {
        "pred_mean_long": "Long-term mean prediction",
        "pred_p95_long": "Long-term p95 prediction",
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    for metric, g in plot_df.groupby("prediction_metric"):
        g = g.sort_values("decile")
        ax.plot(
            g["decile"],
            g["efda_wind_bark_mean_annual_rate_percent"],
            marker="o",
            linewidth=1.8,
            label=label_map.get(metric, metric),
        )

    ax.set_xlabel("Prediction decile")
    ax.set_ylabel("EFDA wind/bark cumulative disturbance rate (%)")
    ax.set_title("EFDA wind/bark disturbance across prediction deciles")
    ax.set_xticks(sorted(plot_df["decile"].unique()))
    ax.legend(frameon=False)

    fig.tight_layout()

    out_png = OUTDIR / "prediction_efda_wind_bark_deciles_combined.png"
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {out_png}")


# ============================================================
# MAIN
# ============================================================

print("Aggregating modelled predictions to long-term hex-level metrics...")
pred_hex = aggregate_prediction_to_hex(PRED_PERIOD_CSV)

print("\nAggregating EFDA wind/bark-beetle disturbance to long-term hex-level metrics...")
efda_hex = aggregate_efda_to_hex(EFDA_HEX_YEAR_CSV)

print("\nJoining prediction and EFDA hex-level tables...")
joined = pred_hex.merge(efda_hex, on="hex_id", how="inner")

joined_csv = OUTDIR / "prediction_efda_wind_bark_longterm_hex_joined.csv"
joined.to_csv(joined_csv, index=False, encoding="utf-8-sig")

print(f"Joined hexagons: {len(joined):,}")
print(f"Saved joined table: {joined_csv}")

metrics = ["pred_mean_long", "pred_p95_long"]

decile_tables = []
for metric in metrics:
    decile_tables.append(summarize_deciles(joined, metric))

decile_summary = pd.concat(decile_tables, ignore_index=True)

decile_csv = OUTDIR / "prediction_efda_wind_bark_decile_summary.csv"
decile_summary.to_csv(decile_csv, index=False, encoding="utf-8-sig")

print(f"Saved decile summary: {decile_csv}")

stats = pd.DataFrame([key_stats(joined, decile_summary, metric) for metric in metrics])

stats_csv = OUTDIR / "prediction_efda_wind_bark_decile_key_stats.csv"
stats.to_csv(stats_csv, index=False, encoding="utf-8-sig")

print(f"Saved key stats: {stats_csv}")

print("\nKey stats:")
print(stats.to_string(index=False))

for metric in metrics:
    plot_single_metric(decile_summary, metric)

plot_combined(decile_summary)

print("\nDone.")
