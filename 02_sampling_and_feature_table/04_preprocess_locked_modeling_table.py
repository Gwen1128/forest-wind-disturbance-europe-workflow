# -*- coding: utf-8 -*-

import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ========= Paths: defaults can be overridden by CLI or environment variables =========
BASE = os.environ.get("DP_WORK_DIR", r"E:/RF B+G")
INPUT_CSV = os.environ.get("DP_FINAL_FEATURES_CSV", os.path.join(BASE, "bfastglobal_FINAL_ALL_FEATURES.csv"))
OUT_CSV   = os.environ.get("DP_CLEAN_CSV", os.path.join(BASE, "clean_encoded_data_LOCKED.csv"))
ENC_JSON  = os.environ.get("DP_ENCODERS_LOCKED_JSON", os.path.join(BASE, "encoders_LOCKED.json"))

# ========= Locked output columns =========
ID_COLS = ["sample_id", "lon", "lat", "label", "event_date", "ref_storm_id"]

FEATURES_NUM = [
    # core signal (weight-only)
    "LAI_departure",

    # wind (meteorology)
    "gust_peak_speed",
    "days_since_gust_peak",
    "gust_peak_percentile",

    # structure / terrain
    "slope",
    "aspect",
    "aspect_sin",
    "aspect_cos",
    "terrain_roughness",
    "tree_height",

    # cross (wind × structure interaction features)
    "windwardness",
    "exposure",
    "height_exposure",
    "roughness_exposure",
    "susceptibility",
]

FEATURES_CAT = ["landcover"]          # interpretation
FEATURES_ENC = ["landcover_enc"]      # modeling

LOCKED_COLS = ID_COLS + FEATURES_NUM + FEATURES_CAT + FEATURES_ENC

# ========= Column harmonization map =========
RENAME_MAP = {
    # ids
    "index": "sample_id",
    "x": "lon",
    "y": "lat",
    "event_time": "event_date",
    "event_date": "event_date",

    # label
    "Label": "label",
    "label": "label",

    # LAI departure
    "mag": "LAI_departure",
    "LAI_departure": "LAI_departure",

    # wind
    "max_wind_event": "gust_peak_speed",
    "max_wind_speed": "gust_peak_speed",
    "gust_peak_speed": "gust_peak_speed",

    "days_since_peak": "days_since_gust_peak",
    "days_since_gust_peak": "days_since_gust_peak",

    "wind_percentile": "gust_peak_percentile",
    "gust_percentile": "gust_peak_percentile",

    # terrain / structure
    "terrain_roughness": "terrain_roughness",
    "terrain_roughness_std5": "terrain_roughness_std5",
    "terrain_roughness_std3": "terrain_roughness_std3",

    "slope": "slope",
    "aspect": "aspect",
    "aspect_sin": "aspect_sin",
    "aspect_cos": "aspect_cos",

    "tree_height": "tree_height",

    # geometry
    "windwardness": "windwardness",

    # landcover
    "landcover": "landcover",
    "landcover_name": "landcover_name",

    # exposure candidates from upstream scripts
    "exposure": "exposure_raw",
    "exposure_signed": "exposure_signed",
    "exposure_intensity": "exposure_intensity",
}

EPS = 1e-6

def log(msg: str):
    print(msg, flush=True)

def pick_first_numeric(df, candidates):
    """Return first candidate column that exists and has any non-NaN numeric values."""
    for c in candidates:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None

def cv_target_encode(series: pd.Series, y: pd.Series, n_splits=5, smooth_k=50, random_state=42):
    """
    CV target encoding for a single categorical series.
    smooth_k: larger => stronger shrinkage to global mean for rare classes.

    Returns:
      enc (np.array float), mapping (dict), global_mean (float)
    """
    s = series.astype("object").fillna("__NA__")
    yb = y.astype(int)

    global_mean = float(yb.mean())

    # global mapping for deployment
    stats = pd.DataFrame({"cat": s, "y": yb}).groupby("cat")["y"].agg(["mean", "count"])
    m = global_mean
    stats["enc"] = (stats["mean"] * stats["count"] + m * smooth_k) / (stats["count"] + smooth_k)
    mapping = stats["enc"].to_dict()

    # OOF encoding
    enc = np.zeros(len(s), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, va in skf.split(np.zeros(len(s)), yb):
        s_tr, y_tr = s.iloc[tr], yb.iloc[tr]
        m_tr = float(y_tr.mean())
        st = pd.DataFrame({"cat": s_tr, "y": y_tr}).groupby("cat")["y"].agg(["mean", "count"])
        st["enc"] = (st["mean"] * st["count"] + m_tr * smooth_k) / (st["count"] + smooth_k)
        mp = st["enc"].to_dict()
        enc[va] = s.iloc[va].map(mp).fillna(m_tr).astype(float).values

    return enc, mapping, global_mean

def main():
    global INPUT_CSV, OUT_CSV, ENC_JSON
    parser = argparse.ArgumentParser(description="Create clean_encoded_data_LOCKED.csv with locked schema.")
    parser.add_argument("--input-csv", default=INPUT_CSV)
    parser.add_argument("--out-csv", default=OUT_CSV)
    parser.add_argument("--encoder-json", default=ENC_JSON)
    args = parser.parse_args()
    INPUT_CSV = args.input_csv
    OUT_CSV = args.out_csv
    ENC_JSON = args.encoder_json

    log(f"{INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)

    # rename
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    # required
    if "sample_id" not in df.columns:
        df["sample_id"] = np.arange(1, len(df) + 1, dtype=np.int64)

    for c in ["lon", "lat"]:
        if c not in df.columns:
            raise ValueError(f"Missing coordinate column: {c}")

    if "label" not in df.columns:
        raise ValueError("Missing label column: label (0/1)")

    if "event_date" not in df.columns:
        raise ValueError("Missing event_date column (event_time/event_date).")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    if df["event_date"].isna().all():
        raise ValueError("event_date parsing failed (all NaN).")

    # ========= LAI_departure =========
    if "LAI_departure" not in df.columns:
        df["LAI_departure"] = np.nan
    df["LAI_departure"] = pd.to_numeric(df["LAI_departure"], errors="coerce")
    df.loc[df["LAI_departure"] > 0, "LAI_departure"] = 0.0
    log("LAI_departure done")

    # ========= terrain_roughness =========
    c_rough = pick_first_numeric(df, ["terrain_roughness_std5", "terrain_roughness_std3", "terrain_roughness"])
    if c_rough is None:
        df["terrain_roughness"] = np.nan
    else:
        df["terrain_roughness"] = pd.to_numeric(df[c_rough], errors="coerce")

    # ========= exposure =========
    # Prefer intensity > signed > raw
    c_expo = pick_first_numeric(df, ["exposure_intensity", "exposure_signed", "exposure_raw"])
    if c_expo is None:
        df["exposure"] = np.nan
    else:
        df["exposure"] = pd.to_numeric(df[c_expo], errors="coerce")

    # ========= Ensure base numeric columns exist =========
    for c in ["gust_peak_speed", "days_since_gust_peak", "gust_peak_percentile",
              "windwardness", "slope", "aspect", "aspect_sin", "aspect_cos", "tree_height"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ========= Cross features =========
    df["height_exposure"] = df["tree_height"] * df["windwardness"]
    df["roughness_exposure"] = df["terrain_roughness"] * df["exposure"]
    df["susceptibility"] = df["tree_height"] / (df["terrain_roughness"].abs() + EPS)

    # ========= Landcover =========
    if "landcover" not in df.columns:
        if "landcover_name" in df.columns:
            df["landcover"] = df["landcover_name"]
        else:
            df["landcover"] = "__NA__"

    log("CV target encoding: landcover ...")
    enc, mapping, global_mean = cv_target_encode(df["landcover"], df["label"])
    df["landcover_enc"] = enc.astype(np.float32)

    # save encoder
    encoders = {
        "landcover": {
            "mapping": {str(k): float(v) for k, v in mapping.items()},
            "global_mean": float(global_mean),
            "smooth_k": 50,
        }
    }
    os.makedirs(os.path.dirname(ENC_JSON), exist_ok=True)
    with open(ENC_JSON, "w", encoding="utf-8") as f:
        json.dump(encoders, f, ensure_ascii=False, indent=2)
    log(f"Encoders saved to: {ENC_JSON}")

    # ========= Fill NA in numeric features =========
    for c in FEATURES_NUM:
        arr = df[c].values.astype(float)
        med = float(np.nanmedian(arr)) if np.isfinite(np.nanmedian(arr)) else 0.0
        n_nan = int(np.isnan(arr).sum())
        if n_nan > 0:
            df[c] = df[c].fillna(med)
            log(f" - Filled NaN in {c} with median {med:.4f}")

    df["landcover"] = df["landcover"].fillna("__NA__").astype(str)

    # ========= Lock schema & write =========
    out = df.copy()
    sid = pd.to_numeric(out["sample_id"], errors="coerce")
    fallback = pd.Series(np.arange(1, len(out) + 1), index=out.index)
    out["sample_id"] = sid.fillna(fallback).astype("int64")
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)

    out = out[LOCKED_COLS].copy()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    log(f"done")
    log("final colums：")
    log(str(list(out.columns)))

if __name__ == "__main__":
    main()
