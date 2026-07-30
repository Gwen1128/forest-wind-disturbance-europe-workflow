# -*- coding: utf-8 -*-
"""
Model-family comparison and wind-only model training for the paper-result reproduction workflow.
LOCKED schema training script for wind-disturbance PU learning.
"""

from __future__ import annotations

import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold  # sklearn >= 1.1
    HAS_SGKF = True
except Exception:
    HAS_SGKF = False

from joblib import dump

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm


# =========================
# Utils
# =========================
def safe_quantile(s, q):
    try:
        return float(pd.Series(s).quantile(q))
    except Exception:
        return np.nan


def compute_metrics(y_true, y_score, prefix=""):
    out = {}
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Defensive: remove non-finite scores
    m = np.isfinite(y_score)
    y_true = y_true[m]
    y_score = y_score[m]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        out[prefix + "roc_auc"] = np.nan
        out[prefix + "prauc"] = np.nan
        for k in [10, 50, 100, 200]:
            out[prefix + f"p@{k}"] = np.nan
        return out

    out[prefix + "roc_auc"] = roc_auc_score(y_true, y_score)
    out[prefix + "prauc"] = average_precision_score(y_true, y_score)

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    for k in [10, 50, 100, 200]:
        kk = min(k, len(y_sorted))
        out[prefix + f"p@{k}"] = float(np.mean(y_sorted[:kk])) if kk > 0 else np.nan
    return out


def compute_stratified_metrics_by_wind(y_true, y_score, wind_speed, quantile_bins, prefix=""):
    """Compute metrics within wind-speed strata defined by quantiles."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    wind_speed = np.asarray(wind_speed)

    m = np.isfinite(wind_speed) & np.isfinite(y_score)
    y_true = y_true[m]
    y_score = y_score[m]
    wind_speed = wind_speed[m]

    if len(y_true) == 0:
        return []

    qs = [float(q) for q in quantile_bins]
    edges = [np.quantile(wind_speed, q) for q in qs]

    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i < len(edges) - 2:
            idx = (wind_speed >= lo) & (wind_speed < hi)
        else:
            idx = (wind_speed >= lo) & (wind_speed <= hi)

        n = int(idx.sum())
        if n == 0:
            continue
        yb = y_true[idx]
        sb = y_score[idx]
        row = {
            "bin": f"q{qs[i]:.2f}-q{qs[i+1]:.2f}",
            "wind_lo": float(lo),
            "wind_hi": float(hi),
            "n": n,
            "pos": int((yb == 1).sum()),
        }
        row.update(compute_metrics(yb, sb, prefix=prefix))
        rows.append(row)
    return rows


def _clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1.0 - eps)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip_prob(p)
    return np.log(p / (1.0 - p))


def fit_glm_offset_binomial(
    X: pd.DataFrame,
    y: np.ndarray,
    offset: np.ndarray,
    sample_weight: np.ndarray | None = None
):
    """Fit Binomial GLM with logit link and an offset using statsmodels."""
    Xc = sm.add_constant(X, has_constant="add")
    if sample_weight is None:
        model = sm.GLM(y, Xc, family=sm.families.Binomial(), offset=offset)
        res = model.fit(maxiter=200, disp=0)
    else:
        sw = np.asarray(sample_weight, dtype=float)
        model = sm.GLM(y, Xc, family=sm.families.Binomial(), offset=offset, freq_weights=sw)
        res = model.fit(maxiter=200, disp=0)
    return res


def predict_glm_offset(res, X: pd.DataFrame, offset: np.ndarray) -> np.ndarray:
    Xc = sm.add_constant(X, has_constant="add")
    p = res.predict(Xc, offset=offset)
    return np.asarray(p, dtype=float)


def predict_glm_component(res, X: pd.DataFrame) -> np.ndarray:
    """Predict the multiplicative-odds component (no offset)."""
    Xc = sm.add_constant(X, has_constant="add")
    lp = np.asarray(Xc @ res.params, dtype=float)
    return 1.0 / (1.0 + np.exp(-lp))


# =========================
# Holdout split
# =========================
def split_holdout_bbox4326(df, bbox_str):
    # bbox_str: "minlon,minlat,maxlon,maxlat"
    minx, miny, maxx, maxy = [float(x) for x in bbox_str.split(",")]
    in_mask = ~(
        (df["lon"] >= minx) & (df["lon"] <= maxx) &
        (df["lat"] >= miny) & (df["lat"] <= maxy)
    )
    hold_mask = ~in_mask
    return in_mask, hold_mask


# =========================
# Shift-reweight (optional)
# =========================
def fit_shift_model(unlabeled_df, holdout_df, shift_mode="gbdt", clip_q=0.99, seed=20251017):
    """
    Domain classifier: IN vs HOLDOUT, to reweight IN samples to match HOLDOUT distribution.
    Uses numeric features only.
    """
    used_cols = [c for c in unlabeled_df.columns if pd.api.types.is_numeric_dtype(unlabeled_df[c])]
    X_in = unlabeled_df[used_cols].copy()
    X_ho = holdout_df[used_cols].copy()

    X = pd.concat([X_in, X_ho], axis=0)
    y = np.concatenate([np.zeros(len(X_in), dtype=int), np.ones(len(X_ho), dtype=int)])

    X = X.fillna(X.median(numeric_only=True))

    if shift_mode == "gbdt" and HAS_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
        )
    else:
        model = LogisticRegression(max_iter=2000, solver="lbfgs")

    model.fit(X, y)
    p = model.predict_proba(X_in)[:, 1]  # P(holdout | x) for IN points
    p = np.clip(p, 1e-6, 1 - 1e-6)

    w = p / (1.0 - p)

    cap = np.quantile(w, clip_q) if np.isfinite(np.quantile(w, clip_q)) else np.nan
    if np.isfinite(cap):
        w = np.minimum(w, cap)
    return w


# =========================
# Feature engineering (LOCKED)
# =========================
def add_derived_features(df, add_cross=False):
    EPS = 1e-6

    for c in [
        "LAI_departure", "gust_peak_speed", "days_since_gust_peak", "gust_peak_percentile",
        "windwardness", "slope", "exposure", "exposure_intensity", "terrain_roughness", "tree_height",
        "ref_storm_id"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Canonicalize exposure
    if "exposure" not in df.columns:
        if "exposure_intensity" in df.columns:
            df["exposure"] = df["exposure_intensity"]
        elif "exposure_signed" in df.columns:
            df["exposure"] = df["exposure_signed"]
        else:
            df["exposure"] = np.nan

    # susceptibility = tree_height / |terrain_roughness|
    if "susceptibility" not in df.columns and {"tree_height", "terrain_roughness"}.issubset(df.columns):
        denom = df["terrain_roughness"].abs() + EPS
        df["susceptibility"] = (df["tree_height"] / denom).clip(upper=1000)

    if add_cross:
        # wind_impact = gust_peak_speed * exposure (kept for compatibility; not used in locked cross_base below)
        if "wind_impact" not in df.columns and {"gust_peak_speed", "exposure"}.issubset(df.columns):
            df["wind_impact"] = df["gust_peak_speed"] * df["exposure"]

        if "height_exposure" not in df.columns and {"tree_height", "windwardness"}.issubset(df.columns):
            df["height_exposure"] = df["tree_height"] * df["windwardness"]

        if "roughness_exposure" not in df.columns and {"terrain_roughness", "exposure"}.issubset(df.columns):
            df["roughness_exposure"] = df["terrain_roughness"] * df["exposure"]


# =========================
# Models
# =========================
def fit_base_model(X, y, sample_weight=None, seed=20251017):
    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
        class_weight=None,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def fit_meta_model(X_meta, y, meta="logit", penalty="l2", C=1.0, seed=20251017):
    if meta == "logit":
        m = LogisticRegression(
            max_iter=5000,
            solver="lbfgs" if penalty == "l2" else "saga",
            penalty=penalty,
            C=float(C),
            random_state=seed,
        )
        m.fit(X_meta, y)
        return m
    raise ValueError(f"Unsupported meta model: {meta}")


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV (LOCKED schema)")
    ap.add_argument("--holdout_mode", default="bbox4326", choices=["none", "bbox4326"])
    ap.add_argument("--holdout_arg", default=None, help="minlon,minlat,maxlon,maxlat for bbox4326")
    ap.add_argument("--family_mode", default="grouped", choices=["grouped"])
    ap.add_argument("--meta", default="logit", choices=["logit"])
    ap.add_argument("--meta_penalty", default="l2", choices=["l2", "l1"])
    ap.add_argument("--meta_C_grid", default="0.01,0.1,1,3,10")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--meta_select_metric", default="prauc", choices=["prauc", "roc_auc"])

    # compatibility args (currently not used for subsampling logic)
    ap.add_argument("--neg_ratio", type=int, default=10)
    ap.add_argument("--neg_match_holdout", action="store_true")

    ap.add_argument("--shift_reweight", action="store_true")
    ap.add_argument("--shift_mode", default="gbdt", choices=["gbdt", "logit"])
    ap.add_argument("--shift_clip_q", type=float, default=0.99)

    ap.add_argument("--add_cross", action="store_true")
    ap.add_argument(
        "--families",
        default="wind,struct,cross",
        help="Comma-separated feature families to use: wind,struct,cross"
    )
    ap.add_argument(
        "--disable_residual",
        action="store_true",
        help="Disable residual model and exclude residual_logit from the meta learner."
    )

    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--output_preds", required=True)
    ap.add_argument("--output_model", required=True)

    # Residual gating (conditional vulnerability model)
    ap.add_argument("--residual_gate", action="store_true",
                    help="Train/evaluate residual model only on strong-wind subset.")
    ap.add_argument("--residual_gate_on", default="gust_peak_percentile",
                    choices=["gust_peak_percentile", "gust_peak_speed"],
                    help="Wind variable used for gating residual model.")
    ap.add_argument("--residual_gate_thr", type=float, default=0.90,
                    help="Threshold for residual gating (percentile in [0,1] or speed in m/s).")

    args = ap.parse_args()

    print(f"Reading data: {args.csv} ...")
    df = pd.read_csv(args.csv)

    must_cols = [
        "label", "lon", "lat", "LAI_departure", "gust_peak_speed",
        "days_since_gust_peak", "gust_peak_percentile", "windwardness",
        "slope", "aspect_sin", "aspect_cos", "terrain_roughness",
        "landcover_enc", "tree_height",
        "ref_storm_id"
    ]
    missing = [c for c in must_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["label"] = df["label"].astype(int)

    # Holdout split
    if args.holdout_mode == "bbox4326":
        if not args.holdout_arg:
            raise ValueError("--holdout_arg is required for bbox4326")
        in_mask, hold_mask = split_holdout_bbox4326(df, args.holdout_arg)
    else:
        in_mask = np.ones(len(df), dtype=bool)
        hold_mask = np.zeros(len(df), dtype=bool)

    df_in = df.loc[in_mask].reset_index(drop=True)
    df_ho = df.loc[hold_mask].reset_index(drop=True)

    pos_in = int((df_in["label"] == 1).sum())
    pos_ho = int((df_ho["label"] == 1).sum()) if len(df_ho) else 0
    print(f"[Split] IN pos={pos_in:,}, HOLDOUT pos={pos_ho:,}")

    # Derived features
    add_derived_features(df_in, add_cross=args.add_cross)
    add_derived_features(df_ho, add_cross=args.add_cross)

    # Residual gating masks
    def make_gate_mask(df0, on, thr):
        v = pd.to_numeric(df0[on], errors="coerce").to_numpy()
        m = np.isfinite(v)

        if on == "gust_peak_percentile":
            vmax = np.nanmax(v[m]) if m.any() else np.nan
            # auto-detect scale: [0,1] vs [0,100]
            if np.isfinite(vmax) and vmax > 1.5:
                # stored as 0-100
                thr_eff = float(thr) * 100.0 if float(thr) <= 1.0 else float(thr)
            else:
                # stored as 0-1
                thr_eff = float(thr) if float(thr) <= 1.0 else float(thr) / 100.0
            m = m & (v >= thr_eff)
        else:
            m = m & (v >= float(thr))

        return m


    gate_in = None
    gate_ho = None
    if args.residual_gate:
        if args.residual_gate_on not in df_in.columns:
            raise ValueError(f"--residual_gate_on {args.residual_gate_on} not found in df columns.")
        gate_in = make_gate_mask(df_in, args.residual_gate_on, args.residual_gate_thr)
        if len(df_ho):
            gate_ho = make_gate_mask(df_ho, args.residual_gate_on, args.residual_gate_thr)

        print(f"[residual-gate] ON: {args.residual_gate_on} >= {args.residual_gate_thr}")
        print(f"[residual-gate] IN gated n={int(gate_in.sum()):,} / {len(df_in):,}")
        if len(df_ho):
            print(f"[residual-gate] HOLDOUT gated n={int(gate_ho.sum()):,} / {len(df_ho):,}")

    # -------------------------------------------------
    # Structural validity mask for residual learning
    # -------------------------------------------------
    struct_valid_in = (
        (df_in["tree_height"] > 2) &
        (df_in["terrain_roughness"] > 0) &
        (df_in["exposure"].abs() > 0)
    )

    print(
        f"[residual-struct] valid n={int(struct_valid_in.sum()):,} / {len(df_in):,}"
    )

    # Base weights: abs(LAI_departure) clipped + epsilon
    w_in = df_in["LAI_departure"].abs().to_numpy()
    w_in = np.where(np.isfinite(w_in), w_in, np.nan)
    w_in = np.where(np.isnan(w_in), np.nanmedian(w_in), w_in)
    cap = np.nanquantile(w_in, 0.99) if np.isfinite(np.nanquantile(w_in, 0.99)) else np.nanmax(w_in)
    w_in = np.clip(w_in, 0.0, cap)
    w_in = w_in + 1e-3

    # Shift reweight (optional): apply only to IN unlabeled
    if args.shift_reweight and len(df_ho):
        unl = df_in[df_in["label"] == 0].copy()
        ho0 = df_ho[df_ho["label"] == 0].copy()
        drop_non_numeric = [c for c in unl.columns if not pd.api.types.is_numeric_dtype(unl[c])]
        print(f"[Shift] drop non-numeric cols for shift model: {drop_non_numeric}")

        w_shift = fit_shift_model(
            unlabeled_df=unl.drop(columns=drop_non_numeric, errors="ignore"),
            holdout_df=ho0.drop(columns=drop_non_numeric, errors="ignore"),
            shift_mode=args.shift_mode,
            clip_q=args.shift_clip_q,
        )
        w_in2 = w_in.copy()
        idx_unl = (df_in["label"] == 0).to_numpy()
        w_in2[idx_unl] = w_in2[idx_unl] * w_shift
        w_in = w_in2

    # Feature groups (LOCKED)
    feature_groups = {}

    wind_base = ["gust_peak_speed", "gust_peak_percentile"]
    group_wind = [c for c in wind_base if c in df_in.columns]
    if group_wind:
        feature_groups["wind"] = group_wind

    struct_base = ["slope", "terrain_roughness", "tree_height", "landcover_enc", "susceptibility"]
    group_struct = [c for c in struct_base if c in df_in.columns]
    if group_struct:
        feature_groups["struct"] = group_struct

    cross_base = ["exposure", "height_exposure", "roughness_exposure"]
    group_cross = [c for c in cross_base if c in df_in.columns]
    if group_cross:
        feature_groups["cross"] = group_cross

    requested_fams = [x.strip() for x in args.families.split(",") if x.strip()]
    allowed_fams = {"wind", "struct", "cross"}
    bad = [x for x in requested_fams if x not in allowed_fams]
    if bad:
        raise ValueError(f"Unknown families in --families: {bad}")

    feature_groups = {k: v for k, v in feature_groups.items() if k in requested_fams}
    if len(feature_groups) == 0:
        raise ValueError("No feature groups selected after --families filtering.")

    print("Feature groups:", json.dumps(feature_groups, indent=2, ensure_ascii=False))

    # Training targets and groups
    y_in = df_in["label"].to_numpy()
    groups = (df_in["lon"].round(6).astype(str) + "_" + df_in["lat"].round(6).astype(str)).to_numpy()

    # Residual CV grouping: leave-storm-out using ref_storm_id (recommended for vulnerability modeling)
    if "ref_storm_id" not in df_in.columns:
        raise ValueError("Missing required column: ['ref_storm_id'] for residual storm-group cross-validation")
    tmp_storm = df_in["ref_storm_id"].astype("string")
    # For missing storm ids, assign unique pseudo-ids to avoid accidental leakage across folds
    tmp_storm = tmp_storm.fillna(pd.Series(np.arange(len(df_in)), index=df_in.index).astype(str).radd("NA_"))
    groups_resid = tmp_storm.astype(str).to_numpy()


    n_splits = int(args.n_splits)
    if HAS_SGKF:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=20251017)
        def split_iter(X, y):
            return splitter.split(X, y, groups=groups)
        print(f"[CV] Using StratifiedGroupKFold with {n_splits} splits (group=lon,lat).")
    else:
        splitter = GroupKFold(n_splits=n_splits)
        def split_iter(X, y):
            return splitter.split(X, y, groups=groups)
        print(f"[CV] Using GroupKFold with {n_splits} splits (group=lon,lat). Note: class balance per fold may vary.")

    # Residual splitter: always leave-storm-out (GroupKFold on ref_storm_id)
    n_storms = int(pd.Series(groups_resid).nunique())
    n_splits_resid = min(n_splits, n_storms)
    if n_splits_resid < 2:
        splitter_resid = None
        print(f"[residual-CV] Skip: unique storms={n_storms} < 2, cannot run storm-group CV.")
    else:
        splitter_resid = GroupKFold(n_splits=n_splits_resid)
        print(f"[residual-CV] Using GroupKFold with {n_splits_resid} splits (group=ref_storm_id; unique storms={n_storms}).")

    def split_iter_resid(X, y):
        if splitter_resid is None:
            return []
        return splitter_resid.split(X, y, groups=groups_resid)


    # =========================
    # Stage A: OOF base models
    # =========================
    wind_strata_rows = []  # wind-speed stratified diagnostics rows

    base_oof = {}

    for fam, cols in feature_groups.items():
        X = df_in[cols].copy()
        X = X.fillna(X.median(numeric_only=True))

        oof_pred = np.zeros(len(df_in), dtype=float)

        for tr_idx, va_idx in split_iter(X, y_in):
            # leakage guard
            g_tr = set(groups[tr_idx])
            g_va = set(groups[va_idx])
            assert len(g_tr.intersection(g_va)) == 0, "Group leakage: same (lon,lat) in train and val!"

            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr = y_in[tr_idx]
            w_tr = w_in[tr_idx]

            m = fit_base_model(X_tr, y_tr, sample_weight=w_tr)
            oof_pred[va_idx] = m.predict_proba(X_va)[:, 1]

        base_oof[fam] = oof_pred

        met = compute_metrics(y_in, oof_pred, prefix=f"{fam}_")
        print(f"[{fam}] OOF performance:")
        print(f"  {fam}: ROC AUC={met[fam+'_roc_auc']:.4f} | PR AUC={met[fam+'_prauc']:.4f} | "
              f"P@10={met[fam+'_p@10']:.3f} P@50={met[fam+'_p@50']:.3f} "
              f"P@100={met[fam+'_p@100']:.3f} P@200={met[fam+'_p@200']:.3f}")

    
    # -------------------------------------------------
    # Residual vulnerability model (OOF; vulnerability component; struct+SAFE_CROSS | wind offset)
    # -------------------------------------------------
    fam_order = list(feature_groups.keys())

    if (not args.disable_residual) and ("wind" in base_oof) and len(base_oof["wind"]) == len(df_in):
        p_wind_oof = _clip_prob(base_oof["wind"])
        off_oof = _logit(p_wind_oof)

        resid_fams = [f for f in fam_order if f != "wind"]
        resid_cols = []
        for f in resid_fams:
            resid_cols.extend(feature_groups.get(f, []))
        resid_cols = [c for c in resid_cols if c in df_in.columns]

        if len(resid_cols) > 0:
            X_resid = df_in[resid_cols].copy()
            X_resid = X_resid.fillna(X_resid.median(numeric_only=True))

            resid_oof = np.full(len(df_in), np.nan, dtype=float)

            for tr_idx, va_idx in split_iter_resid(X_resid, y_in):
                g_tr = set(groups_resid[tr_idx])
                g_va = set(groups_resid[va_idx])
                assert len(g_tr.intersection(g_va)) == 0, "Storm leakage: same ref_storm_id in train and val!"

                # Apply structural validity mask (always), and wind gate if enabled
                if args.residual_gate:
                    tr_idx2 = tr_idx[(gate_in[tr_idx]) & (struct_valid_in[tr_idx])]
                    va_idx2 = va_idx[(gate_in[va_idx]) & (struct_valid_in[va_idx])]
                else:
                    tr_idx2 = tr_idx[struct_valid_in[tr_idx]]
                    va_idx2 = va_idx[struct_valid_in[va_idx]]

                if len(tr_idx2) < 50 or len(va_idx2) == 0:
                    continue

                res = fit_glm_offset_binomial(
                    X_resid.iloc[tr_idx2], y_in[tr_idx2],
                    offset=off_oof[tr_idx2],
                    sample_weight=w_in[tr_idx2],
                )

                # Vulnerability component only (no offset)
                p_vuln = predict_glm_component(res, X_resid.iloc[va_idx2])
                resid_oof[va_idx2] = _clip_prob(p_vuln)

            base_oof["residual"] = resid_oof

            valid = np.isfinite(resid_oof)
            tag = "gated" if args.residual_gate else "all"
            if valid.sum() >= 100 and len(np.unique(y_in[valid])) >= 2:
                met_res = compute_metrics(y_in[valid], resid_oof[valid], prefix="residual_")
                print(f"[residual] OOF performance ({tag}; vulnerability component; struct+SAFE_CROSS | wind offset):")
                print(f"  residual: ROC AUC={met_res['residual_roc_auc']:.4f} | PR AUC={met_res['residual_prauc']:.4f} | "
                      f"P@10={met_res['residual_p@10']:.3f} P@50={met_res['residual_p@50']:.3f} "
                      f"P@100={met_res['residual_p@100']:.3f} P@200={met_res['residual_p@200']:.3f}")
            else:
                print(f"[residual] OOF metrics skipped ({tag}): not enough valid samples or only one class.")
        else:
            print("[residual] OOF skip: no residual columns found (need struct/cross).")
    else:
        if args.disable_residual:
            print("[residual] OOF skipped by --disable_residual")
        else:
            print("[residual] OOF skip: wind family missing; cannot build offset residual model.")

# =========================
    # Meta learner tuning on TRUE OOF
    # =========================
    meta_feature_names = list(fam_order)
    meta_cols = [base_oof[f] for f in fam_order]

    # Add residual component to meta as a signed logit feature (lets meta learn direction)
    if (not args.disable_residual) and ("residual" in base_oof):
        resid_prob = np.where(np.isfinite(base_oof["residual"]), base_oof["residual"], 0.5)
        resid_logit = _logit(_clip_prob(resid_prob))
        meta_cols.append(resid_logit)
        meta_feature_names.append("residual_logit")

    X_meta = np.column_stack(meta_cols)

    meta_Cs = [float(x) for x in args.meta_C_grid.split(",")]
    best_C = meta_Cs[0]
    best_score = -np.inf

    print("[Stage A] Tuning meta-learner with TRUE OOF (stacking CV)...")

    def meta_oof_for_C(C: float) -> np.ndarray:
        oof = np.zeros(len(y_in), dtype=float)
        for tr_idx, va_idx in split_iter(X_meta, y_in):
            g_tr = set(groups[tr_idx])
            g_va = set(groups[va_idx])
            assert len(g_tr.intersection(g_va)) == 0, "Group leakage in meta CV!"

            m = fit_meta_model(X_meta[tr_idx], y_in[tr_idx], meta=args.meta, penalty=args.meta_penalty, C=C)
            oof[va_idx] = m.predict_proba(X_meta[va_idx])[:, 1]
        return oof

    for C in meta_Cs:
        meta_oof_tmp = meta_oof_for_C(C)
        met = compute_metrics(y_in, meta_oof_tmp, prefix="meta_")
        score = met["meta_prauc"] if args.meta_select_metric == "prauc" else met["meta_roc_auc"]
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_C = C

    print(f"[Stage A] Best meta C={best_C} by {args.meta_select_metric} (score={best_score:.6f})")

    meta_oof = meta_oof_for_C(best_C)
    met_meta = compute_metrics(y_in, meta_oof, prefix="meta_")
    print("[META] TRUE OOF performance:")
    print(f"  meta: ROC AUC={met_meta['meta_roc_auc']:.4f} | PR AUC={met_meta['meta_prauc']:.4f} | "
          f"P@10={met_meta['meta_p@10']:.3f} P@50={met_meta['meta_p@50']:.3f} "
          f"P@100={met_meta['meta_p@100']:.3f} P@200={met_meta['meta_p@200']:.3f}")

    # =========================
    # Stage B: refit on full IN
    # =========================
    print("\n[Stage B] Refitting models on full IN data for deployment...")
    base_models = {}
    for fam, cols in feature_groups.items():
        X = df_in[cols].copy()
        X = X.fillna(X.median(numeric_only=True))
        m = fit_base_model(X, y_in, sample_weight=w_in)
        base_models[fam] = {"model": m, "cols": cols}

    # residual full model (optional)
    residual_model_full = None
    residual_cols_full = []
    if (not args.disable_residual) and ("wind" in feature_groups):
        resid_fams = [f for f in fam_order if f != "wind"]
        for f in resid_fams:
            residual_cols_full.extend(feature_groups[f])
        residual_cols_full = [c for c in residual_cols_full if c in df_in.columns]

        if len(residual_cols_full) > 0 and ("wind" in base_oof):
            p_wind_oof_for_offset = _clip_prob(base_oof.get("wind", np.full(len(df_in), 0.5)))
            off_in = _logit(p_wind_oof_for_offset)

            Xr_in = df_in[residual_cols_full].copy()
            Xr_in = Xr_in.fillna(Xr_in.median(numeric_only=True))

            if args.residual_gate:
                idx = gate_in & struct_valid_in
            else:
                idx = struct_valid_in

            Xr_fit = Xr_in.loc[idx].copy()
            y_fit  = y_in[idx]
            off_fit = off_in[idx]
            w_fit  = w_in[idx]


            if len(y_fit) >= 50 and len(np.unique(y_fit)) >= 2:
                residual_model_full = fit_glm_offset_binomial(Xr_fit, y_fit, offset=off_fit, sample_weight=w_fit)
                tag = "gated" if args.residual_gate else "all"
                print(f"[residual] Fitted full-IN residual model ({tag}; struct+SAFE_CROSS | wind offset).")
            else:
                print("[residual] Full model skip: not enough gated samples or only one class.")
        else:
            print("[residual] Full model skip: no residual columns found (need struct/cross) or missing wind OOF.")
    else:
        if args.disable_residual:
            print("[residual] Full model skipped by --disable_residual")
        else:
            print("[residual] Full model skip: wind family missing; cannot build offset model.")

    # meta full model
    # meta full model (same feature order as Stage A)
    meta_cols_full = [base_models[f]["model"].predict_proba(
        df_in[base_models[f]["cols"]].fillna(df_in[base_models[f]["cols"]].median(numeric_only=True))
    )[:, 1] for f in fam_order]

    # Add residual_logit feature if used in Stage A
    if (not args.disable_residual) and ("residual_logit" in meta_feature_names):
        if residual_model_full is not None and len(residual_cols_full) > 0:
            Xr_in = df_in[residual_cols_full].copy()
            Xr_in = Xr_in.fillna(df_in[residual_cols_full].median(numeric_only=True))

            resid_in = _clip_prob(predict_glm_component(residual_model_full, Xr_in))

            # Apply structural validity mask (always) and wind gate if enabled
            resid_in = np.where(struct_valid_in.to_numpy(), resid_in, np.nan)
            if args.residual_gate:
                resid_in = np.where(gate_in, resid_in, np.nan)

            resid_in_prob = np.where(np.isfinite(resid_in), resid_in, 0.5)
            resid_in_logit = _logit(_clip_prob(resid_in_prob))
        else:
            # Fallback to neutral feature if residual model is unavailable
            resid_in_logit = np.zeros(len(df_in), dtype=float)

        meta_cols_full.append(resid_in_logit)

    X_meta_full = np.column_stack(meta_cols_full)
    meta_model_full = fit_meta_model(X_meta_full, y_in, meta=args.meta, penalty=args.meta_penalty, C=best_C)

    # =========================
    # Save OOF preds
    # =========================
    out_oof = df_in[["lon", "lat", "label"]].copy()
    for fam in feature_groups.keys():
        out_oof[f"oof_{fam}"] = base_oof[fam]
    if (not args.disable_residual) and ("residual" in base_oof):
        out_oof["oof_residual"] = base_oof["residual"]
    out_oof["oof_meta"] = meta_oof
    out_oof.to_csv(args.output_preds, index=False)
    print(f"Saved OOF predictions to {args.output_preds}")

    # =========================
    # Evaluation + metrics CSV
    # =========================
    metrics_rows = []
    metrics_rows.append({"split": "IN_OOF", **met_meta})

    if (not args.disable_residual) and ("residual" in base_oof):
        valid = np.isfinite(base_oof["residual"])
        if valid.sum() >= 100 and len(np.unique(y_in[valid])) >= 2:
            met_res_oof = compute_metrics(y_in[valid], base_oof["residual"][valid], prefix="residual_")
            metrics_rows.append({
                "split": "IN_OOF_RESIDUAL_GATED" if args.residual_gate else "IN_OOF_RESIDUAL",
                **met_res_oof
            })

    if len(df_ho):
        print(f"\n[Eval] Scoring HOLDOUT set ({len(df_ho)} samples)...")
        y_ho = df_ho["label"].to_numpy()

        ho_base = {}
        for fam in feature_groups.keys():
            cols = base_models[fam]["cols"]
            Xh = df_ho[cols].copy()
            Xh = Xh.fillna(df_in[cols].median(numeric_only=True))
            ho_base[fam] = base_models[fam]["model"].predict_proba(Xh)[:, 1]

        # (meta HOLDOUT prediction computed after residual component)

        # residual HOLDOUT scoring
        ho_residual = None
        p_full = None
        if (not args.disable_residual) and residual_model_full is not None and len(residual_cols_full) > 0 and "wind" in ho_base:
            off_ho = _logit(_clip_prob(ho_base["wind"]))

            Xr_ho = df_ho[residual_cols_full].copy()
            Xr_ho = Xr_ho.fillna(df_in[residual_cols_full].median(numeric_only=True))

            p_full = _clip_prob(predict_glm_offset(residual_model_full, Xr_ho, offset=off_ho))
            ho_residual = _clip_prob(predict_glm_component(residual_model_full, Xr_ho))


            # Structural validity mask (always applied to residual outputs)
            struct_valid_ho = (
                (pd.to_numeric(df_ho["tree_height"], errors="coerce") > 2).to_numpy() &
                (pd.to_numeric(df_ho["terrain_roughness"], errors="coerce") > 0).to_numpy() &
                (pd.to_numeric(df_ho["exposure"], errors="coerce").abs() > 0).to_numpy()
            )

            ho_residual = np.where(struct_valid_ho, ho_residual, np.nan)
            p_full = np.where(struct_valid_ho, p_full, np.nan)

            if args.residual_gate:
                ho_residual = np.where(gate_ho, ho_residual, np.nan)
                p_full = np.where(gate_ho, p_full, np.nan)

            valid_ho = np.isfinite(ho_residual)
            tag = "gated" if args.residual_gate else "all"
            if valid_ho.sum() >= 100 and len(np.unique(y_ho[valid_ho])) >= 2:
                met_ho_res = compute_metrics(y_ho[valid_ho], ho_residual[valid_ho], prefix="holdout_residual_")
                print(f"[residual] HOLDOUT performance ({tag}; vulnerability component; struct+SAFE_CROSS | wind offset):")
                print(f"  residual: ROC AUC={met_ho_res['holdout_residual_roc_auc']:.4f} | PR AUC={met_ho_res['holdout_residual_prauc']:.4f} | "
                      f"P@10={met_ho_res['holdout_residual_p@10']:.3f} P@50={met_ho_res['holdout_residual_p@50']:.3f} "
                      f"P@100={met_ho_res['holdout_residual_p@100']:.3f} P@200={met_ho_res['holdout_residual_p@200']:.3f}")
                metrics_rows.append({
                    "split": "HOLDOUT_RESIDUAL_GATED" if args.residual_gate else "HOLDOUT_RESIDUAL",
                    **met_ho_res
                })


                # Flip diagnostics (tests for cross-domain sign reversal; nearly free)
                met_ho_res_flip = compute_metrics(
                    y_ho[valid_ho],
                    (1.0 - ho_residual[valid_ho]),
                    prefix="holdout_residual_flip_",
                )
                print(f"[residual-flip] HOLDOUT performance ({tag}; vulnerability component FLIPPED): "
                      f"ROC AUC={met_ho_res_flip['holdout_residual_flip_roc_auc']:.4f} | "
                      f"PR AUC={met_ho_res_flip['holdout_residual_flip_prauc']:.4f} | "
                      f"P@10={met_ho_res_flip['holdout_residual_flip_p@10']:.3f} "
                      f"P@50={met_ho_res_flip['holdout_residual_flip_p@50']:.3f} "
                      f"P@100={met_ho_res_flip['holdout_residual_flip_p@100']:.3f} "
                      f"P@200={met_ho_res_flip['holdout_residual_flip_p@200']:.3f}")
                metrics_rows.append({
                    "split": "HOLDOUT_RESIDUAL_FLIP_GATED" if args.residual_gate else "HOLDOUT_RESIDUAL_FLIP",
                    **met_ho_res_flip
                })

                met_ho_res_full = compute_metrics(y_ho[valid_ho], p_full[valid_ho], prefix="holdout_residual_full_")
                metrics_rows.append({
                    "split": "HOLDOUT_RESIDUAL_FULL_GATED" if args.residual_gate else "HOLDOUT_RESIDUAL_FULL",
                    **met_ho_res_full
                })
            else:
                print(f"[residual] HOLDOUT metrics skipped ({tag}): not enough samples or only one class.")


        # meta HOLDOUT prediction (same feature order as Stage A)
        meta_cols_ho = [ho_base[f] for f in fam_order]

        if (not args.disable_residual) and ("residual_logit" in meta_feature_names):
            if (not args.disable_residual) and (ho_residual is not None):
                resid_ho_prob = np.where(np.isfinite(ho_residual), ho_residual, 0.5)
                resid_ho_logit = _logit(_clip_prob(resid_ho_prob))
            else:
                resid_ho_logit = np.zeros(len(df_ho), dtype=float)
            meta_cols_ho.append(resid_ho_logit)

        X_meta_ho = np.column_stack(meta_cols_ho)
        ho_meta = meta_model_full.predict_proba(X_meta_ho)[:, 1]

        # Wind-stratified diagnostics (HOLDOUT)
        if "gust_peak_speed" in df_ho.columns:
            wind_vals_ho = pd.to_numeric(df_ho["gust_peak_speed"], errors="coerce").to_numpy()
            qbins = [0.00, 0.50, 0.80, 0.95, 1.00]

            for fam, scores in ho_base.items():
                for r in compute_stratified_metrics_by_wind(y_ho, scores, wind_vals_ho, qbins, prefix=f"{fam}_"):
                    r.update({"split": "HOLDOUT", "model": fam})
                    wind_strata_rows.append(r)

            if (not args.disable_residual) and (ho_residual is not None):
                for r in compute_stratified_metrics_by_wind(y_ho, ho_residual, wind_vals_ho, qbins, prefix="residual_"):
                    r.update({"split": "HOLDOUT", "model": "residual"})
                    wind_strata_rows.append(r)

            for r in compute_stratified_metrics_by_wind(y_ho, ho_meta, wind_vals_ho, qbins, prefix="meta_"):
                r.update({"split": "HOLDOUT", "model": "meta"})
                wind_strata_rows.append(r)

        # Meta metrics (HOLDOUT)
        met_ho = compute_metrics(y_ho, ho_meta, prefix="holdout_")
        metrics_rows.append({"split": "HOLDOUT", **met_ho})

        # Save HOLDOUT preds
        out_ho = df_ho[["lon", "lat", "label"]].copy()
        for fam in fam_order:
            out_ho[f"pred_{fam}"] = ho_base[fam]
        if ho_residual is not None:
            out_ho["pred_residual"] = ho_residual
            out_ho["pred_residual_full"] = p_full
        out_ho["pred_meta"] = ho_meta

        holdout_path = os.path.splitext(args.output_preds)[0] + "_HOLDOUT.csv"
        out_ho.to_csv(holdout_path, index=False)
        print(f"Saved HOLDOUT predictions to {holdout_path}")

    # Save metrics
    pd.DataFrame(metrics_rows).to_csv(args.metrics_csv, index=False)
    print(f"Saved metrics to {args.metrics_csv}")

    # Save wind-stratified diagnostics
    if wind_strata_rows:
        diag_path = os.path.splitext(args.metrics_csv)[0] + "_wind_strata.csv"
        pd.DataFrame(wind_strata_rows).to_csv(diag_path, index=False)
        print(f"Saved wind-stratified diagnostics to {diag_path}")

    # Save model artifact
    artifact = {
        "feature_groups": feature_groups,
        "family_order": fam_order,
        "meta_feature_names": meta_feature_names,
        "base_models": base_models,
        "meta_model": meta_model_full,
        "residual_model": residual_model_full,
        "residual_cols": residual_cols_full,
        "meta_C": best_C,
        "schema_note": "LOCKED_cross6; exposure canonicalized from exposure_intensity if provided.",
        "residual_gate": bool(args.residual_gate),
        "residual_gate_on": args.residual_gate_on,
        "residual_gate_thr": float(args.residual_gate_thr),
        "families": requested_fams,
        "disable_residual": bool(args.disable_residual),
    }
    dump(artifact, args.output_model)
    print(f"Saved complete model artifact to {args.output_model}")


if __name__ == "__main__":
    main()
