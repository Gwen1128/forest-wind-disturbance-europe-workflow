# -*- coding: utf-8 -*-
"""Build manuscript Table 1 from original ablation metrics_*.csv files."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

MODEL_FILES = [
    ("Wind", "metrics_wind.csv"),
    ("Wind+structure", "metrics_wind_struct.csv"),
    ("Wind+interaction", "metrics_wind_cross.csv"),
    ("Wind+structure+interaction", "metrics_wind_struct_cross.csv"),
]

METRIC_COLS = {
    "ROC AUC": "holdout_roc_auc",
    "PR AUC": "holdout_prauc",
    "P@50": "holdout_p@50",
    "P@100": "holdout_p@100",
    "P@200": "holdout_p@200",
}

def read_holdout_row(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "split" not in df.columns:
        raise ValueError(f"{path} has no split column. Columns: {list(df.columns)}")
    hit = df[df["split"].astype(str).str.upper() == "HOLDOUT"]
    if hit.empty:
        raise ValueError(f"{path} has no HOLDOUT row. Available splits: {df['split'].tolist()}")
    return hit.iloc[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    rows = []
    for model_name, fname in MODEL_FILES:
        row = read_holdout_row(metrics_dir / fname)
        out = {"Model": model_name}
        for out_col, src_col in METRIC_COLS.items():
            if src_col not in row.index:
                raise ValueError(f"{fname} missing {src_col}. Columns: {list(row.index)}")
            out[out_col] = float(row[src_col]) if pd.notna(row[src_col]) else np.nan
        rows.append(out)

    table = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
