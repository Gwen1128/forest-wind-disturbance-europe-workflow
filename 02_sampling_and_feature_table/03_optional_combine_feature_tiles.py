# -*- coding: utf-8 -*-
"""
Optional helper for tile-based runs.
Merges CSV files whose names match bfast*FINAL_ALL_FEATURES.csv into one feature table.
This is not needed when 02_extract_geospatial_features.py creates a single bfastglobal_FINAL_ALL_FEATURES.csv directly.
"""
import argparse
import os
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=os.environ.get("DP_FEATURE_TILE_FOLDER", r"E:\RF B+G"))
    ap.add_argument("--output", default=os.environ.get("DP_FINAL_FEATURES_CSV", r"E:\RF B+G\bfastglobal_FINAL_ALL_FEATURES.csv"))
    ap.add_argument("--pattern-prefix", default="bfast")
    ap.add_argument("--pattern-suffix", default="FINAL_ALL_FEATURES.csv")
    args = ap.parse_args()

    folder = Path(args.folder)
    csv_files = sorted(
        p for p in folder.iterdir()
        if p.name.startswith(args.pattern_prefix) and p.name.endswith(args.pattern_suffix)
    )
    print(f"Found {len(csv_files)} files")
    if not csv_files:
        raise FileNotFoundError(f"No matching feature CSVs found in {folder}")

    merged = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)

    key = ["x", "y", "event_time"]
    missing = [c for c in key if c not in merged.columns]
    if missing:
        raise ValueError(f"Merged CSV missing key columns: {missing}")

    dup = merged.duplicated(subset=key).sum()
    print(f"Duplicate rows by {key}: {dup}")

    if "index" in merged.columns:
        merged["event_time"] = pd.to_datetime(merged["event_time"], errors="coerce")
        merged = merged.sort_values(["index", "event_time"]).reset_index(drop=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()
