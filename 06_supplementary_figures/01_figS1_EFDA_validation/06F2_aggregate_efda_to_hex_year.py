# -*- coding: utf-8 -*-
"""
Aggregate EFDA 30 m disturbance products to hex-year.

Supports:
- annual disturbance stack
- year-coded disturbance raster
- annual binary raster
- optional annual disturbance agent stack

Outputs:
- efda_hex_year_disturbance.csv
"""
from shapely import wkt
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import bounds as window_bounds
from shapely.geometry import box


# ============================================================
# USER SETTINGS
# ============================================================

EFDA_INVENTORY_CSV = Path(
    r"E:/EFDA_Zenodo_13333034/efda_file_inventory.csv"
)

HEX_GRID_PATH = Path(
    r"E:/RF_BG_REPRO_from_model_dev/outputs/04_fig1_spatial_indicators/hex_indicator_summary_stylematch_4326.csv"
)

OUTDIR = Path(
    r"E:/RF_BG_REPRO_from_model_dev/outputs/06F_efda_external_disturbance"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2003
END_YEAR = 2023

HEX_ID_COL = "hex_id"

# Use these if auto selection fails.
FORCE_DISTURBANCE_FILES = []
FORCE_AGENT_FILES = []

# Band-year mapping for annual stacks.
# EFDA covers 1985-2023. If band descriptions are unavailable,
# assume band 1 = 1985, band 2 = 1986, ...
STACK_START_YEAR = 1985

# Rasterization
ALL_TOUCHED = False

# Minimum valid 30m pixels per hex-year.
MIN_VALID_PIXELS = 10

# Agent codes from EFDA description:
# 1 wind/bark beetle complex, 2 fire, 3 harvest, 4 mixed
AGENT_WIND_BARK = 1
AGENT_FIRE = 2
AGENT_HARVEST = 3
AGENT_MIXED = 4


# ============================================================
# HELPERS
# ============================================================

def detect_year_from_filename(name):
    m = re.search(r"(19[8-9][0-9]|20[0-3][0-9])", name)
    if m:
        return int(m.group(1))
    return None


def select_files_from_inventory(inv):
    if FORCE_DISTURBANCE_FILES:
        disturbance_files = [Path(p) for p in FORCE_DISTURBANCE_FILES]
    else:
        candidates = inv[
            inv["layer_guess"].isin([
                "annual_disturbance_stack",
                "year_of_disturbance",
                "disturbance_other"
            ])
        ].copy()

        # Prefer annual stack over year-coded.
        stack = candidates[candidates["layer_guess"].eq("annual_disturbance_stack")]
        if len(stack) > 0:
            disturbance_files = [Path(p) for p in stack["path"].tolist()]
        else:
            disturbance_files = [Path(p) for p in candidates["path"].tolist()]

    if FORCE_AGENT_FILES:
        agent_files = [Path(p) for p in FORCE_AGENT_FILES]
    else:
        agent_files = [
            Path(p)
            for p in inv.loc[inv["layer_guess"].eq("agent"), "path"].tolist()
        ]

    if not disturbance_files:
        raise ValueError("No disturbance files selected. Check inventory CSV.")

    return disturbance_files, agent_files


def load_hex_grid(path, target_crs):
    """
    Load analyzed forest hexagons from either a vector file or a CSV with WKT geometry.

    The current final hex CSV:
    E:/RF_BG_REPRO_from_model_dev/outputs/04_fig1_spatial_indicators/
    hex_indicator_summary_stylematch_4326.csv

    contains projected polygon WKT geometries in EPSG:3035, despite the
    filename containing 4326. Therefore, CSV geometries are assigned EPSG:3035.
    """
    path = Path(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)

        # Detect geometry column.
        geom_col = None
        for c in ["geometry", "geom", "wkt", "WKT"]:
            if c in df.columns:
                geom_col = c
                break

        if geom_col is None:
            raise ValueError(
                f"Cannot find WKT geometry column in CSV. "
                f"Available columns: {list(df.columns)}"
            )

        # Detect or create hex id.
        if HEX_ID_COL not in df.columns:
            possible_ids = [
                "hex_id", "HEX_ID", "hex", "HEX", "id", "ID",
                "grid_id", "GRID_ID"
            ]
            found = None
            for c in possible_ids:
                if c in df.columns:
                    found = c
                    break

            if found is not None:
                df = df.rename(columns={found: HEX_ID_COL})
            else:
                warnings.warn(
                    f"No '{HEX_ID_COL}' column found. Creating hex_id from row index."
                )
                df[HEX_ID_COL] = np.arange(len(df)).astype(str)

        df = df[[HEX_ID_COL, geom_col]].copy()
        df = df[df[geom_col].notna()].copy()

        df["geometry"] = df[geom_col].apply(wkt.loads)

        gdf = gpd.GeoDataFrame(
            df[[HEX_ID_COL, "geometry"]],
            geometry="geometry",
            crs="EPSG:3035"
        )

    else:
        gdf = gpd.read_file(path)

        if HEX_ID_COL not in gdf.columns:
            raise ValueError(
                f"{HEX_ID_COL} not found in hex grid. "
                f"Available columns: {list(gdf.columns)}"
            )

        if gdf.crs is None:
            raise ValueError("Hex grid CRS is missing.")

        gdf = gdf[[HEX_ID_COL, "geometry"]].copy()

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    if gdf.crs is None:
        raise ValueError("Hex grid CRS is missing after loading.")

    gdf = gdf.to_crs(target_crs)

    gdf = gdf.reset_index(drop=True)
    gdf["_local_idx"] = np.arange(len(gdf), dtype=np.int32)
    gdf[HEX_ID_COL] = gdf[HEX_ID_COL].astype(str)

    print(f"Loaded hex grid: {len(gdf):,} hexagons")
    print(f"Hex CRS after projection: {gdf.crs}")

    return gdf


def init_output(hex_ids):
    years = np.arange(START_YEAR, END_YEAR + 1)

    idx = pd.MultiIndex.from_product(
        [hex_ids, years],
        names=["hex_id", "year"]
    )

    return pd.DataFrame(
        {
            "valid_px": np.zeros(len(idx), dtype=np.int64),
            "disturbed_px_all": np.zeros(len(idx), dtype=np.int64),
            "disturbed_px_agent_wind_bark": np.zeros(len(idx), dtype=np.int64),
            "disturbed_px_agent_fire": np.zeros(len(idx), dtype=np.int64),
            "disturbed_px_agent_harvest": np.zeros(len(idx), dtype=np.int64),
            "disturbed_px_agent_mixed": np.zeros(len(idx), dtype=np.int64),
        },
        index=idx
    )


def pixel_area_km2(src):
    if src.crs and src.crs.is_geographic:
        raise ValueError("EFDA raster is geographic. Reproject to EPSG:3035 first.")

    return abs(src.transform.a * src.transform.e) / 1e6


def infer_format(path, src):
    name = path.name.lower()

    if "stack" in name and src.count > 1:
        return "annual_stack"

    if src.count > 1:
        return "annual_stack"

    year = detect_year_from_filename(path.name)

    if year is not None:
        return "annual_binary"

    return "year_coded"


def band_year(src, band_idx):
    """
    Try band description first; otherwise assume STACK_START_YEAR.
    band_idx is 1-based.
    """
    desc = src.descriptions[band_idx - 1]

    if desc is not None:
        y = detect_year_from_filename(str(desc))
        if y is not None:
            return y

    tags = src.tags(band_idx)

    for v in tags.values():
        y = detect_year_from_filename(str(v))
        if y is not None:
            return y

    return STACK_START_YEAR + band_idx - 1


def rasterize_hexes(hex_win, shape, transform):
    shapes = [
        (geom, int(local_idx))
        for geom, local_idx in zip(hex_win.geometry, hex_win["_local_idx"])
        if geom is not None and not geom.is_empty
    ]

    if not shapes:
        return None

    return rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=-1,
        dtype="int32",
        all_touched=ALL_TOUCHED
    )


def add_counts(counts, local_to_hex, local_indices, year, column, n_hex):
    if year < START_YEAR or year > END_YEAR:
        return

    bc = np.bincount(local_indices, minlength=n_hex)
    active = np.where(bc > 0)[0]

    for li in active:
        hid = local_to_hex[li]
        counts.loc[(hid, year), column] += int(bc[li])


def find_matching_agent_file(dist_path, agent_files):
    """
    Try to find agent raster from the same country folder/name.
    """
    if not agent_files:
        return None

    dist_parts = set([p.lower() for p in dist_path.parts])

    best = None
    best_score = -1

    for af in agent_files:
        parts = set([p.lower() for p in af.parts])
        score = len(dist_parts & parts)

        if score > best_score:
            best = af
            best_score = score

    return best


# ============================================================
# MAIN
# ============================================================

inv = pd.read_csv(EFDA_INVENTORY_CSV)
disturbance_files, agent_files = select_files_from_inventory(inv)

print(f"Selected disturbance files: {len(disturbance_files)}")
print(f"Selected agent files: {len(agent_files)}")

with rasterio.open(disturbance_files[0]) as src0:
    target_crs = src0.crs

hex_gdf = load_hex_grid(HEX_GRID_PATH, target_crs)
hex_ids = hex_gdf[HEX_ID_COL].to_numpy()
local_to_hex = dict(zip(hex_gdf["_local_idx"].to_numpy(), hex_ids))

counts = init_output(hex_ids)

debug = []

for dist_path in disturbance_files:
    dist_path = Path(dist_path)

    if not dist_path.exists():
        warnings.warn(f"Missing disturbance file: {dist_path}")
        continue

    agent_path = find_matching_agent_file(dist_path, agent_files)

    print(f"\nProcessing disturbance: {dist_path.name}")
    if agent_path:
        print(f"  Matching agent: {Path(agent_path).name}")
    else:
        print("  No matching agent file.")

    with rasterio.open(dist_path) as src:
        fmt = infer_format(dist_path, src)
        px_area = pixel_area_km2(src)

        raster_bbox = box(*src.bounds)
        hex_sub = hex_gdf[hex_gdf.intersects(raster_bbox)].copy()

        if hex_sub.empty:
            print("  No hex overlap.")
            continue

        agent_src = None
        if agent_path is not None and Path(agent_path).exists():
            agent_src = rasterio.open(agent_path)

            if agent_src.crs != src.crs:
                warnings.warn("Agent raster CRS differs from disturbance raster. Skipping agent.")
                agent_src.close()
                agent_src = None

        for _, window in src.block_windows(1):
            w_bounds = window_bounds(window, src.transform)
            w_geom = box(*w_bounds)

            hex_win = hex_sub[hex_sub.intersects(w_geom)]
            if hex_win.empty:
                continue

            idx_arr = rasterize_hexes(
                hex_win,
                shape=(int(window.height), int(window.width)),
                transform=src.window_transform(window)
            )

            if idx_arr is None:
                continue

            inside = idx_arr >= 0

            if not inside.any():
                continue

            if fmt == "annual_stack":
                # Read all bands for this window.
                for band_idx in range(1, src.count + 1):
                    year = band_year(src, band_idx)

                    if year < START_YEAR or year > END_YEAR:
                        continue

                    arr = src.read(band_idx, window=window)

                    valid = inside & np.isfinite(arr)

                    if src.nodata is not None:
                        valid &= arr != src.nodata

                    # For annual stack, valid forest denominator is pixels with 0 or 1.
                    valid &= (arr == 0) | (arr == 1)

                    if not valid.any():
                        continue

                    li_valid = idx_arr[valid].astype(np.int32)
                    add_counts(
                        counts, local_to_hex, li_valid, year,
                        "valid_px", len(hex_gdf)
                    )

                    disturbed = valid & (arr == 1)

                    if disturbed.any():
                        li_dist = idx_arr[disturbed].astype(np.int32)
                        add_counts(
                            counts, local_to_hex, li_dist, year,
                            "disturbed_px_all", len(hex_gdf)
                        )

                        if agent_src is not None:
                            # If agent is annual stack, use same band index.
                            # If single-band, use band 1.
                            agent_band = band_idx if agent_src.count >= band_idx else 1
                            agent = agent_src.read(agent_band, window=window)

                            for agent_code, col in [
                                (AGENT_WIND_BARK, "disturbed_px_agent_wind_bark"),
                                (AGENT_FIRE, "disturbed_px_agent_fire"),
                                (AGENT_HARVEST, "disturbed_px_agent_harvest"),
                                (AGENT_MIXED, "disturbed_px_agent_mixed"),
                            ]:
                                m = disturbed & (agent == agent_code)
                                if m.any():
                                    li_agent = idx_arr[m].astype(np.int32)
                                    add_counts(
                                        counts, local_to_hex, li_agent, year,
                                        col, len(hex_gdf)
                                    )

            elif fmt == "annual_binary":
                year = detect_year_from_filename(dist_path.name)

                if year is None or year < START_YEAR or year > END_YEAR:
                    continue

                arr = src.read(1, window=window)

                valid = inside & np.isfinite(arr)
                if src.nodata is not None:
                    valid &= arr != src.nodata

                if not valid.any():
                    continue

                li_valid = idx_arr[valid].astype(np.int32)
                add_counts(counts, local_to_hex, li_valid, year, "valid_px", len(hex_gdf))

                disturbed = valid & (arr > 0)

                if disturbed.any():
                    li_dist = idx_arr[disturbed].astype(np.int32)
                    add_counts(
                        counts, local_to_hex, li_dist, year,
                        "disturbed_px_all", len(hex_gdf)
                    )

            elif fmt == "year_coded":
                arr = src.read(1, window=window)

                valid = inside & np.isfinite(arr)
                if src.nodata is not None:
                    valid &= arr != src.nodata

                # valid denominator repeated across years
                if valid.any():
                    li_valid = idx_arr[valid].astype(np.int32)
                    for year in range(START_YEAR, END_YEAR + 1):
                        add_counts(
                            counts, local_to_hex, li_valid, year,
                            "valid_px", len(hex_gdf)
                        )

                year_values = arr.astype("float64")

                disturbed = (
                    valid
                    & (year_values >= START_YEAR)
                    & (year_values <= END_YEAR)
                )

                if disturbed.any():
                    years_pix = year_values[disturbed].astype(int)
                    li_pix = idx_arr[disturbed].astype(np.int32)

                    for year in np.unique(years_pix):
                        m = years_pix == year
                        add_counts(
                            counts, local_to_hex, li_pix[m], int(year),
                            "disturbed_px_all", len(hex_gdf)
                        )

        if agent_src is not None:
            agent_src.close()

        debug.append(
            {
                "disturbance_file": str(dist_path),
                "agent_file": str(agent_path) if agent_path else "",
                "format": fmt,
                "pixel_area_km2": px_area,
                "overlapping_hexagons": len(hex_sub),
            }
        )

# Finalize.
out = counts.reset_index()
out["hex_id"] = out["hex_id"].astype(str)

# Use pixel area from last processed file; EFDA should have consistent 30 m grid in EPSG:3035.
out["forest_area_km2"] = out["valid_px"] * px_area

for col in [
    "disturbed_px_all",
    "disturbed_px_agent_wind_bark",
    "disturbed_px_agent_fire",
    "disturbed_px_agent_harvest",
    "disturbed_px_agent_mixed",
]:
    area_col = col.replace("px", "area_km2")
    rate_col = col.replace("disturbed_px", "disturbance_rate")

    out[area_col] = out[col] * px_area
    out[rate_col] = np.where(
        out["valid_px"] >= MIN_VALID_PIXELS,
        out[col] / out["valid_px"],
        np.nan
    )

out = out[out["valid_px"] >= MIN_VALID_PIXELS].copy()

out_csv = OUTDIR / "efda_hex_year_disturbance.csv"
out.to_csv(out_csv, index=False, encoding="utf-8-sig")

debug_csv = OUTDIR / "efda_aggregation_debug.csv"
pd.DataFrame(debug).to_csv(debug_csv, index=False, encoding="utf-8-sig")

print(f"\nSaved: {out_csv}")
print(f"Saved: {debug_csv}")
print("Done.")