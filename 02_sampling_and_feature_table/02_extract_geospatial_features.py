# -*- coding: utf-8 -*-
"""
Unified Geospatial Feature Extraction Pipeline
1. DEM (Elevation, Slope, Aspect) + Crop & Reproject
2. Local Wind Features (ERA5)
3. Windwardness, Exposure, Terrain Roughness
4. MODIS Landcover (Yearly)
5. Tree Height (Canopy)
6. WWF Biome/Ecozone

"""

import os
import glob
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from osgeo import gdal
import xarray as xr
import richdem as rd
from shapely.geometry import box
from rasterio.merge import merge
from rasterio.transform import rowcol, array_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform, transform as rio_transform
from scipy.spatial import cKDTree
from scipy.ndimage import generic_filter
from joblib import Parallel, delayed

# ==========================================
#               (CONFIG)
# ==========================================
CONFIG = {
    "input_initial_csv": r"E:/RF B+G/bfast_global.csv",  # 原始输入 CSV
    "work_dir": r"E:/RF B+G",                              # 工作目录 (用于存放中间和最终结果)
    
    # --- 步骤 1: DEM 配置 ---
    "dem_folder": r"E:/DEM tar",
    "dst_crs": "EPSG:3035",
    "target_resolution": 500,
    "dem_buffer_deg": 0.2,
    "dem_tif_filename": "dem_500m_epsg3035.tif",           # 将生成的中间 DEM 文件名
    "dem_bbox_wgs84": [-31.5, 23.75, 41.5, 71.2],          # 固定欧洲范围 [minx, miny, maxx, maxy]
    "dem_cutline_geojson": r"E:/RF B+G/europe_cutline_wgs84.geojson",
    "dem_src_nodata": -32768,
    "dem_dst_nodata": -9999.0,
    "gdal_warp_memory_mb": 2048,
    
    # --- 步骤 2: Wind (ERA5) 配置 ---
    "wind_nc_file": r"E:/ERA/ERA5 16 combine/EU_final_structural_wind_features.nc",
    "wind_window_mode": "days",   # "days" or "steps"
    "wind_window_days": 32,
    "wind_n_jobs": 8,
    
    # --- 步骤 3: Roughness 配置 ---
    "roughness_windows": (3, 5),
    "wind_dir_col": "struct_wind_dir_from_deg",
    "wind_max_col": "max_wind_event",
    
    # --- 步骤 4: Landcover (MODIS) 配置 ---
    "lc_dir": r"E:/MODISLandcover",
    "lc_pattern": "MODIS_LC_Type1_{year}.tif",
    "lc_year_min": 2003,
    "lc_year_max": 2022,
    "lc_drop_fill": True,
    
    # --- 步骤 5: Tree Height 配置 ---
    "tree_tif": r"E:/TREE_HEIGHT/canopy_height_2005_Europe.tif",
    
    # --- 步骤 6: Biome 配置 ---
    "biome_tif": r"E:/biome_wwf_ecoregion.tif",
    "encoders_json": r"E:/RF B+G/encoders.json",           # 最终编码字典路径
    
    # --- 最终输出文件名 ---
    "final_output_csv": r"E:/RF B+G/bfastglobal_FINAL_ALL_FEATURES.csv"
}

# 辅助：Biome Lookup Table
BIOME_LOOKUP = {
    1: "Tropical Moist Broadleaf Forests", 2: "Tropical Dry Broadleaf Forests",
    3: "Tropical Coniferous Forests", 4: "Temperate Broadleaf Forests",
    5: "Temperate Coniferous Forests", 6: "Boreal Forests/Taiga",
    7: "Tropical Grasslands/Savannas", 8: "Temperate Grasslands/Steppes",
    9: "Flooded Grasslands", 10: "Montane Grasslands", 11: "Tundra",
    12: "Mediterranean Forests", 13: "Deserts & Xeric Shrublands", 14: "Mangroves",
}
# 辅助：Landcover IGBP Map
IGBP_MAP = {
    0: "Water", 1: "Evergreen Needleleaf Forest", 2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest", 4: "Deciduous Broadleaf Forest", 5: "Mixed Forests",
    6: "Closed Shrublands", 7: "Open Shrublands", 8: "Woody Savannas", 9: "Savannas",
    10: "Grasslands", 11: "Permanent Wetlands", 12: "Croplands", 13: "Urban and Built-up",
    14: "Cropland/Natural Vegetation Mosaic", 15: "Permanent Snow and Ice",
    16: "Barren or Sparsely Vegetated", 17: "Unclassified", 255: "Fill",
}

# ==========================================
#           工具函数 (Utils)
# ==========================================
def update_encoders_json(path, new_dict):
    """更新 JSON 编码字典，保留已有内容"""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            old = {}
    else:
        old = {}
    
    old.update(new_dict)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=2)
    print(f"📄 Encoders JSON updated: {path}")

# ==========================================
#           Step 1: DEM (fixed Europe cutline + GDAL Warp)
# ==========================================
def _write_cutline_geojson(out_geojson, bbox_wgs84):
    minx, miny, maxx, maxy = bbox_wgs84
    polygon_coords = [[
        [minx, maxy],
        [maxx, maxy],
        [maxx, miny],
        [minx, miny],
        [minx, maxy]
    ]]
    fc = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "europe_cutline"},
            "geometry": {"type": "Polygon", "coordinates": polygon_coords}
        }]
    }
    os.makedirs(os.path.dirname(out_geojson), exist_ok=True)
    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(fc, f)


def step_1_dem(input_csv, output_csv, dem_out_tif):
    print("\n=== STEP 1: DEM Extraction & Processing (fixed Europe cutline) ===")

    df = pd.read_csv(input_csv)
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("CSV must contain x and y columns.")

    # 1) Normalize point coordinates to WGS84 for later point sampling
    x_sample = pd.to_numeric(df["x"], errors="coerce").mean()
    y_sample = pd.to_numeric(df["y"], errors="coerce").mean()
    source_crs = "EPSG:3035"   # 如原始输入点不是 3035，请改这里

    if abs(x_sample) > 180 or abs(y_sample) > 90:
        print(f"⚠️ Detected projected coordinates. Converting from {source_crs} to EPSG:4326...")
        lons, lats = rio_transform(source_crs, "EPSG:4326", df["x"].to_numpy(), df["y"].to_numpy())
        df["x"] = lons
        df["y"] = lats
        print("✅ Coordinate conversion complete. DataFrame updated to WGS84.")
    else:
        print("✅ Input coordinates already appear to be WGS84.")

    # 2) Build DEM directly with GDAL Warp using fixed Europe cutline
    dem_folder = CONFIG["dem_folder"]
    bbox_wgs84 = CONFIG["dem_bbox_wgs84"]
    cutline_geojson = CONFIG["dem_cutline_geojson"]
    dst_crs = CONFIG["dst_crs"]
    target_res = CONFIG["target_resolution"]
    src_nodata = CONFIG["dem_src_nodata"]
    dst_nodata = CONFIG["dem_dst_nodata"]
    warp_memory_mb = CONFIG["gdal_warp_memory_mb"]

    _write_cutline_geojson(cutline_geojson, bbox_wgs84)

    dem_files = glob.glob(os.path.join(dem_folder, "*.[bB][iI][lL]"))
    print(f"   Found {len(dem_files)} DEM tiles.")
    if len(dem_files) == 0:
        raise RuntimeError(f"No .bil files found in {dem_folder}")

    os.makedirs(os.path.dirname(dem_out_tif), exist_ok=True)

    warp_options = gdal.WarpOptions(
        format="GTiff",
        cutlineDSName=cutline_geojson,
        cropToCutline=True,
        cutlineSRS="EPSG:4326",
        dstSRS=dst_crs,
        xRes=target_res,
        yRes=target_res,
        resampleAlg="bilinear",
        srcNodata=src_nodata,
        dstNodata=dst_nodata,
        multithread=True,
        warpMemoryLimit=warp_memory_mb,
        creationOptions=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "BIGTIFF=YES"
        ]
    )

    ds = gdal.Warp(
        destNameOrDestDS=dem_out_tif,
        srcDSOrSrcDSTab=dem_files,
        options=warp_options
    )
    if ds is None:
        raise RuntimeError("GDAL Warp failed. Output DEM was not created.")
    ds = None
    print(f"✅ Projected DEM saved to: {dem_out_tif}")

    # 3) Read projected DEM and compute slope / aspect
    with rasterio.open(dem_out_tif) as src:
        dem_reproj = src.read(1).astype(np.float32)
        dem_reproj[dem_reproj == src.nodata] = np.nan
        dst_transform = src.transform
        dst_crs_read = src.crs

    rd_dem = rd.rdarray(dem_reproj.copy(), no_data=np.nan)
    rd_dem.geotransform = dst_transform.to_gdal()

    slope = np.array(rd.TerrainAttribute(rd_dem, attrib="slope_degrees"), dtype=np.float32)
    aspect = np.array(rd.TerrainAttribute(rd_dem, attrib="aspect"), dtype=np.float32)

    slope[(slope < 0) | (slope > 90)] = np.nan
    aspect[(aspect < 0) | (aspect > 360)] = np.nan

    # 4) Sample DEM / slope / aspect at point locations
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs="EPSG:4326"
    ).to_crs(dst_crs_read)

    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()

    rows, cols = rowcol(dst_transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    def safe_take(arr, r, c):
        ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
        out = np.full(r.shape, np.nan, dtype=np.float32)
        out[ok] = arr[r[ok], c[ok]]
        return out

    gdf["elevation"] = safe_take(dem_reproj, rows, cols)
    gdf["slope"] = safe_take(slope, rows, cols)
    gdf["aspect"] = safe_take(aspect, rows, cols)

    out_df = pd.DataFrame(gdf.drop(columns="geometry"))
    out_df.to_csv(output_csv, index=False)

    print(f"✅ STEP 1 finished. Output saved to: {output_csv}")

# ==========================================
#           步骤 2: Wind Features
# ==========================================
def step_2_wind(input_csv, output_csv):
    print("\n=== STEP 2: Wind Features (ERA5) ===")
    
    df = pd.read_csv(input_csv, parse_dates=["event_time"])
    if "index" not in df.columns:
        df["index"] = df.index 
        
    ds = xr.open_dataset(CONFIG["wind_nc_file"])
    gusts_arr = ds["max_wind_speed"].values
    dirs_arr = ds["wind_direction"].values
    time_axis = pd.to_datetime(ds["time"].values)
    
    # Build KDTree
    lon_vals, lat_vals = ds["longitude"].values, ds["latitude"].values
    grid_lon, grid_lat = np.meshgrid(lon_vals, lat_vals)
    tree = cKDTree(np.column_stack((grid_lon.ravel(), grid_lat.ravel())))
    
    _, idxs = tree.query(df[["x", "y"]].values, k=1)
    n_lon = len(lon_vals)
    df["lat_idx"] = idxs // n_lon
    df["lon_idx"] = idxs % n_lon

    # Extraction Function
    def extract_one(row):
        try:
            sample_id = int(row.index)
            t_obs = pd.to_datetime(row.event_time)
            lat_i, lon_i = int(row.lat_idx), int(row.lon_idx)
            
            # Time Mask
            if CONFIG["wind_window_mode"] == "days":
                t_start = t_obs - pd.Timedelta(days=CONFIG["wind_window_days"])
                time_mask = (time_axis >= t_start) & (time_axis <= t_obs)
            else: # steps
                # Simplified for brevity, usually 'days' is preferred
                time_mask = np.zeros(len(time_axis), dtype=bool) 
            
            if not time_mask.any(): return (sample_id, np.nan, np.nan, np.nan, np.nan)
            
            full_gusts = gusts_arr[:, lat_i, lon_i]
            event_gusts = full_gusts[time_mask]
            
            if event_gusts.size == 0 or np.all(np.isnan(event_gusts)):
                return (sample_id, np.nan, np.nan, np.nan, np.nan)
                
            peak_pos = int(np.nanargmax(event_gusts))
            max_val = float(event_gusts[peak_pos])
            
            # Time delta
            peak_time = pd.to_datetime(time_axis[time_mask][peak_pos])
            days_since = float((t_obs - peak_time) / np.timedelta64(1, "D"))
            
            # Percentile
            hist = full_gusts[~time_mask]
            hist = hist[np.isfinite(hist)]
            pct = float((hist <= max_val).mean() * 100.0) if hist.size else np.nan
            
            # Dir
            full_dirs = dirs_arr[:, lat_i, lon_i]
            peak_dir = float(full_dirs[time_mask][peak_pos])
            
            return (sample_id, max_val, round(days_since, 1), round(pct, 2), round(peak_dir, 1))
        except:
            return (int(row.index), np.nan, np.nan, np.nan, np.nan)

    # Parallel Execution
    results = Parallel(n_jobs=CONFIG["wind_n_jobs"], prefer="threads")(
        delayed(extract_one)(row) for row in df.itertuples(index=False)
    )
    
    cols = ["index", "max_wind_event", "days_since_peak", "wind_percentile", "struct_wind_dir_from_deg"]
    res_df = pd.DataFrame(results, columns=cols)
    
    df_out = df.merge(res_df, on="index", how="left").drop(columns=["lon_idx", "lat_idx"], errors="ignore")
    df_out.to_csv(output_csv, index=False)
    print(f"✅ Step 2 Done. Saved: {output_csv}")
    return output_csv

# ==========================================
#           修复版 Step 3: 强制整数索引
# ==========================================
def step_3_roughness(input_csv, output_csv, dem_tif):
    print("\n=== STEP 3: Exposure & Roughness (Int Fix) ===")
    
    df = pd.read_csv(input_csv)
    
    # 1. Geometry Derived
    wind_dir = np.deg2rad(df[CONFIG["wind_dir_col"]].astype(float))
    aspect = np.deg2rad(df["aspect"].astype(float))
    slope = np.deg2rad(np.clip(df["slope"].astype(float), 0, 90))
    
    windwardness = np.cos(aspect - wind_dir)
    exposure = np.maximum(0.0, windwardness) * np.sin(slope)
    
    # Flat fix
    flat = df["slope"] < 1.0
    windwardness[flat] = 0.0
    exposure[flat] = 0.0
    
    df["windwardness"] = windwardness
    df["exposure"] = exposure
    if CONFIG["wind_max_col"] in df.columns:
        df["exposure_intensity"] = df["exposure"] * df[CONFIG["wind_max_col"]]

    # 2. DEM Roughness
    with rasterio.open(dem_tif) as src:
        dem = src.read(1).astype(float)
        dem[dem == src.nodata] = np.nan
        
        # 准备采样坐标 (WGS84 -> Projected)
        xs, ys = rio_transform("EPSG:4326", src.crs, df.x.values, df.y.values)
        
        # 获取行列号 (float)
        rows_f, cols_f = rowcol(src.transform, xs, ys, op=np.floor)
        
        # ⚡️ 核心修复：Clip 之后必须强制转为 int ⚡️
        h, w = dem.shape
        rows = np.clip(rows_f, 0, h-1).astype(int)
        cols = np.clip(cols_f, 0, w-1).astype(int)

        # Calculate roughness maps and sample
        for win in CONFIG["roughness_windows"]:
            print(f"   Computing roughness std window {win}...")
            # 计算全图 roughness
            r_map = generic_filter(dem, lambda v: np.std(v) if v.size > 0 else np.nan, size=win, mode="nearest")
            
            # 使用整数索引提取
            df[f"terrain_roughness_std{win}"] = r_map[rows, cols]
            
            # 顺便提取 500m 海拔 (只做一次)
            if win == CONFIG["roughness_windows"][0]:
                 df["elev_500m"] = dem[rows, cols]

    df.to_csv(output_csv, index=False)
    print(f"✅ Step 3 Done. Saved: {output_csv}")
    return output_csv
# ==========================================
#           步骤 4: Landcover
# ==========================================
def step_4_landcover(input_csv, output_csv):
    print("\n=== STEP 4: MODIS Landcover ===")
    
    df = pd.read_csv(input_csv)
    # Find Date Col
    date_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    date_col = date_cols[0] if date_cols else "event_time"
    print(f"   Using date column: {date_col}")
    
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = df[date_col].dt.year
    df["landcover"] = np.nan
    
    years = df["year"].dropna().unique().astype(int)
    valid_years = [y for y in years if CONFIG["lc_year_min"] <= y <= CONFIG["lc_year_max"]]
    
    for year in valid_years:
        fname = CONFIG["lc_pattern"].format(year=year)
        path = os.path.join(CONFIG["lc_dir"], fname)
        if not os.path.exists(path):
            print(f"   ⚠️ Missing LC file for {year}")
            continue
            
        mask = df["year"] == year
        sub = df[mask]
        
        with rasterio.open(path) as src:
            xs, ys = rio_transform("EPSG:4326", src.crs, sub.x.values, sub.y.values)
            vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=float)
            
            # Filter
            if CONFIG["lc_drop_fill"]:
                vals[vals == 255] = np.nan
            
            df.loc[mask, "landcover"] = vals
            
    # Encoding
    df["landcover_name"] = df["landcover"].map(lambda x: IGBP_MAP.get(int(x), None) if pd.notna(x) else None)
    df["landcover_enc"] = df["landcover"] # already numeric
    
    # Update JSON
    lc_json = {
        "landcover_name_to_code": {v: k for k, v in IGBP_MAP.items()},
        "landcover_code_to_name": {str(k): v for k, v in IGBP_MAP.items()}
    }
    update_encoders_json(CONFIG["encoders_json"], lc_json)
    
    df.to_csv(output_csv, index=False)
    print(f"✅ Step 4 Done. Saved: {output_csv}")
    return output_csv

# ==========================================
#           步骤 5: Tree Height
# ==========================================
def step_5_treeheight(input_csv, output_csv):
    print("\n=== STEP 5: Tree Height ===")
    
    df = pd.read_csv(input_csv)
    with rasterio.open(CONFIG["tree_tif"]) as src:
        arr = src.read(1)
        nodata = src.nodata
        
        # Transform coords
        xs, ys = rio_transform("EPSG:4326", src.crs, df.x.values, df.y.values)
        rows, cols = rowcol(src.transform, xs, ys)
        
        # Valid bounds
        h, w = arr.shape
        rows = np.array(rows)
        cols = np.array(cols)
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        
        vals = np.full(len(df), np.nan)
        # Only read valid indices
        r_valid, c_valid = rows[valid], cols[valid]
        raw_vals = arr[r_valid, c_valid]
        
        # Handle nodata
        raw_vals = np.where(raw_vals == nodata, np.nan, raw_vals)
        vals[valid] = raw_vals
        
        df["tree_height"] = vals

    df.to_csv(output_csv, index=False)
    print(f"✅ Step 5 Done. Saved: {output_csv}")
    return output_csv

# ==========================================
#           步骤 6: Biome
# ==========================================
def step_6_biome(input_csv, output_csv):
    print("\n=== STEP 6: WWF Biome ===")
    
    df = pd.read_csv(input_csv)
    with rasterio.open(CONFIG["biome_tif"]) as src:
        xs, ys = rio_transform("EPSG:4326", src.crs, df.x.values, df.y.values)
        vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=float)
        
        # Clean
        vals[vals == src.nodata] = np.nan
        vals[np.isin(vals, [0, 255])] = np.nan # Extra invalid
        
        df["biome_num"] = vals
        
    df["biome_name"] = df["biome_num"].map(lambda x: BIOME_LOOKUP.get(int(x), None) if pd.notna(x) else None)
    df["biome_name_enc"] = df["biome_num"]
    
    # Update JSON
    biome_json = {
        "biome_name_to_code": {v: k for k, v in BIOME_LOOKUP.items()},
        "biome_code_to_name": {str(k): v for k, v in BIOME_LOOKUP.items()}
    }
    update_encoders_json(CONFIG["encoders_json"], biome_json)
    
    df.to_csv(output_csv, index=False)
    print(f"✅ Step 6 Done. Final Saved: {output_csv}")
    return output_csv



def apply_env_overrides():
    """Override CONFIG with environment variables set by 00_bat/03_prepare_clean_encoded_data.bat."""
    env_map = {
        "DP_INPUT_BFAST_GLOBAL": "input_initial_csv",
        "DP_WORK_DIR": "work_dir",
        "DP_DEM_FOLDER": "dem_folder",
        "DP_DEM_CUTLINE_GEOJSON": "dem_cutline_geojson",
        "DP_WIND_NC": "wind_nc_file",
        "DP_MODIS_LC_DIR": "lc_dir",
        "DP_TREE_HEIGHT_TIF": "tree_tif",
        "DP_BIOME_TIF": "biome_tif",
        "DP_ENCODERS_JSON": "encoders_json",
        "DP_FINAL_FEATURES_CSV": "final_output_csv",
    }
    for env_name, key in env_map.items():
        val = os.environ.get(env_name)
        if val:
            CONFIG[key] = val

    int_env_map = {
        "DP_WIND_N_JOBS": "wind_n_jobs",
        "DP_GDAL_WARP_MEMORY_MB": "gdal_warp_memory_mb",
    }
    for env_name, key in int_env_map.items():
        val = os.environ.get(env_name)
        if val:
            CONFIG[key] = int(val)

    bbox = os.environ.get("DP_DEM_BBOX_WGS84")
    if bbox:
        CONFIG["dem_bbox_wgs84"] = [float(x.strip()) for x in bbox.split(",")]

# ==========================================
#           主执行逻辑
# ==========================================
if __name__ == "__main__":
    apply_env_overrides()
    # Ensure work dir exists
    if not os.path.exists(CONFIG["work_dir"]):
        os.makedirs(CONFIG["work_dir"])

    # Define intermediate filenames
    f_step1 = os.path.join(CONFIG["work_dir"], "temp_01_dem.csv")
    f_step2 = os.path.join(CONFIG["work_dir"], "temp_02_wind.csv")
    f_step3 = os.path.join(CONFIG["work_dir"], "temp_03_roughness.csv")
    f_step4 = os.path.join(CONFIG["work_dir"], "temp_04_lc.csv")
    f_step5 = os.path.join(CONFIG["work_dir"], "temp_05_tree.csv")
    f_final = CONFIG["final_output_csv"]
    
    dem_tif_path = os.path.join(CONFIG["work_dir"], CONFIG["dem_tif_filename"])

    # --- Chain Execution ---
    try:
        # Step 1: Input Original -> Output Step1 CSV
        step_1_dem(CONFIG["input_initial_csv"], f_step1, dem_tif_path)
        
        # Step 2: Input Step1 -> Output Step2 CSV
        step_2_wind(f_step1, f_step2)
        
        # Step 3: Input Step2 -> Output Step3 CSV (Requires DEM Tif from Step 1)
        step_3_roughness(f_step2, f_step3, dem_tif_path)
        
        # Step 4: Input Step3 -> Output Step4 CSV
        step_4_landcover(f_step3, f_step4)
        
        # Step 5: Input Step4 -> Output Step5 CSV
        step_5_treeheight(f_step4, f_step5)
        
        # Step 6: Input Step5 -> Final Output
        step_6_biome(f_step5, f_final)
        
        print("\n🎉🎉🎉 All Processing Steps Completed Successfully! 🎉🎉🎉")
        print(f"Final Output: {f_final}")
        
    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
