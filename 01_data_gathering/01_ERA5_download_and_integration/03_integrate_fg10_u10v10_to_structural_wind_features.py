import numpy as np
import xarray as xr
import os
import pandas as pd
from dask.diagnostics import ProgressBar
from dask.distributed import Client

# 初始化 Dask 客户端进行并行处理
client = Client(n_workers=8, threads_per_worker=1)
print(client)

# 设定输入输出目录
SAVE_DIR = "/projappl/project_2011073/fg10eu"
UV_DIR   = "/projappl/project_2011073/u10v10EU"

OUT_NAME = "EU_final_structural_wind_features.nc"

import os
import numpy as np
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar

def calculate_wind_direction(u, v):
    """Meteorological wind direction (FROM): 0=N, 90=E, 180=S, 270=W"""
    angle_rad = np.arctan2(-u, -v)
    angle_deg = (np.degrees(angle_rad) + 360) % 360
    return angle_deg


def standardize_longitude(ds):
    """Convert longitude to [-180, 180] and sort."""
    if "longitude" in ds.coords and (ds.longitude > 180).any():
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")
    return ds


def _drop_era5_junk(ds):
    """Drop cfgrib/ERA5 leftover coords/vars that must NOT participate in concat."""
    # drop_vars handles both data_vars and coords (if they are promoted as variables)
    ds = ds.drop_vars(["number", "expver", "valid_time"], errors="ignore")
    # also ensure no leftover non-dimension coords remain
    ds = ds.reset_coords(drop=True)
    return ds


def max_per_grid(wind_file, uv_file, debug_print=False):
    try:
        # 建议：先 open 再 chunk，避免“stored chunks 不对齐”的性能警告
        ds_wind = xr.open_dataset(wind_file).chunk({"valid_time": 24})
        ds_uv   = xr.open_dataset(uv_file).chunk({"valid_time": 24})

        ds_wind = standardize_longitude(ds_wind)
        ds_uv   = standardize_longitude(ds_uv)

        # 只保留需要的变量，减少对齐干扰与内存
        ds_wind = ds_wind[["fg10"]]
        ds_uv   = ds_uv[["u10", "v10"]]

        # 对齐（交集）
        ds_wind, ds_uv = xr.align(ds_wind, ds_uv, join="inner")

        if debug_print:
            print("PAIR:", os.path.basename(wind_file), "<->", os.path.basename(uv_file))
            print("AFTER align:",
                  "wind valid_time:", ds_wind.sizes.get("valid_time", -1),
                  "uv valid_time:",   ds_uv.sizes.get("valid_time", -1),
                  "lat/lon:", ds_wind.sizes.get("latitude", -1), ds_wind.sizes.get("longitude", -1))

        if ds_wind.sizes.get("valid_time", 0) == 0:
            raise ValueError("valid_time 交集为空：wind 与 uv 时间轴未对齐（或文件错配）")

        fg10 = ds_wind["fg10"]
        u10  = ds_uv["u10"]
        v10  = ds_uv["v10"]

        # 1) 16-day max gust
        fg10_max = fg10.max(dim="valid_time", skipna=True).astype(np.float32)

        # 2) peak index per pixel
        t_idx = fg10.argmax(dim="valid_time").astype(np.int32).compute()

        # 3) 3-hour centered rolling mean for structural wind direction
        u_smooth = u10.rolling(valid_time=3, center=True, min_periods=1).mean()
        v_smooth = v10.rolling(valid_time=3, center=True, min_periods=1).mean()

        # vectorized isel at peak time
        u_at = u_smooth.isel(valid_time=t_idx)
        v_at = v_smooth.isel(valid_time=t_idx)

        wind_dir = xr.apply_ufunc(
            calculate_wind_direction,
            u_at, v_at,
            dask="allowed",
            output_dtypes=[np.float32],
        ).astype(np.float32)

        # ---- 关键修复：清掉 isel 带出来的 valid_time 等辅助坐标，否则 concat 会把数据对齐成 NaN ----
        fg10_max = fg10_max.reset_coords(drop=True)
        wind_dir = wind_dir.reset_coords(drop=True)

        # 构造“干净”的 result：只保留维度坐标 + 两个变量
        result = xr.Dataset(
            data_vars={
                "max_wind_speed": (("latitude", "longitude"), fg10_max.data),
                "wind_direction": (("latitude", "longitude"), wind_dir.data),
            },
            coords={
                "latitude": ds_wind["latitude"].data,
                "longitude": ds_wind["longitude"].data,
            }
        )

        result = _drop_era5_junk(result)

        # 用 chunk 起始日做 time 坐标
        time_str = os.path.basename(wind_file).split("chunk_")[-1].replace(".nc", "")
        start_time = pd.to_datetime(time_str.split("_")[0])

        result = result.expand_dims({"time": [start_time]})

        if debug_print:
            nn_ws = int(result["max_wind_speed"].count().values)
            nn_wd = int(result["wind_direction"].count().values)
            print("chunk non-nan ws/wd:", nn_ws, nn_wd)

        return result

    except Exception as e:
        print(f"❌ 处理文件出错 {wind_file}: {e}")
        return None


def aggregate_max_per_grid(SAVE_DIR, UV_DIR, OUT_NAME, debug_print=False):
    wind_files = [
        f for f in os.listdir(SAVE_DIR)
        if f.endswith(".nc") and "era5_chunk_" in f and f != OUT_NAME
    ]
    uv_files = [
        f for f in os.listdir(UV_DIR)
        if f.endswith(".nc") and "era5_chunk_" in f
    ]

    def chunk_key(fn: str) -> str:
        base = os.path.basename(fn)
        k = base.replace(".nc", "")
        if "era5_chunk_" in k:
            k = k.split("era5_chunk_")[-1]
            k = "era5_chunk_" + k
        return k

    wind_map = {chunk_key(f): os.path.join(SAVE_DIR, f) for f in wind_files}
    uv_map   = {chunk_key(f): os.path.join(UV_DIR,   f) for f in uv_files}

    keys = sorted(set(wind_map.keys()) & set(uv_map.keys()))
    if not keys:
        print("❌ 找不到可匹配的 chunk（wind 与 u/v 没有任何共同 key）。")
        print("wind sample:", wind_files[:3])
        print("uv sample:", uv_files[:3])
        return None

    datasets = []
    for k in keys:
        w  = wind_map[k]
        uv = uv_map[k]
        print(f"正在处理块: {k}")

        ds = max_per_grid(w, uv, debug_print=debug_print)
        if ds is None:
            continue

        # 硬检查：每个块至少要有非 NaN
        if int(ds["max_wind_speed"].count().values) == 0:
            print(f"❌ 块 {k} 结果全 NaN（不应发生），已跳过。")
            continue

        datasets.append(ds)

    if not datasets:
        print("❌ 所有块都无有效数据。")
        return None

    print("🔁 正在进行时间维度合并...")
    combined = xr.concat(datasets, dim="time").sortby("time")

    # 最终再做一次保险清理：任何残余坐标一律去掉
    combined = _drop_era5_junk(combined)

    output_file = os.path.join(SAVE_DIR, OUT_NAME)
    print(f"💾 正在保存结果至: {output_file}")

    encoding = {
        "max_wind_speed": {"zlib": True, "complevel": 4, "dtype": "float32",
                           "chunksizes": (1, 190, 293), "_FillValue": np.float32(-9999.0)},
        "wind_direction": {"zlib": True, "complevel": 4, "dtype": "float32",
                           "chunksizes": (1, 190, 293), "_FillValue": np.float32(-9999.0)},
    }

    with ProgressBar():
        combined.to_netcdf(output_file, encoding=encoding)

    print("✅ 处理完成")
    return output_file

if __name__ == "__main__":
    final_nc = aggregate_max_per_grid(SAVE_DIR, UV_DIR, OUT_NAME, debug_print=False)
    print("Output:", final_nc)
