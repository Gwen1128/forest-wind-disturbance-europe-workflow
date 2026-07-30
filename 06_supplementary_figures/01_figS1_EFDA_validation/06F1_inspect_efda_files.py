# -*- coding: utf-8 -*-
"""
Extract selected EFDA country ZIPs and inspect raster files.

"""

from pathlib import Path
import zipfile
import rasterio
import pandas as pd


EFDA_DIR = Path(r"E:/EFDA_Zenodo_13333034")
EXTRACT_DIR = EFDA_DIR / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = EFDA_DIR / "efda_file_inventory.csv"


def extract_zip(zip_path, out_dir):
    country_dir = out_dir / zip_path.stem

    if country_dir.exists() and any(country_dir.rglob("*.tif")):
        print(f"Already extracted: {country_dir}")
        return country_dir

    country_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(country_dir)

    return country_dir


def inspect_tif(path):
    rec = {
        "path": str(path),
        "name": path.name,
        "parent": path.parent.name,
        "country": path.parts[-3] if len(path.parts) >= 3 else "",
        "size_mb": path.stat().st_size / 1024 / 1024,
    }

    try:
        with rasterio.open(path) as src:
            rec.update({
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
                "bounds": tuple(src.bounds),
                "descriptions": "|".join([str(d) for d in src.descriptions]),
            })

            tags = src.tags()
            rec["tags"] = str(tags)

            band_tags = []
            for i in range(1, min(src.count, 5) + 1):
                band_tags.append(str(src.tags(i)))
            rec["first_band_tags"] = "|".join(band_tags)

    except Exception as e:
        rec["error"] = str(e)

    return rec


def guess_layer_type(name):
    n = name.lower()

    if "agent" in n:
        return "agent"
    if "severity" in n or "nbr" in n:
        return "severity"
    if "stack" in n and "disturb" in n:
        return "annual_disturbance_stack"
    if "year" in n and "disturb" in n:
        return "year_of_disturbance"
    if "prob" in n:
        return "probability"
    if "disturb" in n:
        return "disturbance_other"

    return "unknown"


def main():
    zip_files = sorted(EFDA_DIR.glob("*.zip"))

    if not zip_files:
        raise FileNotFoundError(f"No ZIP files found in {EFDA_DIR}")

    all_records = []

    for z in zip_files:
        country_dir = extract_zip(z, EXTRACT_DIR)

        tif_files = sorted(country_dir.rglob("*.tif"))

        print(f"{z.stem}: {len(tif_files)} tif files")

        for tif in tif_files:
            rec = inspect_tif(tif)
            rec["layer_guess"] = guess_layer_type(tif.name)
            all_records.append(rec)

    df = pd.DataFrame(all_records)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved inventory: {OUT_CSV}")

    print("\nLayer guesses:")
    print(df["layer_guess"].value_counts(dropna=False).to_string())

    print("\nCandidate annual disturbance stacks:")
    print(
        df[df["layer_guess"].isin(["annual_disturbance_stack", "year_of_disturbance"])]
        [["country", "name", "count", "crs", "path"]]
        .to_string(index=False)
    )

    print("\nCandidate agent layers:")
    print(
        df[df["layer_guess"].eq("agent")]
        [["country", "name", "count", "crs", "path"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
