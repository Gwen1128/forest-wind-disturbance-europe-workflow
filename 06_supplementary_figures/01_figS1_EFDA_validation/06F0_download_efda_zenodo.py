# -*- coding: utf-8 -*-
"""
Download selected country ZIP files from the European Forest Disturbance Atlas

Zenodo record:
European Forest Disturbance Atlas, v2.1.1.
"""

from pathlib import Path
import requests
from tqdm import tqdm


ZENODO_API = "https://zenodo.org/api/records/13333034"

OUTDIR = Path(r"E:/EFDA_Zenodo_13333034")
OUTDIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = [
    "germany",
    "czechia",
    "poland",
    "sweden",
    "france",
    "finland",
    # "austria",
    # "slovakia",
]

CHUNK_SIZE = 1024 * 1024


def download_file(url, out_path):
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Exists, skip: {out_path}")
        return

    print(f"Downloading: {out_path.name}")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(out_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=out_path.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def main():
    meta = requests.get(ZENODO_API, timeout=60).json()

    files = meta["files"]

    wanted = {f"{c.lower()}.zip" for c in COUNTRIES}

    found = []

    for item in files:
        key = item["key"].lower()

        if key in wanted:
            url = item["links"]["self"]
            out_path = OUTDIR / item["key"]
            found.append(key)
            download_file(url, out_path)

    missing = sorted(wanted - set(found))

    if missing:
        print("Missing files:")
        for m in missing:
            print(f"  {m}")

    print("Done.")


if __name__ == "__main__":
    main()
