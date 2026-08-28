"""Create Figure S4 for the FORWIND Sweden holdout.

The FORWIND files are expected in E:\\FORWIND:
    FORWIND_v2.shp
    FORWIND_v2.shx
    FORWIND_v2.dbf
    FORWIND_v2.prj

Then run:
    python plot_Figure_S4_FORWIND_holdout.py

Required packages:
    numpy, pandas, matplotlib, scikit-learn, basemap

The script deliberately does not require GeoPandas, Fiona, or Cartopy.
"""

from __future__ import annotations

import argparse
import math
import struct
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


# Spatial holdout used in the manuscript.
HOLDOUT_XMIN = 10.96
HOLDOUT_XMAX = 24.17
HOLDOUT_YMIN = 55.25
HOLDOUT_YMAX = 69.06

# Display extent. This focuses on the European part of the database.
MAP_XMIN = -12.0
MAP_XMAX = 42.0
MAP_YMIN = 42.0
MAP_YMAX = 70.0

# Main high-record countries used in panel b.
COMPARISON_COUNTRIES = ["LT", "IT", "DE", "FR", "SE"]
COUNTRY_NAMES = {
    "LT": "Lithuania",
    "IT": "Italy",
    "DE": "Germany",
    "FR": "France",
    "SE": "Sweden",
}

TRAINING_COLOR = "#9E9E9E"
HOLDOUT_COLOR = "#B2182B"


def read_dbf(path: Path) -> pd.DataFrame:
    """Read the attributes needed from a dBASE III file.

    This small reader preserves record order so that the DBF rows can be
    joined directly to records in the accompanying shapefile.
    """
    with path.open("rb") as stream:
        header = stream.read(32)
        record_count = struct.unpack("<I", header[4:8])[0]
        header_length = struct.unpack("<H", header[8:10])[0]
        record_length = struct.unpack("<H", header[10:12])[0]

        fields = []
        offset = 1  # first byte of each record is the deletion flag
        while True:
            descriptor = stream.read(32)
            if descriptor[0] == 0x0D:
                break
            name = descriptor[:11].split(b"\0", 1)[0].decode("latin1")
            field_type = chr(descriptor[11])
            length = descriptor[16]
            decimals = descriptor[17]
            fields.append((name, field_type, length, decimals, offset))
            offset += length

        stream.seek(header_length)
        rows = []
        for _ in range(record_count):
            raw = stream.read(record_length)
            if not raw:
                break
            if raw[:1] == b"*":
                # Keep an empty row so SHP and DBF record indices remain aligned.
                rows.append({})
                continue

            row = {}
            for name, field_type, length, decimals, field_offset in fields:
                text = raw[field_offset : field_offset + length].decode(
                    "latin1", "ignore"
                ).strip()
                if field_type in {"N", "F"}:
                    try:
                        text = float(text) if decimals else int(text)
                    except ValueError:
                        text = np.nan
                row[name] = text
            rows.append(row)

    return pd.DataFrame(rows)


def polygon_centroid(points: np.ndarray, starts: np.ndarray) -> tuple[float, float]:
    """Calculate a polygon centroid from shapefile rings in lon/lat space."""
    ends = list(starts[1:]) + [len(points)]
    weighted_x = 0.0
    weighted_y = 0.0
    signed_area_total = 0.0

    for start, end in zip(starts, ends):
        ring = points[int(start) : int(end)]
        if len(ring) < 3:
            continue
        x = ring[:, 0]
        y = ring[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        signed_area = 0.5 * cross.sum()
        if abs(signed_area) < 1e-16:
            continue
        centroid_x = ((x + x_next) * cross).sum() / (6.0 * signed_area)
        centroid_y = ((y + y_next) * cross).sum() / (6.0 * signed_area)
        weighted_x += centroid_x * signed_area
        weighted_y += centroid_y * signed_area
        signed_area_total += signed_area

    if abs(signed_area_total) > 1e-16:
        return weighted_x / signed_area_total, weighted_y / signed_area_total

    # Fallback for degenerate geometry.
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def read_shapefile_centroids(path: Path) -> pd.DataFrame:
    """Read polygon centroids directly from an ESRI shapefile."""
    rows = []
    with path.open("rb") as stream:
        stream.seek(100)  # fixed shapefile header length
        while True:
            record_header = stream.read(8)
            if not record_header:
                break

            record_number, content_length_words = struct.unpack(">2i", record_header)
            content = stream.read(content_length_words * 2)
            shape_type = struct.unpack("<i", content[:4])[0]

            if shape_type == 0:
                rows.append(
                    {"record_number": record_number, "lon": np.nan, "lat": np.nan}
                )
                continue
            if shape_type not in {5, 15, 25}:
                raise ValueError(f"Unexpected shape type {shape_type}")

            number_of_parts, number_of_points = struct.unpack("<2i", content[36:44])
            starts = np.frombuffer(
                content, dtype="<i4", count=number_of_parts, offset=44
            )
            points_offset = 44 + 4 * number_of_parts
            points = np.frombuffer(
                content,
                dtype="<f8",
                count=2 * number_of_points,
                offset=points_offset,
            ).reshape(-1, 2)
            lon, lat = polygon_centroid(points, starts)
            rows.append({"record_number": record_number, "lon": lon, "lat": lat})

    return pd.DataFrame(rows)


def mean_nearest_neighbour_km(group: pd.DataFrame) -> float:
    """Mean distance from every centroid to its nearest centroid in the country."""
    coordinates = group[["lat", "lon"]].dropna().to_numpy()
    if len(coordinates) < 2:
        return float("nan")
    coordinates_radians = np.radians(coordinates)
    tree = BallTree(coordinates_radians, metric="haversine")
    distances, _ = tree.query(coordinates_radians, k=2)
    return float(distances[:, 1].mean() * 6371.0088)


def occupied_grid_cells(group: pd.DataFrame, cell_size_degrees: float = 0.25) -> int:
    """Count occupied lon/lat grid cells for a simple dispersion diagnostic."""
    valid = group[["lon", "lat"]].dropna()
    longitude_cells = np.floor(valid["lon"].to_numpy() / cell_size_degrees).astype(int)
    latitude_cells = np.floor(valid["lat"].to_numpy() / cell_size_degrees).astype(int)
    return len(Counter(zip(longitude_cells, latitude_cells)))


def build_country_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country, group in data.groupby("Country"):
        area = pd.to_numeric(group.get("Area"), errors="coerce")
        methods = group.get("Methods", pd.Series(dtype=str)).fillna("").astype(str)
        predominant_method = methods.value_counts().index[0] if len(methods) else ""
        occupied_cells = occupied_grid_cells(group)
        rows.append(
            {
                "country": country,
                "n_polygons": len(group),
                "median_polygon_area_database_units": area.median(),
                "mean_nearest_neighbour_km": mean_nearest_neighbour_km(group),
                "occupied_0.25_degree_cells": occupied_cells,
                "occupied_cells_per_1000_polygons": occupied_cells / len(group) * 1000,
                "predominant_mapping_method": predominant_method,
            }
        )
    return pd.DataFrame(rows).set_index("country").sort_values(
        "n_polygons", ascending=False
    )


def make_figure(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    output_format: str,
) -> None:
    plt.rcParams.update(
        {
            # Use Arial when available and fall back cleanly on other systems.
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, (map_axis, bar_axis) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.65),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    # Panel a: reference-record distribution and holdout boundary.
    # Basemap supplies coastlines and national borders without requiring a
    # separate boundary file or an online download.
    europe_map = Basemap(
        projection="cyl",
        llcrnrlon=MAP_XMIN,
        urcrnrlon=MAP_XMAX,
        llcrnrlat=MAP_YMIN,
        urcrnrlat=MAP_YMAX,
        resolution="l",
        suppress_ticks=False,
        ax=map_axis,
    )
    europe_map.drawmapboundary(
        fill_color="white", color="#666666", linewidth=0.55
    )
    europe_map.drawcoastlines(
        color="#777777", linewidth=0.42, zorder=0.4
    )
    europe_map.drawcountries(
        color="#A0A0A0", linewidth=0.32, zorder=0.5
    )

    map_data = data[
        data["lon"].between(MAP_XMIN, MAP_XMAX)
        & data["lat"].between(MAP_YMIN, MAP_YMAX)
    ]
    training = map_data[map_data["Country"] != "SE"]
    sweden = map_data[map_data["Country"] == "SE"]

    map_axis.scatter(
        training["lon"],
        training["lat"],
        s=0.7,
        color=TRAINING_COLOR,
        alpha=0.22,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    map_axis.scatter(
        sweden["lon"],
        sweden["lat"],
        s=1.2,
        color=HOLDOUT_COLOR,
        alpha=0.50,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    holdout_rectangle = Rectangle(
        (HOLDOUT_XMIN, HOLDOUT_YMIN),
        HOLDOUT_XMAX - HOLDOUT_XMIN,
        HOLDOUT_YMAX - HOLDOUT_YMIN,
        fill=False,
        edgecolor=HOLDOUT_COLOR,
        linewidth=1.1,
        linestyle="--",
        zorder=3,
    )
    map_axis.add_patch(holdout_rectangle)
    map_axis.set_xlim(MAP_XMIN, MAP_XMAX)
    map_axis.set_ylim(MAP_YMIN, MAP_YMAX)
    map_axis.set_xlabel("Longitude (°E)")
    map_axis.set_ylabel("Latitude (°N)")
    map_axis.set_aspect(1.0 / math.cos(math.radians(55.0)))
    map_axis.grid(color="#E5E5E5", linewidth=0.45, zorder=0)
    map_axis.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="", markersize=3.2,
                color=TRAINING_COLOR, label="Training records"
            ),
            Line2D(
                [], [], marker="o", linestyle="", markersize=3.2,
                color=HOLDOUT_COLOR, label="Sweden holdout"
            ),
        ],
        # The upper-left part of this panel is empty, so the legend does not
        # obscure any reference records.
        loc="upper left",
        bbox_to_anchor=(0.01, 0.91),
        frameon=False,
        fontsize=7.5,
        handletextpad=0.4,
    )
    map_axis.text(
        0.01, 0.99, "(a)", transform=map_axis.transAxes,
        ha="left", va="top", fontweight="bold", fontsize=9
    )

    # Panel b: dispersion comparison among the principal high-record countries.
    comparison = metrics.loc[COMPARISON_COUNTRIES].copy()
    values = comparison["mean_nearest_neighbour_km"].to_numpy()
    labels = [COUNTRY_NAMES[code] for code in comparison.index]
    colors = [HOLDOUT_COLOR if code == "SE" else TRAINING_COLOR for code in comparison.index]
    bars = bar_axis.bar(labels, values, color=colors, width=0.68)
    bar_axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=7.5)
    bar_axis.set_ylabel("Mean nearest-neighbour distance (km)")
    bar_axis.set_ylim(0, max(values) * 1.24)
    bar_axis.tick_params(axis="x", rotation=25)
    bar_axis.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    bar_axis.set_axisbelow(True)
    bar_axis.spines["top"].set_visible(False)
    bar_axis.spines["right"].set_visible(False)
    bar_axis.text(
        0.01, 0.99, "(b)", transform=bar_axis.transAxes,
        ha="left", va="top", fontweight="bold", fontsize=9
    )

    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.97, wspace=0.31)
    if output_format == "pdf":
        figure.savefig(
            output_dir / "Figure_S4_FORWIND_holdout.pdf",
            bbox_inches="tight",
            facecolor="white",
        )
    elif output_format == "png":
        figure.savefig(
            output_dir / "Figure_S4_FORWIND_holdout.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"E:\FORWIND"),
        help="Directory containing FORWIND_v2.shp and FORWIND_v2.dbf",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <data-dir>/Figure_S4_output",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else data_dir / "Figure_S4_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    shp_path = data_dir / "FORWIND_v2.shp"
    dbf_path = data_dir / "FORWIND_v2.dbf"
    for path in (shp_path, dbf_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    attributes = read_dbf(dbf_path)
    centroids = read_shapefile_centroids(shp_path)
    if len(attributes) != len(centroids):
        raise RuntimeError(
            f"SHP/DBF record mismatch: {len(centroids)} vs {len(attributes)}"
        )

    data = pd.concat(
        [attributes.reset_index(drop=True), centroids.reset_index(drop=True)],
        axis=1,
    )
    metrics = build_country_metrics(data)
    missing = [code for code in COMPARISON_COUNTRIES if code not in metrics.index]
    if missing:
        raise RuntimeError(f"Countries missing from database: {missing}")

    metrics.to_csv(output_dir / "Figure_S4_country_metrics.csv")
    # Render each format with a fresh figure. This keeps both the vector PDF
    # and the high-resolution PNG reliable across Matplotlib backends.
    make_figure(data, metrics, output_dir, "pdf")
    make_figure(data, metrics, output_dir, "png")

    holdout_records = data[data["Country"] == "SE"]
    other_countries_in_rectangle = data[
        data["lon"].between(HOLDOUT_XMIN, HOLDOUT_XMAX)
        & data["lat"].between(HOLDOUT_YMIN, HOLDOUT_YMAX)
        & (data["Country"] != "SE")
    ]
    print(f"Created outputs in: {output_dir}")
    print(f"Sweden records: {len(holdout_records):,}")
    print(f"Non-Sweden records inside holdout rectangle: {len(other_countries_in_rectangle):,}")
    print(metrics.loc[COMPARISON_COUNTRIES, [
        "n_polygons", "mean_nearest_neighbour_km",
        "occupied_0.25_degree_cells", "predominant_mapping_method"
    ]].to_string())


if __name__ == "__main__":
    main()
