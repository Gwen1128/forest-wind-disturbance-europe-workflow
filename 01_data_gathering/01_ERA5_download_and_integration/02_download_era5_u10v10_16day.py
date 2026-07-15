import cdsapi
import os
from datetime import datetime, timedelta

# Initialize the CDS API client
client = cdsapi.Client()

DATASET = "reanalysis-era5-single-levels"
SAVE_DIR = "E:/ERA/u10v10"  # Change this to your desired directory
os.makedirs(SAVE_DIR, exist_ok=True)

# Europe domain (WGS84): North, West, South, East
AREA = [71.2, -31.5, 23.75, 41.5]


def retrieve_data(start_date: datetime, end_date: datetime):
    """
    Retrieve ERA5 u10/v10 for the given date range and save as a NetCDF file.
    IMPORTANT: Uses CDS 'date' range to include ALL days in the window.
    """
    filename = os.path.join(
        SAVE_DIR,
        f"era5_chunk_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.nc",
    )

    if os.path.exists(filename):
        print(f"File already exists, skipping download: {filename}")
        return filename

    request = {
        "product_type": "reanalysis",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        # Key fix: request the full date range (inclusive)
        "date": f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": AREA,  # N/W/S/E
        "format": "netcdf",
    }

    print(f"Requesting u10/v10 from {start_date.date()} to {end_date.date()} ...")
    try:
        client.retrieve(DATASET, request, filename)
        print(f"Downloaded: {filename}")
        return filename
    except Exception as e:
        print(f"Failed to download {start_date.date()} to {end_date.date()}: {e}")
        return None


def retrieve_era5_16_day_chunks(start_date: datetime, end_date: datetime):
    """
    Retrieve ERA5 data in 16-day chunks between start_date and end_date (inclusive).
    """
    current_start = start_date
    results = []

    while current_start <= end_date:
        current_end = current_start + timedelta(days=15)
        if current_end > end_date:
            current_end = end_date

        print(f"Processing chunk: {current_start.date()} to {current_end.date()}")
        filename = retrieve_data(current_start, current_end)
        if filename:
            results.append(filename)

        current_start = current_end + timedelta(days=1)

    return results


if __name__ == "__main__":
    # Define the time range for retrieval
    start_date = datetime(2001, 1, 1)
    end_date = datetime(2023, 12, 31)

    downloaded_files = retrieve_era5_16_day_chunks(start_date, end_date)

    print("Downloaded files:")
    for f in downloaded_files:
        print(f)
