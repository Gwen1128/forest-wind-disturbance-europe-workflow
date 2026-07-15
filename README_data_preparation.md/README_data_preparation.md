# Data-preparation workflow for `clean_encoded_data_LOCKED.csv`

This folder contains the upstream code needed to rebuild the modeling table used by the Table 1 training workflow.

## Main chain

1. `scripts/00_GEE_LAI_extract.js`
   - Run manually in Google Earth Engine.
   - Exports `ForestLAI_YYYYMMDD.tif` files.

2. `scripts/01_bfast_lai_departure_sampling.R`
   - Uses exported LAI rasters and FORWIND polygons.
   - Produces `bfast_global.csv`.
   - Supports checkpoint restart.

3. `scripts/02_extract_geospatial_features.py`
   - Uses `bfast_global.csv`.
   - Adds DEM/slope/aspect, ERA5 wind, exposure/roughness, MODIS landcover, tree height, and biome features.
   - Produces `bfastglobal_FINAL_ALL_FEATURES.csv`.

4. `scripts/04_preprocess_locked_modeling_table.py`
   - Standardizes names into the locked model schema.
   - Clips positive LAI departure values to zero.
   - Builds cross features in preprocessing.
   - CV target-encodes landcover.
   - Produces `clean_encoded_data_LOCKED.csv` and `encoders_LOCKED.json`.

`03_optional_combine_feature_tiles.py` is only needed if feature extraction was run tile-by-tile and produced multiple `bfast*FINAL_ALL_FEATURES.csv` files.

## Windows entry point

Edit `../00_bat/00_config_paths.bat`, then run:

```bat
cd /d <package>\00_bat
03_prepare_clean_encoded_data.bat
```

The BAT cannot run the Google Earth Engine export. It starts from `bfast_global.csv` if that file already exists; otherwise it attempts to run the R script if `Rscript`, LAI folders, and FORWIND shapefile are configured.


## Release-package note

Only the cleaned scripts in `scripts/` are retained in this release package. The duplicate original uploaded scripts with hard-coded local paths were removed to avoid ambiguity.
