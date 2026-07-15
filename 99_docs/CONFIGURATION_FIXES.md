# Configuration fixes in this package

1. Restored the complete original configuration, including data preparation, Table 1, Fig. 2-5, and supplementary figures.
2. Corrected the Table 1 training-script target to `train_model_family_comparison.py`.
3. Defined `TABLE1_HELPER`, `SUPP_SCRIPT`, `FIGS_SUPPLEMENTARY_OUTDIR`, `FIG1_INDICATOR_CSV`, and `HEX_GEOMETRY_CSV`.
4. Separated wall-to-wall parameters from Fig. 1 parameters.
5. Locked the successful wall-to-wall values: `max_wind_speed`, bbox `-11.5,34.0,42.5,72.5`, `gamma=0.55`, `qhi=99`, and `valid_periods_min_per_hex=10`.
6. Preserved the Fig. 1 extent `-11,34,45,72` while reading the exact wall-to-wall geometry CSV.
7. Unified Fig. 4 preparation/output aliases and retained compatibility aliases for older BAT variants.

8. Locked Fig. 5 to the verified reproduction path: Fig. 2 prediction timing plus retained wind-anomaly temporal metrics, with `frequency` as the wind metric.
9. Locked Fig. 5 agreement classes to complete hex-area weighting (`2165 km2` per analysed hex), matching the original merged temporal correspondence table. This should not be replaced by `forest_area_km2` weighting for exact reproduction.
10. Added `00_bat/10A_verify_fig5_locked_hexarea.bat` to verify that Fig. 5 has 3,535 rows, total area weight 7,653,275 km2, and unique area weight 2165 km2.
