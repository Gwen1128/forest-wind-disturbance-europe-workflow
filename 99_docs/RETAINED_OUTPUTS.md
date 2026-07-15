# Retained outputs in the minimal paper-results package

This package keeps only outputs needed to reproduce the manuscript results and figure source data. Intermediate debug products, temporary maps, full OOF predictions, and redundant diagnostic tables are removed by the BAT files after each step.

## Retained process-level inputs

- `retained_intermediates/fig5/TableS_wind_anomaly_temporal_metrics_long_term_frequency.csv`
  - Locked wind-anomaly temporal metric used by Fig. 5 complete-workflow reproduction.
  - Fig. 5 still rebuilds the final merged CSV and final tables from this intermediate and the Fig. 2 prediction temporal table.

- `reference_outputs/fig5_locked/`
  - Reference copies of the verified Fig. 5 tables used only for checking. They are not read by the workflow.

## Retained outputs by step

- `01_ablation_original_pipeline/`
  - `metrics_*.csv`
  - `table1_spatial_holdout_from_original_pipeline.csv`

- `02_locked_main_model/`
  - `stacked_LOCKED_grouped_logit_main_REPRODUCED.joblib`
  - `metrics_LOCKED_grouped_logit_main_REPRODUCED.csv`

- `03_wall2wall_windonly/`
  - `hex_distribution_wall2wall_windonly_final.csv`
  - `hex_period_summary_wall2wall_windonly_final.csv`

- `04_fig1_spatial_indicators/`
  - `hex_indicator_summary_stylematch_4326.csv`
  - `Fig_hex_indicator_suite_stylematch_publication_2x2.png/pdf`

- `05_fig2_monthly_concentration/`
  - `hex_monthly_spatiotemporal_likelihood_metrics.csv`
  - `Fig_peak_month_and_peak_month_share_1x2.png/pdf`

- `06_fig3_wind_anomaly_background/`
  - `wind_spacetime_long_term_5panel_from_products_nc_EPSG3035_centered_transparent.png/pdf`

- `06_fig4_prediction_anomaly_regime_prepare/`
  - `long_term/merged_prediction_anomaly_long_term.csv`

- `07_fig4_prediction_anomaly_regime_plot/`
  - `Fig5_prediction_response_frequency_intensity.png/pdf`
  - `fig5a_mean_prediction_lift_surface.csv`
  - `fig5b_p95_prediction_lift_surface.csv`
  - `fig5c_regime_enrichment_top20.csv`
  - `fig5d_regime_enrichment_sensitivity.csv`

- `08_fig5_temporal_correspondence/`
  - `Fig5_peak_month_correspondence_long_term.png/pdf`
  - `merged_temporal_correspondence_long_term.csv`
  - `TableS_temporal_correspondence_summary_long_term.csv`
  - `TableS_peak_month_agreement_classes_long_term.csv`
  - `TableS_temporal_correlations_long_term.csv`
  - `TableS_prediction_temporal_metrics_long_term.csv`
  - `TableS_wind_anomaly_temporal_metrics_long_term_frequency.csv`



V4 note: standalone main-model training and per-storm analysis files were removed as redundant for the paper-result reproduction workflow.


V4: The final wind-only model is copied from the Table 1 wind-only run; standalone duplicate training and per-storm outputs are not part of the retained outputs.
