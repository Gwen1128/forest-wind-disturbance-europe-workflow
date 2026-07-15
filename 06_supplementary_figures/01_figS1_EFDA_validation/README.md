# Fig. S1 EFDA validation workflow

The scripts `06F0`–`06F3` belong to the Fig. S1 / EFDA spatial-consistency workflow.

Run order:

```text
06F0_download_efda_zenodo.py
06F1_inspect_efda_files.py
06F2_aggregate_efda_to_hex_year.py
06F3_compare_prediction_with_efda.py
```

They are separated from the main Fig. 5 script even though their filenames also start with `06F`.
