# Data

This repository does **not** redistribute the benchmark dataset.

The experiments use:

**Tran, T. T. & Tran, N. T. (2026). _A Real Controlled-Fault Benchmark for Power-Anomaly Detection and Identification on IoT Edge Nodes (Three-Station 1 Hz Dataset)._ Zenodo.**

DOI: https://doi.org/10.5281/zenodo.20565892

Download the original ZIP from Zenodo and pass its local path to the scripts with `--zip`.
The dataset is an external research object and remains subject to the license and attribution terms specified by its original authors on Zenodo.

## Expected station files

The scripts locate the three station feature CSVs inside the downloaded ZIP:

- `station1_normal_features.csv`
- `station2_medium_features.csv`
- `station3_high_features.csv`

No target-site fault labels from the disjoint reference interval are used to fit the classifier or calibrate novelty thresholds in the primary protocol.
