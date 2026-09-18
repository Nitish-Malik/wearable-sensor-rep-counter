# Dataset Setup

The notebooks expect raw MetaMotion accelerometer and gyroscope CSV files. Store them using the following layout:

```text
data/
└── raw/
    └── MetaMotion/
        └── MetaMotion/
            ├── A-bench-heavy_..._Accelerometer_12.500Hz_1.4.4.csv
            ├── A-bench-heavy_..._Gyroscope_25.000Hz_1.4.4.csv
            └── additional sensor CSV files
```

Before publishing the repository, rename this document to `data/raw/README.md` or copy its contents there.

## Dataset Source

Add the original dataset URL and license here before publishing. Do not redistribute the raw data unless its license permits redistribution.

## Generated Files

Running the notebooks creates the following intermediate files:

```text
data/interim/01_data_processed.pkl
data/interim/02_outlier_removed_df.pkl
data/interim/03_feature_eng.pkl
```

These generated files are excluded by `.gitignore` and can be recreated by running the notebook pipeline.

