# ADS-B Anomaly Detector

This is my Master's Project for Rensselaer Polytechnic Institute.

The primary goal of this project is to research, design, implement, and evaluate a system capable of identifying any anomalous data within real-time ADS-B signal streams. This system serves as a proof-of-concept for enhancing the integrity and reliability of ADS-B surveillance data.

## Project Objectives

1. **ADS-B Data Acquisition and Parsing**
   - Gather publicly available ADS-B data via the OpenSky Network, focusing on specific global regions (like the Middle East).
   - Parse, clean, and format the data for rigorous analysis.

2. **Anomaly Detection Model(s)**
   - Research and implement machine learning models to detect anomalous flight patterns. 
   - Uses techniques such as **K-Means Clustering**, **DBSCAN**, and **Isolation Forest**.
   - Leverages historical ADS-B data to establish baselines for normal flight behavior.

3. **Implementation and Evaluation**
   - A pipeline to feed models real-time or historical data and measure performance.
   - Generates visualizations (geographic scatter plots, Voronoi diagrams) to display aircraft tracks and flag detected anomalies.

## Project Structure

- `main.py` - The core pipeline script for data ingestion, feature engineering, modeling, and visualization.
- `src/` - Contains the module logic:
  - `config.py` - Configuration parameters and bounding box definitions.
  - `data_collector.py` - OpenSky Network integration for live and historical data.
  - `detector.py` - Machine learning models (K-Means, DBSCAN, Isolation Forest).
  - `features.py` - Feature engineering for observation and flight-level aggregation.
  - `visualize.py` - Plotting and geographic visualization generators.
- `data/` - Storage for `raw/` and `processed/` parquet/csv files.
- `models/` - Saved state models (e.g., scalers, K-Means centroids).
- `output/` - Output charts, maps, and reports.

## Prerequisites & Installation

This project uses the [`uv`](https://docs.astral.sh/uv/) package manager for lightning-fast dependency management.

1. Install `uv` if you haven't already.
2. Synchronize the environment and install dependencies:
   ```bash
   uv sync
   ```

## Usage

You can run the main pipeline using `uv run`. The pipeline supports three modes: `live`, `historical`, and `file`.

### 1. Live Data Mode (REST API)
Captures live data snapshots directly from OpenSky without requiring authentication.

```bash
uv run main.py --mode live --snapshots 5 --interval 15 --region middle_east
```

### 2. Historical Data Mode
Requires an OpenSky account configured in `traffic`. It will fetch past data for a given region and timeframe.

```bash
uv run main.py --mode historical --start "2025-12-01 00:00" --stop "2025-12-01 01:00" --region iran
```
*(You can pass credentials via `--opensky-username` and `--opensky-password` arguments, or environment variables).*

### 3. File Mode
Process a previously saved raw `.parquet` file directly without reaching out to the OpenSky API.

```bash
uv run main.py --mode file --file data/raw/snapshot_20251201_120000.parquet
```

## Outputs

After running the pipeline, the system will output:
- **Processed Tables:** Located in `data/processed/` (`flight_features.parquet`, `flight_results.parquet`).
- **Models:** Saved to the `models/` directory for potential reuse.
- **Visualizations:** Found in the `output/` directory containing clustering visualizations, geographic scatter maps, and Voronoi anomaly maps.
- **Console Log:** A Top-10 Most Anomalous Flights report ranked by K-means distance.

