"""
ADS-B Anomaly Detector - Main Pipeline
=======================================
Collect ADS-B data from OpenSky, engineer features, run K-means clustering,
and produce visualisations (geographic scatter + Voronoi diagram).

Usage:
    # Live data mode (REST API, no auth required):
    python main.py --mode live --snapshots 5 --interval 15

    # Historical data mode (requires OpenSky account configured in traffic):
    python main.py --mode historical --start "2025-12-01 00:00" --stop "2025-12-01 01:00" --region iran

    # From a previously saved parquet file:
    python main.py --mode file --file data/raw/snapshot_20251201_120000.parquet
"""

import argparse
import logging
import sys

import pandas as pd

from src.config import DEFAULT_K, PROCESSED_DIR
from src.data_collector import (
    collect_live_snapshots,
    ensure_traffic_config_credentials,
    fetch_historical_traffic,
    load_raw,
    resolve_opensky_credentials,
    save_raw,
    traffic_to_dataframe,
)
from src.detector import (
    find_optimal_k,
    prepare_features,
    run_dbscan,
    run_isolation_forest,
    run_kmeans,
    save_model,
    score_anomalies_kmeans,
)
from src.features import (
    aggregate_flight_features,
    compute_observation_features,
    flag_anomalies,
)
from src.visualize import (
    close_all,
    plot_clusters,
    plot_elbow_silhouette,
    plot_geographic,
    plot_voronoi,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def ingest(args) -> pd.DataFrame:
    """Return a raw DataFrame based on the chosen mode."""
    credentials = resolve_opensky_credentials(
        username=args.opensky_username,
        password=args.opensky_password,
    )

    if args.mode == "live":
        logger.info("Collecting %d live snapshots …", args.snapshots)
        df = collect_live_snapshots(
            n_snapshots=args.snapshots,
            interval_sec=args.interval,
            auth=credentials,
        )
        if not df.empty:
            save_raw(df, tag="live")
        return df

    elif args.mode == "historical":
        if credentials is not None:
            ensure_traffic_config_credentials(
                username=credentials[0],
                password=credentials[1],
                overwrite=args.overwrite_traffic_credentials,
            )
        else:
            logger.info(
                "No credentials passed via CLI/env; traffic will use existing ~/.config/traffic/traffic.conf"
            )

        traffic_obj = fetch_historical_traffic(
            start=args.start,
            stop=args.stop,
            region=args.region,
            auth=credentials,
        )
        df = traffic_to_dataframe(traffic_obj)
        if not df.empty:
            save_raw(df, tag=f"hist_{args.region}")
        return df

    elif args.mode == "file":
        logger.info("Loading from file: %s", args.file)
        return load_raw(args.file)

    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)


def run_pipeline(args):
    raw_df = ingest(args)
    if raw_df.empty:
        logger.error("No data available. Exiting.")
        return

    logger.info("Raw data: %d rows, %d columns.", *raw_df.shape)

    obs_df = compute_observation_features(raw_df)
    obs_df = flag_anomalies(obs_df)
    flight_df = aggregate_flight_features(obs_df)
    logger.info("Flight-level features: %d flights.", len(flight_df))

    flight_df.to_parquet(PROCESSED_DIR / "flight_features.parquet", index = False)

    X_scaled, scaler = prepare_features(flight_df)

    k_results = find_optimal_k(X_scaled, k_range = range(2, min(11, len(flight_df))))
    plot_elbow_silhouette(k_results)

    best_k = args.k if args.k else k_results["k"][
        max(range(len(k_results["silhouette"])), key = lambda i: k_results["silhouette"][i])
    ]
    logger.info("Selected K = %d", best_k)

    km, labels = run_kmeans(X_scaled, k = best_k)
    anomaly_distances = score_anomalies_kmeans(X_scaled, km, labels)

    flight_df["cluster"] = labels
    flight_df["anomaly_distance"] = anomaly_distances

    flight_df["dbscan_label"] = run_dbscan(X_scaled)
    flight_df["iforest_label"] = run_isolation_forest(X_scaled)

    flight_df.to_parquet(PROCESSED_DIR / "flight_results.parquet", index = False)

    save_model(km, scaler, tag = "kmeans")

    plot_clusters(X_scaled, labels, anomaly_distances)
    plot_geographic(flight_df)
    plot_voronoi(flight_df)
    close_all()

    top_anomalies = flight_df.nlargest(10, "anomaly_distance")
    print("\n" + "=" * 70)
    print("  TOP-10 MOST ANOMALOUS FLIGHTS (by K-means distance)")
    print("=" * 70)
    print(
        top_anomalies[
            ["icao24", "cluster", "anomaly_distance", "total_anomaly_flags",
             "mean_latitude", "mean_longitude"]
        ].to_string(index = False)
    )
    print("=" * 70)
    print(f"\nOutputs saved to: {PROCESSED_DIR.parent.parent / 'output'}")

def parse_args():
    p = argparse.ArgumentParser(description = "ADS-B Anomaly Detector")
    p.add_argument(
        "--mode", choices = ["live", "historical", "file"], default = "live",
        help = "Data source mode (default: live)",
    )
    p.add_argument("--snapshots", type = int, default = 5, help = "Number of live snapshots")
    p.add_argument("--interval", type = int, default = 15, help = "Seconds between snapshots")
    p.add_argument("--start", type = str, help = "Start datetime (ISO)")
    p.add_argument("--stop", type = str, help = "Stop datetime (ISO)")
    p.add_argument("--region", type = str, default = "middle_east", help = "Region key")
    p.add_argument("--file", type = str, help = "Path to raw parquet file")
    p.add_argument(
        "--opensky-username",
        type = str,
        default = None,
        help = "OpenSky username (or set OPENSKY_USERNAME)",
    )
    p.add_argument(
        "--opensky-password",
        type = str,
        default = None,
        help = "OpenSky password (or set OPENSKY_PASSWORD)",
    )
    p.add_argument(
        "--overwrite-traffic-credentials",
        action = "store_true",
        help = "Overwrite existing credentials in traffic.conf",
    )
    p.add_argument("--k", type = int, default = 0, help = "Force K (0 = auto-select)")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())