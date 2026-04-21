"""
Bulk data collection + anomaly validation runner for ADS-B project.

This script:
1. Pulls historical ADS-B data across configured Middle East regions.
2. Runs the same feature engineering and anomaly pipeline as main.py.
3. Writes per-window outputs and an aggregate validation summary.

Example:
    python batch_collect_validate.py \
      --start "2025-12-01 00:00" \
      --stop "2025-12-02 00:00" \
      --window-hours 2
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import MIDDLE_EAST_BBOX, OUTPUT_DIR, PROCESSED_DIR, REGIONS
from src.data_collector import (
    fetch_historical_traffic,
    resolve_opensky_credentials,
    traffic_to_dataframe,
)
from src.detector import (
    find_optimal_k,
    prepare_features,
    run_dbscan,
    run_isolation_forest,
    run_kmeans,
    score_anomalies_kmeans,
)
from src.features import (
    aggregate_flight_features,
    compute_observation_features,
    flag_anomalies,
)

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    region: str
    start: str
    stop: str
    status: str
    n_raw_rows: int
    n_flights: int
    selected_k: int
    best_silhouette: float
    mean_anomaly_distance: float
    rule_flag_rate: float
    topk_rule_hit_rate: float
    iforest_anomaly_rate: float
    dbscan_noise_rate: float
    jaccard_kmeans_iforest: float
    jaccard_kmeans_dbscan: float
    jaccard_iforest_dbscan: float
    spearman_distance_vs_rules: float
    consensus_rate_2of3: float
    output_file: str


def parse_dt(value: str) -> datetime:
    dt = pd.to_datetime(value, utc=True, errors="raise")
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return dt.to_pydatetime().astimezone(timezone.utc)


def iter_windows(start: datetime, stop: datetime, step_hours: int) -> Iterable[tuple[datetime, datetime]]:
    cur = start
    step = timedelta(hours=step_hours)
    while cur < stop:
        nxt = min(cur + step, stop)
        yield cur, nxt
        cur = nxt


def jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def evaluate_window(
    raw_df: pd.DataFrame,
    region: str,
    window_start: datetime,
    window_stop: datetime,
    top_frac: float,
    forced_k: int,
    min_flights: int,
    save_per_window: bool,
) -> WindowResult:
    if raw_df.empty:
        return WindowResult(
            region=region,
            start=window_start.isoformat(),
            stop=window_stop.isoformat(),
            status="empty",
            n_raw_rows=0,
            n_flights=0,
            selected_k=0,
            best_silhouette=0.0,
            mean_anomaly_distance=0.0,
            rule_flag_rate=0.0,
            topk_rule_hit_rate=0.0,
            iforest_anomaly_rate=0.0,
            dbscan_noise_rate=0.0,
            jaccard_kmeans_iforest=0.0,
            jaccard_kmeans_dbscan=0.0,
            jaccard_iforest_dbscan=0.0,
            spearman_distance_vs_rules=0.0,
            consensus_rate_2of3=0.0,
            output_file="",
        )

    obs_df = compute_observation_features(raw_df)
    obs_df = flag_anomalies(obs_df)
    flight_df = aggregate_flight_features(obs_df)

    if len(flight_df) < min_flights:
        return WindowResult(
            region=region,
            start=window_start.isoformat(),
            stop=window_stop.isoformat(),
            status="too_few_flights",
            n_raw_rows=len(raw_df),
            n_flights=len(flight_df),
            selected_k=0,
            best_silhouette=0.0,
            mean_anomaly_distance=0.0,
            rule_flag_rate=float((flight_df["total_anomaly_flags"] > 0).mean()) if len(flight_df) else 0.0,
            topk_rule_hit_rate=0.0,
            iforest_anomaly_rate=0.0,
            dbscan_noise_rate=0.0,
            jaccard_kmeans_iforest=0.0,
            jaccard_kmeans_dbscan=0.0,
            jaccard_iforest_dbscan=0.0,
            spearman_distance_vs_rules=0.0,
            consensus_rate_2of3=0.0,
            output_file="",
        )

    X_scaled, _ = prepare_features(flight_df)
    k_candidates = list(range(2, min(11, len(flight_df))))
    if not k_candidates:
        return WindowResult(
            region=region,
            start=window_start.isoformat(),
            stop=window_stop.isoformat(),
            status="insufficient_for_clustering",
            n_raw_rows=len(raw_df),
            n_flights=len(flight_df),
            selected_k=0,
            best_silhouette=0.0,
            mean_anomaly_distance=0.0,
            rule_flag_rate=float((flight_df["total_anomaly_flags"] > 0).mean()),
            topk_rule_hit_rate=0.0,
            iforest_anomaly_rate=0.0,
            dbscan_noise_rate=0.0,
            jaccard_kmeans_iforest=0.0,
            jaccard_kmeans_dbscan=0.0,
            jaccard_iforest_dbscan=0.0,
            spearman_distance_vs_rules=0.0,
            consensus_rate_2of3=0.0,
            output_file="",
        )

    k_results = find_optimal_k(X_scaled, k_range=range(2, min(11, len(flight_df))))
    best_idx = int(np.argmax(k_results["silhouette"]))
    best_silhouette = float(k_results["silhouette"][best_idx])
    selected_k = forced_k if forced_k > 1 else int(k_results["k"][best_idx])

    km, labels = run_kmeans(X_scaled, k=selected_k)
    anomaly_distance = score_anomalies_kmeans(X_scaled, km, labels)
    db_labels = run_dbscan(X_scaled)
    if_labels = run_isolation_forest(X_scaled)

    flight_df = flight_df.copy()
    flight_df["cluster"] = labels
    flight_df["anomaly_distance"] = anomaly_distance
    flight_df["dbscan_label"] = db_labels
    flight_df["iforest_label"] = if_labels

    n_flights = len(flight_df)
    top_n = max(1, int(np.ceil(top_frac * n_flights)))
    top_idx = set(flight_df.nlargest(top_n, "anomaly_distance").index.tolist())

    rule_idx = set(flight_df.index[flight_df["total_anomaly_flags"] > 0].tolist())
    iforest_idx = set(flight_df.index[flight_df["iforest_label"] == -1].tolist())
    dbscan_idx = set(flight_df.index[flight_df["dbscan_label"] == -1].tolist())

    topk_rule_hit_rate = float((flight_df.loc[list(top_idx), "total_anomaly_flags"] > 0).mean())
    iforest_rate = float((flight_df["iforest_label"] == -1).mean())
    dbscan_rate = float((flight_df["dbscan_label"] == -1).mean())

    # Consensus = flagged by at least two methods among top-k KMeans, IF, DBSCAN.
    top_mask = flight_df.index.isin(top_idx)
    if_mask = flight_df["iforest_label"].eq(-1).to_numpy()
    db_mask = flight_df["dbscan_label"].eq(-1).to_numpy()
    method_votes = top_mask.astype("int64") + if_mask.astype("int64") + db_mask.astype("int64")
    consensus_rate = float((method_votes >= 2).mean())

    spearman_corr = float(
        flight_df["anomaly_distance"].corr(flight_df["total_anomaly_flags"], method="spearman")
    )
    if np.isnan(spearman_corr):
        spearman_corr = 0.0

    output_file = ""
    if save_per_window:
        stamp = f"{region}_{window_start.strftime('%Y%m%dT%H%M%SZ')}_{window_stop.strftime('%Y%m%dT%H%M%SZ')}"
        out_path = PROCESSED_DIR / f"flight_results_{stamp}.parquet"
        flight_df.to_parquet(out_path, index=False)
        output_file = str(out_path)

    return WindowResult(
        region=region,
        start=window_start.isoformat(),
        stop=window_stop.isoformat(),
        status="ok",
        n_raw_rows=len(raw_df),
        n_flights=n_flights,
        selected_k=selected_k,
        best_silhouette=best_silhouette,
        mean_anomaly_distance=float(flight_df["anomaly_distance"].mean()),
        rule_flag_rate=float((flight_df["total_anomaly_flags"] > 0).mean()),
        topk_rule_hit_rate=topk_rule_hit_rate,
        iforest_anomaly_rate=iforest_rate,
        dbscan_noise_rate=dbscan_rate,
        jaccard_kmeans_iforest=jaccard(top_idx, iforest_idx),
        jaccard_kmeans_dbscan=jaccard(top_idx, dbscan_idx),
        jaccard_iforest_dbscan=jaccard(iforest_idx, dbscan_idx),
        spearman_distance_vs_rules=spearman_corr,
        consensus_rate_2of3=consensus_rate,
        output_file=output_file,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk collect and validate ADS-B anomaly detection.")
    parser.add_argument("--start", required=True, help="UTC start datetime, e.g. '2025-12-01 00:00'")
    parser.add_argument("--stop", required=True, help="UTC stop datetime, e.g. '2025-12-02 00:00'")
    parser.add_argument("--window-hours", type=int, default=2, help="Window size in hours")
    parser.add_argument(
        "--regions",
        default="all",
        help="Comma list of region keys from config.REGIONS; use 'all' for all + middle_east",
    )
    parser.add_argument("--k", type=int, default=0, help="Force K for KMeans (0 means auto by silhouette)")
    parser.add_argument("--top-frac", type=float, default=0.10, help="Top anomaly fraction for KMeans")
    parser.add_argument("--min-flights", type=int, default=20, help="Minimum flights to run full clustering")
    parser.add_argument("--save-per-window", action="store_true", help="Save each window result parquet")
    parser.add_argument("--opensky-username", default=None, help="OpenSky username")
    parser.add_argument("--opensky-password", default=None, help="OpenSky password")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def get_region_map(regions_arg: str) -> dict[str, dict[str, float]]:
    if regions_arg.strip().lower() == "all":
        return {"middle_east": MIDDLE_EAST_BBOX, **REGIONS}

    wanted = [r.strip() for r in regions_arg.split(",") if r.strip()]
    out: dict[str, dict[str, float]] = {}
    for name in wanted:
        if name == "middle_east":
            out[name] = MIDDLE_EAST_BBOX
        elif name in REGIONS:
            out[name] = REGIONS[name]
        else:
            raise ValueError(f"Unknown region '{name}'. Valid: middle_east or {sorted(REGIONS.keys())}")
    return out


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    start_dt = parse_dt(args.start)
    stop_dt = parse_dt(args.stop)
    if stop_dt <= start_dt:
        raise ValueError("--stop must be greater than --start")

    region_map = get_region_map(args.regions)
    creds = resolve_opensky_credentials(args.opensky_username, args.opensky_password)

    logger.info("Running %d regions across windows from %s to %s", len(region_map), start_dt, stop_dt)

    results: list[WindowResult] = []

    for region_name, bounds in region_map.items():
        for win_start, win_stop in iter_windows(start_dt, stop_dt, args.window_hours):
            logger.info("Region=%s Window=%s -> %s", region_name, win_start, win_stop)
            traffic_obj = fetch_historical_traffic(
                start=win_start.strftime("%Y-%m-%d %H:%M"),
                stop=win_stop.strftime("%Y-%m-%d %H:%M"),
                region=region_name if region_name in REGIONS else "middle_east",
                bounds=bounds,
                auth=creds,
            )
            raw_df = traffic_to_dataframe(traffic_obj)
            result = evaluate_window(
                raw_df=raw_df,
                region=region_name,
                window_start=win_start,
                window_stop=win_stop,
                top_frac=args.top_frac,
                forced_k=args.k,
                min_flights=args.min_flights,
                save_per_window=args.save_per_window,
            )
            results.append(result)

    summary_df = pd.DataFrame([asdict(r) for r in results])

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = OUTPUT_DIR / f"validation_summary_{ts}.csv"
    json_path = OUTPUT_DIR / f"validation_summary_{ts}.json"

    summary_df.to_csv(csv_path, index=False)
    with Path(json_path).open("w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    ok_df = summary_df[summary_df["status"] == "ok"]
    print("\n" + "=" * 80)
    print("BULK COLLECTION + VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total windows processed: {len(summary_df)}")
    print(f"Successful windows:      {len(ok_df)}")
    if len(ok_df):
        print(f"Mean silhouette:         {ok_df['best_silhouette'].mean():.4f}")
        print(f"Mean top-k rule hit:     {ok_df['topk_rule_hit_rate'].mean():.4f}")
        print(f"Mean consensus (2 of 3): {ok_df['consensus_rate_2of3'].mean():.4f}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
