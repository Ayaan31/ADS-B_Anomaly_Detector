"""
Process existing raw parquet files locally and generate an aggregate validation summary.
"""
import logging
from dataclasses import asdict
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path

from src.config import RAW_DIR, OUTPUT_DIR
from src.data_collector import load_raw
from batch_collect_validate import evaluate_window

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

def main():
    parquet_files = list(RAW_DIR.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"No parquet files found in {RAW_DIR}")
        return

    logger.info(f"Found {len(parquet_files)} parquet files. Starting processing...")
    results = []

    for file_path in parquet_files:
        logger.info(f"Processing {file_path.name}...")
        try:
            # Load raw data
            raw_df = load_raw(file_path)
            
            if raw_df.empty:
                logger.warning(f"File {file_path.name} is empty. Skipping.")
                continue

            # Deduce region and timestamps (fallback to file modification time if needed)
            # Assuming filenames like hist_iran_20260329_182125.parquet
            parts = file_path.stem.split('_')
            region = parts[1] if len(parts) > 1 else "unknown"
            
            # Using min/max time from the data itself as the window
            if "last_position" in raw_df.columns:
                window_start = raw_df["last_position"].min()
                window_stop = raw_df["last_position"].max()
            else:
                window_start = datetime.now(timezone.utc)
                window_stop = datetime.now(timezone.utc)

            # Evaluate the window using your existing pipeline
            result = evaluate_window(
                raw_df=raw_df,
                region=region,
                window_start=window_start,
                window_stop=window_stop,
                top_frac=0.10,
                forced_k=0,
                min_flights=20,
                save_per_window=True
            )
            
            # Keep track of file name in result if possible or just append
            results.append(result)
            logger.info(f"Finished {file_path.name}: KMeans silhouette = {result.best_silhouette:.3f}, Rule Hit Rate = {result.rule_flag_rate:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    if not results:
        logger.warning("No files were successfully processed.")
        return

    # Aggregate into a summary dataframe
    summary_df = pd.DataFrame([asdict(r) for r in results])
    
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = OUTPUT_DIR / f"local_validation_summary_{ts}.csv"
    summary_df.to_csv(csv_path, index=False)
    
    logger.info("=" * 60)
    logger.info(f"Processing complete! Processed {len(results)} valid files.")
    logger.info(f"Summary saved to: {csv_path}")
    logger.info("=" * 60)
    
    # Print a quick preview of averages for the presentation
    print("\nPresentation Quick Stats (Averages across all files):")
    print(f"Average Outliers Detected (Top K): {summary_df['topk_rule_hit_rate'].mean():.2%}")
    print(f"Jaccard Similarity (KMeans vs IForest): {summary_df['jaccard_kmeans_iforest'].mean():.3f}")
    print(f"Consensus Rate (2/3 Models Agree): {summary_df['consensus_rate_2of3'].mean():.2%}")
    print(f"Rule Flag Rate (Physics violations): {summary_df['rule_flag_rate'].mean():.2%}")

if __name__ == "__main__":
    main()
