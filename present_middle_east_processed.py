"""
Generate presentation visualisations and statistics from the pre-processed Middle East data.
"""
import logging
from pathlib import Path
import pandas as pd

from src.config import PROCESSED_DIR, OUTPUT_DIR
from src.visualize import plot_geographic, plot_voronoi, close_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

def main():
    # ind all processed Middle East parquet files
    me_files = list(PROCESSED_DIR.glob("flight_results_middle_east_*.parquet"))
    
    if not me_files:
        logger.error(f"No Middle East files found in {PROCESSED_DIR}")
        return

    logger.info(f"Loading {len(me_files)} processed files for the Middle East...")
    
    # Combine them into one massive dataframe
    dfs = []
    for file in me_files:
        try:
            df = pd.read_parquet(file)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.error(f"Could not load {file.name}: {e}")
            
    if not dfs:
        logger.error("No valid data loaded.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate in case overlapping windows caused duplicate flights
    original_len = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["icao24"])
    logger.info(f"Combined data contains {len(combined_df)} unique flights (filtered down from {original_len}).")

    # Generate Visualizations
    logger.info("Generating Geographic Scatter map...")
    plot_geographic(combined_df, save=False).savefig(OUTPUT_DIR / "presentation_middle_east_scatter.png", dpi=300)
    
    logger.info("Generating Computational Geometry (Voronoi) map...")
    plot_voronoi(combined_df, save=False).savefig(OUTPUT_DIR / "presentation_middle_east_voronoi.png", dpi=300)
    close_all()
    
    # Print aggregate presentation statistics
    print("\n" + "="*70)
    print(" 📊 MIDDLE EAST AGGREGATE FINDINGS FOR PRESENTATION")
    print("="*70)
    print(f"Total Unique Flights Analyzed: {len(combined_df):,}")
    
    # Physical Rule Violations
    flagged_pct = (combined_df['total_anomaly_flags'] > 0).mean() * 100
    print(f"Flights Violating Physics (Spoofing indicator): {flagged_pct:.2f}%")
    
    # Unsupervised Anomaly Consensus (agreed by K-Means, DBSCAN, IForest)
    if 'dbscan_label' in combined_df.columns and 'iforest_label' in combined_df.columns:
        
        # Define top 10% KDE distance as anomalous for K-Means
        dist_threshold = combined_df['anomaly_distance'].quantile(0.90)
        kmeans_anom = combined_df['anomaly_distance'] > dist_threshold
        dbscan_anom = combined_df['dbscan_label'] == -1
        iforest_anom = combined_df['iforest_label'] == -1
        
        # At least 2 models agree
        consensus = (kmeans_anom.astype(int) + dbscan_anom.astype(int) + iforest_anom.astype(int)) >= 2
        print(f"High-Confidence Anomalies (Model Consensus): {consensus.sum():,} ({consensus.mean()*100:.2f}%)")
    
    print("\nTop 5 Most Anomalous Flights mathematically (furthest from Voronoi centers):")
    top_5 = combined_df.nlargest(5, 'anomaly_distance')[
        ['icao24', 'total_anomaly_flags', 'anomaly_distance', 'mean_latitude', 'mean_longitude']
    ]
    print(top_5.to_string(index=False))
    print("="*70)
    print(f"✅ High-resolution charts saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()