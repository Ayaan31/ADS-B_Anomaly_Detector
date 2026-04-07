"""
Visualization module for ADS-B anomaly detection.

Produces:
  - Elbow / silhouette plots for K selection
  - 2-D cluster scatter plots
  - Geographic scatter (aircraft on map, coloured by anomaly)
  - Voronoi diagram overlaid on the Middle East map
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.decomposition import PCA

from src.config import MIDDLE_EAST_BBOX, OUTPUT_DIR

logger = logging.getLogger(__name__)


# ── K-selection plots ───────────────────────────────────────────────
def plot_elbow_silhouette(k_results: dict, save: bool = True) -> plt.Figure:
    """Side-by-side elbow (inertia) and silhouette plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(k_results["k"], k_results["inertia"], "o-")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")

    ax2.plot(k_results["k"], k_results["silhouette"], "s-", color="orange")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")

    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "elbow_silhouette.png", dpi=150)
        logger.info("Saved elbow/silhouette plot.")
    return fig


# ── Cluster scatter (PCA‑reduced) ──────────────────────────────────
def plot_clusters(
    X_scaled: pd.DataFrame,
    labels: np.ndarray,
    anomaly_scores: Optional[np.ndarray] = None,
    save: bool = True,
) -> plt.Figure:
    """2-D PCA projection of clusters, optionally sized by anomaly score."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    sizes = 20 if anomaly_scores is None else 10 + anomaly_scores * 5
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap="tab10", s=sizes, alpha=0.7, edgecolors="k", linewidths=0.3,
    )
    ax.set_xlabel(f"PC-1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC-2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("K-means Clusters (PCA Projection)")
    fig.colorbar(scatter, ax=ax, label="Cluster")

    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "clusters_pca.png", dpi=150)
        logger.info("Saved cluster scatter plot.")
    return fig


# ── Geographic scatter ──────────────────────────────────────────────
def plot_geographic(
    flight_df: pd.DataFrame,
    label_col: str = "cluster",
    score_col: Optional[str] = "anomaly_distance",
    save: bool = True,
) -> plt.Figure:
    """
    Plot flights on a lon/lat map coloured by cluster or anomaly label.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    if score_col and score_col in flight_df.columns:
        scatter = ax.scatter(
            flight_df["mean_longitude"],
            flight_df["mean_latitude"],
            c=flight_df[score_col],
            cmap="YlOrRd",
            s=15,
            alpha=0.8,
            edgecolors="k",
            linewidths=0.2,
        )
        fig.colorbar(scatter, ax=ax, label="Anomaly Distance")
    else:
        ax.scatter(
            flight_df["mean_longitude"],
            flight_df["mean_latitude"],
            c=flight_df[label_col] if label_col in flight_df.columns else "steelblue",
            cmap="tab10",
            s=15,
            alpha=0.8,
            edgecolors="k",
            linewidths=0.2,
        )

    bbox = MIDDLE_EAST_BBOX
    ax.set_xlim(bbox["lon_min"], bbox["lon_max"])
    ax.set_ylim(bbox["lat_min"], bbox["lat_max"])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("ADS-B Flights – Middle East")
    ax.set_aspect("equal")

    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "geographic_scatter.png", dpi=150)
        logger.info("Saved geographic scatter plot.")
    return fig


# ── Voronoi diagram ────────────────────────────────────────────────
def plot_voronoi(
    flight_df: pd.DataFrame,
    seed_points: Optional[np.ndarray] = None,
    label_col: str = "cluster",
    save: bool = True,
) -> plt.Figure:
    """
    Draw a Voronoi tessellation over the Middle East.

    Parameters
    ----------
    flight_df : DataFrame with mean_longitude, mean_latitude, and label columns.
    seed_points : (N, 2) array of [lon, lat] seed points for the Voronoi cells.
                  Defaults to K-means cluster centres' geographic positions.
    label_col : column to colour anomaly density per Voronoi cell.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    bbox = MIDDLE_EAST_BBOX

    # --- seed points ---
    if seed_points is None:
        # Use per-cluster geographic centroids as seeds
        centroids = (
            flight_df.groupby(label_col)[["mean_longitude", "mean_latitude"]]
            .mean()
            .values
        )
        seed_points = centroids

    # Add far-away dummy points so all Voronoi regions are bounded
    far = 1000
    dummies = np.array([[-far, -far], [far, -far], [-far, far], [far, far]])
    seeds_ext = np.vstack([seed_points, dummies])

    vor = Voronoi(seeds_ext)
    voronoi_plot_2d(
        vor, ax=ax, show_vertices=False, show_points=False,
        line_colors="grey", line_width=0.8, line_alpha=0.6,
    )

    # --- overlay flights coloured by anomaly density per cell ---
    ax.scatter(
        flight_df["mean_longitude"],
        flight_df["mean_latitude"],
        c=flight_df.get("anomaly_distance", flight_df.get(label_col, 0)),
        cmap="YlOrRd",
        s=10,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.2,
    )

    # Highlight seed points
    ax.scatter(
        seed_points[:, 0], seed_points[:, 1],
        c="blue", marker="X", s=120, zorder=5, label="Voronoi Seeds",
    )

    ax.set_xlim(bbox["lon_min"], bbox["lon_max"])
    ax.set_ylim(bbox["lat_min"], bbox["lat_max"])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Voronoi Tessellation – ADS-B Anomaly Density")
    ax.legend()
    ax.set_aspect("equal")

    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "voronoi_diagram.png", dpi=150)
        logger.info("Saved Voronoi diagram.")
    return fig


# ── Close all matplotlib figures (memory hygiene) ───────────────────
def close_all():
    plt.close("all")
