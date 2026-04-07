"""
Anomaly detection models for ADS-B data.

Implements:
  - K-means clustering with automatic K selection (elbow / silhouette)
  - Cluster-based anomaly scoring
  - Optional: Isolation Forest and DBSCAN for comparison
"""

import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_K, MODELS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

# Features used for clustering (adjust as needed)
CLUSTER_FEATURES = [
    "mean_velocity",
    "std_velocity",
    "max_velocity",
    "mean_altitude",
    "std_altitude",
    "mean_vertical_rate",
    "max_vertical_rate",
    "mean_position_jump",
    "max_position_jump",
    "mean_speed_diff",
    "max_speed_diff",
    "mean_acceleration",
    "max_acceleration",
    "mean_turn_rate",
    "max_turn_rate",
    "mean_dt",
    "max_dt",
    "total_anomaly_flags",
    "pct_flagged",
]


# ── Preprocessing ───────────────────────────────────────────────────
def prepare_features(
    flight_df: pd.DataFrame,
    features: Optional[list] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Scale the feature matrix. Returns (scaled DataFrame, fitted scaler).
    """
    features = features or CLUSTER_FEATURES
    X = flight_df[features].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=features, index=X.index
    )
    return X_scaled, scaler


# ── K-means with automatic K selection ──────────────────────────────
def find_optimal_k(
    X_scaled: pd.DataFrame,
    k_range: range = range(2, 11),
) -> dict:
    """
    Run K-means for each k and return inertia + silhouette scores.
    """
    results = {"k": [], "inertia": [], "silhouette": []}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if k > 1 else 0.0
        results["k"].append(k)
        results["inertia"].append(km.inertia_)
        results["silhouette"].append(sil)
        logger.info("K=%d  inertia=%.1f  silhouette=%.4f", k, km.inertia_, sil)

    return results


def run_kmeans(
    X_scaled: pd.DataFrame,
    k: int = DEFAULT_K,
) -> tuple[KMeans, np.ndarray]:
    """
    Fit K-means and return the model + cluster labels.
    """
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    logger.info("K-means (k=%d): silhouette=%.4f", k, silhouette_score(X_scaled, labels))
    return km, labels


def score_anomalies_kmeans(
    X_scaled: pd.DataFrame,
    km: KMeans,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute anomaly score = distance from each point to its cluster centre.
    Points with high distance are more anomalous within their cluster.
    """
    centres = km.cluster_centers_
    distances = np.linalg.norm(X_scaled.values - centres[labels], axis=1)
    return distances


# ── DBSCAN ──────────────────
def run_dbscan(
    X_scaled: pd.DataFrame,
    eps: float = 1.5,
    min_samples: int = 5,
) -> np.ndarray:
    """
    Run DBSCAN. Points labelled -1 are noise / anomalies.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    logger.info("DBSCAN: %d clusters, %d noise points.", n_clusters, n_noise)
    return labels


# ── Isolation Forest ───────────────────────────────
def run_isolation_forest(
    X_scaled: pd.DataFrame,
    contamination: float = 0.05,
) -> np.ndarray:
    """
    Run Isolation Forest. Returns array of -1 (anomaly) / 1 (normal).
    """
    iso = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_estimators=200,
    )
    preds = iso.fit_predict(X_scaled)
    n_anomalies = (preds == -1).sum()
    logger.info("Isolation Forest: %d anomalies detected.", n_anomalies)
    return preds


# ── Persistence ─────────────────────────────────────────────────────
def save_model(model, scaler: StandardScaler, tag: str = "kmeans"):
    """Save model and scaler to disk."""
    joblib.dump(model, MODELS_DIR / f"{tag}_model.pkl")
    joblib.dump(scaler, MODELS_DIR / f"{tag}_scaler.pkl")
    logger.info("Saved model → %s", MODELS_DIR / f"{tag}_model.pkl")


def load_model(tag: str = "kmeans"):
    """Load model and scaler from disk."""
    model = joblib.load(MODELS_DIR / f"{tag}_model.pkl")
    scaler = joblib.load(MODELS_DIR / f"{tag}_scaler.pkl")
    return model, scaler
