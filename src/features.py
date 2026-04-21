"""
Feature engineering for ADS-B anomaly detection.

Computes per-observation and per-flight features that capture indicators
of GPS jamming / spoofing:
  - Position jumps (haversine distance between consecutive reports)
  - Velocity plausibility
  - Altitude rate anomalies
  - Heading consistency
  - Reporting gaps
"""

import logging

import numpy as np
import pandas as pd

from src.config import (
    MAX_ACCELERATION_MS2,
    MAX_TURN_RATE_DEG_S,
    MAX_VELOCITY_MS,
    MAX_VERTICAL_RATE_MS,
)

logger = logging.getLogger(__name__)


def _haversine(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in metres."""
    R = 6_371_000  # Earth radius in metres
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _angle_diff(a, b):
    """Signed angular difference in degrees, wrapped to [-180, 180]."""
    a_arr = np.asarray(a, dtype="float64")
    b_arr = np.asarray(b, dtype="float64")
    return np.mod(a_arr - b_arr + 180.0, 360.0) - 180.0


def compute_observation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-observation columns computed from consecutive ADS-B reports
    within each icao24 group.

    Expects columns: icao24, timestamp (or time_position), latitude, longitude,
    velocity, baro_altitude, vertical_rate, true_track.
    """
    df = df.copy()

    # Ensure we have a usable time column
    if "timestamp" not in df.columns and "time_position" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time_position"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Ensure arithmetic-ready numeric dtypes (Arrow dtypes can fail on modulo ops).
    numeric_cols = [
        "latitude",
        "longitude",
        "velocity",
        "baro_altitude",
        "vertical_rate",
        "true_track",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.sort_values(["icao24", "timestamp"], inplace=True)

    grouped = df.groupby("icao24")

    # Time delta between consecutive reports (seconds)
    df["dt"] = grouped["timestamp"].diff().dt.total_seconds()

    # Position jump (metres)
    df["position_jump_m"] = _haversine(
        grouped["latitude"].shift(),
        grouped["longitude"].shift(),
        df["latitude"],
        df["longitude"],
    )

    # Speed derived from position jump vs. reported velocity
    df["derived_speed_ms"] = df["position_jump_m"] / df["dt"].replace(0, np.nan)
    df["speed_diff_ms"] = (df["derived_speed_ms"] - df["velocity"]).abs()

    # Acceleration (change in velocity / dt)
    df["acceleration_ms2"] = grouped["velocity"].diff().abs() / df["dt"].replace(0, np.nan)

    # Heading change rate (deg/s)
    prev_track = grouped["true_track"].shift()
    heading_delta = _angle_diff(df["true_track"], prev_track)
    df["heading_change_deg"] = np.abs(heading_delta)
    df["turn_rate_deg_s"] = df["heading_change_deg"] / df["dt"].replace(0, np.nan)

    # Altitude change rate derived from barometric altitude
    df["alt_change_rate"] = (
        grouped["baro_altitude"].diff().abs() / df["dt"].replace(0, np.nan)
    )

    logger.info("Computed observation-level features for %d rows.", len(df))
    return df


def flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean flag columns for observations that exceed physical limits.
    These serve as soft labels / pre-filters before ML clustering.
    """
    df = df.copy()

    df["flag_velocity"] = df["velocity"].abs() > MAX_VELOCITY_MS
    df["flag_vertical_rate"] = df["vertical_rate"].abs() > MAX_VERTICAL_RATE_MS
    df["flag_acceleration"] = df["acceleration_ms2"] > MAX_ACCELERATION_MS2
    df["flag_turn_rate"] = df["turn_rate_deg_s"] > MAX_TURN_RATE_DEG_S
    df["flag_position_jump"] = (
        df["position_jump_m"] > df["velocity"].clip(lower=1) * df["dt"] * 3
    )  # jump > 3× expected distance

    flag_cols = [
        "flag_velocity",
        "flag_vertical_rate",
        "flag_acceleration",
        "flag_turn_rate",
        "flag_position_jump",
    ]
    for col in flag_cols:
        df[col] = df[col].fillna(False)

    df["anomaly_score_rules"] = (
        df["flag_velocity"].astype("int64")
        + df["flag_vertical_rate"].astype("int64")
        + df["flag_acceleration"].astype("int64")
        + df["flag_turn_rate"].astype("int64")
        + df["flag_position_jump"].astype("int64")
    )

    n_flagged = (df["anomaly_score_rules"] > 0).sum()
    logger.info(
        "Rule-based flags: %d / %d observations flagged (%.1f%%).",
        n_flagged,
        len(df),
        100 * n_flagged / max(len(df), 1),
    )
    return df


def aggregate_flight_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate observation-level features into a single feature vector
    per flight (icao24). This is the input to K-means / other ML models.
    """
    agg = df.groupby("icao24").agg(
        n_reports=("timestamp", "count"),
        mean_velocity=("velocity", "mean"),
        std_velocity=("velocity", "std"),
        max_velocity=("velocity", "max"),
        mean_altitude=("baro_altitude", "mean"),
        std_altitude=("baro_altitude", "std"),
        mean_vertical_rate=("vertical_rate", lambda x: x.abs().mean()),
        max_vertical_rate=("vertical_rate", lambda x: x.abs().max()),
        mean_position_jump=("position_jump_m", "mean"),
        max_position_jump=("position_jump_m", "max"),
        mean_speed_diff=("speed_diff_ms", "mean"),
        max_speed_diff=("speed_diff_ms", "max"),
        mean_acceleration=("acceleration_ms2", "mean"),
        max_acceleration=("acceleration_ms2", "max"),
        mean_turn_rate=("turn_rate_deg_s", "mean"),
        max_turn_rate=("turn_rate_deg_s", "max"),
        mean_dt=("dt", "mean"),
        max_dt=("dt", "max"),
        total_anomaly_flags=("anomaly_score_rules", "sum"),
        pct_flagged=("anomaly_score_rules", lambda x: (x > 0).mean()),
        mean_latitude=("latitude", "mean"),
        mean_longitude=("longitude", "mean"),
    ).reset_index()

    # Fill NaNs that result from flights with only 1 report
    agg.fillna(0, inplace=True)

    logger.info("Aggregated features for %d flights.", len(agg))
    return agg
