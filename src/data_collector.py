"""
Data collection module for ADS-B data via OpenSky Network.

Provides two approaches:
  1. REST API  – live snapshots (no auth needed, limited history)
  2. traffic / pyopensky – historical queries (requires OpenSky account)
"""

import logging
import os
import time
import types
from configparser import ConfigParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.config import MIDDLE_EAST_BBOX, RAW_DIR, REGIONS

logger = logging.getLogger(__name__)

# ── Column names returned by the OpenSky REST API ───────────────────
STATE_COLUMNS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "geo_altitude",
    "squawk", "spi", "position_source",
]


def resolve_opensky_credentials(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """
    Resolve OpenSky credentials from CLI args first, then environment variables.

    Environment variables:
      - OPENSKY_USERNAME
      - OPENSKY_PASSWORD
    """
    resolved_username = username or os.getenv("OPENSKY_USERNAME")
    resolved_password = password or os.getenv("OPENSKY_PASSWORD")

    if not (resolved_username and resolved_password):
        config_path = Path.home() / ".config" / "traffic" / "traffic.conf"
        if config_path.exists():
            parser = ConfigParser()
            parser.read(config_path)
            if parser.has_section("opensky"):
                resolved_username = resolved_username or parser.get(
                    "opensky", "username", fallback=None
                )
                resolved_password = resolved_password or parser.get(
                    "opensky", "password", fallback=None
                )

    if resolved_username and resolved_password:
        return resolved_username, resolved_password
    return None


def ensure_traffic_config_credentials(
    username: str,
    password: str,
    overwrite: bool = False,
) -> Path:
    """
    Ensure traffic config has OpenSky credentials for historical Trino queries.

    Writes to: ~/.config/traffic/traffic.conf
    """
    config_path = Path.home() / ".config" / "traffic" / "traffic.conf"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    parser = ConfigParser()
    if config_path.exists():
        parser.read(config_path)

    if not parser.has_section("opensky"):
        parser.add_section("opensky")

    # Avoid overwriting existing credentials unless requested.
    if overwrite or not parser.has_option("opensky", "username"):
        parser.set("opensky", "username", username)
    if overwrite or not parser.has_option("opensky", "password"):
        parser.set("opensky", "password", password)

    with config_path.open("w", encoding="utf-8") as f:
        parser.write(f)

    logger.info("OpenSky credentials available for traffic at %s", config_path)
    return config_path


# ─────────────────────────────────────────────────────────────────────
# 1.  REST API collector (live snapshots)
# ─────────────────────────────────────────────────────────────────────
def fetch_live_states(
    bbox: Optional[dict] = None,
    auth: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Fetch current aircraft states from the OpenSky REST API.

    Parameters
    ----------
    bbox : dict, optional
        Bounding box with keys lat_min, lat_max, lon_min, lon_max.
        Defaults to MIDDLE_EAST_BBOX.
    auth : tuple, optional
        (username, password) for authenticated requests (higher rate limits).

    Returns
    -------
    pd.DataFrame  with one row per aircraft.
    """
    bbox = bbox or MIDDLE_EAST_BBOX
    url = "https://opensky-network.org/api/states/all"
    params = {
        "lamin": bbox["lat_min"],
        "lamax": bbox["lat_max"],
        "lomin": bbox["lon_min"],
        "lomax": bbox["lon_max"],
    }

    response = requests.get(url, params=params, auth=auth, timeout=30)
    response.raise_for_status()
    data = response.json()

    states = data.get("states", [])
    if not states:
        logger.warning("No states returned from OpenSky API.")
        return pd.DataFrame(columns=STATE_COLUMNS)

    df = pd.DataFrame(states, columns=STATE_COLUMNS)
    df["callsign"] = df["callsign"].str.strip()
    df["timestamp"] = pd.to_datetime(data["time"], unit="s", utc=True)
    logger.info("Fetched %d live aircraft states.", len(df))
    return df


def collect_live_snapshots(
    n_snapshots: int = 10,
    interval_sec: int = 10,
    bbox: Optional[dict] = None,
    auth: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Collect multiple consecutive live snapshots and concatenate them.
    Useful for building short-term trajectory segments from the REST API.
    """
    frames = []
    for i in range(n_snapshots):
        logger.info("Snapshot %d/%d", i + 1, n_snapshots)
        try:
            df = fetch_live_states(bbox=bbox, auth=auth)
            frames.append(df)
        except requests.RequestException as exc:
            logger.error("Snapshot %d failed: %s", i + 1, exc)
        if i < n_snapshots - 1:
            time.sleep(interval_sec)

    if not frames:
        return pd.DataFrame(columns=STATE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────
# 2.  Historical data via the `traffic` library
# ─────────────────────────────────────────────────────────────────────
def fetch_historical_traffic(
    start: str,
    stop: str,
    region: str = "middle_east",
    bounds: Optional[dict] = None,
    auth: Optional[tuple[str, str]] = None,
):
    """
    Fetch historical ADS-B data using the traffic library (backed by pyopensky).

    Requires an OpenSky account configured in traffic's settings
    (see: https://traffic-viz.github.io/opensky_usage.html).

    Parameters
    ----------
    start : str   ISO datetime, e.g. "2025-12-01 00:00"
    stop  : str   ISO datetime, e.g. "2025-12-01 01:00"
    region : str  Key in REGIONS dict, or "middle_east" for the full bbox.
    bounds : dict Override bounding box.

    Returns
    -------
    traffic.core.Traffic object (or None if no data).
    """
    if bounds is None:
        bounds = REGIONS.get(region, MIDDLE_EAST_BBOX)

    logger.info(
        "Querying OpenSky historical data: %s → %s  [%s]", start, stop, region
    )

    # First try via traffic (legacy path in many examples).
    try:
        # Compatibility shim: traffic<=2.10 imports `impala` from pyopensky,
        # but pyopensky>=2.16 removed that module. Inject a no-op module so
        # Trino-based queries can still initialize.
        import sys
        import pyopensky

        if not hasattr(pyopensky, "impala"):
            impala_module = types.ModuleType("pyopensky.impala")
            sys.modules.setdefault("pyopensky.impala", impala_module)
            setattr(pyopensky, "impala", impala_module)

        from traffic.data import opensky

        t = opensky.history(
            start=start,
            stop=stop,
            bounds=(
                bounds["lon_min"],
                bounds["lat_min"],
                bounds["lon_max"],
                bounds["lat_max"],
            ),
        )
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "traffic.data.opensky path unavailable (%s); falling back to pyopensky.trino",
            exc,
        )
        from pyopensky.trino import Trino

        trino_client = Trino(
            username=auth[0] if auth else None,
            password=auth[1] if auth else None,
        )
        t = trino_client.history(
            start=start,
            stop=stop,
            bounds=(
                bounds["lon_min"],
                bounds["lat_min"],
                bounds["lon_max"],
                bounds["lat_max"],
            ),
        )

    if t is None:
        logger.warning("No historical data returned for the given query.")
        return None

    logger.info("Retrieved %d flights from historical data.", len(t))
    return t


def traffic_to_dataframe(traffic_obj) -> pd.DataFrame:
    """Convert a traffic.core.Traffic object into a flat pandas DataFrame."""
    if traffic_obj is None:
        return pd.DataFrame()

    if isinstance(traffic_obj, pd.DataFrame):
        df = traffic_obj.reset_index(drop=True)
    else:
        df = traffic_obj.data.reset_index(drop=True)

    # Normalize column names between traffic and pyopensky/trino outputs.
    rename_map = {
        "time": "timestamp",
        "lat": "latitude",
        "lon": "longitude",
        "baroaltitude": "baro_altitude",
        "geoaltitude": "geo_altitude",
        "vertrate": "vertical_rate",
        "heading": "true_track",
        "onground": "on_ground",
        "lastposupdate": "time_position",
        "lastcontact": "last_contact",
    }
    df = df.rename(columns=rename_map)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    elif "time_position" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time_position"], utc=True, errors="coerce")

    return df


# ─────────────────────────────────────────────────────────────────────
# 3.  Persistence helpers
# ─────────────────────────────────────────────────────────────────────
def save_raw(df: pd.DataFrame, tag: str = "snapshot") -> str:
    """Save a raw DataFrame to the data/raw directory as Parquet."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RAW_DIR / f"{tag}_{ts}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved raw data → %s", path)
    return str(path)


def load_raw(filename: str) -> pd.DataFrame:
    """Load a raw Parquet file from data/raw/."""
    path = RAW_DIR / filename
    return pd.read_parquet(path)
