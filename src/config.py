"""
Configuration constants for the ADS-B Anomaly Detector.
"""

# ── Middle East bounding box (lat/lon) ──────────────────────────────
# Covers Iran, Iraq, Syria, Israel, Jordan, UAE, Saudi Arabia, etc.
MIDDLE_EAST_BBOX = {
    "lat_min": 12.0,   # Southern tip of Yemen
    "lat_max": 42.0,   # Northern Turkey/Iran border
    "lon_min": 25.0,    # Eastern Mediterranean
    "lon_max": 63.0,    # Eastern Iran / Western Pakistan
}

# ── Sub-regions of interest ─────────────────────────────────────────
REGIONS = {
    "iran": {"lat_min": 25.0, "lat_max": 40.0, "lon_min": 44.0, "lon_max": 63.0},
    "israel": {"lat_min": 29.0, "lat_max": 33.5, "lon_min": 34.0, "lon_max": 36.0},
    "uae": {"lat_min": 22.5, "lat_max": 26.5, "lon_min": 51.0, "lon_max": 56.5},
    "iraq": {"lat_min": 29.0, "lat_max": 37.5, "lon_min": 39.0, "lon_max": 48.5},
    "syria": {"lat_min": 32.0, "lat_max": 37.5, "lon_min": 35.5, "lon_max": 42.5},
}

# ── Data directory paths ────────────────────────────────────────────
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Feature engineering defaults ────────────────────────────────────
# Maximum physically plausible values for commercial aircraft
MAX_VELOCITY_MS = 340.0       # ~Mach 1 (basically impossible for airliners)
MAX_VERTICAL_RATE_MS = 80.0   # ~15,000 ft/min
MAX_ACCELERATION_MS2 = 50.0   # extreme threshold
MAX_TURN_RATE_DEG_S = 10.0    # bank angle limit

# ── Clustering defaults ─────────────────────────────────────────────
DEFAULT_K = 5                  # starting K for K-means (tuned via elbow/silhouette)
RANDOM_STATE = 42
