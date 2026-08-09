"""
bkk.constants
=============
BKK-specific constants, GTFS route-type mappings, and model defaults.

All travel-time / cost quantities are in **seconds** unless noted.
All rate quantities are in **passengers per second** unless noted.

Calendar note
-------------
The BKK GTFS feed does NOT include calendar.txt.
It uses calendar_dates.txt exclusively (exception-based calendar,
valid per the GTFS Schedule specification §calendar_dates).
The framework handles both variants automatically.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# GTFS download URL (from bkk.hu/bkk-partnerek/egyeb/gtfs-menetrendi-adatok-
# programozoknak/)
# ---------------------------------------------------------------------------
GTFS_URL: str = "https://bkk.hu/gtfs/budapest_gtfs.zip"

# Files to attempt to parse from the ZIP.
# calendar.txt is optional (BKK does not include it; other agencies do).
# calendar_dates.txt is the BKK calendar format.
# Both are listed here so the loader will read whichever is present.
GTFS_FILES: tuple[str, ...] = (
    "agency.txt",
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "calendar.txt",         # optional – present in many feeds, absent in BKK
    "calendar_dates.txt",   # present in BKK; used as the sole calendar source
    "shapes.txt",
    "transfers.txt",
    "feed_info.txt",
)

# Required files (absent → raise, not warn).
# calendar.txt intentionally excluded – BKK does not include it.
# Either calendar.txt OR calendar_dates.txt must be present (checked in code).
GTFS_REQUIRED: tuple[str, ...] = (
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
)

# ---------------------------------------------------------------------------
# BKK route_type → mode metadata
# GTFS route_type codes used in the BKK feed:
#   0   Tram (villamos)
#   1   Metro
#   2   Suburban railway / HÉV
#   3   Bus (busz)
#   4   Ferry / boat (hajó)
#   800 Trolleybus (trolibusz)   ← extended GTFS type
# ---------------------------------------------------------------------------
ROUTE_TYPES: dict[int, dict] = {
    0: {
        "name":          "tram",
        "hu":            "villamos",
        "kappa":         0.70,          # mode comfort weight (dimensionless)
        "capacity":      280,           # mean vehicle capacity [pax]
        "speed_kmh":     18.0,          # typical average speed [km/h]
        "walk_radius_m": 400,           # catchment radius for demand prior [m]
        "load_factor":   0.50,          # rho^v default
    },
    1: {
        "name":          "metro",
        "hu":            "metró",
        "kappa":         0.40,
        "capacity":      900,
        "speed_kmh":     35.0,
        "walk_radius_m": 800,
        "load_factor":   0.60,
    },
    2: {
        "name":          "hev",
        "hu":            "HÉV",
        "kappa":         0.55,
        "capacity":      500,
        "speed_kmh":     32.0,
        "walk_radius_m": 800,
        "load_factor":   0.45,
    },
    3: {
        "name":          "bus",
        "hu":            "busz",
        "kappa":         1.00,
        "capacity":      130,
        "speed_kmh":     21.0,
        "walk_radius_m": 400,
        "load_factor":   0.50,
    },
    4: {
        "name":          "ferry",
        "hu":            "hajó",
        "kappa":         1.20,
        "capacity":      100,
        "speed_kmh":     15.0,
        "walk_radius_m": 400,
        "load_factor":   0.40,
    },
    800: {
        "name":          "trolleybus",
        "hu":            "trolibusz",
        "kappa":         0.85,
        "capacity":      135,
        "speed_kmh":     17.0,
        "walk_radius_m": 400,
        "load_factor":   0.50,
    },
}

# Canonical set of modes present in BKK
BKK_ROUTE_TYPES: frozenset[int] = frozenset(ROUTE_TYPES)

# ---------------------------------------------------------------------------
# Cost-matrix parameters   (Eq. 1 in the paper)
# C^v_{ij} = kappa^v * tau_ij  +  gamma * d_ij  +  delta * I_transfer * t_tf
# ---------------------------------------------------------------------------
GAMMA_S_PER_M: float  = 0.01   # distance disutility  [s/m]  (100 m ≡ 1 s)
DELTA_TRANSFER: float = 1.0    # transfer-penalty multiplier (dimensionless)
COST_INFINITY: float  = 1e15   # sentinel for "no direct connection" (never stored)

# ---------------------------------------------------------------------------
# CTMC / entropy-maximisation defaults
# ---------------------------------------------------------------------------
DEFAULT_LAMBDA: float = 1.0 / 300.0   # [s^{-1}]  cost sensitivity (λ^v)
DEFAULT_PEAK_WINDOW: tuple[str, str] = ("07:00:00", "09:00:00")

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------
DEFAULT_HORIZON_S: float    = 900.0     # simulation horizon [s]
DEFAULT_TAU_S: float        = 1.0       # tau-leap step size [s]
DEFAULT_EPSILON_LEAP: float = 0.03      # Cao–Gillespie–Petzold ε criterion
SSA_MAX_EVENTS: int         = 500_000   # safety cap for exact SSA

# ---------------------------------------------------------------------------
# Demand prior defaults
# ---------------------------------------------------------------------------
N_DAY_TOTAL: float    = 4_000_000.0    # BKK published weekday boardings
PHI_PEAK_07_09: float = 0.11           # fraction of daily boardings in 07–09h

# Literature-informed prior coefficients β  (Table 3 in the paper)
# Order: [log(1+f_i), log(1+Pop_i), log(1+POI_i), B_i, I_i, log(1+D_i)]
BETA_PRIOR_MEAN: list[float] = [0.70, 0.60, 0.20, 0.30, 0.50, -0.25]
BETA_PRIOR_STD:  list[float] = [0.10, 0.15, 0.10, 0.15, 0.20,  0.10]

# ---------------------------------------------------------------------------
# Resilience analysis
# ---------------------------------------------------------------------------
RESILIENCE_TOP_K: int   = 100   # candidate stops by importance
KRYLOV_TRUNC: int       = 20    # eigenvalues for Kemeny tail sum
HYBRID_THRESHOLD: int   = 1_000 # N_i above which fluid ODE is used

# ---------------------------------------------------------------------------
# Deák Ferenc tér coordinates (city-centre reference point for D_i feature)
# ---------------------------------------------------------------------------
DEAK_TER_LAT: float = 47.4979
DEAK_TER_LON: float = 19.0548

# ---------------------------------------------------------------------------
# Calendar / date helpers
# ---------------------------------------------------------------------------
# ISO weekday names (Monday=0 … Sunday=6) — used when inferring weekday
# from a date string in calendar_dates.txt
WEEKDAY_NAMES: tuple[str, ...] = (
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
)
