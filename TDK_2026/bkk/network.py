"""
bkk.network
===========
Build modal subgraphs G^v = (S^v, E^v), sparse cost matrices C^v,
and per-stop departure intensities mu^v_i from a parsed GTFSFeed.

All costs are in **seconds**.  No dense matrices are ever materialised;
output is always in CSR/COO form via scipy.sparse or plain edge arrays.

Main class
----------
NetworkBuilder
    .build(feed, peak_window=("07:00:00","09:00:00")) -> NetworkData
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .constants import (
    BKK_ROUTE_TYPES,
    DEFAULT_PEAK_WINDOW,
    DELTA_TRANSFER,
    GAMMA_S_PER_M,
    ROUTE_TYPES,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Haversine distance helper
# ---------------------------------------------------------------------------
def _haversine_m(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in metres."""
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _parse_time_s(time_str):
    """
    Convert GTFS time strings (HH:MM:SS, possibly >24:00 for next-day)
    to seconds-since-midnight as float64.  Invalid entries become NaN.
    """
    parts = time_str.str.split(":", expand=True)
    if parts.shape[1] < 3:
        return pd.Series(np.nan, index=time_str.index)
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce")
    s = pd.to_numeric(parts[2], errors="coerce")
    return h * 3600.0 + m * 60.0 + s


def _parent_or_self(stop_id, parent_station) -> str:
    """Return a usable parent ID without turning missing values into 'nan'."""
    sid = str(stop_id)
    if pd.isna(parent_station):
        return sid
    parent = str(parent_station).strip()
    return parent if parent and parent.lower() != "nan" else sid


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class ModalGraph:
    """Sparse representation of G^v for one mode."""
    route_type:  int
    name:        str
    stop_ids:    np.ndarray     # shape (|S^v|,)   dtype str
    stop_idx:    dict           # stop_id -> integer index in stop_ids
    i_arr:       np.ndarray     # shape (|E^v|,)   origin stop index
    j_arr:       np.ndarray     # shape (|E^v|,)   destination stop index
    cost_arr:    np.ndarray     # shape (|E^v|,)   C^v_{ij} [s]
    mu_arr:      np.ndarray     # shape (|S^v|,)   per-passenger hazard mu^v_i [s^-1]

    @property
    def n_stops(self):
        return len(self.stop_ids)

    @property
    def n_edges(self):
        return len(self.i_arr)

    def cost_matrix_csr(self):
        """Sparse CSR cost matrix of shape (n_stops, n_stops)."""
        n = self.n_stops
        return csr_matrix(
            (self.cost_arr, (self.i_arr, self.j_arr)),
            shape=(n, n),
        )


@dataclass
class NetworkData:
    """All modal graphs produced by NetworkBuilder."""
    all_stop_ids:   np.ndarray
    stop_coords:    pd.DataFrame
    transfer_times: dict
    modal:          dict = field(default_factory=dict)

    def __getitem__(self, route_type):
        return self.modal[route_type]

    @property
    def modes(self):
        return sorted(self.modal)

    def summary(self):
        lines = ["NetworkData summary", "=" * 50]
        lines.append(f"  Total stops (all modes): {len(self.all_stop_ids):,}")
        lines.append(f"  Transfer nodes:          {len(self.transfer_times):,}")
        for rt, mg in sorted(self.modal.items()):
            meta = ROUTE_TYPES.get(rt, {})
            lines.append(
                f"  mode {rt:3d} ({meta.get('name','?'):12s}): "
                f"{mg.n_stops:5,} stops  {mg.n_edges:7,} edges  "
                f"mu_max={mg.mu_arr.max():.6f} s^-1"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
class NetworkBuilder:
    """
    Build modal subgraphs and cost matrices from a GTFSFeed.

    Parameters
    ----------
    kappa_override : dict[int, float], optional
        Override mode comfort weights kappa^v (route_type -> kappa).
    gamma : float
        Distance disutility gamma [s/m].  Default: 0.01 s/m.
    delta : float
        Transfer-penalty multiplier delta.  Default: 1.0.
    route_types : set[int], optional
        Restrict to this subset of route_types.  Default: all BKK types.
    """

    def __init__(
        self,
        kappa_override=None,
        gamma=GAMMA_S_PER_M,
        delta=DELTA_TRANSFER,
        route_types=None,
    ):
        self.kappa_override = kappa_override or {}
        self.gamma          = gamma
        self.delta          = delta
        self.route_types    = route_types or BKK_ROUTE_TYPES

    # ------------------------------------------------------------------ #
    def build(self, feed, peak_window=DEFAULT_PEAK_WINDOW):
        """
        Build the full NetworkData from *feed*.

        Parameters
        ----------
        feed : GTFSFeed
        peak_window : (start_str, end_str)

        Returns
        -------
        NetworkData
        """
        log.info("NetworkBuilder.build(): peak window %s - %s", *peak_window)

        # --- 1. Stop coordinates ------------------------------------------
        stops_df = feed.stops.copy()
        stops_df = stops_df.dropna(subset=["stop_lat", "stop_lon"])
        stops_df["stop_lat"] = stops_df["stop_lat"].astype(float)
        stops_df["stop_lon"] = stops_df["stop_lon"].astype(float)
        stop_coord_map = {
            row.stop_id: (row.stop_lat, row.stop_lon)
            for row in stops_df.itertuples(index=False)
        }

        # --- 1b. Stop-to-station aggregation map --------------------------
        # BKK metro assigns separate stop_ids for each directional platform.
        # Collapse platform stops to parent_station so the subgraph is SCC.
        #
        # stop2node[stop_id] = parent_station  (if non-empty parent_station)
        # stop2node[stop_id] = stop_id         (otherwise -- identity)
        stop2node = {}
        if "parent_station" in stops_df.columns:
            for row in stops_df.itertuples(index=False):
                stop2node[str(row.stop_id)] = _parent_or_self(
                    row.stop_id, getattr(row, "parent_station", "")
                )
            n_mapped = sum(1 for s, n in stop2node.items() if s != n)
            log.info(
                "stop2node: %d platform stops -> parent_station aggregated",
                n_mapped,
            )
        else:
            for row in stops_df.itertuples(index=False):
                stop2node[str(row.stop_id)] = str(row.stop_id)

        # Only pass stop2node if any platform->station mappings exist
        active_stop2node = stop2node if any(s != n for s, n in stop2node.items()) else None

        # --- 2. Transfer times -------------------------------------------
        transfer_times = {}
        if not feed.transfers.empty:
            tf = feed.transfers.copy()
            tf["min_transfer_time"] = pd.to_numeric(
                tf["min_transfer_time"], errors="coerce"
            ).fillna(120.0)
            for sid, grp in tf.groupby("from_stop_id"):
                transfer_times[str(sid)] = float(grp["min_transfer_time"].min())

        # --- 3. Join routes -> trips -> stop_times ------------------------
        route_cols = ["route_id", "route_type"]
        trip_cols  = ["trip_id", "route_id", "service_id"]
        st_cols    = ["trip_id", "stop_id", "stop_sequence",
                      "arrival_time", "departure_time", "shape_dist_traveled"]

        routes_sub = feed.routes[
            feed.routes["route_type"].astype(int).isin(self.route_types)
        ][route_cols].copy()
        routes_sub["route_type"] = routes_sub["route_type"].astype(int)

        trips_sub = feed.trips[trip_cols].merge(routes_sub, on="route_id", how="inner")

        st_available = [c for c in st_cols if c in feed.stop_times.columns]
        st = feed.stop_times[st_available].merge(
            trips_sub[["trip_id", "route_type"]], on="trip_id", how="inner"
        )

        st["dep_s"] = _parse_time_s(st["departure_time"])
        st["arr_s"] = _parse_time_s(st["arrival_time"])
        st = st.dropna(subset=["dep_s", "arr_s"])

        pw_start = _parse_time_s(pd.Series([peak_window[0]])).iloc[0]
        pw_end   = _parse_time_s(pd.Series([peak_window[1]])).iloc[0]
        pw_dur_s = pw_end - pw_start
        st_peak  = st[(st["dep_s"] >= pw_start) & (st["dep_s"] < pw_end)].copy()

        log.info(
            "  stop_times rows (all / peak): %d / %d",
            len(st), len(st_peak),
        )

        # --- 4. Consecutive-stop edges for each mode ----------------------
        st_sorted = st.sort_values(["trip_id", "stop_sequence"])
        modal = {}

        for route_type in sorted(self.route_types):
            mg = self._build_modal_graph(
                route_type     = route_type,
                st_sorted      = st_sorted,
                st_peak        = st_peak,
                stop_coord_map = stop_coord_map,
                transfer_times = transfer_times,
                pw_dur_s       = pw_dur_s,
                stop2node      = active_stop2node,
            )
            if mg is not None:
                modal[route_type] = mg
                log.info(
                    "  mode %3d %-12s: %5d stops  %6d edges",
                    route_type,
                    ROUTE_TYPES.get(route_type, {}).get("name", "?"),
                    mg.n_stops, mg.n_edges,
                )

        # --- 5. Assemble NetworkData --------------------------------------
        all_stop_ids = np.array(
            sorted({sid for mg in modal.values() for sid in mg.stop_ids}),
            dtype=str,
        )

        return NetworkData(
            all_stop_ids   = all_stop_ids,
            stop_coords    = stops_df[["stop_id", "stop_lat", "stop_lon"]],
            transfer_times = transfer_times,
            modal          = modal,
        )

    # ------------------------------------------------------------------ #
    def _build_modal_graph(
        self,
        route_type,
        st_sorted,
        st_peak,
        stop_coord_map,
        transfer_times,
        pw_dur_s,
        stop2node=None,
    ):
        """Build ModalGraph for a single route_type."""
        meta     = ROUTE_TYPES.get(route_type, {})
        kappa    = self.kappa_override.get(route_type, meta.get("kappa", 1.0))

        # --- Consecutive pairs (i, j) for this mode -----------------------
        mode_st = st_sorted[st_sorted["route_type"] == route_type].copy()
        if mode_st.empty:
            return None

        mode_st   = mode_st.reset_index(drop=True)
        mode_next = mode_st.copy()
        mode_next["stop_id_i"] = mode_st["stop_id"].values
        mode_next["dep_s_i"]   = mode_st["dep_s"].values
        mode_next["stop_id_j"] = mode_next.groupby("trip_id")["stop_id"].shift(-1)
        mode_next["arr_s_j"]   = mode_next.groupby("trip_id")["arr_s"].shift(-1)
        mode_next["seq_j"]     = mode_next.groupby("trip_id")["stop_sequence"].shift(-1)

        if "shape_dist_traveled" in mode_next.columns:
            mode_next["dist_j"] = mode_next.groupby("trip_id")[
                "shape_dist_traveled"
            ].shift(-1)
            mode_next["dist_m"] = (
                mode_next["dist_j"] - mode_next["shape_dist_traveled"]
            ).clip(lower=0)
        else:
            mode_next["dist_m"] = np.nan

        pairs = mode_next.dropna(subset=["stop_id_j"]).copy()
        pairs = pairs[pairs["stop_sequence"] < pairs["seq_j"]].copy()

        if pairs.empty:
            return None

        # --- Stop-to-station aggregation ----------------------------------
        # Collapses direction-specific platform stop IDs to their parent_station,
        # turning the metro DAG into a strongly-connected subgraph so that
        # power iteration yields a non-trivial stationary distribution.
        # Only active when stop2node is not None (i.e. platform->station
        # mappings actually exist in the feed).
        if stop2node is not None:
            pairs = pairs.copy()
            pairs["stop_id_i"] = pairs["stop_id_i"].map(
                lambda s: stop2node.get(str(s), str(s))
            )
            pairs["stop_id_j"] = pairs["stop_id_j"].map(
                lambda s: stop2node.get(str(s), str(s))
            )
            pairs = pairs[pairs["stop_id_i"] != pairs["stop_id_j"]].copy()
            if pairs.empty:
                return None

        # --- Travel time [s] ---------------------------------------------
        pairs["tau_s"] = (pairs["arr_s_j"] - pairs["dep_s_i"]).clip(lower=0)

        edge_stats = (
            pairs.groupby(["stop_id_i", "stop_id_j"])
            .agg(
                tau_median=("tau_s", "median"),
                dist_m=("dist_m", "median"),
            )
            .reset_index()
        )

        if edge_stats.empty:
            return None

        # --- Generalised cost C^v_{ij} [s] --------------------------------
        missing_dist = edge_stats["dist_m"].isna()
        if missing_dist.any():
            i_lats = edge_stats.loc[missing_dist, "stop_id_i"].map(
                lambda s: stop_coord_map.get(s, (np.nan, np.nan))[0]
            ).values
            i_lons = edge_stats.loc[missing_dist, "stop_id_i"].map(
                lambda s: stop_coord_map.get(s, (np.nan, np.nan))[1]
            ).values
            j_lats = edge_stats.loc[missing_dist, "stop_id_j"].map(
                lambda s: stop_coord_map.get(s, (np.nan, np.nan))[0]
            ).values
            j_lons = edge_stats.loc[missing_dist, "stop_id_j"].map(
                lambda s: stop_coord_map.get(s, (np.nan, np.nan))[1]
            ).values
            edge_stats.loc[missing_dist, "dist_m"] = _haversine_m(
                i_lats, i_lons, j_lats, j_lons
            )

        if edge_stats["dist_m"].isna().any():
            missing = edge_stats.loc[
                edge_stats["dist_m"].isna(), ["stop_id_i", "stop_id_j"]
            ].head(5).to_dict("records")
            raise ValueError(f"Cannot infer distance for edges with missing coordinates: {missing}")

        edge_stats["t_tf"] = edge_stats["stop_id_i"].map(
            lambda s: transfer_times.get(s, 0.0)
        )

        edge_stats["cost"] = (
            kappa * edge_stats["tau_median"]
            + self.gamma * edge_stats["dist_m"]
            + self.delta * edge_stats["t_tf"]
        )

        # --- Stop index --------------------------------------------------
        all_stops_in_mode = pd.unique(
            np.concatenate([
                edge_stats["stop_id_i"].values,
                edge_stats["stop_id_j"].values,
            ])
        )
        stop_ids = np.array(sorted(all_stops_in_mode), dtype=str)
        stop_idx = {sid: k for k, sid in enumerate(stop_ids)}
        n        = len(stop_ids)

        i_arr    = np.array([stop_idx[s] for s in edge_stats["stop_id_i"]], dtype=np.int32)
        j_arr    = np.array([stop_idx[s] for s in edge_stats["stop_id_j"]], dtype=np.int32)
        cost_arr = edge_stats["cost"].values.astype(np.float64)

        # --- Per-passenger departure hazard mu^v_i [s^-1] ----------------
        # A CTMC generator must have inverse-time units.  Scheduled vehicle
        # frequency is therefore the hazard; vehicle capacity/load factor
        # belong in demand or capacity models and must not multiply Q.
        mu_arr    = np.zeros(n, dtype=np.float64)
        peak_mode = st_peak[st_peak["route_type"] == route_type].copy()
        if not peak_mode.empty and pw_dur_s > 0:
            # Aggregate to station level when stop2node is active
            if stop2node is not None:
                peak_mode = peak_mode.copy()
                peak_mode["stop_id"] = peak_mode["stop_id"].map(
                    lambda s: stop2node.get(str(s), str(s))
                )
            freq = (
                peak_mode.groupby("stop_id")["trip_id"]
                .nunique()
                .rename("n_trips")
            )
            for sid, nt in freq.items():
                k = stop_idx.get(str(sid))
                if k is None:
                    continue
                mu_arr[k] = nt / pw_dur_s

        return ModalGraph(
            route_type = route_type,
            name       = meta.get("name", str(route_type)),
            stop_ids   = stop_ids,
            stop_idx   = stop_idx,
            i_arr      = i_arr,
            j_arr      = j_arr,
            cost_arr   = cost_arr,
            mu_arr     = mu_arr,
        )
