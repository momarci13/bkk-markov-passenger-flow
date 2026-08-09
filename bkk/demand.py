"""
bkk.demand
==========
Prior estimators for stop-level initial passenger counts N_i(0).

Three estimators (Section 5 of the paper):

E1  Service proxy (GTFS only, no external data)
    N_hat_i ∝ f_i · k^v_i · ρ^v
    Anchored to N_peak = phi_peak * N_day_total.

E2  Poisson log-linear model (recommended)
    log λ_i = β_0 + β^T x_i
    x_i = [log(1+f_i), log(1+Pop_i), log(1+POI_i), B_i, I_i, log(1+D_i)]
    β informed by transit-demand meta-analysis (Schiewe & Lindner 2022).
    Anchored to N_peak; only β_0 is pinned by aggregate ridership.

E3  Monte-Carlo uncertainty propagation
    Draw β^(s) ~ N(μ_β, Σ_β), compute N_hat^(s), run simulation.

All estimators return an array of shape (n_stops,) with sum = N_peak.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .constants import (
    BETA_PRIOR_MEAN,
    BETA_PRIOR_STD,
    DEAK_TER_LAT,
    DEAK_TER_LON,
    N_DAY_TOTAL,
    PHI_PEAK_07_09,
)
from .network import NetworkData, _haversine_m

log = logging.getLogger(__name__)


def _peak_hour_ridership(
    n_day: float = N_DAY_TOTAL,
    phi: float   = PHI_PEAK_07_09,
) -> float:
    return n_day * phi


class DemandPrior:
    """
    Hierarchical prior estimator for N_i(0).

    Parameters
    ----------
    n_day_total : float
        BKK published daily ridership (default 4 000 000).
    phi_peak : float
        Fraction of daily boardings in the peak hour (default 0.11).
    beta_mean : array-like, shape (6,)
        Prior mean of log-linear coefficients.
    beta_std : array-like, shape (6,)
        Prior std  of log-linear coefficients.
    rng_seed : int
        Random seed for Monte-Carlo propagation.

    Notes
    -----
    **Identifiability warning**: only β_0 (global scale / intercept) is
    pinned by the aggregate-ridership anchor.  The feature coefficients
    β_1 … β_6 are prior beliefs from the demand-modelling literature,
    *not* maximum-likelihood estimates.  Treat all stop-level counts as
    indicative, not operational.
    """

    def __init__(
        self,
        n_day_total: float          = N_DAY_TOTAL,
        phi_peak:    float          = PHI_PEAK_07_09,
        beta_mean:   Optional[list] = None,
        beta_std:    Optional[list] = None,
        rng_seed:    int            = 42,
    ) -> None:
        self.n_peak    = _peak_hour_ridership(n_day_total, phi_peak)
        self.beta_mean = np.array(beta_mean or BETA_PRIOR_MEAN, dtype=float)
        self.beta_std  = np.array(beta_std  or BETA_PRIOR_STD,  dtype=float)
        self.rng       = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------ #
    #  E1: Service proxy                                                   #
    # ------------------------------------------------------------------ #
    def e1_service_proxy(self, net: NetworkData) -> np.ndarray:
        """
        E1 estimator: proportional to departure intensity μ^v_i.

        Uses the union of per-stop intensities across all modes.

        Parameters
        ----------
        net : NetworkData

        Returns
        -------
        N_hat : np.ndarray, shape (n_global,)
            Anchored so that sum = N_peak.
            Indexed by net.all_stop_ids.
        """
        all_ids   = net.all_stop_ids
        n_global  = len(all_ids)
        sid2idx   = {sid: k for k, sid in enumerate(all_ids)}
        raw       = np.zeros(n_global, dtype=np.float64)

        for rt, mg in net.modal.items():
            for local_k, sid in enumerate(mg.stop_ids):
                g = sid2idx.get(sid)
                if g is not None:
                    raw[g] += mg.mu_arr[local_k]

        raw_sum = raw.sum()
        if raw_sum <= 0:
            raise ValueError("All departure intensities are zero; check mu_arr.")

        N_hat = raw * (self.n_peak / raw_sum)
        log.info(
            "E1 estimator: sum=%.0f  max=%.1f  (N_peak=%.0f)",
            N_hat.sum(), N_hat.max(), self.n_peak,
        )
        return N_hat

    # ------------------------------------------------------------------ #
    #  E2: Poisson log-linear model                                       #
    # ------------------------------------------------------------------ #
    def e2_log_linear(
        self,
        net:             NetworkData,
        pop_i:           Optional[np.ndarray] = None,
        poi_i:           Optional[np.ndarray] = None,
        beta:            Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        E2 estimator: anchored Poisson log-linear model.

        log λ_i = β_0 + β^T x_i
        x_i = [log(1+f_i), log(1+Pop_i), log(1+POI_i), B_i, I_i, log(1+D_i)]

        Parameters
        ----------
        net : NetworkData
        pop_i : array of shape (n_global,), optional
            Catchment population per stop.
            If None, set to 0 (Pop not used).
        poi_i : array of shape (n_global,), optional
            POI count per stop from OSM.
            If None, set to 0 (POI not used).
        beta : array of shape (6,), optional
            Override coefficient vector. Default: prior mean.

        Returns
        -------
        N_hat : np.ndarray, shape (n_global,)  anchored to N_peak.
        """
        beta = np.array(beta if beta is not None else self.beta_mean)
        all_ids  = net.all_stop_ids
        n_global = len(all_ids)
        sid2idx  = {sid: k for k, sid in enumerate(all_ids)}

        # --- Feature 1: peak-hour frequency f_i (trips/h) ---------------
        f_i = np.zeros(n_global, dtype=float)
        for rt, mg in net.modal.items():
            for local_k, sid in enumerate(mg.stop_ids):
                g = sid2idx.get(sid)
                if g is not None:
                    # mu_arr is scheduled departures/s per passenger.
                    f_i[g] += mg.mu_arr[local_k] * 3600.0

        # --- Feature 4: betweenness centrality B_i ----------------------
        B_i = self._betweenness(net, sid2idx, n_global)

        # --- Feature 5: interchange indicator I_i -----------------------
        transfer_sids = set(net.transfer_times.keys())
        I_i = np.array(
            [1.0 if sid in transfer_sids else 0.0 for sid in all_ids],
            dtype=float,
        )

        # --- Feature 6: distance to Deák tér ----------------------------
        coords    = net.stop_coords.set_index("stop_id")
        lats      = np.array([
            coords.at[sid, "stop_lat"] if sid in coords.index else DEAK_TER_LAT
            for sid in all_ids
        ], dtype=float)
        lons      = np.array([
            coords.at[sid, "stop_lon"] if sid in coords.index else DEAK_TER_LON
            for sid in all_ids
        ], dtype=float)
        D_i_km    = _haversine_m(
            lats, lons,
            np.full(n_global, DEAK_TER_LAT),
            np.full(n_global, DEAK_TER_LON),
        ) / 1000.0

        # External features (defaults to zeros if not provided)
        _pop = pop_i if pop_i is not None else np.zeros(n_global)
        _poi = poi_i if poi_i is not None else np.zeros(n_global)

        # --- Feature matrix X (n_global × 6) ----------------------------
        X = np.column_stack([
            np.log1p(f_i),   # β_1
            np.log1p(_pop),  # β_2
            np.log1p(_poi),  # β_3
            B_i,             # β_4
            I_i,             # β_5
            np.log1p(D_i_km),# β_6
        ])

        # --- Anchored prediction -----------------------------------------
        log_raw  = X @ beta                  # (n_global,)
        raw      = np.exp(log_raw - log_raw.max())   # stabilise
        N_hat    = raw * (self.n_peak / raw.sum())

        log.info(
            "E2 estimator: sum=%.0f  max=%.1f  top-stop=%s  (N_peak=%.0f)",
            N_hat.sum(), N_hat.max(),
            all_ids[np.argmax(N_hat)],
            self.n_peak,
        )
        return N_hat

    # ------------------------------------------------------------------ #
    #  E3: Monte-Carlo uncertainty propagation                            #
    # ------------------------------------------------------------------ #
    def e3_monte_carlo(
        self,
        net:    NetworkData,
        n_samples: int = 200,
        pop_i:  Optional[np.ndarray] = None,
        poi_i:  Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        E3 estimator: draw β^(s) ~ N(μ_β, Σ_β) and propagate.

        Parameters
        ----------
        net : NetworkData
        n_samples : int
            Number of Monte-Carlo draws.
        pop_i, poi_i : optional externals (passed to E2).

        Returns
        -------
        mean_N  : np.ndarray, shape (n_global,)
        q05_N   : np.ndarray, shape (n_global,)  5th percentile
        q95_N   : np.ndarray, shape (n_global,)  95th percentile
        """
        log.info("E3 Monte-Carlo: %d samples", n_samples)
        n_global = len(net.all_stop_ids)
        ensemble = np.empty((n_samples, n_global), dtype=float)

        for s in range(n_samples):
            beta_s = self.rng.normal(
                loc   = self.beta_mean,
                scale = self.beta_std,
            )
            ensemble[s] = self.e2_log_linear(
                net, pop_i=pop_i, poi_i=poi_i, beta=beta_s
            )

        mean_N = ensemble.mean(axis=0)
        q05_N  = np.percentile(ensemble, 5,  axis=0)
        q95_N  = np.percentile(ensemble, 95, axis=0)

        log.info(
            "E3 done: mean_sum=%.0f  max_q95=%.1f",
            mean_N.sum(), q95_N.max(),
        )
        return mean_N, q05_N, q95_N

    # ------------------------------------------------------------------ #
    #  Diurnal profile                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def diurnal_scale(N_hat: np.ndarray, hour: int = 8) -> np.ndarray:
        """
        Scale N_hat to a specific hour using a Budapest diurnal profile.

        Parameters
        ----------
        N_hat : array  (represents the peak-hour estimate)
        hour  : int 0–23

        Returns
        -------
        scaled : array  scaled to hourly ridership at *hour*
        """
        # Normalised diurnal profile (sums to 24 ≈ 1 per hour on average)
        # Based on typical Budapest weekday pattern
        phi = {
            0: 0.01, 1: 0.01, 2: 0.005, 3: 0.005, 4: 0.01, 5: 0.03,
            6: 0.07, 7: 0.10, 8: 0.11,  9: 0.08,  10: 0.07, 11: 0.07,
            12: 0.07, 13: 0.07, 14: 0.07, 15: 0.07, 16: 0.09, 17: 0.10,
            18: 0.08, 19: 0.06, 20: 0.04, 21: 0.03, 22: 0.02, 23: 0.01,
        }
        return N_hat * (phi.get(hour, 0.01) / PHI_PEAK_07_09)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _betweenness(
        net:     NetworkData,
        sid2idx: dict[str, int],
        n:       int,
    ) -> np.ndarray:
        """
        Approximate betweenness centrality using NetworkX.
        Falls back to degree centrality for large networks (n > 2000).
        """
        try:
            import networkx as nx

            G = nx.DiGraph()
            for rt, mg in net.modal.items():
                for r in range(mg.n_edges):
                    G.add_edge(
                        mg.stop_ids[mg.i_arr[r]],
                        mg.stop_ids[mg.j_arr[r]],
                    )

            if G.number_of_nodes() > 2000:
                log.info(
                    "Betweenness: n=%d > 2000, using 500-node sample approximation",
                    G.number_of_nodes(),
                )
                bc = nx.betweenness_centrality(
                    G, k=min(500, G.number_of_nodes()), normalized=True
                )
            else:
                bc = nx.betweenness_centrality(G, normalized=True)

            B = np.zeros(n, dtype=float)
            for sid, val in bc.items():
                k = sid2idx.get(str(sid))
                if k is not None:
                    B[k] = val
            return B

        except ImportError:
            log.warning("networkx not installed; B_i set to 0.")
            return np.zeros(n, dtype=float)
