"""
bkk.simulate
============
Three simulation layers for the BKK Markov-chain passenger-flow model.

KFESolver      – Kolmogorov Forward Equation via sparse ODE (Layer 2)
GillespieSSA   – Exact Doob-Gillespie direct method    (Layer 3, exact)
TauLeap        – τ-leaping approximation               (Layer 3, fast)

All simulations conserve the total passenger count Σ_i N_i(t) to
machine precision.

Units: time in seconds, populations as integer counts (SSA/τ-leap)
or float densities (KFE).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from .constants import (
    DEFAULT_HORIZON_S,
    DEFAULT_TAU_S,
    DEFAULT_EPSILON_LEAP,
    SSA_MAX_EVENTS,
)
from .generator import ModalGenerator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    """Result of one simulation run."""
    t_eval:     np.ndarray        # shape (T_steps,)  wall-clock times [s]
    N_hist:     np.ndarray        # shape (T_steps, n_global)  populations
    mode:       str               # 'kfe' | 'ssa' | 'tauleap'
    events:     list  = field(default_factory=list)   # SSA event list
    metadata:   dict  = field(default_factory=dict)

    @property
    def N_final(self) -> np.ndarray:
        return self.N_hist[-1]

    def conservation_error(self) -> float:
        """
        Relative deviation of total passengers over the simulation.
        Should be ≈ 0 for KFE (machine precision) and exactly 0 for SSA.
        """
        N0  = self.N_hist[0].sum()
        Nfin = self.N_hist[-1].sum()
        return float(abs(Nfin - N0) / (N0 + 1e-300))


def _validate_run_inputs(generators, N0_modal, T: float, n_eval: int | None = None) -> None:
    if not generators:
        raise ValueError("generators must not be empty")
    if not np.isfinite(T) or T < 0:
        raise ValueError("T must be a finite non-negative horizon")
    if n_eval is not None and n_eval < 2:
        raise ValueError("n_eval must be at least 2")
    for rt, mg in generators.items():
        values = np.asarray(N0_modal.get(rt, np.zeros(mg.n_stops)), dtype=float)
        if values.shape != (mg.n_stops,):
            raise ValueError(f"N0_modal[{rt}] must have shape ({mg.n_stops},)")
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"N0_modal[{rt}] must be finite and non-negative")


def _global_layout(generators):
    """Return canonical stop ids and local-to-global indices for each mode."""
    stop_ids = np.array(sorted({str(s) for mg in generators.values() for s in mg.stop_ids}))
    index = {sid: k for k, sid in enumerate(stop_ids)}
    local_to_global = {
        rt: np.array([index[str(s)] for s in mg.stop_ids], dtype=np.int32)
        for rt, mg in generators.items()
    }
    return stop_ids, local_to_global


def _integerise_state(N: np.ndarray) -> tuple[np.ndarray, float]:
    """Largest-remainder rounding with a conserved rounded grand total."""
    floors = np.floor(N).astype(np.int64)
    target = int(np.rint(N.sum()))
    remainder = target - int(floors.sum())
    if remainder > 0:
        order = np.argsort((N - floors).ravel())[::-1][:remainder]
        floors.ravel()[order] += 1
    return floors, float(target - N.sum())


# ---------------------------------------------------------------------------
# Utility: build global state vector from ModalGenerator dict
# ---------------------------------------------------------------------------
def allocate_modal_population(
    generators: dict[int, ModalGenerator],
    N0_global:  np.ndarray,   # shape (n_global,)
    all_stop_ids: np.ndarray, # shape (n_global,)
) -> dict[int, np.ndarray]:
    """
    Map global N0 to modal arrays without losing or duplicating passengers.

    At a stop served by m modes, the default assigns N_i/m to each mode.

    Parameters
    ----------
    generators   : route_type → ModalGenerator
    N0_global    : global initial count vector
    all_stop_ids : global stop-id array (matches N0_global index)

    Returns
    -------
    N0_modal : route_type → np.ndarray shape (n_stops_v,)
    """
    N0_global = np.asarray(N0_global, dtype=float)
    all_stop_ids = np.asarray(all_stop_ids, dtype=str)
    if N0_global.shape != (len(all_stop_ids),):
        raise ValueError("N0_global and all_stop_ids must have matching 1-D shapes")
    if np.any(~np.isfinite(N0_global)) or np.any(N0_global < 0):
        raise ValueError("N0_global must contain finite non-negative counts")
    gidx = {sid: k for k, sid in enumerate(all_stop_ids)}
    memberships = np.zeros(len(all_stop_ids), dtype=int)
    for mg in generators.values():
        for sid in mg.stop_ids:
            if sid in gidx:
                memberships[gidx[sid]] += 1
    N0_modal = {}
    for rt, mg in generators.items():
        N_v = np.zeros(mg.n_stops, dtype=np.float64)
        for local_k, sid in enumerate(mg.stop_ids):
            g = gidx.get(sid, None)
            if g is not None:
                if memberships[g] > 0:
                    N_v[local_k] = N0_global[g] / memberships[g]
        N0_modal[rt] = N_v
    return N0_modal


# Backward-compatible private alias used by older notebooks.
_build_global_state = allocate_modal_population


# ---------------------------------------------------------------------------
# Layer 2: Kolmogorov Forward Equation
# ---------------------------------------------------------------------------
class KFESolver:
    """
    Solve the mean-field Kolmogorov Forward Equation (KFE).

        dN^v/dt = N^v · Q^v          (row-vector convention)

    For all modes simultaneously, treating each mode independently.
    Passenger counts are conserved to machine precision by construction
    (Q^v has zero row sums).

    Parameters
    ----------
    method : str
        ODE integrator:
        - ``'lsoda'``    : scipy.integrate.solve_ivp LSODA  (default)
        - ``'rk45'``     : explicit RK45
        - ``'expm'``     : matrix exponential via expm_multiply (fastest
                           for small n, single time point)
    rtol, atol : float
        ODE solver tolerances.

    Examples
    --------
    >>> solver = KFESolver()
    >>> result = solver.solve(generators, N0_modal, T=900.0, n_eval=61)
    """

    def __init__(
        self,
        method: str  = "lsoda",
        rtol:   float = 1e-8,
        atol:   float = 1e-10,
    ) -> None:
        self.method = method.lower()
        self.rtol   = rtol
        self.atol   = atol

    def solve(
        self,
        generators:  dict[int, ModalGenerator],
        N0_modal:    dict[int, np.ndarray],
        T:           float = DEFAULT_HORIZON_S,
        n_eval:      int   = 61,
    ) -> dict[int, SimResult]:
        """
        Solve KFE for each mode in *generators*.

        Parameters
        ----------
        generators : route_type → ModalGenerator
        N0_modal   : route_type → initial population array
        T          : simulation horizon [s]
        n_eval     : number of time-evaluation points (including t=0)

        Returns
        -------
        results : route_type → SimResult
        """
        _validate_run_inputs(generators, N0_modal, T, n_eval)
        t_eval   = np.linspace(0.0, T, n_eval)
        results  = {}

        for rt, mg in generators.items():
            N0_v = N0_modal.get(rt, np.zeros(mg.n_stops))
            log.info(
                "KFE solve: mode %d  n=%d  Σ N0=%.1f  method=%s",
                rt, mg.n_stops, N0_v.sum(), self.method,
            )

            if self.method == "expm":
                res = self._solve_expm(mg.Q, N0_v, T, t_eval)
            else:
                res = self._solve_ode(mg.Q, N0_v, T, t_eval)

            results[rt] = SimResult(
                t_eval  = t_eval,
                N_hist  = res,
                mode    = f"kfe_{self.method}",
                metadata = {
                    "route_type": rt,
                    "conservation_error": abs(res[-1].sum() - N0_v.sum()) / (N0_v.sum() + 1e-300),
                },
            )
            log.info(
                "  Done. conservation_error=%.2e",
                results[rt].conservation_error(),
            )
        return results

    def _solve_ode(
        self,
        Q:      csr_matrix,
        N0:     np.ndarray,
        T:      float,
        t_eval: np.ndarray,
    ) -> np.ndarray:
        """Solve via scipy.integrate.solve_ivp."""
        # Q is CSR; row-vector convention: dN/dt = N Q
        # Transpose for column-vector form: d(N^T)/dt = Q^T N^T
        QT = Q.T.tocsr()

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return QT.dot(y)

        sol = solve_ivp(
            rhs,
            t_span  = [0.0, T],
            y0      = N0.astype(float),
            t_eval  = t_eval,
            method  = self.method.upper() if self.method != "lsoda" else "LSODA",
            rtol    = self.rtol,
            atol    = self.atol,
        )
        if not sol.success:
            log.warning("ODE solver warning: %s", sol.message)
        return sol.y.T   # shape (n_eval, n_stops)

    def _solve_expm(
        self,
        Q:      csr_matrix,
        N0:     np.ndarray,
        T:      float,
        t_eval: np.ndarray,
    ) -> np.ndarray:
        """Solve via expm_multiply at each t_eval point (Krylov subspace)."""
        # p(t) = p(0) exp(Q t) ⟺ p(t)^T = exp(Q^T t) p(0)^T
        QT  = Q.T.tocsr()
        out = np.empty((len(t_eval), len(N0)), dtype=float)
        for k, t in enumerate(t_eval):
            if t == 0.0:
                out[k] = N0
            else:
                out[k] = expm_multiply(QT * t, N0.astype(float))
        return out


# ---------------------------------------------------------------------------
# Layer 3a: Exact Gillespie SSA
# ---------------------------------------------------------------------------
class GillespieSSA:
    """
    Exact Doob-Gillespie direct method for the BKK multi-modal CTMC.

    Produces exact sample paths at the cost of O(a_0 · T) events.
    At full BKK scale (~4×10⁶ pax, metro μ ≈ 11 pax/s) this is
    ~3.6×10⁹ events per 15-min window — infeasible.
    Use :class:`TauLeap` for full-network simulation.

    Suitable for:
    - Sub-networks (≤ 500 stops)
    - Short time windows (≤ 5 min)
    - Validation / unit testing against KFE

    Parameters
    ----------
    max_events : int
        Safety cap on the number of events per run.  Default: 500 000.
    rng_seed : int
        Random seed.
    record_every : int
        Record the state vector every *record_every* events (for trajectories).
    """

    def __init__(
        self,
        max_events:   int = SSA_MAX_EVENTS,
        rng_seed:     int = 42,
        record_every: int = 1_000,
    ) -> None:
        self.max_events   = max_events
        self.rng          = np.random.default_rng(rng_seed)
        self.record_every = record_every

    def run(
        self,
        generators: dict[int, ModalGenerator],
        N0_modal:   dict[int, np.ndarray],
        T:          float = DEFAULT_HORIZON_S,
    ) -> SimResult:
        """
        Run one exact SSA trajectory.

        Parameters
        ----------
        generators : route_type → ModalGenerator
        N0_modal   : route_type → initial integer population array
        T          : simulation horizon [s]

        Returns
        -------
        SimResult with `events` populated.
        """
        _validate_run_inputs(generators, N0_modal, T)
        # --- Build flat channel arrays in a canonical global stop index ----
        # Channels: (v_idx, local_i, local_j, q_ij)
        # Also store rate q from Q sparse matrix off-diagonals
        v_list, i_list, j_list, q_list = [], [], [], []
        rt_list = list(generators.keys())
        stop_ids, local_to_global = _global_layout(generators)

        for v_idx, rt in enumerate(rt_list):
            mg = generators[rt]
            Q  = mg.Q
            Qd = Q.tocoo()
            for r in range(Qd.nnz):
                row, col, val = Qd.row[r], Qd.col[r], Qd.data[r]
                if row != col and val > 0:
                    v_list.append(v_idx)
                    i_list.append(local_to_global[rt][row])
                    j_list.append(local_to_global[rt][col])
                    q_list.append(val)

        v_arr = np.array(v_list, dtype=np.int32)
        i_arr = np.array(i_list, dtype=np.int32)
        j_arr = np.array(j_list, dtype=np.int32)
        q_arr = np.array(q_list, dtype=np.float64)
        R     = len(v_arr)

        # --- State: N[v_idx, local_i] -------------------------------------
        N = np.zeros((len(rt_list), len(stop_ids)), dtype=np.float64)
        for v_idx, rt in enumerate(rt_list):
            N[v_idx, local_to_global[rt]] = N0_modal.get(
                rt, np.zeros(generators[rt].n_stops)
            )

        N, rounding_delta = _integerise_state(N)
        N0_total = N.sum()
        t        = 0.0
        events   = []
        t_hist   = [0.0]
        N_hist   = [N.sum(axis=0).copy()]   # sum over modes

        log.info(
            "GillespieSSA: R=%d channels  N0=%.0f  max_events=%d",
            R, N0_total, self.max_events,
        )

        for ev_count in range(self.max_events):
            # Propensities
            a = N[v_arr, i_arr] * q_arr    # shape (R,)
            a0 = a.sum()
            if a0 <= 0.0:
                log.info("SSA halted: a0=0 at t=%.2f", t)
                break

            # Waiting time
            tau = -np.log(self.rng.random()) / a0
            if t + tau >= T:
                break
            t += tau

            # Channel selection (inverse-CDF; alias optional)
            u = self.rng.random() * a0
            r = int(np.searchsorted(np.cumsum(a), u))
            r = min(r, R - 1)

            v  = v_arr[r]
            ii = i_arr[r]
            jj = j_arr[r]

            N[v, ii] = max(0.0, N[v, ii] - 1)
            N[v, jj] += 1

            events.append((t, rt_list[v], ii, jj))

            if ev_count % self.record_every == 0:
                t_hist.append(t)
                N_hist.append(N.sum(axis=0).copy())

        t_hist.append(T)
        N_hist.append(N.sum(axis=0).copy())

        t_arr = np.array(t_hist)
        N_arr = np.array(N_hist)

        log.info(
            "SSA done: %d events  t_final=%.2f  "
            "conservation_err=%.2e",
            len(events), t,
            abs(N.sum() - N0_total) / (N0_total + 1e-300),
        )
        return SimResult(
            t_eval   = t_arr,
            N_hist   = N_arr,
            mode     = "ssa",
            events   = events,
            metadata = {
                "n_events":          len(events),
                "fraction_moved":    len(events) / (N0_total + 1e-300),
                "conservation_error": abs(N.sum() - N0_total) / (N0_total + 1e-300),
                "stop_ids":           stop_ids,
                "completed_horizon":  t >= T or len(events) < self.max_events,
                "initial_rounding_delta": rounding_delta,
            },
        )

    def ensemble(
        self,
        generators: dict[int, ModalGenerator],
        N0_modal:   dict[int, np.ndarray],
        T:          float = DEFAULT_HORIZON_S,
        M:          int   = 40,
    ) -> list[SimResult]:
        """Run M independent SSA realisations with different random seeds."""
        results = []
        for s in range(M):
            self.rng = np.random.default_rng(s)
            log.info("SSA ensemble run %d/%d", s + 1, M)
            results.append(self.run(generators, N0_modal, T))
        return results


# ---------------------------------------------------------------------------
# Layer 3b: τ-leaping
# ---------------------------------------------------------------------------
class TauLeap:
    """
    Tau-leaping approximation for large-population BKK simulation.

    Advances time by fixed step τ [s] and fires each channel
        Δk^v_{ij} ~ Poisson(a^v_{ij} · τ)
    times, vectorised over all channels simultaneously.

    Conservation note: Δk is capped at N^v_i to prevent negative counts.

    Parameters
    ----------
    tau : float
        Fixed leap size [s].  Default: 1.0 s.
        Smaller τ → more accurate but slower.
        Use ``epsilon_control`` to adapt τ automatically.
    epsilon : float
        Cao–Gillespie–Petzold ε criterion for adaptive step size.
        Set to 0 to use fixed *tau* always.
    rng_seed : int

    Examples
    --------
    >>> leaper = TauLeap(tau=1.0)
    >>> result = leaper.run(generators, N0_modal, T=900.0)
    >>> print(result.conservation_error())  # should be < 1e-6
    """

    def __init__(
        self,
        tau:      float = DEFAULT_TAU_S,
        epsilon:  float = DEFAULT_EPSILON_LEAP,
        rng_seed: int   = 42,
    ) -> None:
        self.tau_fixed = tau
        self.epsilon   = epsilon
        self.rng       = np.random.default_rng(rng_seed)

    def run(
        self,
        generators: dict[int, ModalGenerator],
        N0_modal:   dict[int, np.ndarray],
        T:          float = DEFAULT_HORIZON_S,
        n_eval:     int   = 61,
    ) -> SimResult:
        """
        Run one τ-leap trajectory.

        Parameters
        ----------
        generators : route_type → ModalGenerator
        N0_modal   : route_type → initial array
        T          : simulation horizon [s]
        n_eval     : number of snapshot times for N_hist

        Returns
        -------
        SimResult
        """
        _validate_run_inputs(generators, N0_modal, T, n_eval)
        if not np.isfinite(self.tau_fixed) or self.tau_fixed <= 0:
            raise ValueError("tau must be finite and strictly positive")
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("epsilon must be finite and non-negative")
        # --- Build channels in a canonical global stop index --------------
        v_list, i_list, j_list, q_list = [], [], [], []
        rt_list = list(generators.keys())
        stop_ids, local_to_global = _global_layout(generators)

        for v_idx, rt in enumerate(rt_list):
            mg = generators[rt]
            Q  = mg.Q.tocoo()
            for r in range(Q.nnz):
                row, col, val = Q.row[r], Q.col[r], Q.data[r]
                if row != col and val > 0:
                    v_list.append(v_idx)
                    i_list.append(local_to_global[rt][row])
                    j_list.append(local_to_global[rt][col])
                    q_list.append(val)

        v_arr = np.array(v_list, dtype=np.int32)
        i_arr = np.array(i_list, dtype=np.int32)
        j_arr = np.array(j_list, dtype=np.int32)
        q_arr = np.array(q_list, dtype=np.float64)

        # --- State --------------------------------------------------------
        N = np.zeros((len(rt_list), len(stop_ids)), dtype=np.float64)
        for v_idx, rt in enumerate(rt_list):
            N[v_idx, local_to_global[rt]] = N0_modal.get(
                rt, np.zeros(generators[rt].n_stops)
            ).astype(float)

        N, rounding_delta = _integerise_state(N)
        N0_total  = N.sum()
        t_eval    = np.linspace(0.0, T, n_eval)
        t_hist    = [0.0]
        N_hist    = [N.sum(axis=0).copy()]
        snap_idx  = 1    # next snapshot index

        t         = 0.0
        n_steps   = 0

        log.info(
            "TauLeap: R=%d channels  N0=%.0f  τ=%.2f s  ε=%.3f",
            len(v_arr), N0_total, self.tau_fixed, self.epsilon,
        )

        while t < T:
            tau = self._choose_tau(N, v_arr, i_arr, q_arr)
            if t + tau > T:
                tau = T - t

            # Propensities a[r] = N[v,i] * q[r]
            a = N[v_arr, i_arr] * q_arr   # shape (R,)

            # Poisson samples for each channel
            dk = self.rng.poisson(a * tau).astype(np.int64)   # shape (R,)

            # Atomic outflow clamping — fixes the order-dependency bug:
            # each channel is clamped individually in loop order, so the
            # first channels drain N[v,i] and starve later ones.
            # Fix: compute the *total* proposed outflow per (v,i) first,
            # then scale all channels from that origin proportionally if
            # the total exceeds N[v,i].
            total_out = np.zeros_like(N)                         # (m, n)
            np.add.at(total_out, (v_arr, i_arr), dk)            # sum per origin
            # Oversubscribed origins use an integer multinomial allocation,
            # preserving both total count and relative proposed destinations.
            for v, i in np.argwhere(total_out > N):
                channels = np.flatnonzero((v_arr == v) & (i_arr == i))
                proposed = dk[channels]
                probabilities = proposed / proposed.sum()
                dk[channels] = self.rng.multinomial(int(N[v, i]), probabilities)

            # Apply updates
            np.add.at(N, (v_arr, i_arr), -dk)
            np.add.at(N, (v_arr, j_arr), +dk)
            if np.any(N < 0):
                raise RuntimeError("tau-leap outflow guard failed")

            t      += tau
            n_steps += 1

            # Record every requested time crossed by this approximate leap.
            while snap_idx < len(t_eval) and t >= t_eval[snap_idx]:
                t_hist.append(t_eval[snap_idx])
                N_hist.append(N.sum(axis=0).copy())
                snap_idx += 1

        # Final snapshot
        if t_hist[-1] < T:
            t_hist.append(T)
            N_hist.append(N.sum(axis=0).copy())

        t_arr = np.array(t_hist)
        N_arr = np.array(N_hist)

        cons_err = abs(N.sum() - N0_total) / (N0_total + 1e-300)
        log.info(
            "TauLeap done: %d steps  conservation_err=%.2e",
            n_steps, cons_err,
        )
        return SimResult(
            t_eval   = t_arr,
            N_hist   = N_arr,
            mode     = "tauleap",
            metadata = {
                "n_steps":           n_steps,
                "tau_used":          self.tau_fixed,
                "conservation_error": cons_err,
                "stop_ids":          stop_ids,
                "initial_rounding_delta": rounding_delta,
            },
        )

    def _choose_tau(
        self,
        N:     np.ndarray,
        v_arr: np.ndarray,
        i_arr: np.ndarray,
        q_arr: np.ndarray,
    ) -> float:
        """Conservative propensity-ratio heuristic or fixed τ.

        This is not the full Cao–Gillespie–Petzold algorithm, which requires
        species-wise drift and variance bounds.
        """
        if self.epsilon <= 0:
            return self.tau_fixed

        a     = N[v_arr, i_arr] * q_arr
        a0    = a.sum()
        if a0 <= 0:
            return self.tau_fixed

        # Estimate |∂a_r/∂t| ≈ |q_r · (Σ_incoming - Σ_outgoing)|
        # Simplified: use a0 / max(a) as a proxy
        a_max = a.max()
        if a_max <= 0:
            return self.tau_fixed

        tau_eps = self.epsilon * a0 / a_max
        return max(min(tau_eps, self.tau_fixed * 10), self.tau_fixed * 0.1)
