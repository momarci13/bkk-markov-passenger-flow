"""
bkk.resilience
==============
Network resilience analysis via node-removal perturbation.

For each candidate stop i*, compute:
    ΔK        Kemeny-constant increase
    Δg/g      Spectral-gap collapse fraction
    ΔE/E      Network-efficiency loss fraction

Aggregate into a criticality score C_i = sqrt(I_tilde * V_tilde).

See Section 10 of the paper and Algorithm 3 (alg:resilience).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigs

from .constants import KRYLOV_TRUNC, RESILIENCE_TOP_K
from .generator import ModalGenerator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class BaselineMetrics:
    """Network metrics for the intact network."""
    route_type:    int
    kemeny:        float          # K(P)
    spectral_gap:  float          # g(P) = 1 - |λ_2|
    efficiency:    float          # E(P)
    stationary_pi: np.ndarray     # shape (n,)
    eigenvalues:   np.ndarray     # shape (K_trunc,)  non-principal
    irreducible:   bool           # whether standard Kemeny is defined


@dataclass
class StopMetrics:
    """Vulnerability metrics for one candidate stop."""
    stop_id:       str
    stop_local_k:  int            # index in modal stop array
    importance:    float          # stationary departure flux π^Q_i μ_i
    delta_kemeny:  float          # K(P^{-i}) - K(P)
    delta_gap:     float          # (g - g^{-i}) / g
    delta_eff:     float          # (E - E^{-i}) / E
    vulnerability: float          # aggregated V_i ∈ [0,1]
    criticality:   float          # C_i = sqrt(I_tilde * V_tilde)


@dataclass
class ResilienceReport:
    """Full resilience analysis result for one mode."""
    route_type: int
    baseline:   BaselineMetrics
    stops:      list[StopMetrics] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for s in self.stops:
            rows.append({
                "stop_id":       s.stop_id,
                "importance":    s.importance,
                "delta_kemeny":  s.delta_kemeny,
                "delta_gap":     s.delta_gap,
                "delta_eff":     s.delta_eff,
                "vulnerability": s.vulnerability,
                "criticality":   s.criticality,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("criticality", ascending=False).reset_index(drop=True)
            df.index = df.index + 1
            df.index.name = "rank"
        return df


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------
class ResilienceAnalyser:
    """
    Node-removal resilience analysis for the BKK Markov-chain model.

    Parameters
    ----------
    top_k : int
        Number of candidate stops to analyse (pre-filtered by importance).
        Default: 100.
    krylov_trunc : int
        Number of non-principal eigenvalues for truncated Kemeny sum.
        Default: 20.
    mu_v : dict[int, np.ndarray], optional
        Per-stop departure intensities μ^v_i (for importance score).
        Provided automatically when using ``analyse_all``.

    Examples
    --------
    >>> analyser = ResilienceAnalyser(top_k=50)
    >>> for rt, mg in generators.items():
    ...     report = analyser.analyse(mg)
    ...     print(report.to_dataframe().head(10))
    """

    def __init__(
        self,
        top_k:        int = RESILIENCE_TOP_K,
        krylov_trunc: int = KRYLOV_TRUNC,
        dense_max:    int = 1500,
    ) -> None:
        self.top_k        = top_k
        self.krylov_trunc = krylov_trunc
        self.dense_max    = dense_max

    # ------------------------------------------------------------------ #
    def analyse(
        self,
        mg:    ModalGenerator,
        mu_v:  Optional[np.ndarray] = None,
    ) -> ResilienceReport:
        """
        Full resilience analysis for one modal generator.

        Parameters
        ----------
        mg   : ModalGenerator
        mu_v : departure intensities μ^v_i, shape (n,).
               Used to compute the stationary departure flux
               I_i = π^Q_i μ_i = π^P_i / Σ_k π^P_k / μ_k.
               If None, uses uniform μ = 1.

        Returns
        -------
        ResilienceReport
        """
        n    = mg.n_stops
        P    = mg.P
        if mu_v is not None:
            mu = np.asarray(mu_v, dtype=float)
        elif mg.mu_arr is not None:
            mu = np.asarray(mg.mu_arr, dtype=float)
        else:
            mu = np.ones(n)
        if mu.shape != (n,) or np.any(~np.isfinite(mu)) or np.any(mu < 0):
            raise ValueError(f"importance weights must have shape ({n},) and be non-negative")

        log.info(
            "ResilienceAnalyser: mode %d  n=%d  top_k=%d",
            mg.route_type, n, self.top_k,
        )

        # --- Baseline metrics -------------------------------------------
        baseline = self._baseline(P, mg.route_type)

        # --- Importance scores ------------------------------------------
        pi    = baseline.stationary_pi
        imp   = self.departure_flux(pi, mu)   # stationary departures [s^-1]

        # Restrict to top_k candidates
        k_candidates = min(self.top_k, n)
        cand_idx     = np.argsort(imp)[::-1][:k_candidates]

        log.info(
            "  Baseline K=%.2f  g=%.4f  E=%.4f",
            baseline.kemeny, baseline.spectral_gap, baseline.efficiency,
        )

        # --- Per-candidate perturbation ---------------------------------
        stop_metrics = []
        for local_k in cand_idx:
            sid = mg.stop_ids[local_k]
            sm  = self._perturb(
                P, local_k, n, baseline, imp[local_k], sid
            )
            stop_metrics.append(sm)

        # --- Min-max normalise and compute criticality ------------------
        if stop_metrics:
            imp_vals = np.array([s.importance    for s in stop_metrics])
            vul_vals = np.array([s.vulnerability for s in stop_metrics])

            def _minmax(x: np.ndarray) -> np.ndarray:
                lo, hi = x.min(), x.max()
                return (x - lo) / (hi - lo + 1e-300)

            I_tilde = _minmax(imp_vals)
            V_tilde = _minmax(vul_vals)
            C       = np.sqrt(I_tilde * V_tilde)

            for i, sm in enumerate(stop_metrics):
                sm.criticality = float(C[i])

        log.info("  Analysis done: %d candidates evaluated.", len(stop_metrics))

        return ResilienceReport(
            route_type = mg.route_type,
            baseline   = baseline,
            stops      = stop_metrics,
        )

    def analyse_all(
        self,
        generators: dict[int, ModalGenerator],
        N0_modal:   Optional[dict[int, np.ndarray]] = None,
    ) -> dict[int, ResilienceReport]:
        """Analyse all modes. Optionally pass N0_modal for importance scaling."""
        reports = {}
        for rt, mg in generators.items():
            mu_v = None
            if N0_modal and rt in N0_modal:
                # Expected stationary departures: N_mode * pi_i * mu_i.
                hazards = mg.mu_arr if mg.mu_arr is not None else np.ones(mg.n_stops)
                mu_v = hazards * float(np.asarray(N0_modal[rt]).sum())
            reports[rt] = self.analyse(mg, mu_v=mu_v)
        return reports

    # ------------------------------------------------------------------ #
    #  Internal: baseline metrics                                          #
    # ------------------------------------------------------------------ #
    def _baseline(self, P: csr_matrix, route_type: int) -> BaselineMetrics:
        n = P.shape[0]
        P = self._make_stochastic(P)
        irreducible = self._is_irreducible(P)

        # Stationary distribution π: left-null-space of (P^T - I)
        pi = self._stationary(P, n)

        # Non-principal eigenvalues of P (ARPACK)
        evals = self._top_eigenvalues(P, n)

        kemeny       = self._kemeny_from_evals(evals, n) if irreducible else float("nan")
        spectral_gap = 1.0 - float(np.abs(evals).max()) if len(evals) else 0.0
        efficiency   = self._network_efficiency(P, pi)

        return BaselineMetrics(
            route_type    = route_type,
            kemeny        = kemeny,
            spectral_gap  = spectral_gap,
            efficiency    = efficiency,
            stationary_pi = pi,
            eigenvalues   = evals,
            irreducible   = irreducible,
        )

    # ------------------------------------------------------------------ #
    #  Internal: single node removal                                       #
    # ------------------------------------------------------------------ #
    def _perturb(
        self,
        P:         csr_matrix,
        local_k:   int,
        n:         int,
        baseline:  BaselineMetrics,
        importance: float,
        sid:       str,
    ) -> StopMetrics:
        """Compute post-removal metrics for stop *local_k*."""
        # Build reduced transition matrix P^{-k}: delete row/col k, renormalise
        P_red = self._remove_node(P, local_k)
        n_red = n - 1

        irreducible_red = self._is_irreducible(P_red)
        evals_red     = self._top_eigenvalues(P_red, n_red)
        pi_red        = self._stationary(P_red, n_red)

        kemeny_red = self._kemeny_from_evals(evals_red, n_red) if irreducible_red else float("nan")
        # Clamp spectral gap to [0, 1] — eigenvalues are already clipped to
        # [0, 1] in _top_eigenvalues, so max(|λ|) ≤ 1 and gap ≥ 0 always.
        gap_red = float(np.clip(1.0 - np.abs(evals_red).max(), 0.0, 1.0)) \
                  if len(evals_red) else 0.0
        eff_red       = self._network_efficiency(P_red, pi_red)

        if np.isfinite(baseline.kemeny) and np.isfinite(kemeny_red):
            delta_K = max(0.0, (kemeny_red - baseline.kemeny) /
                          (abs(baseline.kemeny) + 1e-300))
        elif baseline.irreducible and not irreducible_red:
            delta_K = 1.0
        else:
            delta_K = 0.0
        delta_g   = (baseline.spectral_gap - gap_red) / (baseline.spectral_gap + 1e-300)
        delta_eff = (baseline.efficiency - eff_red) / (baseline.efficiency + 1e-300)

        delta_g   = max(0.0, delta_g)
        delta_eff = max(0.0, delta_eff)

        # Aggregate vulnerability (unweighted mean of three normalised signals)
        # (normalisation to [0,1] across candidates is done after all stops)
        vul = (delta_K + delta_g + delta_eff) / 3.0   # pre-normalisation

        return StopMetrics(
            stop_id      = sid,
            stop_local_k = local_k,
            importance   = float(importance),
            delta_kemeny = float(delta_K),
            delta_gap    = float(delta_g),
            delta_eff    = float(delta_eff),
            vulnerability = float(vul),
            criticality  = 0.0,   # filled later
        )

    # ------------------------------------------------------------------ #
    #  Utility methods                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def departure_flux(pi_jump: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Stationary departure flux of the CTMC from the jump-chain distribution.

        ``pi_jump`` is stationary for the embedded chain P, not for Q.  The
        CTMC stationary law is pi^Q_i ∝ pi^P_i / mu_i, hence the departure
        flux is pi^Q_i mu_i = pi^P_i / sum_k (pi^P_k / mu_k).  The former
        ``pi^P_i * mu_i`` weighted stops by mu_i twice.  Stops with mu_i = 0
        have no departures and receive zero flux.
        """
        pi_jump = np.asarray(pi_jump, dtype=float)
        mu = np.asarray(mu, dtype=float)
        active = mu > 0
        flux = np.zeros_like(pi_jump)
        denom = float(np.sum(pi_jump[active] / mu[active]))
        if denom > 0:
            flux[active] = pi_jump[active] / denom
        return flux

    @staticmethod
    def _remove_node(P: csr_matrix, k: int) -> csr_matrix:
        """Return P with row/col k deleted and rows renormalised."""
        keep = np.arange(P.shape[0]) != k
        P_red = P.tocsr()[keep][:, keep].tolil()
        row_sums = np.asarray(P_red.sum(axis=1)).ravel()
        for row in np.flatnonzero(row_sums == 0):
            P_red[row, row] = 1.0
        row_sums = np.asarray(P_red.sum(axis=1)).ravel()
        return csr_matrix(P_red.multiply((1.0 / row_sums)[:, None]))

    @staticmethod
    def _make_stochastic(P: csr_matrix) -> csr_matrix:
        """Return a row-stochastic copy, making dead ends absorbing."""
        out = P.tolil(copy=True)
        sums = np.asarray(out.sum(axis=1)).ravel()
        for i in np.flatnonzero(sums == 0):
            out[i, i] = 1.0
        sums = np.asarray(out.sum(axis=1)).ravel()
        return csr_matrix(out.multiply((1.0 / sums)[:, None]))

    @staticmethod
    def _is_irreducible(P: csr_matrix) -> bool:
        if P.shape[0] <= 1:
            return True
        count, _ = connected_components(P, directed=True, connection="strong")
        return count == 1

    def _top_eigenvalues(self, P: csr_matrix, n: int) -> np.ndarray:
        """Compute top non-principal eigenvalues of P.

        Strategy
        --------
        - For matrices with n ≤ dense_max (default 1500): dense
          ``numpy.linalg.eigvals`` gives the full spectrum, so Kemeny's
          constant is exact (no tail approximation).  Formerly n ≤ 50.
          to avoid ARPACK instability.  The 18-stop tram subgraph returned
          |λ| ~ 10^{291} from ARPACK (non-convergence artefact), causing
          delta_gap overflow.
        - For large matrices: ARPACK sparse eigensolver (much faster).
        - Always clip eigenvalue magnitudes to [0, 1] — row-stochastic
          matrices have spectral radius ≤ 1; anything larger is numerical
          noise and must be discarded before Kemeny/gap calculations.
        """
        dense = n <= self.dense_max
        k = n - 1 if dense else min(self.krylov_trunc, n - 2)
        if k < 1:
            return np.array([])

        try:
            if dense:
                # Dense path: exact and stable for small matrices
                vals = np.linalg.eigvals(P.toarray())
            else:
                # Sparse ARPACK path for large matrices
                vals, _ = eigs(P.T.tocsr(), k=k + 1, which="LM",
                               tol=1e-6, maxiter=1000)

            vals = np.asarray(vals, dtype=complex)
            vals = vals[np.isfinite(vals)]
            vals = vals[np.abs(vals) <= 1.0 + 1e-7]
            if len(vals):
                principal = int(np.argmin(np.abs(vals - 1.0)))
                if abs(vals[principal] - 1.0) < 1e-5:
                    vals = np.delete(vals, principal)
            vals = vals[np.argsort(np.abs(vals))[::-1]]
            return vals[:k]

        except Exception as exc:
            log.debug("eigs failed for n=%d: %s", n, exc)
            return np.array([])

    @staticmethod
    def _kemeny_from_evals(evals: np.ndarray, n: int) -> float:
        """
        Kemeny constant from non-principal eigenvalues.
        K(P) = Σ_{k=2}^{n} 1 / (1 - λ_k)
        Tail correction: (n - len(evals)) / (1 - mean_tail_lambda)
        """
        if len(evals) == 0:
            return float(n)   # degenerate
        safe = np.where(np.abs(1.0 - evals) < 1e-9, 1.0 - 1e-9, evals)
        contrib = np.sum(1.0 / (1.0 - safe))
        n_tail  = n - 1 - len(evals)
        if n_tail > 0 and len(evals) > 0:
            contrib += n_tail  # neutral zero-eigenvalue tail approximation
        return float(np.real_if_close(contrib).real)

    @staticmethod
    def _stationary(P: csr_matrix, n: int) -> np.ndarray:
        """
        Compute the uniform-start Cesàro stationary distribution.

        Cesàro averaging converges for periodic and reducible finite chains;
        for reducible chains the documented uniform start selects one member
        of the non-unique stationary family.
        """
        if n == 0:
            return np.array([])
        pi = np.ones(n, dtype=float) / n
        avg = np.zeros(n, dtype=float)
        P_csr = ResilienceAnalyser._make_stochastic(P)
        for iteration in range(1, 10001):
            pi_new = pi @ P_csr   # row-vector convention
            pi_new /= pi_new.sum() + 1e-300
            avg += (pi_new - avg) / iteration
            if iteration > 100 and np.linalg.norm(avg @ P_csr - avg, 1) < 1e-10:
                return avg / avg.sum()
            pi = pi_new
        log.warning("Cesàro stationary distribution did not converge in 10000 iterations.")
        return avg / (avg.sum() + 1e-300)

    @staticmethod
    def _network_efficiency(P: csr_matrix, pi: np.ndarray) -> float:
        """
        Latora–Marchiori efficiency E = (1 / n(n-1)) sum_{i != j} 1 / d_ij
        with d_ij the directed hop distance in the support graph of P.

        Exact all-sources BFS (scipy.sparse.csgraph).  The former version
        summed over the first 200 stop indices only, i.e. a sample selected
        by stop-id sort order rather than at random.  ``pi`` is unused and
        kept for API compatibility.
        """
        from scipy.sparse.csgraph import shortest_path

        n = P.shape[0]
        if n <= 1:
            return 1.0
        A = P.tocsr().copy()
        A.setdiag(0.0)
        A.eliminate_zeros()
        A.data[:] = 1.0
        total = 0.0
        for start in range(0, n, 512):
            idx = np.arange(start, min(n, start + 512))
            d = shortest_path(A, directed=True, unweighted=True, indices=idx)
            d[np.arange(len(idx)), idx] = np.inf
            finite = np.isfinite(d) & (d > 0)
            total += float((1.0 / d[finite]).sum())
        return total / (n * (n - 1))
