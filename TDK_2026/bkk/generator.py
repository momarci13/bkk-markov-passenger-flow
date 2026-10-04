"""
bkk.generator
=============
Assemble the CTMC generator matrix Q^v (CSR) for each modal subgraph.

Mathematical formulation
------------------------
Off-diagonal rates (Eq. 4 in the paper):
    q^v_{ij} = μ^v_i · π^v_{j|i}

Wilson destination kernel (row-wise softmax, log-sum-exp stabilised):
    π^v_{j|i} = exp(-λ^v · C^v_{ij}) / Σ_{k ∈ N^v(i)} exp(-λ^v · C^v_{ik})

Generator diagonal:
    Q^v_{ii} = -Σ_{j ≠ i} q^v_{ij}     (row sums = 0)

Units
-----
λ^v  [s^{-1}],  C^v_{ij}  [s]  ⟹  λ^v · C^v_{ij}  dimensionless
q^v_{ij}  [s^{-1}]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from .constants import DEFAULT_LAMBDA
from .network import ModalGraph

log = logging.getLogger(__name__)


@dataclass
class ModalGenerator:
    """CTMC generator and transition matrix for one mode."""
    route_type: int
    name:       str
    n_stops:    int
    stop_ids:   np.ndarray   # shape (n,)  str
    Q:          csr_matrix   # shape (n, n)  generator
    P:          csr_matrix   # shape (n, n)  DTMC transition (embedded jump chain)
    lambda_v:   float        # [s^{-1}]
    mu_arr:      np.ndarray | None = None  # per-passenger hazards [s^-1]

    # ------------------------------------------------------------------ #
    def validate(self, tol: float = 1e-8) -> dict[str, float]:
        """
        Run diagnostic checks; return dict of metrics.
        Raises AssertionError if a hard constraint is violated.

        Notes
        -----
        Terminal stops (no outgoing edges) are valid absorbing states;
        their P row sums to 0 in the embedded-jump-chain representation.
        The row-stochasticity check is applied only to non-absorbing rows.
        """
        row_sums_Q = np.array(self.Q.sum(axis=1)).ravel()
        row_sums_P = np.array(self.P.sum(axis=1)).ravel()

        # Non-absorbing rows: those with at least one outgoing edge
        non_absorbing = row_sums_P > tol
        max_q_rowsum  = float(np.abs(row_sums_Q).max())
        if non_absorbing.any():
            max_p_rowsum = float(np.abs(row_sums_P[non_absorbing] - 1.0).max())
        else:
            max_p_rowsum = 0.0   # all absorbing – degenerate but valid
        n_absorbing   = int((~non_absorbing).sum())

        # Off-diagonal Q non-negativity
        if self.Q.nnz > 0:
            Qcoo = self.Q.tocoo()
            off_mask = Qcoo.row != Qcoo.col
            min_q_offdiag = float(Qcoo.data[off_mask].min()) if off_mask.any() else 0.0
        else:
            min_q_offdiag = 0.0

        assert max_q_rowsum < tol, (
            f"Generator row-sum violation: max|Σ_j Q_ij| = {max_q_rowsum:.2e}"
        )
        assert max_p_rowsum < tol, (
            f"P row-stochasticity violation (non-absorbing rows): "
            f"max|Σ_j P_ij - 1| = {max_p_rowsum:.2e}"
        )
        assert min_q_offdiag >= -tol, (
            f"Negative off-diagonal Q entry: min = {min_q_offdiag:.2e}"
        )

        return {
            "max_Q_rowsum_abs": max_q_rowsum,
            "max_P_rowsum_dev": max_p_rowsum,
            "min_Q_offdiag":    min_q_offdiag,
            "n_absorbing":      n_absorbing,
            "nnz_Q":            self.Q.nnz,
            "nnz_P":            self.P.nnz,
        }


class GeneratorBuilder:
    """
    Build CTMC generator matrices Q^v from a :class:`~bkk.network.ModalGraph`.

    Parameters
    ----------
    lambda_v : float or dict[int, float]
        Entropy–cost sensitivity λ^v [s^{-1}].
        Either a single float (applied to all modes) or a dict mapping
        route_type → λ^v.
        Default: 1/300 s^{-1} (penalises ~5-min travel by factor e^{-1}).

    Examples
    --------
    >>> from bkk import GTFSLoader, NetworkBuilder, GeneratorBuilder
    >>> feed = GTFSLoader(cache_dir="data/").download().parse(weekday="monday")
    >>> net  = NetworkBuilder().build(feed)
    >>> gen  = GeneratorBuilder(lambda_v=1/300)
    >>> for rt, mg in net.modal.items():
    ...     modal_gen = gen.build(mg)
    ...     metrics   = modal_gen.validate()
    ...     print(f"mode {rt}: {metrics}")
    """

    def __init__(
        self,
        lambda_v: float | dict[int, float] = DEFAULT_LAMBDA,
    ) -> None:
        self._lambda_default = (
            lambda_v if isinstance(lambda_v, float) else DEFAULT_LAMBDA
        )
        self._lambda_map: dict[int, float] = (
            lambda_v if isinstance(lambda_v, dict) else {}
        )

    def _lam(self, route_type: int) -> float:
        return self._lambda_map.get(route_type, self._lambda_default)

    # ------------------------------------------------------------------ #
    def build(self, mg: ModalGraph) -> ModalGenerator:
        """
        Build the generator and embedded-jump-chain for *mg*.

        Parameters
        ----------
        mg : ModalGraph

        Returns
        -------
        ModalGenerator
        """
        lam = self._lam(mg.route_type)
        n   = mg.n_stops
        log.info(
            "GeneratorBuilder: mode %d (%s)  n=%d  |E|=%d  λ=%.2e",
            mg.route_type, mg.name, n, mg.n_edges, lam,
        )

        # --- Compute Wilson softmax rates ---------------------------------
        # Group edges by origin stop i
        # For each i: log_w_k = -λ · C^v_{i,j_k}  (j_k ∈ N^v(i))
        # Stabilise: subtract row max before exp (log-sum-exp trick)

        i_arr    = mg.i_arr
        j_arr    = mg.j_arr
        cost_arr = mg.cost_arr
        mu_arr   = mg.mu_arr

        # Sort by origin for efficient groupby
        order  = np.argsort(i_arr, kind="stable")
        i_s    = i_arr[order]
        j_s    = j_arr[order]
        c_s    = cost_arr[order]

        # Allocate rate array (same length as edges)
        q_vals = np.empty(len(order), dtype=np.float64)

        # Group boundaries
        boundaries = np.where(np.diff(i_s))[0] + 1
        groups     = np.split(np.arange(len(order)), boundaries)

        for grp_idx in groups:
            orig  = i_s[grp_idx[0]]
            costs = c_s[grp_idx]

            # Log-sum-exp stabilised softmax
            log_w   = -lam * costs
            log_w  -= log_w.max()          # stability shift (cancels in softmax)
            w       = np.exp(log_w)
            pi      = w / w.sum()          # destination probabilities π^v_{j|i}

            q_vals[grp_idx] = mu_arr[orig] * pi   # per-passenger rate [s^-1]

        # --- Assemble CSR generator Q -------------------------------------
        # Off-diagonals
        Q_lil = lil_matrix((n, n), dtype=np.float64)
        for r in range(len(i_s)):
            Q_lil[i_s[r], j_s[r]] = q_vals[r]

        # Diagonal: Q_{ii} = -Σ_{j≠i} Q_{ij}
        for i in range(n):
            row_offdiag_sum = Q_lil.getrow(i).sum()
            Q_lil[i, i]     = -row_offdiag_sum

        Q_csr = Q_lil.tocsr()
        Q_csr.eliminate_zeros()

        # --- Assemble transition matrix P (embedded jump chain) -----------
        # P^v_{ij} = π^v_{j|i} for j≠i, P^v_{ii} = 0
        # Reuse the group structure to recompute pi cleanly.
        P_lil = lil_matrix((n, n), dtype=np.float64)
        for grp_idx in groups:
            orig  = i_s[grp_idx[0]]
            costs = c_s[grp_idx]
            dests = j_s[grp_idx]

            log_w  = -lam * costs
            log_w -= log_w.max()
            w      = np.exp(log_w)
            pi     = w / w.sum()            # destination probabilities

            for k, j in enumerate(dests):
                P_lil[orig, j] = pi[k]

        P_csr = P_lil.tocsr()
        P_csr.eliminate_zeros()

        mg_gen = ModalGenerator(
            route_type = mg.route_type,
            name       = mg.name,
            n_stops    = n,
            stop_ids   = mg.stop_ids,
            Q          = Q_csr,
            P          = P_csr,
            lambda_v   = lam,
            mu_arr      = mg.mu_arr.copy(),
        )
        metrics = mg_gen.validate()
        log.info(
            "  Validation OK: max|ΣQ_ij|=%.1e  max|ΣP_ij-1|=%.1e",
            metrics["max_Q_rowsum_abs"],
            metrics["max_P_rowsum_dev"],
        )
        return mg_gen

    def build_all(
        self,
        net,  # NetworkData
    ) -> dict[int, ModalGenerator]:
        """Build generators for all modes in *net*."""
        return {rt: self.build(mg) for rt, mg in net.modal.items()}
