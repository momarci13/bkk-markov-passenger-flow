"""
bkk/demand_data.py
==================
Data-driven travel demand for the open network model (bkk.linemodel).

Three layers replace the departure-proportional prior lambda_h ∝ f_h:

1. residential population on a raster (Meta HRSL, 30 m), assigned to hubs by
   a logit access-hub choice with mode-specific access constants, calibrated
   to the published trips by mode                     -> P_h
2. district public-transport propensity from car ownership or the census
   commuting mode share                                -> g_K
3. an origin-destination matrix: census district flows Q_KL distributed to
   hubs, or a doubly constrained gravity model         -> W_od

and the source vector is calibrated to published boardings per mode by
bounded least squares on district multipliers mu_K.  The model is
linear in the source vector, so the boardings of mode m are

    b_m(s) = s · M e_m,     M = (-T)^{-1} B,

with B_{h,m} the boarding hazard out of hub h into segments of mode m, and a
district multiplier changes b by the column G_{.K} = sum_{h in K} pi_h M_{h,.}.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import lsq_linear, nnls
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# 1. population catchments
# ---------------------------------------------------------------------------
# Access hierarchy: a hub belongs to the class of its highest-order mode.
HUB_CLASSES = (1, 109, 0, 11, 3, 4)          # metro, HEV, tram, trolleybus, bus, boat


def hub_class(hub_modes: list[set]) -> np.ndarray:
    """Route type of the highest-order mode serving each hub (-1: none)."""
    out = np.full(len(hub_modes), -1)
    for h, ms in enumerate(hub_modes):
        for m in HUB_CLASSES:
            if m in ms:
                out[h] = m
                break
    return out


@dataclass
class AccessChoice:
    """
    Logit access-hub choice of the residents of each raster cell c:

        omega_ch = exp(V_ch) / sum_{h' in C(c)} exp(V_ch'),
        V_ch     = -d_ch / ell + delta_{k(h)},

    where C(c) holds the hubs of class k within the class access radius R_k,
    and delta_k is the access constant of class k (bus = 0).  Then
    P_h = sum_c omega_ch P_c; every covered cell is split completely, so
    sum_h P_h + U = sum_c P_c with U the population outside every C(c).
    """
    idx: np.ndarray        # (n, J) candidate hub ids, -1 = none
    dist: np.ndarray       # (n, J) distances [m]
    cls: np.ndarray        # (n, J) class position in ``classes``
    pop: np.ndarray        # (n,) residents of the cells
    n_hubs: int
    classes: tuple

    def assign(self, delta: dict | None = None, ell_m: float = 250.0) -> tuple[np.ndarray, float]:
        dk = np.array([(delta or {}).get(k, 0.0) for k in self.classes])
        ok = self.idx >= 0
        V = np.where(ok, -self.dist / ell_m + dk[self.cls], -np.inf)
        covered = ok.any(axis=1)
        Vmax = np.where(covered, V.max(axis=1), 0.0)
        E = np.where(ok, np.exp(V - Vmax[:, None]), 0.0)
        Z = E.sum(axis=1)
        wgt = np.where(covered[:, None], E * (self.pop / np.where(covered, Z, 1.0))[:, None], 0.0)
        P = np.bincount(self.idx[ok], weights=wgt[ok], minlength=self.n_hubs)
        return P, float(self.pop[~covered].sum())


def build_access(hub_xy: np.ndarray, hub_cls: np.ndarray, cell_xy: np.ndarray, pop: np.ndarray,
                 radius_m: dict, k_max: dict | None = None,
                 active: np.ndarray | None = None) -> AccessChoice:
    """Candidate hubs per cell: the k_max[k] nearest active hubs of each class within radius_m[k]."""
    act = np.ones(len(hub_xy), dtype=bool) if active is None else np.asarray(active, dtype=bool)
    keep = pop > 0
    cxy, cp = cell_xy[keep], pop[keep]
    classes = tuple(k for k in radius_m if np.any(act & (hub_cls == k)))
    I, D, C = [], [], []
    for j, k in enumerate(classes):
        ids = np.flatnonzero(act & (hub_cls == k))
        kk = min((k_max or {}).get(k, 8), len(ids))
        d, i = cKDTree(hub_xy[ids]).query(cxy, k=kk, distance_upper_bound=radius_m[k])
        d, i = d.reshape(len(cxy), kk), i.reshape(len(cxy), kk)
        miss = ~np.isfinite(d)
        I.append(np.where(miss, -1, ids[np.minimum(i, len(ids) - 1)]))
        D.append(np.where(miss, 0.0, d))
        C.append(np.full(d.shape, j))
    return AccessChoice(np.hstack(I), np.hstack(D), np.hstack(C), cp, len(hub_xy), classes)


def hub_catchments(hub_xy: np.ndarray, cell_xy: np.ndarray, pop: np.ndarray,
                   radius_m: float = 600.0, ell_m: float = 250.0,
                   active: np.ndarray | None = None, k_max: int = 64) -> tuple[np.ndarray, float]:
    """Single-class special case (delta = 0): softmax of -d/ell over hubs within R."""
    acc = build_access(hub_xy, np.zeros(len(hub_xy), dtype=int), cell_xy, pop,
                       {0: radius_m}, {0: k_max}, active)
    return acc.assign(None, ell_m)


def calibrate_access(access: AccessChoice, M: np.ndarray, modes: list, target_share: np.ndarray,
                     free: tuple = (1, 109, 0, 11), ell_m: float = 250.0, g: np.ndarray | None = None,
                     bounds: tuple[float, float] = (-2.0, 2.0)) -> tuple[dict, dict]:
    """
    Access constants delta_k (k in ``free``; bus fixed at 0) such that the
    boardings-by-mode shares of the source vector pi = g * P(delta) match the
    target:

        s_m(delta) = (pi(delta) M)_m / sum_m' (pi(delta) M)_m'.

    Residuals are log-share differences over the modes with a positive target;
    with |free| = (number of such modes) - 1 the system is square and, when
    the target is attainable, solved exactly.  Nonlinear least squares
    (trust-region reflective, scipy.optimize.least_squares).  The default box
    |delta| <= 2 means |delta| * ell <= 500 m for ell = 250 m: the access
    advantage of a class cannot exceed the difference of the rail and bus
    access radii.
    """
    from scipy.optimize import least_squares
    act = target_share > 0
    gg = 1.0 if g is None else g

    def shares(x):
        P, _ = access.assign(dict(zip(free, x)), ell_m)
        b = (gg * P) @ M
        return b / b.sum()

    def resid(x):
        return np.log(np.maximum(shares(x)[act], 1e-12)) - np.log(target_share[act])

    sol = least_squares(resid, np.zeros(len(free)), bounds=bounds, method="trf")
    return dict(zip(free, sol.x)), {"cost": float(sol.cost), "nfev": int(sol.nfev),
                                    "shares": shares(sol.x).tolist(),
                                    "shares_delta0": shares(np.zeros(len(free))).tolist(),
                                    "status": int(sol.status)}


# ---------------------------------------------------------------------------
# 2. public-transport propensity
# ---------------------------------------------------------------------------
def propensity(cars_per_1000: np.ndarray | None, mean_share: float = 0.5, b: float = 1.0,
               observed_share: np.ndarray | None = None) -> np.ndarray:
    """
    g_K = sigma(a - b m_K / 1000), with a chosen so that g at the mean car
    ownership equals ``mean_share``.  An observed census mode share overrides
    the logistic form; without car data the layer is switched off (g = 1).
    """
    if observed_share is not None:
        return np.asarray(observed_share, dtype=float)
    if cars_per_1000 is None:
        return None
    m = np.asarray(cars_per_1000, dtype=float) / 1000.0
    a = np.log(mean_share / (1.0 - mean_share)) + b * m.mean()
    return 1.0 / (1.0 + np.exp(-(a - b * m)))


# ---------------------------------------------------------------------------
# 3. origin-destination weights
# ---------------------------------------------------------------------------
def gravity_ipf(o: np.ndarray, a: np.ndarray, T: np.ndarray, beta: float,
                tol: float = 1e-8, max_iter: int = 500) -> tuple[np.ndarray, dict]:
    """
    Doubly constrained gravity model (Furness / Sinkhorn balancing)

        W_od = r_o c_d o_o a_d exp(-beta t_od),   o != d, t_od < inf,

    with row sums o and column sums a scaled to the same total.  T is the full
    (H, H) matrix of shortest expected times [s].  Rows or columns without any
    reachable partner cannot be balanced and are reported.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        K = np.where(np.isfinite(T), np.exp(-beta * np.where(np.isfinite(T), T, 0.0)), 0.0)
    np.fill_diagonal(K, 0.0)
    o = np.asarray(o, float)
    a = np.asarray(a, float) * o.sum() / a.sum()
    K *= (o > 0)[:, None]
    K *= (a > 0)[None, :]
    ok_r = K.sum(1) > 0
    ok_c = K.sum(0) > 0
    rt = np.where(ok_r, o, 0.0)
    ct = np.where(ok_c, a, 0.0)
    ct *= rt.sum() / ct.sum()
    r = np.ones(len(o))
    c = np.ones(len(o))
    err = np.inf
    for it in range(max_iter):
        r = np.where(ok_r, rt / np.maximum(K @ c, 1e-300), 0.0)
        cs = K.T @ r
        c = np.where(ok_c, ct / np.maximum(cs, 1e-300), 0.0)
        err = float(np.abs((K @ c) * r - rt).sum() / rt.sum())
        if err < tol:
            break
    W = r[:, None] * K * c[None, :]
    return W, {"iterations": it + 1, "row_error": err,
               "unbalanced_origin_mass": float(o[~ok_r].sum() / o.sum())}


def gravity_production(o: np.ndarray, a: np.ndarray, T: np.ndarray, beta: float) -> np.ndarray:
    """
    Production-constrained gravity model (Wilson 1971)

        W_od = o_o a_d exp(-beta t_od) / sum_{d'} a_d' exp(-beta t_od'),

    so every row sums to the origin's journeys o_o, while the attraction a
    (a proxy, not a measured total) only shapes the destination choice.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        F = np.where(np.isfinite(T), np.exp(-beta * np.where(np.isfinite(T), T, 0.0)), 0.0)
    np.fill_diagonal(F, 0.0)
    K = F * np.asarray(a, float)[None, :]
    rs = K.sum(axis=1)
    return np.where(rs[:, None] > 0, K * (np.asarray(o, float) / np.where(rs > 0, rs, 1.0))[:, None],
                    0.0)


def mean_od_time(W: np.ndarray, T: np.ndarray) -> float:
    ok = W > 0
    return float((W[ok] * T[ok]).sum() / W[ok].sum())


def calibrate_beta(o: np.ndarray, a: np.ndarray, T: np.ndarray, target_mean_s: float,
                   lo: float = 1e-5, hi: float = 2e-2, n: int = 40) -> float:
    """
    Deterrence beta of the production-constrained gravity model such that the
    OD-weighted mean shortest time equals the target (the model's own
    phase-type mean journey time).  Each row is a Gibbs distribution in t_od
    with parameter beta, whose mean is decreasing in beta (its derivative is
    minus the variance), so bisection on log beta applies.
    """
    for _ in range(n):
        mid = np.sqrt(lo * hi)
        if mean_od_time(gravity_production(o, a, T, mid), T) > target_mean_s:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def census_od_weights(Q: np.ndarray, district: np.ndarray, o: np.ndarray, a: np.ndarray,
                      T: np.ndarray, beta: float) -> np.ndarray:
    """
    Distribute census district flows Q_KL to hub pairs:

        W_od = Q_{K(o)L(d)} (o_o a_d F(t_od)) / Z_KL,   F(t) = exp(-beta t),
        Z_KL = sum_{o in K, d in L} o_o a_d F(t_od),

    so every block sums exactly to Q_KL.  ``district`` holds integer district
    ids 0..K-1 (negative = outside, weight 0).
    """
    with np.errstate(over="ignore", invalid="ignore"):
        F = np.where(np.isfinite(T), np.exp(-beta * np.where(np.isfinite(T), T, 0.0)), 0.0)
    np.fill_diagonal(F, 0.0)
    base = o[:, None] * a[None, :] * F
    nK = Q.shape[0]
    d = np.asarray(district)
    ind = np.zeros((nK, len(d)))
    ins = d >= 0
    ind[d[ins], np.flatnonzero(ins)] = 1.0
    Z = ind @ base @ ind.T
    scale = np.where(Z > 0, Q / np.where(Z > 0, Z, 1.0), 0.0)
    W = np.zeros_like(base)
    W[np.ix_(ins, ins)] = base[np.ix_(ins, ins)] * scale[np.ix_(d[ins], d[ins])]
    return W


# ---------------------------------------------------------------------------
# 4. NNLS calibration to boardings per mode
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    mu: np.ndarray                 # district multipliers (groups)
    rates: np.ndarray              # calibrated hub source rates [journeys/s]
    G: np.ndarray                  # (modes, groups) design matrix [boardings/s]
    modes: list
    target: np.ndarray             # target boardings per mode [1/s]
    fitted: np.ndarray
    prior_fit: np.ndarray          # boardings per mode at mu = 1
    rho: float
    rho_path: list = field(default_factory=list)


def mode_response(model, modes: list) -> np.ndarray:
    """
    M = (-T)^{-1} B restricted to hub rows: M[h, m] is the expected number of
    boardings of mode m made by one journey that starts at hub h.
    """
    net = model.net
    H, n = net.n_hubs, model.n_states
    B = np.zeros((n, len(modes)))
    for k, m in enumerate(modes):
        sel = net.seg_mode == m
        B[:H, k] = np.bincount(net.seg_from[sel], weights=model.q_board[sel], minlength=H)
    lu = splu(csc_matrix(-model.T))
    return lu.solve(B)[:H]


def design_matrix(M: np.ndarray, prior: np.ndarray, group: np.ndarray, n_groups: int) -> np.ndarray:
    """G_{mK} = sum_{h in K} pi_h M_{h m}."""
    G = np.zeros((M.shape[1], n_groups))
    for K in range(n_groups):
        k = group == K
        G[:, K] = prior[k] @ M[k]
    return G


def solve_multipliers(G: np.ndarray, target: np.ndarray, rho: float,
                      weight: np.ndarray | None = None,
                      extra: tuple[np.ndarray, np.ndarray] | None = None,
                      bounds: tuple[float, float] = (0.0, np.inf)) -> np.ndarray:
    """
    min_{lo <= mu <= hi} || D (G mu - b) ||^2 + || A_x mu - y_x ||^2 + rho || mu - 1 ||^2

    with D = diag(weight) (default 1/b, relative residuals), solved as one
    stacked least-squares problem: Lawson-Hanson NNLS for bounds (0, inf),
    bounded-variable least squares (Stark-Parker BVLS) otherwise.  For
    rho > 0 the objective is strictly convex, so the minimiser is unique.
    """
    D = 1.0 / np.maximum(target, 1e-300) if weight is None else weight
    A = [D[:, None] * G]
    y = [D * target]
    if extra is not None:
        A.append(extra[0]); y.append(extra[1])
    n = G.shape[1]
    A.append(np.sqrt(rho) * np.eye(n)); y.append(np.sqrt(rho) * np.ones(n))
    A, y = np.vstack(A), np.concatenate(y)
    if bounds == (0.0, np.inf):
        return nnls(A, y, maxiter=50 * n)[0]
    return lsq_linear(A, y, bounds=bounds, method="bvls").x


def calibrate(model, prior: np.ndarray, group: np.ndarray, n_groups: int,
              modes: list, target: np.ndarray, weight: np.ndarray | None = None,
              rho: float = 1e-3, bounds: tuple[float, float] = (0.5, 2.0),
              extra=None, rhos: np.ndarray | None = None) -> Calibration:
    """
    Fit district multipliers mu in [lo, hi] to the target boardings per mode.
    ``prior`` must already be scaled so that its total boardings equal the
    target total, so mu = 1 is the uncalibrated prior.  The ridge path over
    ``rhos`` (without bounds, NNLS) is returned for the L-curve diagnostic.
    """
    M = mode_response(model, modes)
    G = design_matrix(M, prior, group, n_groups)
    D = 1.0 / np.maximum(target, 1e-300) if weight is None else weight
    act = D > 0
    path = []
    for r in (np.logspace(-6, 1, 36) if rhos is None else rhos):
        mu_r = solve_multipliers(G, target, r, D, extra)
        res = (G @ mu_r - target)[act] / target[act]
        path.append({"rho": float(r), "rms_rel": float(np.sqrt(np.mean(res ** 2))),
                     "dev": float(np.linalg.norm(mu_r - 1.0))})
    mu = solve_multipliers(G, target, rho, D, extra, bounds)
    rates = prior * mu[group]
    return Calibration(mu=mu, rates=rates, G=G, modes=list(modes), target=target,
                       fitted=G @ mu, prior_fit=G.sum(axis=1), rho=rho, rho_path=path)
