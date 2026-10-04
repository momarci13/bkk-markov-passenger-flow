"""
bkk.linemodel
=============
Line-aware open Markov network for transit passenger flow (model v3).

The legacy stop-hop chain (``bkk.generator``) lets every passenger alight at
every stop, ignores in-vehicle time in the dynamics, and uses the raw vehicle
frequency as waiting hazard.  This module replaces it with a continuous-time
Markov model on an augmented state space

    X = H  (waiting at hub h)   ∪   S  (riding segment s = (line, i -> j))

with the following transitions (all rates in s^-1):

* boarding         h -> s     q_s = theta_s * a_s,        s departs from h
* in-vehicle exit  s -> .     nu_s = 1 / tau_s
      continue on the same line   s -> s'   (1 - alpha_s) r_{ss'}
      alight at hub(j)            s -> h'   alpha_s + (1 - alpha_s) r_{s,end}
* after alighting  transfer (stay at h') with prob. p_tr, else leave the system.

theta_s = 2 f_s / (1 + CV_s^2) is the exponential hazard whose mean equals the
renewal-theoretic mean wait E[W] = E[H](1 + CV^2)/2 for random passenger
arrivals; a_s = exp(-lambda (c_s - min_{s' in h} c_{s'})) is an acceptance
probability, so boarding choices follow the minimum-relative-entropy kernel
pi_{s|h} ∝ theta_s exp(-lambda c_s).  alpha_s = 1 - exp(-d_s / D) makes
alighting a Poisson process in distance with mean leg length D.

External demand enters hubs as independent Poisson streams with rates
lambda_h.  The transient part T of the generator is a sub-generator, so the
journey time is phase-type PH(alpha, T); the stationary occupation of every
state is independent Poisson with mean vector L = lambda (-T)^{-1}.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import eigs, expm_multiply, spsolve

from .constants import ROUTE_TYPES
from .network import _haversine_m, _parse_time_s

log = logging.getLogger(__name__)

_SUFFIX = re.compile(r"\s+(M\+H|M|H)$")


def normalise_stop_name(name: str) -> str:
    """Strip BKK interchange suffixes (' M', ' H', ' M+H') used on surface stops."""
    return _SUFFIX.sub("", str(name).strip())


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# ---------------------------------------------------------------------------
# Network container
# ---------------------------------------------------------------------------
@dataclass
class LineNetwork:
    """Hubs and line segments active in one service window."""
    hub_name:  np.ndarray        # (H,) str
    hub_lat:   np.ndarray        # (H,)
    hub_lon:   np.ndarray        # (H,)
    stop_hub:  dict              # raw stop_id -> hub index
    seg_route: np.ndarray        # (S,) route_id
    seg_mode:  np.ndarray        # (S,) route_type
    seg_from:  np.ndarray        # (S,) hub index of origin stop
    seg_to:    np.ndarray        # (S,) hub index of destination stop
    seg_xy:    np.ndarray        # (S, 4) lat_i, lon_i, lat_j, lon_j
    seg_n:     np.ndarray        # (S,) departures in window
    seg_freq:  np.ndarray        # (S,) f_s [s^-1]
    seg_cv2:   np.ndarray        # (S,) squared headway CV (1.0 if unknown)
    seg_tau:   np.ndarray        # (S,) median in-vehicle time [s]
    seg_dist:  np.ndarray        # (S,) segment length [m]
    succ:      csr_matrix        # (S, S) continuation shares r_{ss'}
    seg_end:   np.ndarray        # (S,) share of trips terminating after s
    window_s:  float
    meta:      dict = field(default_factory=dict)

    @property
    def n_hubs(self) -> int:
        return len(self.hub_name)

    @property
    def n_segments(self) -> int:
        return len(self.seg_route)

    def hub_departures(self) -> np.ndarray:
        """Scheduled departures per second leaving each hub, f_h = sum_s f_s."""
        return np.bincount(self.seg_from, weights=self.seg_freq, minlength=self.n_hubs)

    def hub_modes(self) -> list[set]:
        modes = [set() for _ in range(self.n_hubs)]
        for h, m in zip(self.seg_from, self.seg_mode):
            modes[h].add(int(m))
        return modes


def build_line_network(
    feed,
    window: tuple[str, str] = ("07:00:00", "09:00:00"),
    hub_radius_m: float = 400.0,
    tau_min_s: float = 20.0,
) -> LineNetwork:
    """
    Build hubs and line segments from a date-filtered :class:`GTFSFeed`.

    Hubs: platforms are first mapped to ``parent_station``; stops whose
    normalised names agree and lie within ``hub_radius_m`` (single linkage)
    form one interchange hub.  Segments are consecutive stop pairs of trips
    whose departure from the segment origin lies inside the window.
    """
    w0 = float(_parse_time_s(pd.Series([window[0]])).iloc[0])
    w1 = float(_parse_time_s(pd.Series([window[1]])).iloc[0])
    W = w1 - w0
    if W <= 0:
        raise ValueError("window end must be after window start")

    stops = feed.stops.copy()
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)
    coords = stops.set_index("stop_id")[["stop_lat", "stop_lon", "stop_name"]]

    st = feed.stop_times[["trip_id", "stop_id", "stop_sequence",
                          "arrival_time", "departure_time"]].copy()
    st = st.merge(feed.trips[["trip_id", "route_id"]], on="trip_id")
    st = st.merge(feed.routes[["route_id", "route_type"]], on="route_id")
    st["route_type"] = st["route_type"].astype(int)
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"])
    st["dep"] = _parse_time_s(st["departure_time"].astype(str))
    st["arr"] = _parse_time_s(st["arrival_time"].astype(str))
    st = st.dropna(subset=["dep", "arr"]).sort_values(["trip_id", "stop_sequence"])

    g = st.groupby("trip_id", sort=False)
    st["stop_j"] = g["stop_id"].shift(-1)
    st["arr_j"] = g["arr"].shift(-1)
    st["stop_k"] = g["stop_id"].shift(-2)
    pairs = st.dropna(subset=["stop_j"])
    pairs = pairs[(pairs["dep"] >= w0) & (pairs["dep"] < w1)].copy()
    if pairs.empty:
        raise ValueError("no segment departures inside the window")

    # --- hubs ---------------------------------------------------------------
    parent = {}
    if "parent_station" in stops.columns:
        for sid, par in zip(stops["stop_id"], stops["parent_station"]):
            parent[sid] = par if isinstance(par, str) and par.strip() else sid
    used = pd.unique(np.concatenate([pairs["stop_id"].values, pairs["stop_j"].values]))
    node_of = {s: parent.get(s, s) for s in used}
    nodes = np.array(sorted(set(node_of.values())))
    nlat = coords.loc[nodes, "stop_lat"].to_numpy()
    nlon = coords.loc[nodes, "stop_lon"].to_numpy()
    nname = np.array([normalise_stop_name(x) for x in coords.loc[nodes, "stop_name"]])

    uf = _UnionFind(len(nodes))
    by_name: dict[str, list[int]] = {}
    for k, nm in enumerate(nname):
        by_name.setdefault(nm, []).append(k)
    for idx in by_name.values():
        for a_pos, a in enumerate(idx):
            for b in idx[a_pos + 1:]:
                if _haversine_m(nlat[a], nlon[a], nlat[b], nlon[b]) <= hub_radius_m:
                    uf.union(a, b)
    root = np.array([uf.find(k) for k in range(len(nodes))])
    roots, node_hub = np.unique(root, return_inverse=True)
    H = len(roots)
    hub_lat = np.bincount(node_hub, weights=nlat, minlength=H) / np.bincount(node_hub, minlength=H)
    hub_lon = np.bincount(node_hub, weights=nlon, minlength=H) / np.bincount(node_hub, minlength=H)
    hub_name = nname[roots]
    node_index = {n: k for k, n in enumerate(nodes)}
    stop_hub = {s: int(node_hub[node_index[n]]) for s, n in node_of.items()}

    # --- segments -----------------------------------------------------------
    pairs["tau"] = (pairs["arr_j"] - pairs["dep"]).clip(lower=0.0)
    key = ["route_id", "stop_id", "stop_j"]
    agg = pairs.groupby(key, sort=True).agg(
        route_type=("route_type", "first"),
        n=("trip_id", "size"),
        tau=("tau", "median"),
    ).reset_index()

    def _cv2(deps: np.ndarray) -> float:
        if len(deps) < 3:
            return 1.0
        h = np.diff(np.sort(deps))
        m = h.mean()
        return float(h.var() / m**2) if m > 0 else 1.0

    cv2 = pairs.groupby(key, sort=True)["dep"].apply(lambda s: _cv2(s.to_numpy()))
    agg["cv2"] = cv2.to_numpy()

    S = len(agg)
    lat_i = coords.loc[agg["stop_id"], "stop_lat"].to_numpy()
    lon_i = coords.loc[agg["stop_id"], "stop_lon"].to_numpy()
    lat_j = coords.loc[agg["stop_j"], "stop_lat"].to_numpy()
    lon_j = coords.loc[agg["stop_j"], "stop_lon"].to_numpy()
    dist = np.maximum(_haversine_m(lat_i, lon_i, lat_j, lon_j), 50.0)
    seg_from = np.array([stop_hub[s] for s in agg["stop_id"]], dtype=np.int64)
    seg_to = np.array([stop_hub[s] for s in agg["stop_j"]], dtype=np.int64)

    # --- continuation shares r_{ss'} -----------------------------------------
    seg_index = {k: n for n, k in enumerate(zip(agg["route_id"], agg["stop_id"], agg["stop_j"]))}
    s_idx = np.array([seg_index[k] for k in zip(pairs["route_id"], pairs["stop_id"], pairs["stop_j"])])
    has_next = pairs["stop_k"].notna().to_numpy()
    nxt = np.full(len(pairs), -1, dtype=np.int64)
    nk = list(zip(pairs["route_id"].values[has_next], pairs["stop_j"].values[has_next],
                  pairs["stop_k"].values[has_next]))
    # a continuation outside the window still counts as "stays on board":
    # map it to the segment if it exists in the window network, else to the end
    nxt[has_next] = [seg_index.get(k, -1) for k in nk]
    counts = np.bincount(s_idx, minlength=S).astype(float)
    cont = nxt >= 0
    M = coo_matrix((np.ones(cont.sum()), (s_idx[cont], nxt[cont])), shape=(S, S)).tocsr()
    M.sum_duplicates()
    succ = csr_matrix(diags(1.0 / counts) @ M)
    seg_end = 1.0 - np.asarray(succ.sum(axis=1)).ravel()
    seg_end = np.clip(seg_end, 0.0, 1.0)

    net = LineNetwork(
        hub_name=hub_name, hub_lat=hub_lat, hub_lon=hub_lon, stop_hub=stop_hub,
        seg_route=agg["route_id"].to_numpy(), seg_mode=agg["route_type"].to_numpy(int),
        seg_from=seg_from, seg_to=seg_to,
        seg_xy=np.column_stack([lat_i, lon_i, lat_j, lon_j]),
        seg_n=agg["n"].to_numpy(int), seg_freq=agg["n"].to_numpy(float) / W,
        seg_cv2=agg["cv2"].to_numpy(float),
        seg_tau=np.maximum(agg["tau"].to_numpy(float), tau_min_s),
        seg_dist=dist, succ=succ, seg_end=seg_end, window_s=W,
        meta={"window": window, "hub_radius_m": hub_radius_m, "n_trip_segments": len(pairs)},
    )
    log.info("LineNetwork: %d hubs, %d segments, %d trip-segments in window",
             H, S, len(pairs))
    return net


# ---------------------------------------------------------------------------
# Open Markov model
# ---------------------------------------------------------------------------
@dataclass
class ModelParams:
    lam:           float = 1.0 / 300.0    # cost sensitivity lambda [s^-1]
    mean_leg_m:    float = 3500.0         # D: mean in-vehicle leg length [m]
    p_transfer:    float = 0.30           # transfer probability after alighting
    headway_correction: bool = True       # theta = 2f/(1+CV^2) instead of f
    kappa:         dict | None = None     # route_type -> comfort weight


class OpenNetworkModel:
    """
    Sub-generator T on X = H ∪ S, exit-rate vector t (T 1 + t = 0) and the
    exact first-moment / Poisson occupancy solution.

    State order: hubs 0..H-1, then segments H..H+S-1.
    """

    def __init__(self, net: LineNetwork, params: ModelParams | None = None) -> None:
        self.net = net
        self.p = params or ModelParams()
        self._build()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        net, p = self.net, self.p
        H, S = net.n_hubs, net.n_segments
        n = H + S
        kappa = {rt: m["kappa"] for rt, m in ROUTE_TYPES.items()}
        if p.kappa:
            kappa.update(p.kappa)

        # boarding hazard theta_s and acceptance a_s
        theta = net.seg_freq.copy()
        if p.headway_correction:
            theta = 2.0 * theta / (1.0 + net.seg_cv2)
        kap = np.array([kappa.get(int(m), 1.0) for m in net.seg_mode])
        cost = kap * 1000.0 * net.seg_tau / net.seg_dist     # perceived s per km
        cmin = np.full(H, np.inf)
        np.minimum.at(cmin, net.seg_from, cost)
        accept = np.exp(-p.lam * (cost - cmin[net.seg_from]))
        q_board = theta * accept

        nu = 1.0 / net.seg_tau
        alpha = 1.0 - np.exp(-net.seg_dist / p.mean_leg_m)
        p_alight = alpha + (1.0 - alpha) * net.seg_end

        board_out = np.bincount(net.seg_from, weights=q_board, minlength=H)
        can_wait = board_out > 0
        # a passenger alighting where nothing departs in the window must leave
        p_tr_eff = np.where(can_wait[net.seg_to], p.p_transfer, 0.0)

        rows, cols, vals = [], [], []
        # hub -> segment (boarding)
        rows.append(net.seg_from); cols.append(H + np.arange(S)); vals.append(q_board)
        # segment -> segment (stay on board)
        sc = net.succ.tocoo()
        rows.append(H + sc.row); cols.append(H + sc.col)
        vals.append(nu[sc.row] * (1.0 - alpha[sc.row]) * sc.data)
        # segment -> hub (alight and transfer)
        rows.append(H + np.arange(S)); cols.append(net.seg_to)
        vals.append(nu * p_alight * p_tr_eff)
        off = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n)).tocsr()
        off.sum_duplicates()

        exit_rate = np.zeros(n)
        exit_rate[H:] = nu * p_alight * (1.0 - p_tr_eff)
        # hubs without departures receive no inflow (no source, p_tr_eff = 0);
        # a nominal exit keeps -T a nonsingular M-matrix without affecting L
        exit_rate[:H][~can_wait] = 1.0
        out = np.asarray(off.sum(axis=1)).ravel() + exit_rate
        self.T = (off - diags(out)).tocsr()
        self.exit = exit_rate
        self.theta, self.accept, self.q_board = theta, accept, q_board
        self.cost, self.alpha, self.p_alight = cost, alpha, p_alight
        self.hub_active = can_wait
        self.n_states = n

    # ------------------------------------------------------------------ #
    def demand_prior(self, beta: float = 1.0) -> np.ndarray:
        """Origin weights lambda_h ∝ f_h^beta on hubs with departures (sums to 1)."""
        f = self.net.hub_departures()
        w = np.where(self.hub_active, f, 0.0) ** beta
        w[~self.hub_active] = 0.0
        return w / w.sum()

    def source_vector(self, hub_rates: np.ndarray) -> np.ndarray:
        src = np.zeros(self.n_states)
        src[: self.net.n_hubs] = hub_rates
        return src

    def steady_state(self, hub_rates: np.ndarray) -> np.ndarray:
        """Mean stationary occupancy L solving L T + lambda = 0."""
        src = self.source_vector(hub_rates)
        L = spsolve((-self.T).T.tocsc(), src)
        return np.asarray(L).ravel()

    def flows(self, L: np.ndarray) -> dict[str, np.ndarray]:
        """Passenger flows [pax/s] implied by occupancy L."""
        H = self.net.n_hubs
        seg_flow = L[H:] / self.net.seg_tau                 # entries = exits of s
        boarding = np.bincount(self.net.seg_from, weights=L[self.net.seg_from] * self.q_board,
                               minlength=H)
        alight = np.bincount(self.net.seg_to, weights=seg_flow * self.p_alight, minlength=H)
        board_seg = L[self.net.seg_from] * self.q_board
        return {"segment": seg_flow, "boarding": boarding, "alighting": alight,
                "boarding_segment": board_seg}

    def scale_to_boardings(self, shape: np.ndarray, boardings_per_s: float) -> np.ndarray:
        """Return hub source rates ∝ shape whose total boarding flow equals target."""
        L1 = self.steady_state(shape)
        b1 = self.flows(L1)["boarding"].sum()
        return shape * (boardings_per_s / b1)

    def mean_journey_time(self, hub_rates: np.ndarray) -> float:
        """E[journey] = alpha (-T)^{-1} 1 with alpha = lambda / Lambda (= sum L / Lambda)."""
        L = self.steady_state(hub_rates)
        return float(L.sum() / hub_rates.sum())

    def transient_total(self, hub_rates: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        """Total mean occupancy from an empty system: N(t) = L (I - e^{T t})."""
        L = self.steady_state(hub_rates)
        TT = self.T.T.tocsr()
        out = expm_multiply(TT, L, start=0.0, stop=float(t_eval[-1]),
                            num=len(t_eval), endpoint=True)
        return L.sum() - out.sum(axis=1)

    def relaxation_time(self) -> float:
        """1 / spectral abscissa gap of T: slowest decay rate of the transient part."""
        vals = eigs(self.T.tocsc(), k=1, sigma=0.0, which="LM", return_eigenvectors=False)
        return float(1.0 / abs(vals[0].real))

    # ------------------------------------------------------------------ #
    def embedded_jump_chain(self) -> tuple[csr_matrix, np.ndarray]:
        """Row-stochastic jump matrix on X ∪ {exit} and holding rates."""
        rate = -self.T.diagonal()
        off = self.T - diags(self.T.diagonal())
        n = self.n_states
        ext = coo_matrix((self.exit[self.exit > 0], (np.flatnonzero(self.exit > 0),
                          np.full((self.exit > 0).sum(), n))), shape=(n, n + 1))
        Pfull = (diags(1.0 / rate) @ csr_matrix(
            (off.data, off.indices, off.indptr), shape=(n, n + 1)) + diags(1.0 / rate) @ ext)
        return csr_matrix(Pfull), rate

    def sample_journeys(self, origin_prob: np.ndarray, m: int, rng=None,
                        max_jumps: int = 100_000) -> dict[str, np.ndarray]:
        """
        Exact Monte-Carlo (Doob–Gillespie per independent passenger) journeys.

        Returns total journey times and the time spent in each state, so that
        Lambda * E[time in x] can be compared with the exact L_x.
        """
        rng = rng or np.random.default_rng(0)
        P, rate = self.embedded_jump_chain()
        n = self.n_states
        P.sort_indices()
        rows = np.repeat(np.arange(n), np.diff(P.indptr))
        # key = row + cumulative probability within the row: one global sorted
        # array, so a single searchsorted samples every passenger's next state
        cs = np.concatenate([[0.0], np.cumsum(P.data)])
        key = rows + (cs[1:] - cs[P.indptr[rows]])
        key[P.indptr[1:] - 1] = np.arange(n) + 1.0      # exact row ends
        H = self.net.n_hubs
        state = rng.choice(H, size=m, p=origin_prob[:H] / origin_prob[:H].sum())
        alive = np.ones(m, dtype=bool)
        total = np.zeros(m)
        occupancy = np.zeros(n)
        for _ in range(max_jumps):
            idx = np.flatnonzero(alive)
            if idx.size == 0:
                break
            s = state[idx]
            dt = rng.exponential(1.0 / rate[s])
            total[idx] += dt
            np.add.at(occupancy, s, dt)
            u = rng.random(idx.size)
            pos = np.searchsorted(key, s + u, side="left")
            nxt = P.indices[pos]
            done = nxt == n
            state[idx] = np.where(done, 0, nxt)
            alive[idx[done]] = False
        return {"journey_time": total, "occupancy_time": occupancy / m}

    # ------------------------------------------------------------------ #
    def travel_graph(self, closed_hub: int | None = None, scenario: str = "closure") -> csr_matrix:
        """
        Weighted digraph for expected shortest travel times.

        Nodes are the CTMC states.  hub -> s costs the mean wait 1/theta_s,
        s -> s' and s -> hub(j) cost the in-vehicle time tau_s.
        scenario='closure': no boarding/alighting at the hub (vehicles pass).
        scenario='failure': the hub and every segment touching it are removed.
        """
        net = self.net
        H, S = net.n_hubs, net.n_segments
        seg = np.arange(S)
        ok_seg = np.ones(S, dtype=bool)
        if closed_hub is not None and scenario == "failure":
            ok_seg = (net.seg_from != closed_hub) & (net.seg_to != closed_hub)
        board_ok = ok_seg & (self.theta > 0)
        alight_ok = ok_seg.copy()
        if closed_hub is not None:
            board_ok &= net.seg_from != closed_hub
            alight_ok &= net.seg_to != closed_hub
        sc = net.succ.tocoo()
        cont_ok = ok_seg[sc.row] & ok_seg[sc.col]
        r = np.concatenate([net.seg_from[board_ok], H + sc.row[cont_ok], H + seg[alight_ok]])
        c = np.concatenate([H + seg[board_ok], H + sc.col[cont_ok], net.seg_to[alight_ok]])
        w = np.concatenate([1.0 / self.theta[board_ok], net.seg_tau[sc.row[cont_ok]],
                            net.seg_tau[alight_ok]])
        G = coo_matrix((w, (r, c)), shape=(H + S, H + S)).tocsr()
        G.sum_duplicates()  # duplicates cannot occur, kept for safety
        return G

    def hub_travel_times(self, G: csr_matrix, origins: np.ndarray, chunk: int = 256) -> np.ndarray:
        """Shortest expected times [s] from origin hubs to all hubs."""
        H = self.net.n_hubs
        out = np.empty((len(origins), H))
        for a in range(0, len(origins), chunk):
            d = dijkstra(G, directed=True, indices=origins[a:a + chunk])
            out[a:a + chunk] = d[:, :H]
        return out

    def efficiency(self, weights: np.ndarray, closed_hub: int | None = None,
                   scenario: str = "closure", origins: np.ndarray | None = None) -> float:
        """
        Demand-weighted Latora–Marchiori efficiency
            E = sum_{o != d} w_o w_d / t_od  /  sum_{o != d} w_o w_d   [s^-1].
        OD pairs touching a closed hub count as unreachable (1/t = 0).
        """
        H = self.net.n_hubs
        origins = np.flatnonzero(weights > 0) if origins is None else origins
        G = self.travel_graph(closed_hub, scenario)
        Tm = self.hub_travel_times(G, origins)
        inv = np.zeros_like(Tm)
        finite = np.isfinite(Tm) & (Tm > 0)
        inv[finite] = 1.0 / Tm[finite]
        inv[np.arange(len(origins)), origins] = 0.0
        if closed_hub is not None:
            inv[:, closed_hub] = 0.0
            inv[origins == closed_hub, :] = 0.0
        wo = weights[origins]
        num = float(wo @ inv @ weights)
        den = float(wo.sum() * weights.sum() - (wo * weights[origins]).sum())
        return num / den
