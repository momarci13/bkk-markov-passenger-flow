"""
bkk.scenario
============
New-line scenarios for the line-aware open Markov model (bkk.linemodel).

* :func:`add_lines` appends new lines (both directions, optional through-running
  into existing lines) to a :class:`LineNetwork`, so the full CTMC predicts
  the boardings and section loads of the new line.
* :class:`MinPlusEvaluator` screens many candidate lines exactly.  For a line
  with stations a_1..a_k and in-line costs c_ab (mean wait at a plus riding
  time a -> b), the new shortest expected times are

      t'_od = min( t_od , min_{a,b} t_oa + c_ab + t_bd ),

  which is exact for paths that use the new line once.  The demand-weighted
  efficiency gain of a set of lines is then a facility-location type set
  function, so the greedy choice of lines is the natural heuristic
  (Nemhauser, Wolsey & Fisher 1978).
* :func:`generate_candidates` builds corridor lines between pairs of major hubs
  with mode-specific speed, headway, station spacing and length limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from .linemodel import LineNetwork
from .network import _haversine_m


# ---------------------------------------------------------------------------
# Line definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModeSpec:
    """Design parameters of a new line of one mode."""
    mode:          int            # GTFS route_type (1 metro, 0 tram)
    speed_kmh:     float          # commercial speed incl. stops
    headway_s:     float          # peak headway (regular: CV^2 = 0)
    spacing_m:     float          # minimum station spacing
    corridor_m:    float          # half-width of the station search corridor
    min_len_m:     float
    max_len_m:     float
    detour:        float          # route length / straight-line length
    cost_per_km:   float          # relative construction cost index
    may_cross_danube: bool = True


METRO = ModeSpec(mode=1, speed_kmh=33.0, headway_s=180.0, spacing_m=800.0, corridor_m=450.0,
                 min_len_m=4000.0, max_len_m=14000.0, detour=1.10, cost_per_km=10.0)
TRAM = ModeSpec(mode=0, speed_kmh=18.0, headway_s=300.0, spacing_m=350.0, corridor_m=250.0,
                min_len_m=2000.0, max_len_m=10000.0, detour=1.20, cost_per_km=1.0,
                may_cross_danube=False)   # street-routed trams use path lengths instead


@dataclass
class NewLine:
    """A new two-way line through existing hubs (in order)."""
    name:      str
    spec:      ModeSpec
    hubs:      list[int]
    headway_s: float | None = None
    # through-running (existing segment indices):
    feed_forward:  list[int] = field(default_factory=list)  # continue into forward first seg
    exit_forward:  list[int] = field(default_factory=list)  # forward last seg continues here
    feed_reverse:  list[int] = field(default_factory=list)
    exit_reverse:  list[int] = field(default_factory=list)
    one_way:   bool = False
    # street-routed lines: segment lengths along the street path [m] and the
    # path itself (lat, lon) for maps; None -> straight segments x detour
    seg_len_m: list[float] | None = None
    path_latlon: np.ndarray | None = None

    @property
    def headway(self) -> float:
        return self.headway_s if self.headway_s is not None else self.spec.headway_s


def _pair_dist(net: LineNetwork, a: int, b: int) -> float:
    return float(_haversine_m(net.hub_lat[a], net.hub_lon[a], net.hub_lat[b], net.hub_lon[b]))


def segment_lengths(net: LineNetwork, line: NewLine) -> np.ndarray:
    """Route length [m] of each inter-station segment."""
    if line.seg_len_m is not None:
        return np.asarray(line.seg_len_m, dtype=float)
    return np.array([line.spec.detour * _pair_dist(net, a, b)
                     for a, b in zip(line.hubs, line.hubs[1:])])


def line_length_m(net: LineNetwork, line: NewLine) -> float:
    return float(segment_lengths(net, line).sum())


def line_cost(net: LineNetwork, line: NewLine) -> float:
    return line.spec.cost_per_km * line_length_m(net, line) / 1000.0


def segment_times(net: LineNetwork, line: NewLine) -> np.ndarray:
    """In-vehicle times [s] between consecutive stations."""
    v = line.spec.speed_kmh / 3.6
    return np.maximum(20.0, segment_lengths(net, line) / v)


def find_segment(net: LineNetwork, route_id: str, from_hub: int, to_hub: int) -> int:
    k = np.flatnonzero((net.seg_route == route_id) & (net.seg_from == from_hub)
                       & (net.seg_to == to_hub))
    if len(k) != 1:
        raise KeyError(f"segment {route_id}: {from_hub}->{to_hub} not unique ({len(k)})")
    return int(k[0])


# ---------------------------------------------------------------------------
# Network augmentation
# ---------------------------------------------------------------------------
def add_lines(net: LineNetwork, lines: Sequence[NewLine]) -> LineNetwork:
    """Return a new LineNetwork with the lines appended (stations at existing hubs)."""
    S0 = net.n_segments
    rid, mode, frm, to, xy, n, freq, cv2, tau, dist = ([] for _ in range(10))
    chains = []          # (direction segments, feed list, exit list)
    for line in lines:
        dirs = [list(line.hubs)] if line.one_way else [list(line.hubs), list(line.hubs[::-1])]
        feeds = [line.feed_forward, line.feed_reverse]
        exits = [line.exit_forward, line.exit_reverse]
        n_dep = net.window_s / line.headway
        for d, hubs in enumerate(dirs):
            idx = []
            lens = None if line.seg_len_m is None else (
                list(line.seg_len_m) if d == 0 else list(line.seg_len_m)[::-1])
            sub = replace(line, hubs=hubs, seg_len_m=lens)
            for (a, b), t, ln in zip(zip(hubs, hubs[1:]), segment_times(net, sub),
                                     segment_lengths(net, sub)):
                idx.append(S0 + len(rid))
                rid.append(f"NEW:{line.name}")
                mode.append(line.spec.mode)
                frm.append(a); to.append(b)
                xy.append([net.hub_lat[a], net.hub_lon[a], net.hub_lat[b], net.hub_lon[b]])
                n.append(int(round(n_dep))); freq.append(1.0 / line.headway); cv2.append(0.0)
                tau.append(t); dist.append(float(ln))
            chains.append((idx, feeds[d], exits[d]))
    S = S0 + len(rid)
    succ = lil_matrix((S, S))
    old = net.succ.tocoo()
    for r, c, v in zip(old.row, old.col, old.data):
        succ[r, c] = v
    for idx, feed, ex in chains:
        for s, s_next in zip(idx, idx[1:]):
            succ[s, s_next] = 1.0
        for f in feed:                      # on-board passengers now continue into the line
            succ[f, :] = 0.0
            succ[f, idx[0]] = 1.0
        for e in ex:
            succ[idx[-1], e] = 1.0 / len(ex)
    succ = csr_matrix(succ)
    seg_end = np.clip(1.0 - np.asarray(succ.sum(axis=1)).ravel(), 0.0, 1.0)

    cat = lambda a, b, dt=None: np.concatenate([a, np.asarray(b, dtype=dt or a.dtype)])
    return LineNetwork(
        hub_name=net.hub_name, hub_lat=net.hub_lat, hub_lon=net.hub_lon, stop_hub=net.stop_hub,
        seg_route=cat(net.seg_route.astype(object), rid, object),
        seg_mode=cat(net.seg_mode, mode), seg_from=cat(net.seg_from, frm),
        seg_to=cat(net.seg_to, to),
        seg_xy=np.vstack([net.seg_xy, np.asarray(xy).reshape(-1, 4)]),
        seg_n=cat(net.seg_n, n), seg_freq=cat(net.seg_freq, freq),
        seg_cv2=cat(net.seg_cv2, cv2), seg_tau=cat(net.seg_tau, tau),
        seg_dist=cat(net.seg_dist, dist), succ=succ, seg_end=seg_end,
        window_s=net.window_s, meta={**net.meta, "added_lines": [l.name for l in lines]},
    )


# ---------------------------------------------------------------------------
# Exact min-plus screening
# ---------------------------------------------------------------------------
class MinPlusEvaluator:
    """
    Demand-weighted efficiency of the hub travel-time matrix and its exact
    update when one new line (without through-running) is added.

    T  : (len(origins), H) shortest expected times from origin hubs; every
         hub that may become a station needs a row (pass all hubs as
         origins -- zero-weight hubs do not affect the efficiency)
    w  : (H,) demand weights; OD weight w_o w_d
    """

    def __init__(self, T: np.ndarray, origins: np.ndarray, weights: np.ndarray) -> None:
        self.T = T.copy()
        self.origins = np.asarray(origins)
        self.w = np.asarray(weights, dtype=float)
        self.row = {int(h): k for k, h in enumerate(self.origins)}
        wo = self.w[self.origins]
        self.den = float(wo.sum() * self.w.sum() - (wo * self.w[self.origins]).sum())
        self._diag = (np.arange(len(self.origins)), self.origins)

    def efficiency(self, T: np.ndarray | None = None) -> float:
        T = self.T if T is None else T
        with np.errstate(divide="ignore"):
            inv = np.where(np.isfinite(T) & (T > 0), 1.0 / T, 0.0)
        inv[self._diag] = 0.0
        return float(self.w[self.origins] @ inv @ self.w) / self.den

    def in_line_costs(self, net: LineNetwork, line: NewLine) -> np.ndarray:
        """c_ab = mean wait at a (1/theta, theta = 2/headway) + riding time a -> b."""
        k = len(line.hubs)
        t = segment_times(net, line)
        cum = np.concatenate([[0.0], np.cumsum(t)])
        ride = np.abs(cum[:, None] - cum[None, :])
        if line.one_way:
            ride = np.where(np.arange(k)[:, None] < np.arange(k)[None, :], ride, np.inf)
        C = line.headway / 2.0 + ride
        C[np.arange(k), np.arange(k)] = np.inf
        return C

    def updated(self, net: LineNetwork, line: NewLine, T: np.ndarray | None = None) -> np.ndarray:
        T = self.T if T is None else T
        st = np.asarray(line.hubs)
        rows = np.array([self.row[int(h)] for h in st])
        C = self.in_line_costs(net, line)
        A = T[:, st]                                   # o -> a
        M = np.full_like(A, np.inf)                    # o -> (board a, ride) -> b
        for a in range(len(st)):
            np.minimum(M, A[:, [a]] + C[a][None, :], out=M)
        out = T.copy()
        for b in range(len(st)):
            np.minimum(out, M[:, [b]] + T[rows[b]][None, :], out=out)
            out[:, st[b]] = np.minimum(out[:, st[b]], M[:, b])
        return out

    def gain(self, net: LineNetwork, line: NewLine, base_eff: float | None = None) -> float:
        e0 = self.efficiency() if base_eff is None else base_eff
        return (self.efficiency(self.updated(net, line)) - e0) / e0

    def add(self, net: LineNetwork, line: NewLine) -> None:
        self.T = self.updated(net, line)


# ---------------------------------------------------------------------------
# Candidate generation and greedy selection
# ---------------------------------------------------------------------------
def _local_xy(net: LineNetwork) -> tuple[np.ndarray, np.ndarray]:
    lat0 = np.radians(np.mean(net.hub_lat))
    x = np.radians(net.hub_lon) * 6_371_000.0 * np.cos(lat0)
    y = np.radians(net.hub_lat) * 6_371_000.0
    return x, y


def generate_candidates(
    net: LineNetwork,
    weights: np.ndarray,
    spec: ModeSpec,
    endpoints: Sequence[int],
    station_pool: np.ndarray,
    crosses_danube: Callable[[float, float, float, float], bool] | None = None,
) -> list[NewLine]:
    """
    Straight corridor lines between endpoint pairs.  Stations are hubs of
    ``station_pool`` within ``corridor_m`` of the corridor axis, accepted in
    order of demand weight subject to ``spacing_m``, then ordered along the axis.
    """
    x, y = _local_xy(net)
    pool = np.asarray(station_pool)
    out = []
    ends = list(endpoints)
    for i, a in enumerate(ends):
        for b in ends[i + 1:]:
            ax, ay, bx, by = x[a], y[a], x[b], y[b]
            L = np.hypot(bx - ax, by - ay)
            if not (spec.min_len_m <= spec.detour * L <= spec.max_len_m):
                continue
            if not spec.may_cross_danube and crosses_danube is not None \
                    and crosses_danube(net.hub_lon[a], net.hub_lat[a], net.hub_lon[b], net.hub_lat[b]):
                continue
            ux, uy = (bx - ax) / L, (by - ay) / L
            px, py = x[pool] - ax, y[pool] - ay
            along = px * ux + py * uy
            across = np.abs(-px * uy + py * ux)
            inside = (along > 0) & (along < L) & (across <= spec.corridor_m)
            cand = pool[inside]
            cand = cand[np.argsort(-weights[cand])]
            chosen = [a, b]
            for h in cand:
                if all(np.hypot(x[h] - x[c], y[h] - y[c]) >= spec.spacing_m for c in chosen):
                    chosen.append(int(h))
            order = sorted(chosen, key=lambda h: (x[h] - ax) * ux + (y[h] - ay) * uy)
            mode_name = {1: "M", 0: "V"}.get(spec.mode, str(spec.mode))
            out.append(NewLine(name=f"{mode_name}:{net.hub_name[a]}–{net.hub_name[b]}",
                               spec=spec, hubs=order))
    return out


def first_round_gains(evaluator: MinPlusEvaluator, net: LineNetwork,
                      candidates: Sequence[NewLine]) -> np.ndarray:
    """Relative efficiency gain of each candidate added alone."""
    e0 = evaluator.efficiency()
    return np.array([evaluator.gain(net, c, e0) for c in candidates])


def generate_street_candidates(
    net: LineNetwork,
    weights: np.ndarray,
    spec: ModeSpec,
    endpoints: Sequence[int],
    station_pool: np.ndarray,
    router,
    station_radius_m: float = 150.0,
    max_detour: float = 1.6,
) -> list[NewLine]:
    """
    Surface lines routed on a street (sub)graph (``bkk.streets.Router``).

    Each endpoint pair is joined by the shortest street path; candidates whose
    path is longer than ``max_detour`` times the straight distance or outside
    the length limits are dropped.  Stations are pool hubs within
    ``station_radius_m`` of the path, accepted by demand weight subject to the
    spacing measured *along the path*; segment lengths are path distances.
    """
    from scipy.spatial import cKDTree
    from .streets import project, unproject

    G = router.graph
    pool = np.asarray(station_pool)
    px_h, py_h = project(net.hub_lat, net.hub_lon)
    tree_pool = cKDTree(np.column_stack([px_h[pool], py_h[pool]]))
    ends = [int(e) for e in endpoints if router.hub_node[e] >= 0]
    out = []
    mode_name = {1: "M", 0: "V"}.get(spec.mode, str(spec.mode))
    for i, a in enumerate(ends):
        dist, pred = router.paths_from(router.hub_node[a])
        for b in ends[i + 1:]:
            nb = router.hub_node[b]
            L = dist[nb]
            if not np.isfinite(L) or not (spec.min_len_m <= L <= spec.max_len_m):
                continue
            straight = np.hypot(px_h[a] - px_h[b], py_h[a] - py_h[b])
            if L > max_detour * max(straight, 1.0):
                continue
            path = router.extract(pred, router.hub_node[a], nb)
            if len(path) < 2:
                continue
            P = G.xy[path]
            cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
            near = set()
            for lst in tree_pool.query_ball_point(P, station_radius_m):
                near.update(lst)
            cand = pool[sorted(near)]
            if len(cand):
                ptree = cKDTree(P)
                _, pos = ptree.query(np.column_stack([px_h[cand], py_h[cand]]))
                along = dict(zip(cand.tolist(), cum[pos].tolist()))
            else:
                along = {}
            along[a], along[b] = 0.0, float(cum[-1])
            chosen = [a, b]
            for h in sorted(along, key=lambda h: -weights[h]):
                if h in (a, b):
                    continue
                if all(abs(along[h] - along[c]) >= spec.spacing_m for c in chosen):
                    chosen.append(int(h))
            order = sorted(chosen, key=lambda h: along[h])
            pos_m = np.array([along[h] for h in order])
            lat, lon = unproject(P[:, 0], P[:, 1])
            out.append(NewLine(name=f"{mode_name}:{net.hub_name[a]}–{net.hub_name[b]}",
                               spec=spec, hubs=order,
                               seg_len_m=np.maximum(np.diff(pos_m), 1.0).tolist(),
                               path_latlon=np.column_stack([lat, lon])))
    return out


def greedy_select(
    evaluator: MinPlusEvaluator,
    net: LineNetwork,
    candidates: Sequence[NewLine],
    k: int,
    per_cost: bool = True,
    initial_gains: np.ndarray | None = None,
    key: Callable[[NewLine], object] | None = None,
) -> list[dict]:
    """
    Lazy greedy: repeatedly add the candidate with the largest marginal
    efficiency gain (per unit cost if ``per_cost``).  Marginal gains are
    re-evaluated lazily, which is exact when gains only shrink as lines are
    added (diminishing returns).  ``initial_gains`` (gains on the empty
    package, relative to E0) can be passed to reuse a first-round evaluation.
    ``key`` makes candidates exclusive: once a line is picked, candidates
    with the same key (e.g. the same corridor) are skipped.
    """
    import heapq

    costs = np.array([line_cost(net, c) for c in candidates])
    e0 = evaluator.efficiency()
    if initial_gains is None:
        initial_gains = first_round_gains(evaluator, net, candidates)
    heap = []
    for i, c in enumerate(candidates):
        g = initial_gains[i]
        prio = g / costs[i] if per_cost else g
        heap.append((-prio, i, 0))
    heapq.heapify(heap)
    picked, rnd, used = [], 0, set()
    while heap and len(picked) < k:
        _, i, stamp = heapq.heappop(heap)
        if key is not None and key(candidates[i]) in used:
            continue
        if stamp == rnd:
            if key is not None:
                used.add(key(candidates[i]))
            e_before = evaluator.efficiency()
            evaluator.add(net, candidates[i])
            e_after = evaluator.efficiency()
            picked.append({"index": i, "line": candidates[i], "cost": float(costs[i]),
                           "marginal_gain": (e_after - e_before) / e0,
                           "cumulative_gain": (e_after - e0) / e0})
            rnd += 1
            continue
        g = evaluator.gain(net, candidates[i])
        g = g * evaluator.efficiency() / e0          # relative to the original E0
        prio = g / costs[i] if per_cost else g
        heapq.heappush(heap, (-prio, i, rnd))
    return picked
