"""
bkk.streets
===========
Street graph for routing new surface lines, built from GTFS ``shapes.txt``.

Every shape (the drawn path of a bus, trolleybus or tram route) is densified
and snapped to a square grid; grid cells are nodes and consecutive cells along
a shape are edges.  Both directions of a street and parallel routes on the
same street fall into the same cells, so the graph is the network of streets
that already carry large vehicles.  Each edge records which modes and how
many distinct bus routes use it, which gives a road-quality proxy:

    main road  <=>  tram track  or  trolleybus  or  >= 3 bus routes

New tram lines are routed on the main-road subgraph by shortest path, so a
line from Rákóczi tér to Deák Ferenc tér follows the Nagykörút and Rákóczi út
(Blaha Lujza tér, Uránia, Astoria) instead of a straight line through blocks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

LAT0 = 47.5          # reference latitude of the local metric projection
R_EARTH = 6_371_000.0


def project(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Local equirectangular projection to metres (accurate to <0.1 % in Budapest)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = np.radians(lon) * R_EARTH * np.cos(np.radians(LAT0))
    y = np.radians(lat) * R_EARTH
    return x, y


def unproject(x, y) -> tuple[np.ndarray, np.ndarray]:
    lat = np.degrees(np.asarray(y) / R_EARTH)
    lon = np.degrees(np.asarray(x) / (R_EARTH * np.cos(np.radians(LAT0))))
    return lat, lon


@dataclass
class StreetGraph:
    xy:        np.ndarray        # (V, 2) node coordinates [m]
    adj:       csr_matrix        # (V, V) symmetric, edge length [m]
    n_bus:     csr_matrix        # (V, V) number of distinct bus routes on the edge
    tram:      csr_matrix        # (V, V) 1 if a tram uses the edge
    trolley:   csr_matrix        # (V, V) 1 if a trolleybus uses the edge
    cell_m:    float

    @property
    def n_nodes(self) -> int:
        return len(self.xy)

    def main_roads(self, min_bus_routes: int = 3) -> csr_matrix:
        """Subgraph of edges on tram track, trolleybus streets or busy bus streets."""
        A = self.adj.tocoo()
        nb = np.asarray(self.n_bus[A.row, A.col]).ravel()
        tr = np.asarray(self.tram[A.row, A.col]).ravel()
        tb = np.asarray(self.trolley[A.row, A.col]).ravel()
        keep = (tr > 0.5) | (tb > 0.5) | (nb >= min_bus_routes - 1e-6)
        return csr_matrix((A.data[keep], (A.row[keep], A.col[keep])), shape=A.shape)


def build_street_graph(shapes: pd.DataFrame, shape_routes: pd.DataFrame,
                       cell_m: float = 40.0, step_m: float = 15.0) -> StreetGraph:
    """
    Parameters
    ----------
    shapes : GTFS shapes.txt (shape_id, shape_pt_sequence, shape_pt_lat, shape_pt_lon)
    shape_routes : (shape_id, route_id, route_type) pairs from trips/routes
    cell_m : grid size; 40 m merges the two directions of an avenue
    step_m : densification step (< cell_m / 2 keeps consecutive cells adjacent)
    """
    sh = shapes.sort_values(["shape_id", "shape_pt_sequence"])
    x, y = project(sh["shape_pt_lat"].to_numpy(), sh["shape_pt_lon"].to_numpy())
    sid = sh["shape_id"].to_numpy()
    info = shape_routes.drop_duplicates(["shape_id", "route_id"])
    cells_all, edges = {}, []
    edge_bus: dict[tuple[int, int], set] = {}
    edge_tram: set = set()
    edge_trolley: set = set()

    def cell_id(cx, cy):
        key = (cx, cy)
        if key not in cells_all:
            cells_all[key] = len(cells_all)
        return cells_all[key]

    routes_of = {k: g for k, g in info.groupby("shape_id")}
    bounds = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        s = sid[a]
        if s not in routes_of or b - a < 2:
            continue
        px, py = x[a:b], y[a:b]
        seg = np.hypot(np.diff(px), np.diff(py))
        cum = np.r_[0.0, np.cumsum(seg)]
        t = np.arange(0.0, cum[-1] + step_m, step_m)
        qx, qy = np.interp(t, cum, px), np.interp(t, cum, py)
        cx = np.floor(qx / cell_m).astype(int)
        cy = np.floor(qy / cell_m).astype(int)
        ids = [cell_id(i, j) for i, j in zip(cx, cy)]
        g = routes_of[s]
        rtypes = set(g["route_type"].astype(int))
        bus_routes = set(g.loc[g["route_type"].astype(int) == 3, "route_id"])
        for u, v in zip(ids[:-1], ids[1:]):
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            edges.append(e)
            if bus_routes:
                edge_bus.setdefault(e, set()).update(bus_routes)
            if 0 in rtypes:
                edge_tram.add(e)
            if 11 in rtypes or 800 in rtypes:
                edge_trolley.add(e)

    keys = np.array(list(cells_all.keys()), dtype=float)
    order = np.array(list(cells_all.values()))
    xy = np.empty((len(cells_all), 2))
    xy[order] = (keys + 0.5) * cell_m
    E = np.array(sorted(set(edges)), dtype=int)
    V = len(xy)
    length = np.hypot(*(xy[E[:, 0]] - xy[E[:, 1]]).T)

    def sym(vals):
        r = np.r_[E[:, 0], E[:, 1]]
        c = np.r_[E[:, 1], E[:, 0]]
        return csr_matrix((np.r_[vals, vals], (r, c)), shape=(V, V))

    nb = np.array([len(edge_bus.get(tuple(e), ())) for e in E], dtype=float)
    tr = np.array([1.0 if tuple(e) in edge_tram else 0.0 for e in E])
    tb = np.array([1.0 if tuple(e) in edge_trolley else 0.0 for e in E])
    # a tiny offset keeps zero-valued attributes as explicit sparse entries
    return StreetGraph(xy=xy, adj=sym(length), n_bus=sym(nb + 1e-9), tram=sym(tr + 1e-9),
                       trolley=sym(tb + 1e-9), cell_m=cell_m)


@dataclass
class Router:
    """Shortest paths on a (sub)graph with hubs snapped to its largest component."""
    graph:   StreetGraph
    sub:     csr_matrix
    hub_node: np.ndarray          # (H,) snapped node or -1
    hub_snap_m: np.ndarray        # (H,) snapping distance

    @classmethod
    def build(cls, graph: StreetGraph, sub: csr_matrix, hub_lat, hub_lon,
              max_snap_m: float = 120.0) -> "Router":
        n_comp, lab = connected_components(sub, directed=False)
        deg = np.diff(sub.indptr)
        sizes = np.bincount(lab[deg > 0], minlength=n_comp)
        giant = int(np.argmax(sizes))
        nodes = np.flatnonzero((lab == giant) & (deg > 0))
        tree = cKDTree(graph.xy[nodes])
        hx, hy = project(hub_lat, hub_lon)
        d, k = tree.query(np.column_stack([hx, hy]))
        hub_node = np.where(d <= max_snap_m, nodes[k], -1)
        return cls(graph=graph, sub=sub, hub_node=hub_node, hub_snap_m=d)

    def paths_from(self, src_node: int) -> tuple[np.ndarray, np.ndarray]:
        dist, pred = dijkstra(self.sub, directed=False, indices=src_node,
                              return_predecessors=True)
        return dist, pred

    @staticmethod
    def extract(pred: np.ndarray, src: int, dst: int) -> list[int]:
        path = [dst]
        while path[-1] != src:
            p = pred[path[-1]]
            if p < 0:
                return []
            path.append(int(p))
        return path[::-1]
