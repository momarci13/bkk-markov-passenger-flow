"""
scripts/tdk_common.py
=====================
Shared set-up for the scenario scripts: baseline model, street router with the
tram-bridge rule, district labels, and the travel-time risk / accessibility
measures used in the paper.
"""
from __future__ import annotations

import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from shapely.geometry import Point
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from budapest_basemap import BUDA, CSEPEL, eov, load_districts  # noqa: E402

from bkk.linemodel import ModelParams, OpenNetworkModel  # noqa: E402
from bkk.streets import Router, build_street_graph, unproject  # noqa: E402

N_DAY, PHI = 3.4e6, 0.20
BASE_PARAMS = ModelParams(lam=1 / 300, mean_leg_m=3500.0, p_transfer=0.30)


def hub(net, name: str) -> int:
    """Hub with this name; among same-name hubs the one with most departures."""
    k = np.flatnonzero(net.hub_name == name)
    if len(k) == 0:
        raise KeyError(name)
    return int(k[np.argmax(net.hub_departures()[k])])


def load_base():
    net = pickle.load(open("data/cache/net_20260609.pkl", "rb"))
    model = OpenNetworkModel(net, BASE_PARAMS)
    w = model.demand_prior()
    lam = model.scale_to_boardings(w, N_DAY * PHI / 7200.0)
    origins = np.arange(net.n_hubs)
    T0 = model.hub_travel_times(model.travel_graph(), origins)
    return net, model, w, lam, origins, T0


def geography(net):
    """District polygons, Danube geometry and the district of every hub."""
    districts = load_districts()
    buda = unary_union([g for n, g in districts.items() if n in BUDA])
    rest = unary_union([g for n, g in districts.items() if n not in BUDA])
    csepel = unary_union([g for n, g in districts.items() if n in CSEPEL])
    river = buda.boundary.intersection(rest.buffer(60))
    arm = csepel.boundary.intersection(unary_union(
        [g for n, g in districts.items() if n not in BUDA | CSEPEL]).buffer(60))
    water = unary_union([river, arm])
    hx, hy = eov(net.hub_lon, net.hub_lat)
    label = np.array([next((n for n, g in districts.items() if g.contains(Point(x, y))), "")
                      for x, y in zip(hx, hy)], dtype=object)
    return districts, water, label


def street_router(net, water):
    """Main-road router; trams cross the Danube only on today's tram bridges."""
    cache = Path("data/cache/streets.pkl")
    if cache.exists():
        G = pickle.load(open(cache, "rb"))
    else:
        with zipfile.ZipFile("data/budapest_gtfs.zip") as z:
            sh = pd.read_csv(z.open("shapes.txt"), dtype={"shape_id": str})
            tr = pd.read_csv(z.open("trips.txt"), dtype=str, usecols=["route_id", "shape_id"])
            ro = pd.read_csv(z.open("routes.txt"), dtype=str, usecols=["route_id", "route_type"])
        G = build_street_graph(sh, tr.drop_duplicates().merge(ro, on="route_id"))
        pickle.dump(G, open(cache, "wb"))
    main = G.main_roads(min_bus_routes=3)
    A = main.tocoo()
    up = A.row < A.col
    r_, c_ = A.row[up], A.col[up]
    la1, lo1 = unproject(G.xy[r_, 0], G.xy[r_, 1])
    la2, lo2 = unproject(G.xy[c_, 0], G.xy[c_, 1])
    x1, y1 = eov(lo1, la1)
    x2, y2 = eov(lo2, la2)
    segs = shapely.linestrings(np.stack([np.column_stack([x1, y1]),
                                         np.column_stack([x2, y2])], axis=1))
    drop = shapely.intersects(segs, water) & ~(np.asarray(G.tram[r_, c_]).ravel() > 0.5)
    keep = ~drop
    rr, cc, vv = r_[keep], c_[keep], A.data[up][keep]
    main = csr_matrix((np.r_[vv, vv], (np.r_[rr, cc], np.r_[cc, rr])), shape=main.shape)
    return G, main, Router.build(G, main, net.hub_lat, net.hub_lon, max_snap_m=120.0), int(drop.sum())


def es_travel_time(T: np.ndarray, origins: np.ndarray, w: np.ndarray, alpha: float = 0.9) -> float:
    """
    Demand-weighted Expected Shortfall of shortest expected travel times:
    the mean of t_od over the worst (1 - alpha) share of OD demand w_o w_d
    (Acerbi & Tasche 2002), reachable pairs only, in minutes.
    """
    W = np.outer(w[origins], w)
    W[np.arange(len(origins)), origins] = 0.0
    ok = np.isfinite(T) & (W > 0)
    t, q = T[ok], W[ok]
    order = np.argsort(t)[::-1]
    t, q = t[order], q[order] / q.sum()
    cum = np.cumsum(q)
    tail = 1.0 - alpha
    k = np.searchsorted(cum, tail)
    take = q[:k].sum()
    es = (np.dot(t[:k], q[:k]) + (tail - take) * t[k]) / tail
    return float(es / 60.0)


def accessibility(T: np.ndarray, origins: np.ndarray, w: np.ndarray) -> np.ndarray:
    """A_o = sum_d w_d / t_od  [s^-1] for every origin hub."""
    with np.errstate(divide="ignore"):
        inv = np.where(np.isfinite(T) & (T > 0), 1.0 / T, 0.0)
    inv[np.arange(len(origins)), origins] = 0.0
    return inv @ w


def district_change(A0, A1, w, label, names) -> dict:
    """Demand-weighted relative accessibility change per district."""
    out = {}
    for n in names:
        k = label == n
        if w[k].sum() > 0:
            out[n] = float((w[k] @ A1[k]) / (w[k] @ A0[k]) - 1.0)
    return out
