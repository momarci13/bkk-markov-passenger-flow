"""
scripts/tdk_common.py
=====================
Shared set-up for the scenario scripts: baseline model, street router with the
tram-bridge rule, district labels, and the travel-time risk / accessibility
measures used in the paper.
"""
from __future__ import annotations

import os
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

from bkk.demand_data import gravity_production  # noqa: E402
from bkk.linemodel import ModelParams, OpenNetworkModel  # noqa: E402
from bkk.scenario import node_mass, od_rows  # noqa: E402
from bkk.streets import Router, build_street_graph, unproject  # noqa: E402

N_DAY, PHI = 3.4e6, 0.20
BASE_PARAMS = ModelParams(lam=1 / 300, mean_leg_m=3500.0, p_transfer=0.30)

# Demand variants:
#   "population"          HRSL residents assigned by the calibrated logit access
#                         choice (tdk_demand_data.py); OD W_od = lam_o a_d / sum a
#                         (production-constrained, no deterrence: beta = 0)
#   "population_gravity"  the same with the calibrated deterrence beta
#   "departures"          lambda_h ∝ f_h, OD weight w_o w_d
# Results of the non-default variants go to a subfolder.
DEMAND = os.environ.get("TDK_DEMAND", "population")
RES = Path("data/results") if DEMAND == "population" else Path("data/results") / DEMAND
RES.mkdir(parents=True, exist_ok=True)


def hub(net, name: str) -> int:
    """Hub with this name; among same-name hubs the one with most departures."""
    k = np.flatnonzero(net.hub_name == name)
    if len(k) == 0:
        raise KeyError(name)
    return int(k[np.argmax(net.hub_departures()[k])])


def load_net(zip_path: str = "data/budapest_gtfs.zip", date: str = "20260609"):
    """Line network of the pinned service day (built and cached on first use)."""
    cache = Path(f"data/cache/net_{date}.pkl")
    if not cache.exists():
        from bkk import GTFSLoader
        from bkk.linemodel import build_line_network
        net = build_line_network(GTFSLoader().load(zip_path).parse(date_filter=date))
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(net, open(cache, "wb"))
    return pickle.load(open(cache, "rb"))


def _beta(z) -> float:
    return float(z["beta"]) if DEMAND == "population_gravity" else 0.0


def demand(model, T0, variant: str | None = None):
    """
    Source rates lam [journeys/s] and demand weights w for a demand variant
    (default: TDK_DEMAND).  w is a vector (OD weight w_o w_d) for "departures"
    and an (H, H) OD matrix W [journeys/s] with row sums lam otherwise.
    """
    variant = variant or DEMAND
    if variant == "departures":
        w = model.demand_prior()
        return model.scale_to_boardings(w, N_DAY * PHI / 7200.0), w
    z = np.load("data/cache/demand_population.npz")
    lam = z["rates"]
    beta = float(z["beta"]) if variant == "population_gravity" else 0.0
    return lam, gravity_production(lam, z["attraction"], T0, beta)


def od_matrix(lam, model, T0):
    """OD weights for modified source rates (same attraction and beta)."""
    if DEMAND == "departures":
        return lam / lam.sum()
    z = np.load("data/cache/demand_population.npz")
    return gravity_production(lam, z["attraction"], T0, _beta(z))


def load_base():
    net = load_net()
    model = OpenNetworkModel(net, BASE_PARAMS)
    origins = np.arange(net.n_hubs)
    T0 = model.hub_travel_times(model.travel_graph(), origins)
    lam, w = demand(model, T0)
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
    the mean of t_od over the worst (1 - alpha) share of OD demand W_od
    (W_od = w_o w_d for a weight vector; Acerbi & Tasche 2002), reachable
    pairs only, in minutes.
    """
    W = od_rows(w, origins)
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
    """
    A_o = sum_d W_od / t_od / sum_d W_od  [s^-1]  for every origin hub
    (for a weight vector: sum_d w_d / t_od / sum_d w_d).
    """
    with np.errstate(divide="ignore"):
        inv = np.where(np.isfinite(T) & (T > 0), 1.0 / T, 0.0)
    W = od_rows(w, origins)
    rs = W.sum(axis=1)
    return np.where(rs > 0, (W * inv).sum(axis=1) / np.where(rs > 0, rs, 1.0), 0.0)


def origin_mass(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    return w if w.ndim == 1 else w.sum(axis=1)


def district_change(A0, A1, w, label, names) -> dict:
    """Origin-mass-weighted relative accessibility change per district."""
    m = origin_mass(w)
    out = {}
    for n in names:
        k = label == n
        if m[k].sum() > 0:
            out[n] = float((m[k] @ A1[k]) / (m[k] @ A0[k]) - 1.0)
    return out
