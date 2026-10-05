#!/usr/bin/env python3
"""
scripts/tdk_scenarios.py
========================
Network-extension scenarios for the TDK paper: which new metro / tram lines
raise the demand-weighted efficiency of the Budapest network most per unit
construction cost, and how the planned M5 (north-south regional rapid
railway) compares.

    python scripts/tdk_scenarios.py          (after tdk_analysis.py)

Outputs: data/results/scenarios.json, data/results/new_lines.csv
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from budapest_basemap import BUDA, CSEPEL, eov, load_districts  # noqa: E402

from bkk.linemodel import ModelParams, OpenNetworkModel  # noqa: E402
from bkk.scenario import (  # noqa: E402
    METRO, TRAM, MinPlusEvaluator, NewLine, add_lines, find_segment,
    first_round_gains, generate_candidates, generate_street_candidates, greedy_select,
    line_cost, line_length_m,
)
from bkk.streets import Router, build_street_graph  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("scen")
N_DAY, PHI = 3.4e6, 0.20
K_LINES = 5
N_END = int(__import__("os").environ.get("SCEN_NEND", 60))   # override for smoke tests
RES = Path(__import__("os").environ.get("SCEN_OUT", "data/results"))


def hub(net, name: str) -> int:
    """Hub with this name; among same-name hubs the one with most departures."""
    k = np.flatnonzero(net.hub_name == name)
    if len(k) == 0:
        raise KeyError(name)
    return int(k[np.argmax(net.hub_departures()[k])])


def line_predictions(model: OpenNetworkModel, lam: np.ndarray, name: str) -> dict:
    """Peak-hour boardings and maximum section load of a new line (pax/h)."""
    net = model.net
    L = model.steady_state(lam)
    F = model.flows(L)
    k = net.seg_route == f"NEW:{name}"
    return {"boardings_ph": float(F["boarding_segment"][k].sum() * 3600),
            "max_section_ph": float(F["segment"][k].max() * 3600)}


def main() -> None:
    t0 = time.perf_counter()
    net = pickle.load(open("data/cache/net_20260609.pkl", "rb"))
    base_params = ModelParams(lam=1 / 300, mean_leg_m=3500.0, p_transfer=0.30)
    model = OpenNetworkModel(net, base_params)
    w = model.demand_prior()
    lam = model.scale_to_boardings(w, N_DAY * PHI / 7200.0)     # fixed demand
    H = net.n_hubs
    origins = np.arange(H)
    T0 = model.hub_travel_times(model.travel_graph(), origins)
    ev0 = MinPlusEvaluator(T0, origins, w)
    E0 = ev0.efficiency()
    log.info("baseline efficiency %.4e  (%.1f s)", E0, time.perf_counter() - t0)

    # --- geography: city limits and the Danube ------------------------------
    districts = load_districts()
    city = unary_union(list(districts.values()))
    buda = unary_union([g for n, g in districts.items() if n in BUDA])
    rest = unary_union([g for n, g in districts.items() if n not in BUDA])
    river = buda.boundary.intersection(rest.buffer(60))
    csepel = unary_union([g for n, g in districts.items() if n in CSEPEL])
    arm = csepel.boundary.intersection(unary_union(
        [g for n, g in districts.items() if n not in BUDA | CSEPEL]).buffer(60))
    water = unary_union([river, arm])
    hx, hy = eov(net.hub_lon, net.hub_lat)
    in_city = np.array([city.contains(Point(x, y)) for x, y in zip(hx, hy)])

    def crosses(lon1, lat1, lon2, lat2) -> bool:
        x, y = eov([lon1, lon2], [lat1, lat2])
        return LineString(list(zip(x, y))).intersects(water)

    pool = np.flatnonzero(model.hub_active & in_city)
    ends = pool[np.argsort(-w[pool])][:N_END]

    # --- street graph from GTFS shapes; trams run on main roads only --------
    sg_cache = Path("data/cache/streets.pkl")
    if sg_cache.exists():
        G = pickle.load(open(sg_cache, "rb"))
    else:
        import zipfile
        with zipfile.ZipFile("data/budapest_gtfs.zip") as z:
            sh = pd.read_csv(z.open("shapes.txt"), dtype={"shape_id": str})
            tr = pd.read_csv(z.open("trips.txt"), dtype=str, usecols=["route_id", "shape_id"])
            ro = pd.read_csv(z.open("routes.txt"), dtype=str, usecols=["route_id", "route_type"])
        G = build_street_graph(sh, tr.drop_duplicates().merge(ro, on="route_id"))
        pickle.dump(G, open(sg_cache, "wb"))
    main = G.main_roads(min_bus_routes=3)
    # Danube crossings: trams only on bridges that carry trams today
    # (Margit, Petőfi, Rákóczi híd) -- not on the historic Lánchíd etc.
    import shapely
    from bkk.streets import unproject
    A = main.tocoo()
    up = A.row < A.col
    r_, c_ = A.row[up], A.col[up]
    la1, lo1 = unproject(G.xy[r_, 0], G.xy[r_, 1])
    la2, lo2 = unproject(G.xy[c_, 0], G.xy[c_, 1])
    x1, y1 = eov(lo1, la1)
    x2, y2 = eov(lo2, la2)
    segs = shapely.linestrings(np.stack([np.column_stack([x1, y1]),
                                         np.column_stack([x2, y2])], axis=1))
    crossing = shapely.intersects(segs, water)
    has_tram = np.asarray(G.tram[r_, c_]).ravel() > 0.5
    drop = crossing & ~has_tram
    keep = ~drop
    from scipy.sparse import csr_matrix as _csr
    rr, cc, vv = r_[keep], c_[keep], A.data[up][keep]
    main = _csr((np.r_[vv, vv], (np.r_[rr, cc], np.r_[cc, rr])), shape=main.shape)
    log.info("main-road graph: %d river-crossing edges without tram removed", int(drop.sum()))
    router = Router.build(G, main, net.hub_lat, net.hub_lon, max_snap_m=120.0)
    tram_pool = pool[router.hub_node[pool] >= 0]

    # --- candidates and greedy selection -------------------------------------
    cands = [x for x in generate_candidates(net, w, METRO, ends, pool, crosses_danube=crosses)
             if len(x.hubs) >= 3]
    n_m = len(cands)
    cands += [x for x in generate_street_candidates(net, w, TRAM, ends, tram_pool, router)
              if len(x.hubs) >= 3]
    log.info("candidates: %d metro (straight tunnels), %d tram (street-routed)",
             n_m, len(cands) - n_m)
    # the worked example of the paper: Rákóczi tér -> Széchenyi István tér by tram
    ex = generate_street_candidates(net, w, TRAM, [hub(net, "Rákóczi tér"),
                                    hub(net, "Széchenyi István tér")], tram_pool, router)
    R = {"street_graph": {"nodes": int(G.n_nodes), "edges": int(G.adj.nnz // 2),
                          "main_edges": int(main.nnz // 2),
                          "main_km": float(main.sum() / 2 / 1000),
                          "all_km": float(G.adj.sum() / 2 / 1000),
                          "tram_station_hubs": int(len(tram_pool)),
                          "bridge_edges_removed": int(drop.sum()),
                          "active_city_hubs": int(len(pool))},
         "example_route": {"stations": [str(net.hub_name[h]) for h in ex[0].hubs],
                           "length_km": line_length_m(net, ex[0]) / 1000} if ex else None,
         "baseline_efficiency": E0, "n_candidates": len(cands),
         "n_metro": sum(c.spec.mode == 1 for c in cands),
         "n_tram": sum(c.spec.mode == 0 for c in cands),
         "endpoints": int(len(ends)), "cost_ratio": METRO.cost_per_km / TRAM.cost_per_km}

    cache = Path(f"data/cache/scen_gains_streets_{N_END}.pkl")
    t1 = time.perf_counter()
    if cache.exists():
        g1 = pickle.load(open(cache, "rb"))
    else:
        g1 = first_round_gains(ev0, net, cands)
        pickle.dump(g1, open(cache, "wb"))
    R["first_round_seconds"] = time.perf_counter() - t1
    log.info("first-round gains: %.0f s", R["first_round_seconds"])

    def run_greedy(cand_list, per_cost=True):
        # one line per corridor (unordered endpoint pair)
        return greedy_select(MinPlusEvaluator(T0, origins, w), net, cand_list, K_LINES, per_cost,
                             initial_gains=g1,
                             key=lambda c: frozenset((c.hubs[0], c.hubs[-1])))

    t1 = time.perf_counter()
    picked = run_greedy(cands)
    R["greedy_seconds"] = time.perf_counter() - t1
    log.info("greedy done in %.0f s", R["greedy_seconds"])

    # sensitivity to the cost ratio and the pure-gain ranking
    def recost(ratio):
        m = replace(METRO, cost_per_km=ratio)
        return [replace(c, spec=m) if c.spec.mode == 1 else c for c in cands]
    sens, sens_detail = {}, {}
    def brief(picks, ratio):
        return [{"name": p["line"].name, "mode": "metró" if p["line"].spec.mode == 1 else "villamos",
                 "from": str(net.hub_name[p["line"].hubs[0]]),
                 "to": str(net.hub_name[p["line"].hubs[-1]]),
                 "stations": [str(net.hub_name[h]) for h in p["line"].hubs],
                 "length_km": line_length_m(net, p["line"]) / 1000,
                 "marginal_gain": p["marginal_gain"], "cumulative_gain": p["cumulative_gain"]}
                for p in picks]
    for ratio in (5.0, 20.0):
        pk = run_greedy(recost(ratio))
        sens[f"ratio_{ratio:g}"] = [p["line"].name for p in pk]
        sens_detail[f"ratio_{ratio:g}"] = brief(pk, ratio)
    pk = run_greedy(cands, per_cost=False)
    sens["pure_gain"] = [p["line"].name for p in pk]
    sens_detail["pure_gain"] = brief(pk, None)
    R["selection_detail"] = sens_detail
    R["selection_sensitivity"] = sens
    base_names = [p["line"].name for p in picked]
    R["overlap_with_base"] = {k: len(set(v) & set(base_names)) for k, v in sens.items()}

    # --- exact check and CTMC predictions for the selected package -----------
    lines = [p["line"] for p in picked]
    for i, l in enumerate(lines, start=1):
        l.name = f"U{i}"
    aug = OpenNetworkModel(add_lines(net, lines), base_params)
    E_pkg = aug.efficiency(w, origins=origins)
    R["package_gain_exact"] = (E_pkg - E0) / E0
    R["package_gain_minplus"] = picked[-1]["cumulative_gain"]
    rows = []
    for i, (p, l) in enumerate(zip(picked, lines), start=1):
        pred = line_predictions(aug, lam, l.name)
        rows.append({
            "rank": i, "name": l.name, "mode": "metró" if l.spec.mode == 1 else "villamos",
            "from": str(net.hub_name[l.hubs[0]]), "to": str(net.hub_name[l.hubs[-1]]),
            "stations": [str(net.hub_name[h]) for h in l.hubs],
            "length_km": line_length_m(net, l) / 1000, "cost": p["cost"],
            "marginal_gain": p["marginal_gain"], "cumulative_gain": p["cumulative_gain"],
            "gain_per_cost": p["marginal_gain"] / p["cost"], **pred,
        })
    R["selected"] = rows

    # --- planned M5 benchmark (two central variants, H5/H6/H7 through-running)
    def m5(variant: str):
        mid = {"A": "Lehel tér", "B": "Nyugati pályaudvar"}[variant]
        core = ["Margit híd, budai hídfő", "Margitsziget / Margit híd", "Szent István park",
                mid, "Astoria", "Kálvin tér", "Boráros tér"]
        h = [hub(net, n) for n in core]
        mh, bat = hub(net, "Margit híd, budai hídfő"), hub(net, "Batthyány tér")
        # H5 neighbour of Margit híd on the Szentendre side (not Batthyány tér)
        nb = net.seg_to[(net.seg_route == "H5") & (net.seg_from == mh)]
        szv = int([x for x in nb if x != bat][0])
        kvh, bor = hub(net, "Közvágóhíd"), hub(net, "Boráros tér")
        h5_in, h5_out = find_segment(net, "H5", szv, mh), find_segment(net, "H5", mh, szv)
        h6_out = list(np.flatnonzero((net.seg_route == "H6") & (net.seg_from == kvh)))
        h6_in = list(np.flatnonzero((net.seg_route == "H6") & (net.seg_to == kvh)))
        h7_out = list(np.flatnonzero((net.seg_route == "H7") & (net.seg_from == bor)))
        h7_in = list(np.flatnonzero((net.seg_route == "H7") & (net.seg_to == bor)))
        spec = replace(METRO, detour=1.15)
        # two service patterns share the core, each at twice the core headway
        a = NewLine(f"M5{variant}-H6", spec, h + [kvh], headway_s=2 * METRO.headway_s,
                    feed_forward=[h5_in], exit_forward=h6_out,
                    feed_reverse=h6_in, exit_reverse=[h5_out])
        b = NewLine(f"M5{variant}-H7", spec, h, headway_s=2 * METRO.headway_s,
                    exit_forward=h7_out, feed_reverse=h7_in, exit_reverse=[h5_out])
        return [a, b]

    bench = {}
    for v in ("A", "B"):
        ls = m5(v)
        aug_m = OpenNetworkModel(add_lines(net, ls), base_params)
        E_m = aug_m.efficiency(w, origins=origins)
        cost = line_cost(net, ls[0])
        preds = [line_predictions(aug_m, lam, l.name) for l in ls]
        bench[v] = {
            "stations": [str(net.hub_name[x]) for x in ls[0].hubs],
            "length_km": line_length_m(net, ls[0]) / 1000, "cost": cost,
            "gain": (E_m - E0) / E0, "gain_per_cost": (E_m - E0) / E0 / cost,
            "boardings_ph": sum(p["boardings_ph"] for p in preds),
            "max_section_ph": max(p["max_section_ph"] for p in preds),
        }
        log.info("M5%s: gain %.4f  per cost %.2e", v, bench[v]["gain"], bench[v]["gain_per_cost"])
    R["m5"] = bench
    # rank of M5 among all candidates by gain per cost (first round)
    first = sorted((g / line_cost(net, c) for g, c in zip(g1, cands)), reverse=True)
    R["m5_rank_gain_per_cost"] = {v: int(np.searchsorted(-np.array(first),
                                                          -bench[v]["gain_per_cost"]) + 1)
                                  for v in bench}
    R["seconds"] = time.perf_counter() - t0

    json.dump(R, open(RES / "scenarios.json", "w"), indent=2, ensure_ascii=False, default=float)
    geo = []
    for r, l in zip(rows, lines):
        for k, h_ in enumerate(l.hubs):
            geo.append({"line": r["name"], "mode": r["mode"], "order": k,
                        "hub": str(net.hub_name[h_]), "lat": net.hub_lat[h_], "lon": net.hub_lon[h_]})
    for k_, rec in enumerate(sens_detail["pure_gain"], start=1):
        for k, name in enumerate(rec["stations"]):
            h_ = hub(net, name) if (net.hub_name == name).sum() == 1 else None
            if h_ is None:
                continue
            geo.append({"line": f"G{k_}", "mode": rec["mode"], "order": k, "hub": name,
                        "lat": net.hub_lat[h_], "lon": net.hub_lon[h_]})
    for v in ("A", "B"):
        for k, h_ in enumerate(m5(v)[0].hubs):
            geo.append({"line": f"M5{v}", "mode": "metró", "order": k,
                        "hub": str(net.hub_name[h_]), "lat": net.hub_lat[h_], "lon": net.hub_lon[h_]})
    pd.DataFrame(geo).to_csv(RES / "new_lines.csv", index=False)
    paths = []
    for r, l in zip(rows, lines):
        if l.path_latlon is not None:
            for k, (la, lo) in enumerate(l.path_latlon):
                paths.append({"line": r["name"], "order": k, "lat": la, "lon": lo})
    pd.DataFrame(paths, columns=["line", "order", "lat", "lon"]).to_csv(
        RES / "new_line_paths.csv", index=False)
    log.info("wrote scenarios (%.0f s)", R["seconds"])


if __name__ == "__main__":
    main()
