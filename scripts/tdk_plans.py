#!/usr/bin/env python3
"""
scripts/tdk_plans.py
====================
Current Budapest development plans in the line-aware Markov model, and the
best additional lines once they are built.

Plans (stations at existing hubs; new stops without an existing hub, e.g.
Bertalan Lajos utca or Nádorkert, cannot carry demand in the model):
  * Bajcsy tram  (Pesti fonódó II): Lehel tér - Deák tér, through-running
    14 -> 47 and 12 -> 49
  * Budai fonódó II: Szent Gellért tér - Műegyetem rakpart - Budafoki út /
    Dombóvári út (2.8 km)
  * Budafoki út tram: Dombóvári út - Savoya Park (street-routed)
  * M5 (variant B, Nyugati pályaudvar), H5/H6/H7 through-running

    python scripts/tdk_plans.py        (after tdk_analysis.py, tdk_scenarios.py)

Outputs: data/results/plans.json, data/results/plan_lines.csv
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from tdk_common import (BASE_PARAMS, DEMAND, RES as C_RES, demand, accessibility, district_change, es_travel_time,
                        geography, hub, load_base, street_router)

from bkk.linemodel import OpenNetworkModel
from bkk.scenario import (METRO, TRAM, MinPlusEvaluator, NewLine, add_lines, node_mass,
                          first_round_gains, generate_candidates,
                          generate_street_candidates, greedy_select, line_cost,
                          line_length_m)
from bkk.streets import unproject

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("plans")
RES = Path(__import__("os").environ.get("PLANS_OUT", str(C_RES)))
DISTRICTS = ["IV. kerület", "XIII. kerület", "V. kerület", "XI. kerület", "XXII. kerület",
             "XXI. kerület", "IX. kerület", "III. kerület"]


def routed_line(net, router, name, spec, hubs, headway=None, **through):
    """Line through the given hubs, each leg routed on the main-road graph."""
    G = router.graph
    lens, pts = [], []
    for a, b in zip(hubs, hubs[1:]):
        na, nb = router.hub_node[a], router.hub_node[b]
        if na < 0 or nb < 0:
            raise ValueError(f"{net.hub_name[a]} or {net.hub_name[b]} not on a main road")
        dist, pred = router.paths_from(na)
        path = router.extract(pred, na, nb)
        lens.append(float(dist[nb]))
        pts.append(G.xy[path])
    P = np.vstack(pts)
    lat, lon = unproject(P[:, 0], P[:, 1])
    return NewLine(name, spec, list(hubs), headway_s=headway, seg_len_m=lens,
                   path_latlon=np.column_stack([lat, lon]), **through)


def segs(net, route, frm=None, to=None):
    k = net.seg_route == route
    if frm is not None:
        k &= net.seg_from == frm
    if to is not None:
        k &= net.seg_to == to
    return [int(x) for x in np.flatnonzero(k)]


def predictions(model, lam, names):
    L = model.steady_state(lam)
    F = model.flows(L)
    k = np.isin(model.net.seg_route, [f"NEW:{n}" for n in names])
    return {"boardings_ph": float(F["boarding_segment"][k].sum() * 3600),
            "max_section_ph": float(F["segment"][k].max() * 3600) if k.any() else 0.0}


def main() -> None:
    t0 = time.perf_counter()
    net, model, w, lam, origins, T0 = load_base()
    districts, water, label = geography(net)
    G, main_g, router, n_bridge = street_router(net, water)
    E0 = MinPlusEvaluator(T0, origins, w).efficiency()
    ES0 = es_travel_time(T0, origins, w)
    A0 = accessibility(T0, origins, w)
    log.info("baseline E=%.4e  ES90=%.1f min", E0, ES0)

    # ---------------------------------------------------------------- plans
    H = lambda n: hub(net, n)
    lehel, deak = H("Lehel tér"), H("Deák Ferenc tér")
    bajcsy_hubs = [lehel, H("Nyugati pályaudvar"), H("Báthory utca / Bajcsy-Zsilinszky út"),
                   H("Arany János utca"), H("Bajcsy-Zsilinszky út"), deak]
    R_ = {"14": "3140", "12": "3120", "47": "3470", "49": "3490"}
    tram5 = replace(TRAM)
    bajcsy = []
    for north, south in (("14", "47"), ("12", "49")):
        bajcsy.append(routed_line(
            net, router, f"Bajcsy-{north}-{south}", tram5, bajcsy_hubs, headway=300.0,
            feed_forward=segs(net, R_[north], to=lehel),
            exit_forward=segs(net, R_[south], frm=deak),
            feed_reverse=segs(net, R_[south], to=deak),
            exit_reverse=segs(net, R_[north], frm=lehel)))
    for l in bajcsy:                                     # official length: 2.3 km
        l.seg_len_m = (np.asarray(l.seg_len_m) * 2300.0 / np.sum(l.seg_len_m)).tolist()

    bf_hubs = [H("Szent Gellért tér - Műegyetem"), H("Petőfi híd, budai hídfő"),
               H("Magyar tudósok körútja"), H("Infopark (Pázmány Péter sétány)"),
               H("Budafoki út / Dombóvári út")]
    straight = np.array([np.hypot(*(np.array(_xy(net, a)) - np.array(_xy(net, b))))
                         for a, b in zip(bf_hubs, bf_hubs[1:])])
    bf2 = NewLine("BudaiFonodo2", tram5, bf_hubs, headway_s=300.0,
                  seg_len_m=(straight * 2800.0 / straight.sum()).tolist())   # official 2.8 km

    # the planned street is fixed (Budafoki út), so route on the full street graph
    from bkk.streets import Router as _Router
    any_router = _Router.build(G, G.adj, net.hub_lat, net.hub_lon, max_snap_m=120.0)
    any_pool = np.flatnonzero(model.hub_active & (any_router.hub_node >= 0))
    bud = generate_street_candidates(net, w, TRAM, [H("Budafoki út / Dombóvári út"),
                                                    H("Savoya Park")], any_pool, any_router)
    budafoki = replace(bud[0], name="Budafoki", headway_s=300.0)

    def m5b():
        core = ["Margit híd, budai hídfő", "Margitsziget / Margit híd", "Szent István park",
                "Nyugati pályaudvar", "Astoria", "Kálvin tér", "Boráros tér"]
        h = [H(n) for n in core]
        mh, bat = H("Margit híd, budai hídfő"), H("Batthyány tér")
        nb = net.seg_to[(net.seg_route == "H5") & (net.seg_from == mh)]
        szv = int([x for x in nb if x != bat][0])
        kvh, bor = H("Közvágóhíd"), H("Boráros tér")
        spec = replace(METRO, detour=1.15)
        a = NewLine("M5-H6", spec, h + [kvh], headway_s=2 * METRO.headway_s,
                    feed_forward=segs(net, "H5", szv, mh), exit_forward=segs(net, "H6", frm=kvh),
                    feed_reverse=segs(net, "H6", to=kvh), exit_reverse=segs(net, "H5", mh, szv))
        b = NewLine("M5-H7", spec, h, headway_s=2 * METRO.headway_s,
                    exit_forward=segs(net, "H7", frm=bor), feed_reverse=segs(net, "H7", to=bor),
                    exit_reverse=segs(net, "H5", mh, szv))
        return [a, b]

    plans = {"bajcsy": bajcsy, "budai_fonodo2": [bf2], "budafoki": [budafoki], "m5": m5b()}
    labels = {"bajcsy": "Bajcsy-Zsilinszky úti villamos (Deák tér -- Lehel tér)",
              "budai_fonodo2": "Budai fonódó II. (Szent Gellért tér -- Dombóvári út)",
              "budafoki": "Budafoki úti villamos (Dombóvári út -- Savoya Park)",
              "m5": "M5 (B: Nyugati pu.), HÉV-átmenettel"}

    def evaluate(lines):
        aug = OpenNetworkModel(add_lines(net, lines), BASE_PARAMS)
        T = aug.hub_travel_times(aug.travel_graph(), origins)
        ev = MinPlusEvaluator(T, origins, w)
        return aug, T, ev.efficiency()

    R = {"baseline": {"E": E0, "ES90_min": ES0}, "bridge_edges_removed": n_bridge,
         "demand": DEMAND}
    # the same travel-time matrices under the other demand weights (robustness)
    alt = {v: demand(model, T0, v)[1] for v in ("departures", "population", "population_gravity")
           if v != DEMAND}
    alt_base = {v: (MinPlusEvaluator(T0, origins, ww).efficiency(), es_travel_time(T0, origins, ww))
                for v, ww in alt.items()}
    robust = {v: {"baseline_ES90_min": b[1]} for v, b in alt_base.items()}

    def robustness(key, T):
        for v, ww in alt.items():
            E_v = MinPlusEvaluator(T, origins, ww).efficiency()
            robust[v][key] = {"gain": (E_v - alt_base[v][0]) / alt_base[v][0],
                              "ES90_min": es_travel_time(T, origins, ww)}

    out = {}
    for key, lines in plans.items():
        aug, T, E = evaluate(lines)
        robustness(key, T)
        A1 = accessibility(T, origins, w)
        rec = {"label": labels[key], "mode": "metró" if lines[0].spec.mode == 1 else "villamos",
               "stations": [str(net.hub_name[h]) for h in lines[0].hubs],
               "length_km": line_length_m(net, lines[0]) / 1000,
               "cost": line_cost(net, lines[0]),
               "gain": (E - E0) / E0, "ES90_min": es_travel_time(T, origins, w),
               "district": district_change(A0, A1, w, label, DISTRICTS),
               **predictions(aug, lam, [l.name for l in lines])}
        rec["gain_per_cost"] = rec["gain"] / rec["cost"]
        out[key] = rec
        log.info("%-14s gain %.4f  ES %.2f  board %.0f", key, rec["gain"], rec["ES90_min"],
                 rec["boardings_ph"])
    R["plans"] = out

    # all four together
    pkg_lines = [l for ls in plans.values() for l in ls]
    aug_p, T_p, E_p = evaluate(pkg_lines)
    robustness("package", T_p)
    R["robustness"] = robust
    A_p = accessibility(T_p, origins, w)
    R["package"] = {"gain": (E_p - E0) / E0, "ES90_min": es_travel_time(T_p, origins, w),
                    "sum_of_parts": sum(v["gain"] for v in out.values()),
                    "district": district_change(A0, A_p, w, label, DISTRICTS)}
    log.info("package gain %.4f (sum of parts %.4f)", R["package"]["gain"],
             R["package"]["sum_of_parts"])

    # ------------------------------------- best additional lines after the plans
    from tdk_common import geography as _g  # noqa: F401  (documented dependency)
    from shapely.geometry import LineString, Point
    from budapest_basemap import eov
    city = None
    for g in districts.values():
        city = g if city is None else city.union(g)
    hx, hy = eov(net.hub_lon, net.hub_lat)
    in_city = np.array([city.contains(Point(x, y)) for x, y in zip(hx, hy)])
    pool = np.flatnonzero(model.hub_active & in_city)
    n_end = int(__import__("os").environ.get("PLANS_NEND", 60))   # override for smoke tests
    ends = pool[np.argsort(-node_mass(w)[pool])][:n_end]

    def crosses(lon1, lat1, lon2, lat2):
        x, y = eov([lon1, lon2], [lat1, lat2])
        return LineString(list(zip(x, y))).intersects(water)

    cands = [c for c in generate_candidates(net, w, METRO, ends, pool, crosses_danube=crosses)
             if len(c.hubs) >= 3]
    cands += [c for c in generate_street_candidates(net, w, TRAM, ends,
                                                    pool[router.hub_node[pool] >= 0], router)
              if len(c.hubs) >= 3]
    ev_p = MinPlusEvaluator(T_p, origins, w)
    cache = Path(f"data/cache/plans_gains_{n_end}_{DEMAND}.pkl")
    if cache.exists():
        g1 = pickle.load(open(cache, "rb"))
    else:
        g1 = first_round_gains(ev_p, net, cands)
        pickle.dump(g1, open(cache, "wb"))
    picked = greedy_select(ev_p, net, cands, 5, per_cost=True, initial_gains=g1,
                           key=lambda c: frozenset((c.hubs[0], c.hubs[-1])))
    by_name = {c.name: c for c in cands}               # before any renaming
    new = [replace(p["line"], name=f"J{i}") for i, p in enumerate(picked, start=1)]
    aug_n = OpenNetworkModel(add_lines(net, pkg_lines + new), BASE_PARAMS)
    T_n = aug_n.hub_travel_times(aug_n.travel_graph(), origins)
    E_n = MinPlusEvaluator(T_n, origins, w).efficiency()
    rows = []
    for i, (p, l) in enumerate(zip(picked, new), start=1):
        rows.append({"rank": i, "name": l.name,
                     "mode": "metró" if l.spec.mode == 1 else "villamos",
                     "from": str(net.hub_name[l.hubs[0]]), "to": str(net.hub_name[l.hubs[-1]]),
                     "stations": [str(net.hub_name[h]) for h in l.hubs],
                     "length_km": line_length_m(net, l) / 1000, "cost": p["cost"],
                     # gains relative to E0, on top of the plans
                     "marginal_gain": p["marginal_gain"] * E_p / E0,
                     "gain_per_cost": p["marginal_gain"] * E_p / E0 / p["cost"],
                     **predictions(aug_n, lam, [l.name])})
    R["after_plans"] = {"selected": rows, "gain_total_vs_E0": (E_n - E0) / E0,
                        "ES90_min": es_travel_time(T_n, origins, w),
                        "district": district_change(A0, accessibility(T_n, origins, w), w,
                                                    label, DISTRICTS)}

    # how the earlier (plan-free) proposals U1..U5 interact with the plans
    sc = json.load(open(RES / "scenarios.json"))
    inter = []
    ev0 = MinPlusEvaluator(T0, origins, w)
    ev_p = MinPlusEvaluator(T_p, origins, w)            # fresh: plans only, no J-lines
    for r in sc["selected"]:
        key = f"{'M' if r['mode'] == 'metró' else 'V'}:{r['from']}–{r['to']}"
        c = by_name.get(key)
        if c is None:
            continue
        g_base = ev0.gain(net, c, E0)
        g_plan = ev_p.gain(net, c, E_p) * E_p / E0
        inter.append({"name": r["name"], "from": r["from"], "to": r["to"],
                      "gain_alone": g_base, "gain_after_plans": g_plan,
                      "ratio": g_plan / g_base if g_base > 0 else float("nan")})
    R["interaction"] = inter
    R["seconds"] = time.perf_counter() - t0
    json.dump(R, open(RES / "plans.json", "w"), indent=2, ensure_ascii=False, default=float)

    geo = []
    def add_geo(name, kind, line):
        if line.path_latlon is not None:
            pts = line.path_latlon
        else:
            pts = np.column_stack([net.hub_lat[line.hubs], net.hub_lon[line.hubs]])
        for k, (la, lo) in enumerate(pts):
            geo.append({"line": name, "kind": kind,
                        "mode": "metró" if line.spec.mode == 1 else "villamos",
                        "order": k, "lat": la, "lon": lo, "station": 0})
        for k, h_ in enumerate(line.hubs):
            geo.append({"line": name, "kind": kind,
                        "mode": "metró" if line.spec.mode == 1 else "villamos",
                        "order": k, "lat": net.hub_lat[h_], "lon": net.hub_lon[h_], "station": 1})
    add_geo("Bajcsy", "terv", bajcsy[0])
    add_geo("BF2", "terv", bf2)
    add_geo("Budafoki", "terv", budafoki)
    add_geo("M5", "terv", plans["m5"][0])
    for l in new:
        add_geo(l.name, "javaslat", l)
    pd.DataFrame(geo).to_csv(RES / "plan_lines.csv", index=False)
    log.info("wrote plans (%.0f s)", R["seconds"])


def _xy(net, h):
    from bkk.streets import project
    x, y = project(net.hub_lat[h], net.hub_lon[h])
    return float(x), float(y)


if __name__ == "__main__":
    main()
