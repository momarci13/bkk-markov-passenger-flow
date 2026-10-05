#!/usr/bin/env python3
"""
scripts/tdk_plans_demand.py
===========================
Future-demand scenario for the South-Buda plans: BKK expects about 18 000 new
homes and 40 000 new residents between Dombóvári út and Savoya Park by 2030.
Their journeys are added as extra Poisson sources at the hubs within 600 m of
the planned Budafoki út / Műegyetem rakpart stations; the model is linear, so
the new demand simply adds to the source vector.

Assumption: 2 boardings per resident per workday (3.4 million boardings for
about 1.7 million residents), 20 % of them in the 07-09 window.

    python scripts/tdk_plans_demand.py     (after tdk_plans.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tdk_common import BASE_PARAMS, RES, hub, load_base, od_matrix

from bkk.linemodel import OpenNetworkModel
from bkk.network import _haversine_m
from bkk.scenario import MinPlusEvaluator, NewLine, TRAM, add_lines

NEW_RESIDENTS = 40_000
BOARD_PER_RES = 2.0
PHI = 0.20
CATCH_M = 600.0


def main() -> None:
    net, model, w, lam, origins, T0 = load_base()
    P = json.load(open(RES / "plans.json"))
    bf = P["plans"]["budai_fonodo2"]
    bu = P["plans"]["budafoki"]
    st_names = bf["stations"] + bu["stations"][1:]
    st = [hub(net, n) for n in st_names]

    # catchment hubs of the new neighbourhoods: Dombóvári út -> Savoya Park
    south = [hub(net, n) for n in bu["stations"]]
    d = np.min([_haversine_m(net.hub_lat, net.hub_lon, net.hub_lat[s], net.hub_lon[s])
                for s in south], axis=0)
    catch = np.flatnonzero((d <= CATCH_M) & model.hub_active)

    # extra boardings in the window, converted to journeys with the base legs/journey
    base_board = model.flows(model.steady_state(lam))["boarding"].sum()
    legs = base_board / lam.sum()
    extra_board = NEW_RESIDENTS * BOARD_PER_RES * PHI / 7200.0
    lam_new = lam.copy()
    lam_new[catch] += extra_board / legs / len(catch)
    w_new = od_matrix(lam_new, model, T0)

    # rebuild the two South-Buda lines exactly as in tdk_plans.py
    def line(rec, name):
        hubs = [hub(net, n) for n in rec["stations"]]
        L = rec["length_km"] * 1000.0
        straight = np.array([_haversine_m(net.hub_lat[a], net.hub_lon[a],
                                          net.hub_lat[b], net.hub_lon[b])
                             for a, b in zip(hubs, hubs[1:])])
        return NewLine(name, TRAM, hubs, headway_s=300.0,
                       seg_len_m=(straight * L / straight.sum()).tolist())
    lines = [line(bf, "BF2"), line(bu, "Budafoki")]
    aug = OpenNetworkModel(add_lines(net, lines), BASE_PARAMS)

    def boardings(lam_vec):
        F = aug.flows(aug.steady_state(lam_vec))
        k = np.isin(aug.net.seg_route, ["NEW:BF2", "NEW:Budafoki"])
        return float(F["boarding_segment"][k].sum() * 3600)

    Ta = aug.hub_travel_times(aug.travel_graph(), origins)
    out = {}
    for tag, ww, ll in (("today", w, lam), ("2030", w_new, lam_new)):
        ev0 = MinPlusEvaluator(T0, origins, ww)
        eva = MinPlusEvaluator(Ta, origins, ww)
        e0, e1 = ev0.efficiency(), eva.efficiency()
        out[tag] = {"gain": (e1 - e0) / e0, "boardings_ph": boardings(ll)}
    R = {"new_residents": NEW_RESIDENTS, "boardings_per_resident": BOARD_PER_RES,
         "catchment_hubs": int(len(catch)), "extra_boardings_ph": extra_board * 3600,
         "scenarios": out,
         "boardings_ratio": out["2030"]["boardings_ph"] / out["today"]["boardings_ph"],
         "gain_ratio": out["2030"]["gain"] / out["today"]["gain"]}
    json.dump(R, open(RES / "plans_demand.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
