#!/usr/bin/env python3
"""
scripts/tdk_analysis.py
=======================
Reproduces every number of the TDK paper (tdk/tdk_dolgozat.tex) with the
line-aware open Markov model (bkk.linemodel) on one pinned service day.

    python scripts/tdk_analysis.py --zip data/budapest_gtfs.zip --date 20260609

Outputs (data/results/):
    results.json        scalar results quoted in the paper
    hubs.csv            per-hub flows, waiting stock and resilience metrics
    segments.csv        per-segment passenger flows (for the flow map)
    transient.csv       total mean occupancy from an empty system
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bkk import GTFSLoader
from bkk.linemodel import ModelParams, OpenNetworkModel, build_line_network

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tdk")

MODE_HU = {0: "villamos", 1: "metró", 3: "busz", 4: "hajó", 11: "trolibusz", 109: "HÉV"}


def legacy_trip_counts(loader, date: str, weekday: str) -> dict:
    exact = loader.parse(date_filter=date)
    union = loader.parse(weekday=weekday)
    return {"exact_trips": int(len(exact.trips)), "weekday_union_trips": int(len(union.trips)),
            "inflation": float(len(union.trips) / len(exact.trips))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/budapest_gtfs.zip")
    ap.add_argument("--date", default="20260609")
    ap.add_argument("--out", default="data/results")
    ap.add_argument("--n-day", type=float, default=4.0e6, help="weekday boardings")
    ap.add_argument("--phi", type=float, default=0.20, help="share of boardings in 07-09")
    ap.add_argument("--candidates", type=int, default=60)
    ap.add_argument("--mc", type=int, default=200_000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R: dict = {"date": args.date}

    with zipfile.ZipFile(args.zip) as z:
        info = pd.read_csv(z.open("feed_info.txt"), dtype=str).iloc[0].to_dict()
    R["feed"] = info

    cache = Path("data/cache") / f"net_{args.date}.pkl"
    loader = GTFSLoader().load(args.zip)
    R["calendar"] = legacy_trip_counts(loader, args.date, "tuesday")
    if cache.exists():
        net = pickle.load(open(cache, "rb"))
    else:
        feed = loader.parse(date_filter=args.date)
        net = build_line_network(feed)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(net, open(cache, "wb"))
    H, S = net.n_hubs, net.n_segments

    # ---------------------------------------------------------------- network
    modes = {}
    for m in sorted(set(net.seg_mode.tolist())):
        k = net.seg_mode == m
        modes[MODE_HU.get(m, str(m))] = {
            "segments": int(k.sum()),
            "departures_window": int(net.seg_n[k].sum()),
            "median_cv2": float(np.median(net.seg_cv2[k][net.seg_n[k] >= 3])),
        }
    many = net.seg_n >= 3
    R["network"] = {
        "hubs": H, "segments": S, "states": H + S,
        "trip_segments": int(net.meta["n_trip_segments"]),
        "modes": modes,
        "median_cv2": float(np.median(net.seg_cv2[many])),
        "share_cv2_below_0.1": float(np.mean(net.seg_cv2[many] < 0.1)),
        "median_tau_s": float(np.median(net.seg_tau)),
        "median_dist_m": float(np.median(net.seg_dist)),
        "single_departure_segments": int((net.seg_n == 1).sum()),
    }

    # ----------------------------------------------------------- base model
    base = ModelParams(lam=1 / 300, mean_leg_m=3500.0, p_transfer=0.30)
    model = OpenNetworkModel(net, base)
    B_rate = args.n_day * args.phi / 7200.0                      # boardings / s
    shape = model.demand_prior(beta=1.0)
    lam = model.scale_to_boardings(shape, B_rate)
    L = model.steady_state(lam)
    F = model.flows(L)
    waiting, riding = L[:H].sum(), L[H:].sum()
    Lambda = lam.sum()
    R["base"] = {
        "params": {"lambda_per_s": base.lam, "mean_leg_m": base.mean_leg_m,
                   "p_transfer": base.p_transfer, "n_day": args.n_day, "phi": args.phi},
        "boardings_per_s": B_rate, "journeys_per_s": float(Lambda),
        "legs_per_journey": float(B_rate / Lambda),
        "in_system": float(L.sum()), "waiting": float(waiting), "riding": float(riding),
        "mean_journey_min": float(L.sum() / Lambda / 60),
        "mean_wait_per_boarding_min": float(waiting / B_rate / 60),
        "mean_ride_per_leg_min": float(riding / B_rate / 60),
        "max_segment_flow_pph": float(F["segment"].max() * 3600),
        "busiest_hub_boardings_pph": float(F["boarding"].max() * 3600),
        "poisson_cv_busiest_hub_wait": float(1 / np.sqrt(L[:H].max())),
    }
    # modal split of boardings and passenger-km
    by_mode_b = pd.Series(F["boarding_segment"]).groupby(net.seg_mode).sum()
    by_mode_km = pd.Series(F["segment"] * net.seg_dist).groupby(net.seg_mode).sum()
    R["modal_split"] = {
        MODE_HU.get(int(m), str(m)): {"boardings_share": float(by_mode_b[m] / by_mode_b.sum()),
                                      "pkm_share": float(by_mode_km[m] / by_mode_km.sum())}
        for m in by_mode_b.index
    }
    # waiting-time consequence of the hazard correction and the calendar bug
    th = model.theta
    R["hazard"] = {
        "mean_theta_over_f": float(np.average(th / net.seg_freq, weights=net.seg_n)),
    }
    m_f = OpenNetworkModel(net, ModelParams(lam=base.lam, mean_leg_m=base.mean_leg_m,
                                            p_transfer=base.p_transfer,
                                            headway_correction=False))
    lam_f = m_f.scale_to_boardings(shape, B_rate)
    Lf = m_f.steady_state(lam_f)
    R["hazard"]["wait_per_boarding_min_uncorrected"] = float(Lf[:H].sum() / B_rate / 60)

    # --------------------------------------------------------- Monte Carlo
    t0 = time.perf_counter()
    mc = model.sample_journeys(model.source_vector(lam), m=args.mc, rng=np.random.default_rng(42))
    jt = mc["journey_time"]
    occ_mc = mc["occupancy_time"] * Lambda
    big = L > 1.0
    R["monte_carlo"] = {
        "journeys": args.mc, "seconds": time.perf_counter() - t0,
        "mean_journey_mc_min": float(jt.mean() / 60),
        "se_min": float(jt.std() / np.sqrt(len(jt)) / 60),
        "rel_err_mean_journey": float(abs(jt.mean() - L.sum() / Lambda) / (L.sum() / Lambda)),
        "median_rel_err_occupancy_L_gt_1": float(np.median(np.abs(occ_mc[big] - L[big]) / L[big])),
        "journey_q50_min": float(np.quantile(jt, 0.5) / 60),
        "journey_q90_min": float(np.quantile(jt, 0.9) / 60),
    }

    # ------------------------------------------------------------ transient
    t_eval = np.linspace(0, 7200, 121)
    Nt = model.transient_total(lam, t_eval)
    frac = Nt / L.sum()
    pd.DataFrame({"t_s": t_eval, "N": Nt, "frac": frac}).to_csv(out / "transient.csv", index=False)
    R["transient"] = {
        "t50_min": float(np.interp(0.5, frac, t_eval) / 60),
        "t90_min": float(np.interp(0.9, frac, t_eval) / 60),
        "t95_min": float(np.interp(0.95, frac, t_eval) / 60),
        "frac_at_120min": float(frac[-1]),
    }
    t_rel = model.relaxation_time()
    R["transient"]["relaxation_time_min"] = t_rel / 60

    # ---------------------------------------------------------- sensitivity
    base_board = F["boarding"]
    sens = []
    grid = [("lam", 0.0), ("lam", 1 / 60), ("mean_leg_m", 2500.0), ("mean_leg_m", 5000.0),
            ("p_transfer", 0.2), ("p_transfer", 0.4)]
    for key, val in grid:
        p = ModelParams(lam=base.lam, mean_leg_m=base.mean_leg_m, p_transfer=base.p_transfer)
        setattr(p, key, val)
        mm = OpenNetworkModel(net, p)
        ll = mm.scale_to_boardings(mm.demand_prior(), B_rate)
        LL = mm.steady_state(ll)
        FF = mm.flows(LL)
        rho = spearmanr(base_board, FF["boarding"]).statistic
        rho_seg = spearmanr(F["segment"], FF["segment"]).statistic
        sens.append({"param": key, "value": val,
                     "mean_journey_min": float(LL.sum() / ll.sum() / 60),
                     "waiting_share": float(LL[:H].sum() / LL.sum()),
                     "spearman_hub_boardings": float(rho),
                     "spearman_segment_flows": float(rho_seg)})
    R["sensitivity"] = sens

    # ----------------------------------------------------------- resilience
    w = shape                                         # OD weights w_o w_d
    origins = np.flatnonzero(w > 0)
    throughput = F["boarding"] + F["alighting"]
    cand = np.argsort(throughput)[::-1][: args.candidates]

    def eff_from(Tm, drop=None):
        inv = np.zeros_like(Tm)
        ok = np.isfinite(Tm) & (Tm > 0)
        inv[ok] = 1.0 / Tm[ok]
        inv[np.arange(len(origins)), origins] = 0.0
        wo = w[origins].copy()
        wd = w.copy()
        if drop is not None:            # restrict to OD pairs not touching drop
            wo[origins == drop] = 0.0
            wd[drop] = 0.0
        num = float(wo @ inv @ wd)
        den = float(wo.sum() * wd.sum() - (wo * wd[origins]).sum())
        return num / den

    t0 = time.perf_counter()
    T0 = model.hub_travel_times(model.travel_graph(), origins)
    E0 = eff_from(T0)
    R["resilience_baseline"] = {"efficiency_per_s": E0,
                                "harmonic_mean_time_min": 1 / E0 / 60}
    rows = []
    for h in cand:
        rec = {"hub": int(h)}
        E_rest0 = eff_from(T0, drop=h)
        for sc in ("closure", "failure"):
            Tm = model.hub_travel_times(model.travel_graph(int(h), sc), origins)
            Tm[:, h] = np.inf
            Tm[origins == h, :] = np.inf
            E_all = eff_from(Tm)
            E_rest = eff_from(Tm, drop=h)
            rec[f"dE_{sc}"] = (E0 - E_all) / E0                 # total loss
            rec[f"dE_net_{sc}"] = (E_rest0 - E_rest) / E_rest0  # loss for other OD pairs
        rows.append(rec)
        log.info("hub %-30s closure %.4f (net %.4f) failure %.4f (net %.4f)",
                 net.hub_name[h], rec["dE_closure"], rec["dE_net_closure"],
                 rec["dE_failure"], rec["dE_net_failure"])
    R["resilience_seconds"] = time.perf_counter() - t0
    res = pd.DataFrame(rows).set_index("hub")

    # ------------------------------------------------------------- outputs
    hub_modes = net.hub_modes()
    hubs = pd.DataFrame({
        "name": net.hub_name, "lat": net.hub_lat, "lon": net.hub_lon,
        "modes": [",".join(MODE_HU.get(m, str(m)) for m in sorted(s)) for s in hub_modes],
        "departures_ph": net.hub_departures() * 3600,
        "origin_rate_ph": lam * 3600,
        "boardings_ph": F["boarding"] * 3600,
        "alightings_ph": F["alighting"] * 3600,
        "waiting_mean": L[:H],
        "mean_wait_min": np.where(F["boarding"] > 0, L[:H] / np.maximum(F["boarding"], 1e-300) / 60,
                                  np.nan),
    })
    hubs = hubs.join(res)
    hubs.index.name = "hub"
    hubs.to_csv(out / "hubs.csv")
    segs = pd.DataFrame({
        "route_id": net.seg_route, "mode": net.seg_mode,
        "lat_i": net.seg_xy[:, 0], "lon_i": net.seg_xy[:, 1],
        "lat_j": net.seg_xy[:, 2], "lon_j": net.seg_xy[:, 3],
        "from_hub": net.seg_from, "to_hub": net.seg_to,
        "departures": net.seg_n, "cv2": net.seg_cv2, "tau_s": net.seg_tau,
        "flow_ph": F["segment"] * 3600, "riding_mean": L[H:],
    })
    segs.to_csv(out / "segments.csv", index=False)

    top_board = hubs.sort_values("boardings_ph", ascending=False).head(10)
    R["top_boarding_hubs"] = top_board[["name", "modes", "boardings_ph", "waiting_mean",
                                        "mean_wait_min"]].to_dict("records")
    for sc in ("closure", "failure"):
        top = hubs.dropna(subset=[f"dE_{sc}"]).sort_values(f"dE_net_{sc}", ascending=False).head(10)
        R[f"top_{sc}"] = top[["name", "modes", "boardings_ph", f"dE_{sc}",
                              f"dE_net_{sc}"]].to_dict("records")
    sub = hubs.dropna(subset=["dE_closure"])
    R["resilience_rank_corr"] = {
        "spearman_net_closure_vs_throughput": float(spearmanr(
            sub["boardings_ph"] + sub["alightings_ph"], sub["dE_net_closure"]).statistic),
        "spearman_net_closure_vs_failure": float(spearmanr(
            sub["dE_net_closure"], sub["dE_net_failure"]).statistic),
    }
    json.dump(R, open(out / "results.json", "w"), indent=2, ensure_ascii=False, default=float)
    log.info("wrote %s", out / "results.json")


if __name__ == "__main__":
    main()
