#!/usr/bin/env python3
"""
scripts/tdk_tex_numbers.py
==========================
Turn data/results/{results.json,hubs.csv} into LaTeX macros and the
criticality table, so the paper never contains hand-copied numbers.

    tdk/generated/numbers.tex
    tdk/generated/table_critical.tex
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RES = Path("data/results")
OUT = Path("tdk/generated")


def hu(x: float, nd: int = 1) -> str:
    """Hungarian number format: decimal comma, thin-space thousands."""
    s = f"{x:,.{nd}f}".replace(",", "X").replace(".", "{,}").replace("X", "\\,")
    return s


def hui(x: float) -> str:
    return hu(round(x), 0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    R = json.load(open(RES / "results.json"))
    n, b, mc, tr = R["network"], R["base"], R["monte_carlo"], R["transient"]
    sens = {(s["param"], round(s["value"], 6)): s for s in R["sensitivity"]}
    m = {}
    m["FEEDVER"] = R["feed"]["feed_version"].replace("_", "\\_")
    m["NTRIPS"] = hui(R["calendar"]["exact_trips"])
    m["NUNION"] = hui(R["calendar"]["weekday_union_trips"])
    m["INFL"] = hu(R["calendar"]["inflation"], 2)
    m["NHUBS"] = hui(n["hubs"])
    m["NSEGS"] = hui(n["segments"])
    m["NSTATES"] = hui(n["states"])
    m["NTRIPSEG"] = hui(n["trip_segments"])
    m["MEDCVSQ"] = hu(n["median_cv2"], 3)
    m["SHARECVLOW"] = hu(100 * n["share_cv2_below_0.1"], 0)
    m["MEDCVSQMETRO"] = hu(n["modes"]["metró"]["median_cv2"], 3)
    m["MEDTAU"] = hui(n["median_tau_s"])
    m["MEDDIST"] = hui(n["median_dist_m"])
    m["NSINGLE"] = hui(n["single_departure_segments"])
    for mode, key in (("busz", "BUS"), ("villamos", "TRAM"), ("trolibusz", "TROLLEY"),
                      ("metró", "METRO"), ("HÉV", "HEV"), ("hajó", "FERRY")):
        m[f"SEG{key}"] = hui(n["modes"][mode]["segments"])
        m[f"DEP{key}"] = hui(n["modes"][mode]["departures_window"])
    m["THETAF"] = hu(R["hazard"]["mean_theta_over_f"], 2)
    m["BOARDPH"] = hui(b["boardings_per_s"] * 3600)
    m["JOURNPH"] = hui(b["journeys_per_s"] * 3600)
    m["LEGS"] = hu(b["legs_per_journey"], 2)
    m["INSYS"] = hui(b["in_system"])
    m["WAITING"] = hui(b["waiting"])
    m["RIDING"] = hui(b["riding"])
    m["WAITSHARE"] = hu(100 * b["waiting"] / b["in_system"], 1)
    m["EJOURNEY"] = hu(b["mean_journey_min"], 1)
    m["WAITBOARD"] = hu(b["mean_wait_per_boarding_min"], 2)
    m["WAITBOARDOLD"] = hu(R["hazard"]["wait_per_boarding_min_uncorrected"], 2)
    m["RIDELEG"] = hu(b["mean_ride_per_leg_min"], 1)
    m["MAXSEGFLOW"] = hui(b["max_segment_flow_pph"])
    m["MAXHUBBOARD"] = hui(b["busiest_hub_boardings_pph"])
    m["MCN"] = hui(mc["journeys"])
    m["MCSEC"] = hu(mc["seconds"], 2)
    m["MCMEAN"] = hu(mc["mean_journey_mc_min"], 2)
    m["MCSE"] = hu(mc["se_min"], 3)
    m["MCRELERR"] = hu(100 * mc["rel_err_mean_journey"], 2)
    m["MCOCCERR"] = hu(100 * mc["median_rel_err_occupancy_L_gt_1"], 1)
    m["EJOURNEYTWO"] = hu(b["mean_journey_min"], 2)
    m["JQFIFTY"] = hu(mc["journey_q50_min"], 1)
    m["JQNINETY"] = hu(mc["journey_q90_min"], 1)
    m["TFIFTY"] = hu(tr["t50_min"], 0)
    m["TNINETY"] = hu(tr["t90_min"], 0)
    m["TNINETYFIVE"] = hu(tr["t95_min"], 0)
    m["TRELAX"] = hu(tr["relaxation_time_min"], 0)
    s0, s60 = sens[("lam", 0.0)], sens[("lam", round(1 / 60, 6))]
    m["SLAMZEROJ"] = hu(s0["mean_journey_min"], 1)
    m["SLAMSIXTYJ"] = hu(s60["mean_journey_min"], 1)
    m["SLAMSIXTYW"] = hu(100 * s60["waiting_share"], 1)
    m["SLAMZEROW"] = hu(100 * s0["waiting_share"], 1)
    m["SLAMSIXTYRHO"] = hu(s60["spearman_segment_flows"], 3)
    d25, d50 = sens[("mean_leg_m", 2500.0)], sens[("mean_leg_m", 5000.0)]
    m["SDLOW"] = hu(d25["mean_journey_min"], 1)
    m["SDHIGH"] = hu(d50["mean_journey_min"], 1)
    p2, p4 = sens[("p_transfer", 0.2)], sens[("p_transfer", 0.4)]
    m["SPLOW"] = hu(p2["mean_journey_min"], 1)
    m["SPHIGH"] = hu(p4["mean_journey_min"], 1)
    m["SMINRHOHUB"] = hu(min(s["spearman_hub_boardings"] for s in R["sensitivity"]), 3)
    m["SMINRHOSEG"] = hu(min(s["spearman_segment_flows"] for s in R["sensitivity"]), 3)
    rb = R["resilience_baseline"]
    m["EZERO"] = hu(rb["efficiency_per_s"] * 1e4, 2)
    m["EHARM"] = hu(rb["harmonic_mean_time_min"], 1)
    m["RESSEC"] = hu(R["resilience_seconds"] / 60, 0)
    rc = R["resilience_rank_corr"]
    m["RHOTHRU"] = hu(rc["spearman_net_closure_vs_throughput"], 2)
    m["RHOCF"] = hu(rc["spearman_net_closure_vs_failure"], 2)
    tf = R["top_failure"]
    for k, rec in enumerate(tf[:3]):
        m[f"FAILNAME{'ABC'[k]}"] = rec["name"]
        m[f"FAILNET{'ABC'[k]}"] = hu(100 * rec["dE_net_failure"], 1)
    ch = R["choice"]
    m["CHACTIVE"] = hui(ch["active_hubs"])
    m["CHMULTI"] = hui(ch["hubs_multi_segment"])
    m["CHMULTIB"] = hu(100 * ch["boarding_share_multi_segment"], 1)
    m["CHLINEB"] = hu(100 * ch["boarding_share_multi_line"], 1)
    m["CHMODE"] = hui(ch["hubs_multi_mode"])
    m["CHMODEB"] = hu(100 * ch["boarding_share_multi_mode"], 1)
    m["DEAKZERO"] = hu(100 * ch["deak_metro_share_lambda0"], 0)
    m["DEAKBASE"] = hu(100 * ch["deak_metro_share_base"], 0)
    m["SUPPLYSHARE"] = hu(100 * R["supply_share_window"], 1)
    m["NDAY"] = hu(R["base"]["params"]["n_day"] / 1e6, 1)
    if "modal_split" in R:
        ms = R["modal_split"]
        for mode, key in (("metró", "METRO"), ("busz", "BUS"), ("villamos", "TRAM"),
                          ("trolibusz", "TROLLEY"), ("HÉV", "HEV")):
            m[f"MSB{key}"] = hu(100 * ms[mode]["boardings_share"], 1)
            m[f"MSK{key}"] = hu(100 * ms[mode]["pkm_share"], 1)

    scen = RES / "scenarios.json"
    if scen.exists():
        S = json.load(open(scen))
        sg = S["street_graph"]
        m["SGNODES"] = hui(sg["nodes"])
        m["SGALLKM"] = hui(sg["all_km"])
        m["SGMAINKM"] = hui(sg["main_km"])
        m["SGTRAMHUBS"] = hui(sg["tram_station_hubs"])
        m["SGBRIDGE"] = hui(sg["bridge_edges_removed"])
        if S.get("example_route"):
            m["EXROUTE"] = " -- ".join(S["example_route"]["stations"])
            m["EXLEN"] = hu(S["example_route"]["length_km"], 2)
        m["NCAND"] = hui(S["n_candidates"])
        m["NCANDM"] = hui(S["n_metro"])
        m["NCANDV"] = hui(S["n_tram"])
        m["NEND"] = hui(S["endpoints"])
        m["COSTRATIO"] = hu(S["cost_ratio"], 0)
        m["PKGGAIN"] = hu(100 * S["package_gain_exact"], 2)
        m["PKGGAINMP"] = hu(100 * S["package_gain_minplus"], 2)
        for k, rec in enumerate(S["selected"]):
            L = "ABCDE"[k]
            m[f"UNAME{L}"] = f"{rec['from']} -- {rec['to']}"
            m[f"UMODE{L}"] = rec["mode"]
            m[f"UGAIN{L}"] = hu(100 * rec["marginal_gain"], 2)
            m[f"ULEN{L}"] = hu(rec["length_km"], 1)
            m[f"UBOARD{L}"] = hui(rec["boardings_ph"])
            m[f"USECT{L}"] = hui(rec["max_section_ph"])
        for v in ("A", "B"):
            b5 = S["m5"][v]
            m[f"MFGAIN{v}"] = hu(100 * b5["gain"], 2)
            m[f"MFLEN{v}"] = hu(b5["length_km"], 1)
            m[f"MFBOARD{v}"] = hui(b5["boardings_ph"])
            m[f"MFSECT{v}"] = hui(b5["max_section_ph"])
            m[f"MFRANK{v}"] = hui(S["m5_rank_gain_per_cost"][v])
            m[f"MFGPC{v}"] = hu(1e4 * b5["gain_per_cost"], 2)
        best = S["selected"][0]
        m["UGPCA"] = hu(1e4 * best["gain_per_cost"], 2)
        det = S["selection_detail"]
        r5m = [r for r in det["ratio_5"] if r["mode"] == "metró"]
        for k, r in enumerate(r5m[:2]):
            L = "AB"[k]
            m[f"RFIVENAME{L}"] = f"{r['from']} -- {r['to']}"
            m[f"RFIVEGAIN{L}"] = hu(100 * r["marginal_gain"], 2)
            m[f"RFIVELEN{L}"] = hu(r["length_km"], 1)
        m["RFIVECUM"] = hu(100 * det["ratio_5"][-1]["cumulative_gain"], 2)
        for k, r in enumerate(det["pure_gain"][:3]):
            L = "ABC"[k]
            m[f"PGNAME{L}"] = f"{r['from']} -- {r['to']}"
            m[f"PGGAIN{L}"] = hu(100 * r["marginal_gain"], 2)
            m[f"PGLEN{L}"] = hu(r["length_km"], 1)
        m["PGCUM"] = hu(100 * det["pure_gain"][-1]["cumulative_gain"], 2)
        ov = S["overlap_with_base"]
        m["OVFIVE"] = hui(ov["ratio_5"])
        m["OVTWENTY"] = hui(ov["ratio_20"])
        m["OVPURE"] = hui(ov["pure_gain"])
        rows = []
        for rec in S["selected"]:
            rows.append(f"{rec['name']} & {rec['mode']} & {rec['from']} -- {rec['to']} & "
                        f"{len(rec['stations'])} & {hu(rec['length_km'], 1)} & "
                        f"{hu(100 * rec['marginal_gain'], 2)} & "
                        f"{hu(1e4 * rec['gain_per_cost'], 2)} & {hui(rec['boardings_ph'])} \\\\")
        rows.append("\\midrule")
        for v, lab in (("A", "M5 (A: Lehel tér)"), ("B", "M5 (B: Nyugati pu.)")):
            b5 = S["m5"][v]
            rows.append(f"{lab} & metró & Margit híd -- Közvágóhíd & {len(b5['stations'])} & "
                        f"{hu(b5['length_km'], 1)} & {hu(100 * b5['gain'], 2)} & "
                        f"{hu(1e4 * b5['gain_per_cost'], 2)} & {hui(b5['boardings_ph'])} \\\\")
        with open(OUT / "table_lines.tex", "w") as f:
            f.write("% generated by scripts/tdk_tex_numbers.py -- do not edit\n")
            f.write("\n".join(rows) + "\n")

    plans = RES / "plans.json"
    if plans.exists():
        P = json.load(open(plans))
        m["ESZERO"] = hu(P["baseline"]["ES90_min"], 1)
        tag = {"bajcsy": "BAJ", "budai_fonodo2": "BF", "budafoki": "BU", "m5": "MF"}
        for k, t in tag.items():
            r = P["plans"][k]
            m[f"PL{t}GAIN"] = hu(100 * r["gain"], 2)
            m[f"PL{t}GAINTHREE"] = hu(100 * r["gain"], 3)
            m[f"PL{t}BOARD"] = hui(r["boardings_ph"])
            m[f"PL{t}SECT"] = hui(r["max_section_ph"])
            m[f"PL{t}LEN"] = hu(r["length_km"], 1)
            m[f"PL{t}ES"] = hu(r["ES90_min"], 1)
            m[f"PL{t}NST"] = hui(len(r["stations"]))
            for d, dv in r["district"].items():
                m[f"PL{t}D{d.split('.')[0]}"] = hu(100 * dv, 2)
        m["PLBUSTATIONS"] = " -- ".join(P["plans"]["budafoki"]["stations"])
        m["PLMFESDIFF"] = hu(P["baseline"]["ES90_min"] - P["plans"]["m5"]["ES90_min"], 1)
        pk = P["package"]
        m["PKGPLGAIN"] = hu(100 * pk["gain"], 2)
        m["PKGPLSUM"] = hu(100 * pk["sum_of_parts"], 2)
        m["PKGPLES"] = hu(pk["ES90_min"], 1)
        for d, dv in pk["district"].items():
            m[f"PKGD{d.split('.')[0]}"] = hu(100 * dv, 2)
        ap = P["after_plans"]
        m["AFTERGAIN"] = hu(100 * ap["gain_total_vs_E0"], 2)
        m["AFTERES"] = hu(ap["ES90_min"], 1)
        for d, dv in ap["district"].items():
            m[f"AFTD{d.split('.')[0]}"] = hu(100 * dv, 2)
        for k, r in enumerate(ap["selected"]):
            L = "ABCDE"[k]
            m[f"JNAME{L}"] = f"{r['from']} -- {r['to']}"
            m[f"JLEN{L}"] = hu(r["length_km"], 1)
            m[f"JGAIN{L}"] = hu(100 * r["marginal_gain"], 2)
            m[f"JBOARD{L}"] = hui(r["boardings_ph"])
        for r in P["interaction"]:
            m["INTU" + "ABCDE"[int(r["name"][1:]) - 1]] = hu(r["ratio"], 2)
        rows = []
        lab = {"bajcsy": "Bajcsy-villamos", "budai_fonodo2": "Budai fonódó II.",
               "budafoki": "Budafoki úti villamos", "m5": "M5 (B)"}
        ends = {"bajcsy": "Lehel tér -- Deák Ferenc tér",
                "budai_fonodo2": "Szent Gellért tér -- Dombóvári út",
                "budafoki": "Dombóvári út -- Savoya Park",
                "m5": "Margit híd -- Közvágóhíd"}
        for k in ("bajcsy", "budai_fonodo2", "budafoki", "m5"):
            r = P["plans"][k]
            rows.append(f"{lab[k]} & {r['mode']} & {ends[k]} & {len(r['stations'])} & "
                        f"{hu(r['length_km'], 1)} & {hu(100 * r['gain'], 2)} & "
                        f"{hu(1e4 * r['gain_per_cost'], 2)} & {hui(r['boardings_ph'])} \\\\")
        rows.append(f"\\textit{{A négy terv együtt}} & & & & & {hu(100 * pk['gain'], 2)} & & \\\\")
        rows.append("\\midrule")
        for r in ap["selected"]:
            rows.append(f"{r['name']} & {r['mode']} & {r['from']} -- {r['to']} & "
                        f"{len(r['stations'])} & {hu(r['length_km'], 1)} & "
                        f"{hu(100 * r['marginal_gain'], 2)} & {hu(1e4 * r['gain_per_cost'], 2)} & "
                        f"{hui(r['boardings_ph'])} \\\\")
        with open(OUT / "table_plans.tex", "w") as f:
            f.write("% generated by scripts/tdk_tex_numbers.py -- do not edit\n")
            f.write("\n".join(rows) + "\n")
    dem = RES / "plans_demand.json"
    if dem.exists():
        D = json.load(open(dem))
        m["DEMRES"] = hui(D["new_residents"])
        m["DEMEXTRA"] = hui(D["extra_boardings_ph"])
        m["DEMCATCH"] = hui(D["catchment_hubs"])
        m["DEMBTODAY"] = hui(D["scenarios"]["today"]["boardings_ph"])
        m["DEMBFUT"] = hui(D["scenarios"]["2030"]["boardings_ph"])
        m["DEMGTODAY"] = hu(100 * D["scenarios"]["today"]["gain"], 3)
        m["DEMGFUT"] = hu(100 * D["scenarios"]["2030"]["gain"], 3)
        m["DEMBRATIO"] = hu(D["boardings_ratio"], 1)
        m["DEMGRATIO"] = hu(D["gain_ratio"], 1)

    # ------------------------------------------------ population-based demand
    dj = RES / "demand.json"
    if dj.exists():
        Dm = json.load(open(dj))
        po, ac, nn, od = Dm["population"], Dm["access"], Dm["district_nnls"], Dm["od"]
        m["POPCITY"] = hui(po["city"])
        m["POPHRSL"] = hui(po["hrsl_city_raw"])
        m["POPSCALE"] = hu(po["scale"], 3)
        m["POPCOVER"] = hu(100 * po["covered_city_share"], 1)
        for k, t in (("metró", "METRO"), ("HÉV", "HEV"), ("villamos", "TRAM"),
                     ("trolibusz", "TROLLEY")):
            m[f"DELTA{t}"] = hu(ac["delta"][k], 2)
            m[f"DELTAM{t}"] = hui(ac["metres_equivalent"][k])
            m[f"DELTAODDS{t}"] = hu(ac["odds_vs_bus"][k], 1)
        m["NNLSGROUPS"] = hui(nn["n_groups"])
        m["NNLSRANK"] = hui(nn["rank_G"])
        m["NNLSRMSPRIOR"] = hu(100 * nn["rms_rel_prior"], 0)
        m["NNLSRMSFREE"] = hu(100 * nn["rms_rel_unbounded"], 0)
        m["NNLSRMSBOX"] = hu(100 * nn["rms_rel_box"], 0)
        m["NNLSZERO"] = hui(nn["mu_unbounded_zero"])
        m["NNLSMAX"] = hu(nn["mu_unbounded_max"], 1)
        m["NNLSBOXB"] = hui(nn["mu_box_at_bounds"])
        m["NNLSMETRO"] = hu(100 * nn["metro_share_unbounded"], 1)
        m["GRAVBETA"] = hu(od["beta_per_min"], 3)
        m["GRAVMEAN"] = hu(od["target_mean_min"], 1)
        m["SPEARDEP"] = hu(Dm["spearman_rates_vs_departures"], 2)
        m["OUTSHARE"] = hu(100 * Dm["outside_share_of_journeys"], 1)
        ms = Dm["modal_split"]
        tg = Dm["target_share"]
        short = (("busz", "BUS"), ("villamos", "TRAM"), ("metró", "METRO"),
                 ("trolibusz", "TROLLEY"), ("HÉV", "HEV"))
        for k, t in short:
            m[f"TGT{t}"] = hu(100 * tg[k], 1)
            m[f"DEPMS{t}"] = hu(100 * ms["departures_prior"][k], 1)
            m[f"POPMS{t}"] = hu(100 * ms["population_delta0"][k], 1)
        rows = []
        for lab, src in (("BKK, 2022 \\citep{bkkmodal}", tg),
                         ("indulásarányos igény", ms["departures_prior"]),
                         ("népesség, $\\delta=0$", ms["population_delta0"]),
                         ("népesség, kalibrált $\\delta$", ms["calibrated"])):
            rows.append(lab + " & " + " & ".join(hu(100 * src[k], 1) for k, _ in short) + " \\\\")
        with open(OUT / "table_modal.tex", "w") as f:
            f.write("% generated by scripts/tdk_tex_numbers.py -- do not edit\n")
            f.write("\n".join(rows) + "\n")
    if plans.exists() and "robustness" in P:
        tag = {"bajcsy": "BAJ", "budai_fonodo2": "BF", "budafoki": "BU", "m5": "MF",
               "package": "PKG"}
        for v, vt in (("departures", "DEP"), ("population_gravity", "GRAV")):
            rb = P["robustness"].get(v)
            if rb is None:
                continue
            m[f"RB{vt}ESZERO"] = hu(rb["baseline_ES90_min"], 1)
            for k, t in tag.items():
                m[f"RB{vt}{t}GAIN"] = hu(100 * rb[k]["gain"], 2)
                m[f"RB{vt}{t}ES"] = hu(rb[k]["ES90_min"], 1)
    old = RES / "departures" / "plans.json"
    if old.exists():
        Po = json.load(open(old))
        ap_o = Po["after_plans"]["selected"]
        m["OLDJNAMES"] = "; ".join(f"{r['from']} -- {r['to']}" for r in ap_o)
        m["OLDMFGAIN"] = hu(100 * Po["plans"]["m5"]["gain"], 2)
        new_pairs = {frozenset((r["from"], r["to"])) for r in P["after_plans"]["selected"]}
        m["JOVERLAP"] = hui(sum(frozenset((r["from"], r["to"])) in new_pairs for r in ap_o))

    with open(OUT / "numbers.tex", "w") as f:
        f.write("% generated by scripts/tdk_tex_numbers.py -- do not edit\n")
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    hubs = pd.read_csv(RES / "hubs.csv")
    top = hubs.dropna(subset=["dE_net_closure"]).sort_values("dE_net_closure",
                                                             ascending=False).head(10)
    short = {"villamos": "V", "metró": "M", "busz": "B", "trolibusz": "T", "HÉV": "H",
             "hajó": "Hj"}
    rows = []
    for r, (_, h) in enumerate(top.iterrows(), start=1):
        modes = "".join(short[x] for x in str(h.modes).split(","))
        rows.append(f"{r} & {h['name']} & {modes} & {hui(h.boardings_ph)} & "
                    f"{hu(100 * h.dE_closure, 2)} & {hu(100 * h.dE_net_closure, 2)} & "
                    f"{hu(100 * h.dE_net_failure, 2)} \\\\")
    with open(OUT / "table_critical.tex", "w") as f:
        f.write("% generated by scripts/tdk_tex_numbers.py -- do not edit\n")
        f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
