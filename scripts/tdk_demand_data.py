#!/usr/bin/env python3
"""
scripts/tdk_demand_data.py
==========================
Builds the population-based travel demand used by the TDK scripts
(TDK_DEMAND=population, the default) and stores it in data/cache/demand_population.npz
with diagnostics in data/results/demand.json.

Layers (each falls back when its table is missing; see data/external/manual/README.md):
  * Meta HRSL 30 m population grid, read as a cloud-optimised GeoTIFF window
    and rescaled within the city to the 2022 census total;
  * optional KSH district population, car ownership, census mode share and
    census district OD tables (manual CSV exports);
  * BKK trips by mode for the NNLS calibration of district multipliers.

The OD matrix W is not stored: tdk_common.load_base rebuilds it from the
stored source rates, attraction and deterrence beta (a few seconds).

    python scripts/tdk_demand_data.py
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.stats import spearmanr

from tdk_common import BASE_PARAMS, N_DAY, PHI, geography, load_net  # noqa: E402
from budapest_basemap import eov  # noqa: E402

from bkk.demand_data import (build_access, calibrate, calibrate_access, calibrate_beta,
                             census_od_weights, gravity_production, hub_class, mean_od_time,
                             mode_response, propensity)
from bkk.linemodel import OpenNetworkModel

HRSL_URL = ("https://dataforgood-fb-data.s3.amazonaws.com/hrsl-cogs/hrsl_general/v1.5/"
            "cog_globallat_40_lon_10_general-v1.5.2.tif")
BBOX = (18.92, 47.34, 19.34, 47.62)            # lon_min, lat_min, lon_max, lat_max
CENSUS_TOTAL = 1_685_342                       # KSH Census 2022, Budapest residents
MANUAL = Path("data/external/manual")
CACHE = Path("data/cache")
RES = Path("data/results")
MODES = [3, 0, 1, 11, 109, 4]                  # bus, tram, metro, trolleybus, HEV, boat
# access radius [m] and number of candidate hubs per class (metro and HEV
# stations are walked to from farther away than bus stops)
ACCESS_R = {1: 1000.0, 109: 1000.0, 0: 600.0, 11: 500.0, 3: 500.0, 4: 500.0}
ACCESS_K = {1: 3, 109: 3, 0: 6, 11: 6, 3: 12, 4: 2}
ELL = 250.0                                    # distance decay of access utility [m]
MODE_HU = {0: "villamos", 1: "metró", 3: "busz", 4: "hajó", 11: "trolibusz", 109: "HÉV"}


def load_hrsl() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Population per 30 m cell and cell-centre lon/lat inside BBOX."""
    cache = CACHE / "hrsl_bp.npz"
    if not cache.exists():
        import rasterio
        from rasterio.windows import from_bounds
        os.environ.setdefault("CURL_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")
        with rasterio.open("/vsicurl/" + HRSL_URL) as src:
            win = from_bounds(*BBOX, src.transform)
            a = src.read(1, window=win)
            tr = src.window_transform(win)
        a = np.where(np.isfinite(a) & (a > 0), a, 0.0).astype(np.float32)
        np.savez_compressed(cache, pop=a, transform=np.array(tr)[:6])
    z = np.load(cache)
    pop, (sx, _, x0, _, sy, y0) = z["pop"].astype(float), z["transform"]
    r, c = np.nonzero(pop > 0)
    lon = x0 + (c + 0.5) * sx
    lat = y0 + (r + 0.5) * sy
    return pop[r, c], lon, lat


def read_manual(name: str) -> pd.DataFrame | None:
    p = MANUAL / name
    return pd.read_csv(p, comment="#") if p.exists() else None


def sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def main() -> None:
    net = load_net()
    model = OpenNetworkModel(net, BASE_PARAMS)
    H = net.n_hubs
    B_rate = N_DAY * PHI / 7200.0
    districts, _, label = geography(net)
    names = sorted(districts)
    gid = {n: k for k, n in enumerate(names)}
    OUT = len(names)                                     # group of hubs outside the city
    group = np.array([gid.get(l, OUT) for l in label])
    prov = {}

    # ---------------------------------------------------- population grid
    P_c, lon, lat = load_hrsl()
    cx, cy = eov(lon, lat)
    cell_d = np.full(len(P_c), -1)
    for n, g in districts.items():
        cell_d[shapely.contains_xy(g, cx, cy)] = gid[n]
    inside = cell_d >= 0
    hrsl_city = float(P_c[inside].sum())
    dp = read_manual("district_population.csv")
    if dp is not None:
        for n, v in zip(dp["district"], dp["population"]):
            k = cell_d == gid[n]
            P_c[k] *= v / P_c[k].sum()
        prov["population"] = "KSH district totals (manual CSV) x HRSL pattern"
    else:
        P_c[inside] *= CENSUS_TOTAL / hrsl_city
        prov["population"] = "HRSL pattern scaled to the 2022 census city total"
    hx, hy = eov(net.hub_lon, net.hub_lat)
    access = build_access(np.column_stack([hx, hy]), hub_class(net.hub_modes()),
                          np.column_stack([cx, cy]), P_c, ACCESS_R, ACCESS_K, model.hub_active)
    city_pop = float(P_c[inside].sum())

    # ------------------------------------------------ propensity g_K
    share = read_manual("district_pt_share.csv")
    cars = read_manual("district_cars.csv")
    g_K = None
    if share is not None:
        g_K = propensity(None, observed_share=share.set_index("district").loc[names, "pt_share"])
        prov["propensity"] = "census public-transport commuting share"
    elif cars is not None:
        g_K = propensity(cars.set_index("district").loc[names, "cars_per_1000"].to_numpy())
        prov["propensity"] = "logistic in cars per 1,000 residents"
    else:
        prov["propensity"] = "not available: g = 1 (layer off)"
    g = np.ones(H) if g_K is None else np.r_[g_K, 1.0][group]
    # ------------------------------------------------ calibration targets
    ms = read_manual("bkk_modal_split.csv")
    share_m = dict(zip(ms["route_type"].astype(int), ms["share"]))
    target_share = np.array([share_m.get(m, 0.0) for m in MODES])
    target_share = target_share / target_share.sum()
    prov["calibration_target"] = "BKK trips by mode 2022 (manual CSV, secondary source)"
    M = mode_response(model, MODES)

    # (a) district multipliers: linear, but not identified by the modal split
    P0, _ = access.assign(None, ELL)
    prior0 = model.scale_to_boardings(np.where(model.hub_active, P0 * g, 0.0), B_rate)
    target = np.where(target_share > 0, target_share, 1e-12) * B_rate
    weight = np.where(target_share > 0, 1.0 / target, 0.0)
    cal_free = calibrate(model, prior0, group, OUT + 1, MODES, target, weight=weight,
                         rho=1e-6, bounds=(0.0, np.inf))
    cal_box = calibrate(model, prior0, group, OUT + 1, MODES, target, weight=weight,
                        rho=1e-3, bounds=(0.5, 2.0), rhos=np.array([1e-3]))
    rms = lambda b: float(np.sqrt(np.mean(((b - target) / target)[target_share > 0] ** 2)))
    sv = np.linalg.svd(weight[:, None] * cal_free.G, compute_uv=False)

    # (b) access constants of the logit station choice: identified
    delta, acc_info = calibrate_access(access, M, MODES, target_share, ell_m=ELL, g=g)
    P_h, U = access.assign(delta, ELL)
    U_city = city_pop - float(P_h[group < OUT].sum())
    prior = np.where(model.hub_active, P_h * g, 0.0)
    rates = model.scale_to_boardings(prior, B_rate)

    # ------------------------------------------------ OD weights
    f = net.hub_departures()
    attraction = np.where(model.hub_active, f, 0.0)
    origins = np.arange(H)
    T0 = model.hub_travel_times(model.travel_graph(), origins)
    t_mean = model.mean_journey_time(rates)
    od = read_manual("commuting_od.csv")
    beta = calibrate_beta(rates, attraction, T0, t_mean)
    if od is not None:
        Q = np.zeros((OUT, OUT))
        for o_, d_, q in od[["origin", "destination", "commuters"]].itertuples(index=False):
            Q[gid[o_], gid[d_]] += q
        W = census_od_weights(Q, np.where(group < OUT, group, -1), rates, attraction, T0, beta)
        ipf = {}
        prov["od"] = "census district OD distributed to hubs"
    else:
        W, ipf = gravity_production(rates, attraction, T0, beta), {}
        prov["od"] = "production-constrained gravity (no census OD available)"

    # ------------------------------------------------ diagnostics
    dep_prior = model.scale_to_boardings(model.demand_prior(), B_rate)

    def split(lam):
        F = model.flows(model.steady_state(lam))
        b = pd.Series(F["boarding_segment"]).groupby(net.seg_mode).sum()
        return {MODE_HU[m]: float(b.get(m, 0.0) / b.sum()) for m in MODES}

    act = model.hub_active
    R = {
        "date": str(date.today()),
        "provenance": prov,
        "sources": {"hrsl": HRSL_URL, "hrsl_cache_sha": sha(CACHE / "hrsl_bp.npz"),
                    "modal_split_sha": sha(MANUAL / "bkk_modal_split.csv"),
                    "census_total": CENSUS_TOTAL},
        "population": {"hrsl_city_raw": hrsl_city, "city": city_pop,
                       "scale": city_pop / hrsl_city,
                       "covered_city_share": 1.0 - U_city / city_pop,
                       "hub_catchment_total": float(P_h.sum()), "uncovered_bbox": U},
        "access": {"radius_m": {MODE_HU[k]: v for k, v in ACCESS_R.items()}, "ell_m": ELL,
                   "delta": {MODE_HU[k]: float(v) for k, v in delta.items()},
                   "odds_vs_bus": {MODE_HU[k]: float(np.exp(v)) for k, v in delta.items()},
                   "metres_equivalent": {MODE_HU[k]: float(v * ELL) for k, v in delta.items()},
                   **acc_info},
        "district_nnls": {
            "rms_rel_prior": rms(cal_free.prior_fit),
            "rms_rel_unbounded": rms(cal_free.fitted),
            "rms_rel_box": rms(cal_box.fitted),
            "mu_unbounded_zero": int((cal_free.mu < 1e-9).sum()),
            "mu_unbounded_max": float(cal_free.mu.max()),
            "mu_box_at_bounds": int(((cal_box.mu <= 0.5 + 1e-9) | (cal_box.mu >= 2 - 1e-9)).sum()),
            "n_groups": OUT + 1, "rank_G": int((sv > sv[0] * 1e-10).sum()),
            "singular_values": sv.tolist(),
            "metro_share_unbounded": float(cal_free.fitted[2] / cal_free.fitted[:5].sum()),
            "rho_path": cal_free.rho_path,
        },
        "target_share": {MODE_HU[m]: float(t) for m, t in zip(MODES, target_share)},
        "modal_split": {"departures_prior": split(dep_prior), "population_delta0": split(prior0),
                        "calibrated": split(rates)},
        "spearman_rates_vs_departures": float(spearmanr(rates[act], dep_prior[act]).statistic),
        "spearman_delta0_vs_departures": float(spearmanr(prior0[act], dep_prior[act]).statistic),
        "outside_share_of_journeys": float(rates[group == OUT].sum() / rates.sum()),
        "od": {"beta_per_s": beta, "beta_per_min": beta * 60, "target_mean_min": t_mean / 60,
               "mean_od_min": mean_od_time(W, T0) / 60, **ipf},
        "district_journeys_share": {n: float(rates[group == gid[n]].sum() / rates.sum())
                                    for n in names},
        "district_population": {n: float(P_c[cell_d == gid[n]].sum()) for n in names},
    }
    np.savez_compressed(CACHE / "demand_population.npz", rates=rates, attraction=attraction,
                        beta=beta, P_h=P_h, group=group, prior0=prior0,
                        delta=np.array([delta[k] for k in sorted(delta)]))
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(RES / "demand.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: R[k] for k in ("provenance", "population", "access", "modal_split", "od",
                                        "spearman_rates_vs_departures")},
                     indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in R["district_nnls"].items() if k != "rho_path"}, indent=1))


if __name__ == "__main__":
    main()
