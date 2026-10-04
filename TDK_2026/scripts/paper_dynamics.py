#!/usr/bin/env python3
"""Supplementary computations for the paper: network statistics, KFE
trajectories, SSA/tau-leap ensembles against the KFE mean, and the lambda
sweep.  Writes JSON to data/paper_dynamics.json."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bkk import (DemandPrior, GeneratorBuilder, GTFSLoader, GillespieSSA,  # noqa: E402
                 KFESolver, NetworkBuilder, TauLeap, allocate_modal_population)
from bkk.simulate import _global_layout  # noqa: E402

logging.basicConfig(level=logging.WARNING)
OUT: dict = {}


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=str(REPO / "data" / "budapest_gtfs.zip"))
    args = ap.parse_args()
    t_start = time.time()
    loader = GTFSLoader(cache_dir=str(REPO / "data"))
    loader.load(args.zip)
    feed = loader.parse(weekday="monday")
    fi = feed.feed_info
    OUT["feed_info"] = {c: str(fi[c].iloc[0]) for c in fi.columns}
    OUT["raw"] = dict(stops=len(feed.stops), routes=len(feed.routes),
                      trips=len(feed.trips), stop_times=len(feed.stop_times),
                      shapes=len(feed.shapes))
    sid2name = dict(zip(feed.stops["stop_id"].astype(str), feed.stops["stop_name"]))

    net = NetworkBuilder().build(feed, peak_window=("07:00:00", "09:00:00"))
    gens = GeneratorBuilder(lambda_v=1.0 / 300.0).build_all(net)

    # ---------------- network statistics per mode
    modes = {}
    for rt, mg in net.modal.items():
        n, E = mg.n_stops, mg.n_edges
        mu = mg.mu_arr
        cost = mg.cost_arr
        modes[rt] = dict(
            n=n, E=E, fill=E / n**2, mean_mu=float(mu.mean()),
            mean_mu_pos=float(mu[mu > 0].mean()) if (mu > 0).any() else 0.0,
            median_headway_s=float(np.median(1 / mu[mu > 0])) if (mu > 0).any() else None,
            median_cost_s=float(np.median(cost)),
            outdeg_mean=float(E / n),
            q_rowsum=gens[rt].validate()["max_Q_rowsum_abs"],
            p_rowdev=gens[rt].validate()["max_P_rowsum_dev"],
        )
    OUT["modes"] = modes
    OUT["n_global"] = int(len(net.all_stop_ids))
    OUT["R"] = int(sum(m["E"] for m in modes.values()))

    # ---------------- prior
    N0 = DemandPrior().e1_service_proxy(net)
    N0_modal = allocate_modal_population(gens, N0, net.all_stop_ids)
    ids = net.all_stop_ids
    served = {sid: [] for sid in ids}
    fglob = np.zeros(len(ids))
    gidx = {s: k for k, s in enumerate(ids)}
    for rt, mg in net.modal.items():
        for k, sid in enumerate(mg.stop_ids):
            served[sid].append(rt)
            fglob[gidx[sid]] += mg.mu_arr[k] * 3600.0
    top = np.argsort(N0)[::-1][:10]
    OUT["prior_top10"] = [dict(stop_id=str(ids[k]), name=sid2name.get(str(ids[k]), str(ids[k])),
                               modes=sorted(served[ids[k]]), f_per_h=float(fglob[k]),
                               N0=float(N0[k])) for k in top]
    OUT["prior_total"] = float(N0.sum())
    OUT["prior_max"] = float(N0.max())
    OUT["prior_gini"] = gini(N0)
    OUT["prior_top50_share"] = float(np.sort(N0)[::-1][:50].sum() / N0.sum())

    # ---------------- KFE trajectory (global coordinates)
    stop_ids_g, l2g = _global_layout(gens)
    T = 900.0
    t_eval = np.array([0, 60, 120, 300, 600, 900], float)
    kfe = KFESolver(method="expm").solve(gens, N0_modal, T=T, n_eval=16)
    tgrid = kfe[next(iter(kfe))].t_eval
    Ng = np.zeros((len(tgrid), len(stop_ids_g)))
    for rt, res in kfe.items():
        Ng[:, l2g[rt]] += res.N_hist
    OUT["kfe_cons_err"] = float(max(r.conservation_error() for r in kfe.values()))
    # evaluate at requested times by an explicit expm run
    kfe2 = {}
    from scipy.sparse.linalg import expm_multiply
    Nt = np.zeros((len(t_eval), len(stop_ids_g)))
    for rt, mg in gens.items():
        QT = mg.Q.T.tocsr()
        for a, t in enumerate(t_eval):
            v = N0_modal[rt] if t == 0 else expm_multiply(QT * t, N0_modal[rt])
            Nt[a, l2g[rt]] += v
    g_order = np.argsort(Nt[0])[::-1]
    top4 = g_order[:5]
    top50 = g_order[:50]
    OUT["kfe_times"] = t_eval.tolist()
    OUT["kfe_top"] = [dict(stop_id=str(stop_ids_g[k]), name=sid2name.get(str(stop_ids_g[k]), ""),
                           N=[float(x) for x in Nt[:, k]]) for k in top4]
    OUT["kfe_top50_sum"] = [float(x) for x in Nt[:, top50].sum(axis=1)]
    OUT["kfe_total"] = [float(x) for x in Nt.sum(axis=1)]
    OUT["kfe_gini"] = [gini(Nt[a]) for a in range(len(t_eval))]
    # largest stops at T
    endtop = np.argsort(Nt[-1])[::-1][:5]
    OUT["kfe_end_top"] = [dict(stop_id=str(stop_ids_g[k]), name=sid2name.get(str(stop_ids_g[k]), ""),
                               N0=float(Nt[0, k]), NT=float(Nt[-1, k])) for k in endtop]
    # relaxation: distance to the T=3600 state
    OUT["relax"] = {}
    for rt, mg in gens.items():
        mu = mg.mu_arr
        OUT["relax"][str(rt)] = float(np.median(1 / mu[mu > 0])) if (mu > 0).any() else None

    # ---------------- exact SSA ensemble on the metro sub-network vs KFE
    metro = {1: gens[1]}
    N0m = {1: N0_modal[1]}
    a0 = float((N0m[1] * gens[1].mu_arr).sum()) if gens[1].mu_arr is not None else None
    OUT["metro_a0"] = a0
    T_ssa = 300.0
    exp_events = a0 * T_ssa
    OUT["metro_T_ssa"] = T_ssa
    OUT["metro_expected_events"] = exp_events
    M = 20
    ssa = GillespieSSA(max_events=int(exp_events * 1.5) + 1000, record_every=10**9)
    finals, completed, nev = [], [], []
    t0 = time.time()
    for s in range(M):
        ssa.rng = np.random.default_rng(1000 + s)
        r = ssa.run(metro, N0m, T=T_ssa)
        finals.append(r.N_final)
        completed.append(bool(r.metadata["completed_horizon"]))
        nev.append(r.metadata["n_events"])
    OUT["metro_ssa_time_s"] = time.time() - t0
    finals = np.array(finals)
    ssa_ids = r.metadata["stop_ids"]
    kfe_m = expm_multiply(gens[1].Q.T.tocsr() * T_ssa, N0m[1])
    # map metro local -> ssa global (identical ordering since single mode sorted)
    loc = {s: k for k, s in enumerate(gens[1].stop_ids)}
    kfe_on_ssa = np.array([kfe_m[loc[s]] for s in ssa_ids])
    mean = finals.mean(0)
    q05, q95 = np.percentile(finals, 5, 0), np.percentile(finals, 95, 0)
    sd = finals.std(0, ddof=1)
    z = (mean - kfe_on_ssa) / (sd / np.sqrt(M) + 1e-12)
    order = np.argsort(kfe_on_ssa)[::-1][:6]
    OUT["metro_ssa"] = dict(
        M=M, all_completed=all(completed), mean_events=float(np.mean(nev)),
        rows=[dict(name=sid2name.get(str(ssa_ids[k]), str(ssa_ids[k])), mean=float(mean[k]),
                   q05=float(q05[k]), q95=float(q95[k]), kfe=float(kfe_on_ssa[k]),
                   cv=float(sd[k] / mean[k]) if mean[k] > 0 else None,
                   poisson_cv=float(1 / np.sqrt(kfe_on_ssa[k])) if kfe_on_ssa[k] > 0 else None)
              for k in order],
        max_abs_z=float(np.max(np.abs(z))),
        frac_within_2se=float(np.mean(np.abs(z) < 2)),
        rel_l1=float(np.abs(mean - kfe_on_ssa).sum() / kfe_on_ssa.sum()),
    )

    # ---------------- tau-leap ensemble on the full network vs KFE at T
    Mtl = 10
    tl_final = []
    tl_steps = []
    for s in range(Mtl):
        tl = TauLeap(tau=1.0, rng_seed=2000 + s).run(gens, N0_modal, T=T, n_eval=2)
        tl_final.append(tl.N_final)
        tl_steps.append(tl.metadata["n_steps"])
        tl_ids = tl.metadata.get("stop_ids", stop_ids_g)
    tl_final = np.array(tl_final)
    kfeT = Nt[-1]
    m_tl = tl_final.mean(0)
    OUT["tauleap"] = dict(M=Mtl, mean_steps=float(np.mean(tl_steps)),
                          cons_err=float(max(abs(f.sum() - N0.sum()) for f in tl_final)),
                          rel_l1=float(np.abs(m_tl - kfeT).sum() / kfeT.sum()),
                          top=[dict(name=sid2name.get(str(stop_ids_g[k]), ""), kfe=float(kfeT[k]),
                                    mean=float(m_tl[k]), q05=float(np.percentile(tl_final[:, k], 5)),
                                    q95=float(np.percentile(tl_final[:, k], 95)))
                               for k in np.argsort(kfeT)[::-1][:5]])

    # ---------------- lambda sweep on the bus sub-network
    from scipy.sparse.linalg import eigs  # noqa: F401
    lams = np.logspace(-5, -1, 25)
    sweep = []
    mg_bus = net.modal[3]
    for lam in lams:
        g = GeneratorBuilder(lambda_v=float(lam)).build(mg_bus)
        P = g.P.tocsr()
        # row entropy averaged over rows with >=1 outgoing edge
        H = []
        for i in range(P.shape[0]):
            row = P.data[P.indptr[i]:P.indptr[i + 1]]
            row = row[row > 0]
            if len(row):
                H.append(float(-(row * np.log(row)).sum()))
        # stationary: Cesaro average of uniform start (400 steps)
        p = np.full(P.shape[0], 1 / P.shape[0])
        acc = np.zeros_like(p)
        PT = P.T.tocsr()
        for _ in range(400):
            p = PT @ p
            s = p.sum()
            if s > 0:
                p = p / s
            acc += p
        sweep.append(dict(lam=float(lam), H=float(np.mean(H)), gini=gini(acc)))
    OUT["lambda_sweep"] = sweep
    OUT["runtime_s"] = time.time() - t_start

    (REPO / "data" / "paper_dynamics.json").write_text(json.dumps(OUT, indent=1, default=str),
                                                       encoding="utf-8")
    print("done", OUT["runtime_s"])


if __name__ == "__main__":
    main()
