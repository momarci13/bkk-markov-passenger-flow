#!/usr/bin/env python3
"""
scripts/run_pipeline.py
=======================
Full BKK pipeline:
  1. Download GTFS from bkk.hu/gtfs/budapest_gtfs.zip
  2. Parse and filter to Monday weekday service
  3. Build sparse modal subgraphs and cost matrices
  4. Assemble CTMC generator Q^v for each mode
  5. Estimate initial passenger counts (E1 + E2)
  6. Solve KFE over 15-minute horizon
  7. Run τ-leaping stochastic simulation
  8. Compute network resilience (top-100 stops)
  9. Print summary table

Usage
-----
    python scripts/run_pipeline.py [--zip data/budapest_gtfs.zip]
                                   [--lambda-v 0.00333]
                                   [--horizon 900]
                                   [--tau 1.0]
                                   [--no-resilience]
"""
import argparse
import logging
import time
import warnings
from pathlib import Path

import numpy as np

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Avoid numba JIT warmup noise ──────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning, module="numba")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip",           default="data/budapest_gtfs.zip")
    p.add_argument("--cache-dir",     default="data/")
    p.add_argument("--lambda-v",      type=float, default=1.0 / 300.0)
    p.add_argument("--horizon",       type=float, default=900.0)
    p.add_argument("--tau",           type=float, default=1.0)
    p.add_argument("--ensemble",      type=int,   default=10)
    p.add_argument("--top-k",         type=int,   default=100)
    p.add_argument("--no-resilience", action="store_true")
    p.add_argument("--download",      action="store_true",
                   help="Download the GTFS ZIP (requires network access).")
    args = p.parse_args()

    # ── 1. Download / load ─────────────────────────────────────────────────
    from bkk import GTFSLoader, NetworkBuilder, GeneratorBuilder
    from bkk import (
        DemandPrior, KFESolver, GillespieSSA, TauLeap,
        allocate_modal_population,
    )
    from bkk import ResilienceAnalyser

    loader = GTFSLoader(cache_dir=args.cache_dir)
    if args.download:
        log.info("Downloading GTFS from bkk.hu …")
        loader.download()
    else:
        log.info("Loading GTFS from %s", args.zip)
        loader.load(args.zip)

    # ── 2. Parse ───────────────────────────────────────────────────────────
    t0   = time.perf_counter()
    feed = loader.parse(weekday="monday")
    log.info("Parsed in %.1f s", time.perf_counter() - t0)
    print(feed.summary())

    # ── 3. Build network ───────────────────────────────────────────────────
    t0  = time.perf_counter()
    net = NetworkBuilder().build(feed, peak_window=("07:00:00", "09:00:00"))
    log.info("Network built in %.1f s", time.perf_counter() - t0)
    print(net.summary())

    # ── 4. Build generators ────────────────────────────────────────────────
    t0   = time.perf_counter()
    gens = GeneratorBuilder(lambda_v=args.lambda_v).build_all(net)
    log.info("Generators built in %.1f s", time.perf_counter() - t0)

    for rt, mg in gens.items():
        m = mg.validate()
        log.info(
            "  mode %3d %-12s: n=%5d  |E|=%6d  "
            "max|ΣQ|=%.1e  max|ΣP-1|=%.1e",
            rt, mg.name, mg.n_stops, mg.Q.nnz,
            m["max_Q_rowsum_abs"], m["max_P_rowsum_dev"],
        )

    # ── 5. Demand prior ────────────────────────────────────────────────────
    prior     = DemandPrior()
    N0_global = prior.e1_service_proxy(net)
    log.info(
        "E1 prior: sum=%.0f  max=%.1f  top-stop=%s",
        N0_global.sum(), N0_global.max(),
        net.all_stop_ids[np.argmax(N0_global)],
    )

    # Map to per-mode local arrays
    N0_modal = allocate_modal_population(gens, N0_global, net.all_stop_ids)
    allocated = sum(x.sum() for x in N0_modal.values())
    if not np.isclose(allocated, N0_global.sum()):
        raise RuntimeError("Modal allocation did not conserve the global population")

    # Print top-10 stops
    print("\n" + "=" * 60)
    print("TOP-10 STOPS BY PRIOR ESTIMATE")
    print("=" * 60)
    top10_idx = np.argsort(N0_global)[::-1][:10]
    for rank, k in enumerate(top10_idx, start=1):
        sid = net.all_stop_ids[k]
        # Look up stop name
        name_row = feed.stops[feed.stops["stop_id"] == sid]["stop_name"]
        name     = name_row.iloc[0] if len(name_row) else sid
        print(f"  {rank:2d}. {name:<40s}  N_hat={N0_global[k]:.1f}")

    # ── 6. KFE ─────────────────────────────────────────────────────────────
    log.info("Running KFE (T=%.0f s, LSODA) …", args.horizon)
    t0      = time.perf_counter()
    solver  = KFESolver(method="lsoda")
    kfe_res = solver.solve(gens, N0_modal, T=args.horizon, n_eval=61)
    log.info("KFE done in %.1f s", time.perf_counter() - t0)

    for rt, res in kfe_res.items():
        log.info(
            "  mode %d: conservation_err=%.2e  N_final_sum=%.1f",
            rt, res.conservation_error(), res.N_final.sum(),
        )

    # ── 7. τ-leaping ────────────────────────────────────────────────────────
    log.info("Running τ-leap (T=%.0f s, τ=%.1f s) …", args.horizon, args.tau)
    t0      = time.perf_counter()
    leaper  = TauLeap(tau=args.tau, rng_seed=42)
    tau_res = leaper.run(gens, N0_modal, T=args.horizon, n_eval=61)
    log.info(
        "τ-leap done in %.1f s: steps=%d  conservation_err=%.2e",
        time.perf_counter() - t0,
        tau_res.metadata["n_steps"],
        tau_res.conservation_error(),
    )

    # ── 8. SSA ensemble (small) ────────────────────────────────────────────
    if args.ensemble > 0:
        log.info("Running SSA ensemble (%d runs) …", args.ensemble)
        t0         = time.perf_counter()
        ssa        = GillespieSSA(max_events=50_000, rng_seed=0)
        ssa_runs   = ssa.ensemble(gens, N0_modal, T=args.horizon,
                                  M=args.ensemble)
        log.info("SSA ensemble done in %.1f s", time.perf_counter() - t0)

        # Compare SSA mean vs KFE at final time
        print("\n" + "=" * 60)
        print("SSA vs KFE COMPARISON (top-5 stops, t=%.0fs)" % args.horizon)
        print("=" * 60)

        # Aggregate KFE final counts across modes
        kfe_final = np.zeros(len(net.all_stop_ids))
        for rt, res in kfe_res.items():
            mg    = gens[rt]
            sid2g = {sid: k for k, sid in enumerate(net.all_stop_ids)}
            for lk, sid in enumerate(mg.stop_ids):
                g = sid2g.get(sid)
                if g is not None:
                    kfe_final[g] += res.N_final[lk] if lk < len(res.N_final) else 0.0

        top5 = np.argsort(N0_global)[::-1][:5]
        print(f"  {'Stop':<40s}  {'SSA mean':>10s}  {'KFE':>8s}")
        for k in top5:
            sid    = net.all_stop_ids[k]
            name_r = feed.stops[feed.stops["stop_id"] == sid]["stop_name"]
            name   = name_r.iloc[0] if len(name_r) else sid
            ssa_vals = []
            for r in ssa_runs:
                result_idx = {s: i for i, s in enumerate(r.metadata["stop_ids"])}
                ssa_vals.append(r.N_final[result_idx[sid]])
            ssa_mean = np.mean(ssa_vals)
            print(
                f"  {name:<40s}  {ssa_mean:>10.1f}  {kfe_final[k]:>8.1f}"
            )

    # ── 9. Resilience ─────────────────────────────────────────────────────
    if not args.no_resilience:
        log.info("Running resilience analysis (top-%d) …", args.top_k)
        t0       = time.perf_counter()
        analyser = ResilienceAnalyser(top_k=args.top_k)
        reports  = analyser.analyse_all(gens, N0_modal)
        log.info("Resilience done in %.1f s", time.perf_counter() - t0)

        print("\n" + "=" * 70)
        print("RESILIENCE TOP-15 STOPS")
        print("=" * 70)

        import pandas as pd
        all_dfs = []
        for rt, rep in reports.items():
            df = rep.to_dataframe()
            if df.empty:
                continue
            df.insert(0, "mode", rep.baseline.route_type)
            # Map stop_id → stop_name
            sid2name = dict(zip(
                feed.stops["stop_id"], feed.stops["stop_name"]
            ))
            df["stop_name"] = df["stop_id"].map(sid2name).fillna(df["stop_id"])
            all_dfs.append(df)

        if all_dfs:
            combined = (
                pd.concat(all_dfs, ignore_index=True)
                .sort_values("criticality", ascending=False)
                .head(15)
                .reset_index(drop=True)
            )
            combined.index += 1
            print(
                combined[["stop_name", "importance",
                           "delta_kemeny", "delta_eff", "criticality"]]
                .to_string(float_format="{:.3f}".format)
            )

            out = Path("data/resilience_scores.csv")
            out.parent.mkdir(exist_ok=True)
            (pd.concat(all_dfs, ignore_index=True)
             .sort_values("criticality", ascending=False)
             .to_csv(out, index=False))
            log.info("Saved resilience CSV to %s", out)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
