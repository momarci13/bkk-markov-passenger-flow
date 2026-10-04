"""
bkk.cli
=======
Command-line entry points registered in pyproject.toml.

Usage
-----
    bkk-download   [--cache-dir PATH] [--force]
    bkk-build      [--zip PATH] [--out PATH] [--weekday STR] [--lambda FLOAT]
    bkk-simulate   [--net PATH] [--mode {kfe,ssa,tauleap}] [--horizon FLOAT]
                   [--ensemble INT] [--out PATH]
    bkk-resilience [--net PATH] [--top-k INT] [--out PATH]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("bkk.cli")


# ---------------------------------------------------------------------------
def cmd_download() -> None:
    """bkk-download: fetch the BKK GTFS ZIP."""
    p = argparse.ArgumentParser(description="Download BKK GTFS.")
    p.add_argument("--cache-dir", default="data/",
                   help="Directory to store the ZIP (default: data/)")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if cached.")
    args = p.parse_args()

    from .gtfs import GTFSLoader
    GTFSLoader(cache_dir=args.cache_dir).download(force=args.force)
    log.info("Done. ZIP saved to %s/budapest_gtfs.zip", args.cache_dir)


# ---------------------------------------------------------------------------
def cmd_build() -> None:
    """bkk-build: parse GTFS → NetworkData + GeneratorBuilder → save."""
    p = argparse.ArgumentParser(description="Build BKK network and generators.")
    p.add_argument("--zip",      default="data/budapest_gtfs.zip")
    p.add_argument("--out",      default="data/network.npz",
                   help="Output file for NetworkData serialisation.")
    p.add_argument("--weekday",  default="monday",
                   help="Weekday to filter (monday … sunday).")
    p.add_argument("--lambda-v", type=float, default=1.0 / 300.0,
                   help="Entropy-cost λ^v [s^{-1}]. Default: 1/300.")
    args = p.parse_args()

    from .gtfs      import GTFSLoader
    from .network   import NetworkBuilder
    from .generator import GeneratorBuilder

    feed = GTFSLoader().load(args.zip).parse(weekday=args.weekday)
    net  = NetworkBuilder().build(feed)
    gens = GeneratorBuilder(lambda_v=args.lambda_v).build_all(net)

    log.info(net.summary())
    for rt, mg in gens.items():
        m = mg.validate()
        log.info(
            "mode %d: nnz_Q=%d  max|ΣQ|=%.1e  max|ΣP-1|=%.1e",
            rt, m["nnz_Q"], m["max_Q_rowsum_abs"], m["max_P_rowsum_dev"],
        )
    log.info("Build complete.")


# ---------------------------------------------------------------------------
def cmd_simulate() -> None:
    """bkk-simulate: run KFE, SSA, or tau-leap simulation."""
    p = argparse.ArgumentParser(description="Run BKK passenger-flow simulation.")
    p.add_argument("--zip",      default="data/budapest_gtfs.zip")
    p.add_argument("--mode",     choices=["kfe", "ssa", "tauleap"],
                   default="kfe")
    p.add_argument("--horizon",  type=float, default=900.0,
                   help="Simulation horizon [s]. Default: 900 (15 min).")
    p.add_argument("--ensemble", type=int, default=1,
                   help="Number of SSA realisations (SSA mode only).")
    p.add_argument("--lambda-v", type=float, default=1.0 / 300.0)
    p.add_argument("--weekday",  default="monday")
    p.add_argument("--tau",      type=float, default=1.0,
                   help="τ-leap step size [s]. Default: 1.0.")
    args = p.parse_args()

    from .gtfs      import GTFSLoader
    from .network   import NetworkBuilder
    from .generator import GeneratorBuilder
    from .demand    import DemandPrior
    from .simulate  import KFESolver, GillespieSSA, TauLeap, allocate_modal_population

    feed  = GTFSLoader().load(args.zip).parse(weekday=args.weekday)
    net   = NetworkBuilder().build(feed)
    gens  = GeneratorBuilder(lambda_v=args.lambda_v).build_all(net)
    prior = DemandPrior()

    N0_global = prior.e1_service_proxy(net)
    N0_modal = allocate_modal_population(gens, N0_global, net.all_stop_ids)

    if args.mode == "kfe":
        solver  = KFESolver()
        results = solver.solve(gens, N0_modal, T=args.horizon)
        for rt, res in results.items():
            log.info(
                "KFE mode %d: conservation_err=%.2e  N_final_sum=%.1f",
                rt, res.conservation_error(), res.N_final.sum(),
            )

    elif args.mode == "ssa":
        ssa = GillespieSSA()
        if args.ensemble == 1:
            res = ssa.run(gens, N0_modal, T=args.horizon)
            log.info("SSA: events=%d  conservation_err=%.2e",
                     res.metadata["n_events"], res.conservation_error())
        else:
            results = ssa.ensemble(gens, N0_modal, T=args.horizon,
                                   M=args.ensemble)
            log.info("SSA ensemble (%d runs) done.", len(results))

    elif args.mode == "tauleap":
        leaper = TauLeap(tau=args.tau)
        res    = leaper.run(gens, N0_modal, T=args.horizon)
        log.info(
            "τ-leap: steps=%d  conservation_err=%.2e",
            res.metadata["n_steps"], res.conservation_error(),
        )

    log.info("Simulation complete.")


# ---------------------------------------------------------------------------
def cmd_resilience() -> None:
    """bkk-resilience: run node-removal criticality analysis."""
    p = argparse.ArgumentParser(description="BKK resilience analysis.")
    p.add_argument("--zip",     default="data/budapest_gtfs.zip")
    p.add_argument("--top-k",  type=int, default=100)
    p.add_argument("--weekday", default="monday")
    p.add_argument("--lambda-v", type=float, default=1.0 / 300.0)
    p.add_argument("--out",    default="data/resilience_scores.csv")
    args = p.parse_args()

    from .gtfs       import GTFSLoader
    from .network    import NetworkBuilder
    from .generator  import GeneratorBuilder
    from .demand     import DemandPrior
    from .resilience import ResilienceAnalyser
    from .simulate   import allocate_modal_population

    feed     = GTFSLoader().load(args.zip).parse(weekday=args.weekday)
    net      = NetworkBuilder().build(feed)
    gens     = GeneratorBuilder(lambda_v=args.lambda_v).build_all(net)
    prior    = DemandPrior()
    N0       = prior.e1_service_proxy(net)

    N0_modal = allocate_modal_population(gens, N0, net.all_stop_ids)

    analyser = ResilienceAnalyser(top_k=args.top_k)
    reports  = analyser.analyse_all(gens, N0_modal)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for rt, rep in reports.items():
        df = rep.to_dataframe()
        df.insert(0, "route_type", rt)
        all_rows.append(df)

    if all_rows:
        import pandas as pd
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(out, index=False)
        log.info("Resilience results saved to %s", out)
        print(combined.head(20).to_string())
    else:
        log.warning("No resilience results produced.")
