#!/usr/bin/env python3
"""Pinned-feed reproduction run that regenerates every number in the TDK paper.

The paper carries no hand-typed empirical value.  Every figure quoted in the
results chapter is a LaTeX macro or table fragment emitted by this script, so a
reader can recompute the whole results chapter from one GTFS archive:

    python scripts/tdk_rerun.py --zip data/budapest_gtfs.zip

Outputs
-------
paper/tables/_numbers.tex       LaTeX \\newcommand macros (network size, feed
                                provenance, conservation errors, ...)
paper/tables/t_network.tex      Network summary by mode
paper/tables/t_prior_top10.tex  Top-10 stops by the E1 demand prior
paper/tables/t_resilience.tex   Top-15 stops by empirical criticality
data/resilience_scores.csv      Full resilience score table

Feed pinning
------------
The draft withheld its stop rankings because the run behind them could not be
tied to a specific feed.  This script records the SHA-256 of the archive and
the ``feed_info.txt`` version and validity dates, and prints them into the
paper, so the ranking is attached to an identifiable input.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("tdk_rerun")

REPO = Path(__file__).resolve().parent.parent
TABDIR = REPO / "paper" / "tables"
DATADIR = REPO / "data"

MODE_LABEL = {0: "Tram", 1: "Metro", 2: "Suburban rail", 3: "Bus",
              4: "Ferry", 800: "Trolleybus"}

SOURCE_LINE = (
    r"own computation from the pinned BKK GTFS feed "
    r"(\texttt{budapest\_gtfs.zip}, CC0-1.0), \texttt{scripts/tdk\_rerun.py}"
)

NUM: dict[str, str] = {}


def macro(name: str, value, fmt: str = "{:,.0f}") -> None:
    """Register a LaTeX macro, with a thin space as the thousands separator."""
    s = value if isinstance(value, str) else fmt.format(value)
    NUM[name] = s.replace(",", r"\,")


def latex_escape(s: str) -> str:
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def write_table(path: Path, header: list[str], rows: list[list[str]],
                aligns: str, caption: str, label: str,
                source: str = SOURCE_LINE) -> None:
    """Write a booktabs table with number, caption and source *below* it.

    The Corvinus TDK call requires a serial number, a title and a source line
    under every table and figure.
    """
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             rf"\begin{{tabular}}{{{aligns}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              rf"\forras{{{source}}}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", path.relative_to(REPO))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=str(DATADIR / "budapest_gtfs.zip"),
                    help="path to the BKK GTFS archive")
    ap.add_argument("--lambda-v", type=float, default=1.0 / 300.0,
                    help="entropy-cost multiplier lambda^v, in 1/s")
    ap.add_argument("--horizon", type=float, default=900.0,
                    help="KFE/tau-leap horizon T, in seconds")
    ap.add_argument("--tau", type=float, default=1.0, help="tau-leap step, s")
    ap.add_argument("--top-k", type=int, default=100,
                    help="candidate stops per mode for the resilience sweep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--download", action="store_true",
                    help="fetch the feed from bkk.hu instead of reading --zip")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s  %(levelname)-7s  %(message)s")
    TABDIR.mkdir(parents=True, exist_ok=True)
    DATADIR.mkdir(parents=True, exist_ok=True)

    from bkk import (DemandPrior, GeneratorBuilder, GTFSLoader, KFESolver,
                     NetworkBuilder, ResilienceAnalyser, TauLeap,
                     allocate_modal_population)

    # ---------------------------------------------------------------- feed
    loader = GTFSLoader(cache_dir=str(DATADIR))
    if args.download:
        loader.download()
        zip_path = DATADIR / "budapest_gtfs.zip"
    else:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            log.error("GTFS archive not found: %s\n"
                      "Download it with:  python scripts/tdk_rerun.py --download",
                      zip_path)
            return 2
        loader.load(str(zip_path))

    digest = sha256_of(zip_path)
    macro("feedSha", digest[:16])
    macro("feedBytes", zip_path.stat().st_size / 1e6, "{:,.1f}")
    log.info("feed sha256[:16] = %s  (%.1f MB)", digest[:16],
             zip_path.stat().st_size / 1e6)

    feed = loader.parse(weekday="monday")
    fi = feed.feed_info
    for key, col in (("feedVersion", "feed_version"),
                     ("feedStart", "feed_start_date"),
                     ("feedEnd", "feed_end_date")):
        val = str(fi[col].iloc[0]) if (len(fi) and col in fi.columns) else "n/a"
        macro(key, latex_escape(val))

    macro("nStopsRaw", len(feed.stops))
    macro("nRoutes", len(feed.routes))
    macro("nTrips", len(feed.trips))
    macro("nStopTimes", len(feed.stop_times))
    macro("nShapes", len(feed.shapes))

    # ---------------------------------------------------------------- network
    t0 = time.perf_counter()
    net = NetworkBuilder().build(feed, peak_window=("07:00:00", "09:00:00"))
    log.info("network built in %.1f s", time.perf_counter() - t0)
    macro("nStops", len(net.all_stop_ids))

    gens = GeneratorBuilder(lambda_v=args.lambda_v).build_all(net)
    macro("lambdaV", args.lambda_v, "{:.5f}")
    macro("nModes", len(gens))

    worst_q, worst_p = 0.0, 0.0
    for mg in gens.values():
        m = mg.validate()
        worst_q = max(worst_q, m["max_Q_rowsum_abs"])
        worst_p = max(worst_p, m["max_P_rowsum_dev"])
    macro("maxQRowsum", f"{worst_q:.1e}".replace("e-0", r"\times10^{-") + "}")
    macro("maxPRowdev", f"{worst_p:.1e}".replace("e-0", r"\times10^{-") + "}")

    # ---------------------------------------------------------------- demand
    prior = DemandPrior()
    N0 = prior.e1_service_proxy(net)
    macro("nTotalPrior", N0.sum())
    N0_modal = allocate_modal_population(gens, N0, net.all_stop_ids)
    if not np.isclose(sum(v.sum() for v in N0_modal.values()), N0.sum()):
        raise RuntimeError("modal allocation did not conserve the population")

    sid2name = dict(zip(feed.stops["stop_id"], feed.stops["stop_name"]))
    n_edges_total = 0
    net_rows = []
    for rt in sorted(net.modal):
        mg_net = net.modal[rt]
        mg_gen = gens.get(rt)
        mu = float(np.mean(mg_gen.mu)) if mg_gen is not None and hasattr(mg_gen, "mu") else float("nan")
        pop = float(N0_modal[rt].sum()) if rt in N0_modal else 0.0
        n_edges_total += mg_net.n_edges
        net_rows.append([
            MODE_LABEL.get(rt, str(rt)), str(rt),
            f"{mg_net.n_stops:,}".replace(",", r"\,"),
            f"{mg_net.n_edges:,}".replace(",", r"\,"),
            "---" if np.isnan(mu) else f"{mu:.3f}",
            f"{pop:,.0f}".replace(",", r"\,"),
        ])
    net_rows.append([r"\textbf{Total}", "---",
                     f"{len(net.all_stop_ids):,}".replace(",", r"\,"),
                     f"{n_edges_total:,}".replace(",", r"\,"),
                     "---", f"{N0.sum():,.0f}".replace(",", r"\,")])
    macro("nEdges", n_edges_total)

    write_table(
        TABDIR / "t_network.tex",
        ["Mode", r"\texttt{route\_type}", "$|S^v|$", "$|E^v|$",
         r"Mean $\mu^v_i$ (s$^{-1}$)", r"$\sum_i N_i(0)$"],
        net_rows, "llcccc",
        "Network summary by mode after GTFS ingestion, peak window "
        "07:00--09:00 on Monday service.",
        "tab:network")

    top10 = np.argsort(N0)[::-1][:10]
    write_table(
        TABDIR / "t_prior_top10.tex",
        ["Rank", "Stop", r"$\hat{N}_i(0)$"],
        [[str(r), latex_escape(str(sid2name.get(net.all_stop_ids[k],
                                                net.all_stop_ids[k]))),
          f"{N0[k]:.1f}"] for r, k in enumerate(top10, start=1)],
        "rlr",
        r"Top-10 stops by the E1 service-proxy prior $\hat{N}_i(0)$.",
        "tab:prior-top10")

    # ---------------------------------------------------------------- dynamics
    kfe = KFESolver(method="lsoda").solve(gens, N0_modal, T=args.horizon,
                                          n_eval=61)
    macro("kfeMaxConsErr",
          f"{max(r.conservation_error() for r in kfe.values()):.1e}"
          .replace("e-0", r"\times10^{-") + "}")

    tau = TauLeap(tau=args.tau, rng_seed=args.seed).run(
        gens, N0_modal, T=args.horizon, n_eval=61)
    macro("tauSteps", tau.metadata["n_steps"])
    macro("tauConsErr",
          f"{tau.conservation_error():.1e}".replace("e-0", r"\times10^{-") + "}")
    macro("horizon", args.horizon)

    # ---------------------------------------------------------------- resilience
    t0 = time.perf_counter()
    reports = ResilienceAnalyser(top_k=args.top_k).analyse_all(gens, N0_modal)
    log.info("resilience done in %.1f s", time.perf_counter() - t0)

    frames = []
    for rt, rep in reports.items():
        df = rep.to_dataframe()
        if df.empty:
            continue
        df.insert(0, "mode", rep.baseline.route_type)
        df["stop_name"] = df["stop_id"].map(sid2name).fillna(df["stop_id"])
        frames.append(df)

    if not frames:
        log.warning("no resilience rows produced; skipping the ranking table")
        (TABDIR / "_numbers.tex").write_text(
            "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(NUM.items())),
            encoding="utf-8")
        return 1

    combined = pd.concat(frames, ignore_index=True) \
                 .sort_values("criticality", ascending=False) \
                 .reset_index(drop=True)
    combined.to_csv(DATADIR / "resilience_scores.csv", index=False)
    log.info("wrote data/resilience_scores.csv (%d rows)", len(combined))

    top15 = combined.head(15)
    rows = []
    for rank, (_, r) in enumerate(top15.iterrows(), start=1):
        rows.append([
            str(rank),
            latex_escape(str(r["stop_name"])),
            MODE_LABEL.get(int(r["mode"]), str(r["mode"])),
            f"{r['importance']:.2f}",
            f"{r['delta_kemeny']:.0f}",
            f"{r['delta_eff']:.3f}",
            f"{r['criticality']:.3f}",
        ])
    write_table(
        TABDIR / "t_resilience.tex",
        ["Rank", "Stop", "Mode", "$I_i$", r"$\Delta K$", r"$\Delta E$", "$C_i$"],
        rows, "rllcccc",
        r"Top-15 stops by empirical criticality $C_i$. $I_i$ is passenger "
        r"throughput (pax\,s$^{-1}$), $\Delta K$ the increase in Kemeny's "
        r"constant on removal (reported only for chains that stay irreducible), "
        r"$\Delta E$ the loss of network efficiency.",
        "tab:resilience")

    n_metro = int((top15["mode"] == 1).sum())
    macro("topFifteenMetro", n_metro)
    macro("topStop", latex_escape(str(top15.iloc[0]["stop_name"])))
    macro("topStopCrit", top15.iloc[0]["criticality"], "{:.3f}")
    macro("secondStop", latex_escape(str(top15.iloc[1]["stop_name"])))
    macro("secondStopCrit", top15.iloc[1]["criticality"], "{:.3f}")
    macro("nResilienceRows", len(combined))

    # ---------------------------------------------------------------- provenance
    macro("runStamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    macro("pyVersion", platform.python_version())
    macro("numpyVersion", np.__version__)
    macro("pandasVersion", pd.__version__)

    (TABDIR / "_numbers.tex").write_text(
        "% Generated by scripts/tdk_rerun.py -- do not edit by hand.\n"
        + "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(NUM.items()))
        + "\n", encoding="utf-8")
    log.info("wrote paper/tables/_numbers.tex (%d macros)", len(NUM))

    # The paper's tables come from paper_tables.py, which needs the dynamics
    # results as well; run both so the tables always match this run.
    import subprocess
    here = Path(__file__).resolve().parent
    for script, extra in (("paper_dynamics.py", ["--zip", str(zip_path)]),
                          ("paper_tables.py", []),
                          ("paper_augment.py", ["--zip", str(zip_path)])):
        log.info("running scripts/%s", script)
        subprocess.run([sys.executable, str(here / script), *extra],
                       check=True, cwd=str(REPO))
    log.info("DONE -- now rebuild the paper: cd paper && ./forditas.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
