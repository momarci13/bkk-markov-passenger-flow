#!/usr/bin/env python3
"""Write the LaTeX table fragments used in paper/paper.tex.

Inputs:  data/paper_dynamics.json   (scripts/paper_dynamics.py)
         data/resilience_scores.csv (scripts/tdk_rerun.py)
Outputs: paper/tables/t_*.tex
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TAB = REPO / "paper" / "tables"
MODE = {0: "Tram", 1: "Metro", 2: "Suburban rail", 3: "Bus", 4: "Ferry", 800: "Trolleybus"}
SRC = r"own computation from the BKK GTFS Schedule feed (CC0-1.0)"


def tex(s: str) -> str:
    rep = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
           "á": r"\'a", "é": r"\'e", "í": r"\'{\i}", "ó": r"\'o", "ö": r"\"o",
           "ő": r"\H{o}", "ú": r"\'u", "ü": r"\"u", "ű": r"\H{u}",
           "Á": r"\'A", "É": r"\'E", "Í": r"\'I", "Ó": r"\'O", "Ö": r"\"O",
           "Ő": r"\H{O}", "Ú": r"\'U", "Ü": r"\"U", "Ű": r"\H{U}"}
    return "".join(rep.get(c, c) for c in str(s))


def num(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}".replace(",", r"\,")


def table(name, cols, header, rows, caption, label, notes=None, source=SRC):
    out = [r"\begin{table}[htbp]", r"\centering", r"\small",
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header + r" \\", r"\midrule"]
    for r in rows:
        out.append(r if r.strip() in (r"\midrule", r"\hline") else r + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if notes:
        out.append(r"\par\smallskip{\footnotesize " + notes + "}")
    out += [rf"\forras{{{source}}}", r"\end{table}", ""]
    (TAB / name).write_text("\n".join(out), encoding="utf-8")


def main():
    d = json.loads((REPO / "data" / "paper_dynamics.json").read_text(encoding="utf-8"))

    # ---------------------------------------------------------- network
    rows = []
    for rt in sorted(d["modes"], key=int):
        m = d["modes"][rt]
        rows.append(" & ".join([MODE[int(rt)], rt, num(m["n"]), num(m["E"]),
                                f"{m['outdeg_mean']:.2f}", f"{100*m['fill']:.3f}",
                                num(m["median_headway_s"]), f"{m['median_cost_s']:.0f}"]))
    rows.append(r"\midrule")
    rows.append(" & ".join([r"\textbf{Total}", "---", num(d["n_global"]), num(d["R"]),
                            "---", "---", "---", "---"]))
    table("t_network.tex", "llrrrrrr",
          r"Mode & \texttt{rt} & $|S^v|$ & $|E^v|$ & $|E^v|/|S^v|$ & Fill (\%) & "
          r"$\mathrm{med}\,1/\mu^v_i$ (s) & $\mathrm{med}\,C^v_{ij}$ (s)",
          rows,
          r"Network summary by mode (weekday service, 07:00--09:00 peak window).",
          "tab:network",
          notes=r"Fill $=|E^v|/|S^v|^2$. $1/\mu^v_i$ is the scheduled headway at "
                r"stops with at least one peak departure. The total stop count is "
                r"over distinct nodes; a node served by several modes is counted once.")

    # ---------------------------------------------------------- prior
    rows = []
    for k, r in enumerate(d["prior_top10"], 1):
        rows.append(" & ".join([str(k), tex(r["name"]), r"\texttt{" + tex(r["stop_id"]) + "}",
                                "/".join(MODE[m] for m in r["modes"]),
                                f"{r['f_per_h']:.0f}", f"{r['N0']:.0f}"]))
    table("t_prior_top10.tex", "rlllrr",
          r"Rk & Stop & \texttt{stop\_id} & Mode & $f_i$ (dep.\,h$^{-1}$) & $\hat N_i(0)$",
          rows, r"Top-10 stops by the E1 prior $\hat N_i(0)$, anchored to "
                r"$N_{\mathrm{peak}}=440{,}000$.", "tab:prior-top10",
          notes=r"Metro platforms are aggregated to their parent station; surface "
                r"stops keep one \texttt{stop\_id} per platform, so a name can "
                r"appear more than once.")

    # ---------------------------------------------------------- KFE trajectory
    t = d["kfe_times"]
    hdr = "Stop & " + " & ".join(("$t=0$" if x == 0 else f"{x:.0f}\\,s") for x in t)
    rows = []
    for r in d["kfe_top"]:
        rows.append(tex(r["name"]) + " & " + " & ".join(f"{x:.0f}" for x in r["N"]))
    rows.append(r"\midrule")
    rows.append("Sum, initial top-50 & " + " & ".join(num(x) for x in d["kfe_top50_sum"]))
    rows.append("Gini of $N(t)$ & " + " & ".join(f"{x:.3f}" for x in d["kfe_gini"]))
    table("t_kfe.tex", "l" + "r" * len(t), hdr, rows,
          r"Mean-field (KFE) passenger counts at the five stops with the largest "
          r"initial count, full network, $\lambda^v=1/300$\,s$^{-1}$.", "tab:kfe-traj")

    # ---------------------------------------------------------- SSA vs KFE (metro)
    m = d["metro_ssa"]
    rows = []
    for r in m["rows"]:
        rows.append(" & ".join([tex(r["name"]), f"{r['mean']:.1f}", f"{r['q05']:.0f}",
                                f"{r['q95']:.0f}", f"{r['kfe']:.1f}",
                                f"{100*r['cv']:.1f}", f"{100*r['poisson_cv']:.1f}"]))
    table("t_ssa_kfe.tex", "lrrrrrr",
          r"Station & $\bar N^{\mathrm{SSA}}$ & $N^{05}$ & $N^{95}$ & KFE & "
          r"CV (\%) & $N^{-1/2}$ (\%)",
          rows,
          rf"Exact SSA ensemble ($M={m['M']}$) against the KFE mean on the metro "
          rf"sub-network at $T={d['metro_T_ssa']:.0f}$\,s, six largest stations.",
          "tab:ssa-kfe",
          notes=r"CV: ensemble coefficient of variation; $N^{-1/2}$: the Poisson "
                r"reference $1/\sqrt{N^{\mathrm{KFE}}}$.")

    # ---------------------------------------------------------- resilience
    # Component rule: V^gap is undefined on every modal chain of this feed
    # (metro: periodic, g=0; tram/bus: reducible, g=0) and is not used.
    # V^Kem is used only where the baseline chain is irreducible (metro).
    csv = REPO / "data" / "resilience_scores.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)

    def mm(x):
        x = np.asarray(x, float)
        return (x - x.min()) / (x.max() - x.min() + 1e-300)

    out = []
    for mode, g in df.groupby("mode"):
        if len(g) < 3:
            continue                     # ferry: two nodes, no ranking
        g = g.copy()
        comps = [mm(g["delta_eff"])]
        if mode == 1:
            comps.append(mm(g["delta_kemeny"]))
        g["V"] = np.mean(comps, axis=0)
        g["C"] = np.sqrt(mm(g["importance"]) * mm(g["V"]))
        out.append(g)
    res = pd.concat(out)
    res.to_csv(REPO / "data" / "criticality_paper.csv", index=False)
    summ = {}
    for mode, g in res.groupby("mode"):
        g = g.sort_values("C", ascending=False)
        summ[int(mode)] = dict(n=len(g), vk1=int((g["delta_kemeny"] >= 1).sum()),
                               top=g.head(10)[["stop_name", "importance", "delta_kemeny",
                                               "delta_eff", "C"]].to_dict("records"))
    (REPO / "data" / "criticality_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))

    met = res[res["mode"] == 1].sort_values("C", ascending=False).head(12)
    rows = []
    for k, (_, r) in enumerate(met.iterrows(), 1):
        rows.append(" & ".join([str(k), tex(r["stop_name"]), f"{r['importance']:.1f}",
                                f"{r['delta_kemeny']:.2f}", f"{r['delta_eff']:.3f}",
                                f"{r['C']:.3f}"]))
    table("t_resilience.tex", "rlrrrr",
          r"Rk & Station & $I_i$ (pax\,s$^{-1}$) & $V^{\mathrm{Kem}}$ & "
          r"$V^{\mathrm{eff}}$ & $C_i$",
          rows, r"Metro: top-12 stations by criticality $C_i$ (all 48 stations evaluated).",
          "tab:resilience",
          notes=r"$V^{\mathrm{Kem}}=1$: removal disconnects the metro chain.")

    rows = []
    for mode in (0, 3):
        g = res[res["mode"] == mode].sort_values("C", ascending=False).head(6)
        for k, (_, r) in enumerate(g.iterrows(), 1):
            rows.append(" & ".join([str(k), tex(r["stop_name"]), MODE[mode],
                                    f"{r['importance']:.1f}", f"{r['delta_eff']:.4f}",
                                    f"{r['C']:.3f}"]))
        if mode == 0:
            rows.append(r"\midrule")
    table("t_resilience_surface.tex", "rllrrr",
          r"Rk & Stop & Mode & $I_i$ (pax\,s$^{-1}$) & $V^{\mathrm{eff}}$ & $C_i$",
          rows, r"Tram and bus: top-6 stops by criticality $C_i$ among the 100 "
                r"candidates with the largest stationary throughput.",
          "tab:resilience-surface",
          notes=r"The tram and bus chains are reducible, so $C_i$ uses the "
                r"efficiency component only.")


if __name__ == "__main__":
    main()
