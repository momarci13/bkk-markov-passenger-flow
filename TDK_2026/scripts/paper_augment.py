#!/usr/bin/env python3
"""Robustness-driven link augmentation for the metro and tram layers.

Which new links would make the scheduled network most robust against the
loss of a single station?  For every mode layer the script

1. builds a station-level directed graph with the generalised costs C_ij of
   Section 3 (tram platforms are merged into stop areas);
2. computes, for every failure k, the all-pairs cost matrix D^{(-k)} of the
   graph without k;
3. scores a set of new bidirectional links A by the demand-weighted
   vulnerability functional

       V_k(A) = 1 - E_k(G_A - k) / E_k(G_A),
       J(A)   = alpha * mean_k V_k(A) + (1 - alpha) * max_k V_k(A),

   where E_k is the omega-weighted efficiency over OD pairs not involving k;
4. selects links greedily, using the exact single-edge shortcut update
       D'_ij = min(D_ij, D_ia + c + D_bj, D_ib + c + D_aj),
   so no shortest-path problem is re-solved after the initial sweep;
5. checks the greedy choice against exhaustive search (metro, B = 2),
   evaluates the officially planned projects on the same functional, and
   reports Kemeny's constant, the spectral gap and the fragmentation count of
   the augmented metro chain.

Usage:
    python scripts/paper_augment.py --zip data/budapest_gtfs.zip

Outputs: data/augment_results.json, paper/tables/t_aug_*.tex
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
TAB = REPO / "paper" / "tables"
DATA = REPO / "data"
log = logging.getLogger("paper_augment")

SRC = (r"own computation from the BKK GTFS Schedule feed (CC0-1.0), "
       r"\texttt{scripts/paper\_augment.py}")

# Approximate centre line of the main Danube branch through Budapest
# (lat, lon), north to south.  Used only to forbid new tram links that would
# need a new river crossing; metro links may cross (tunnels, as M2 and M4 do).
DANUBE = np.array([
    [47.600, 19.060], [47.560, 19.050], [47.535, 19.049], [47.514, 19.046],
    [47.499, 19.044], [47.490, 19.050], [47.4855, 19.0555], [47.477, 19.064],
    [47.463, 19.075], [47.440, 19.085], [47.400, 19.080]])


def pest_side(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """True for points east of the Danube centre line."""
    lat_line, lon_line = DANUBE[::-1, 0], DANUBE[::-1, 1]
    return lon > np.interp(lat, lat_line, lon_line)


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def tex(s: str) -> str:
    rep = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
           "á": r"\'a", "é": r"\'e", "í": r"\'{\i}", "ó": r"\'o", "ö": r"\"o",
           "ő": r"\H{o}", "ú": r"\'u", "ü": r"\"u", "ű": r"\H{u}",
           "Á": r"\'A", "É": r"\'E", "Í": r"\'I", "Ó": r"\'O", "Ö": r"\"O",
           "Ő": r"\H{O}", "Ú": r"\'U", "Ü": r"\"U", "Ű": r"\H{U}"}
    return "".join(rep.get(c, c) for c in str(s))


# --------------------------------------------------------------------------
# Station-level layer
# --------------------------------------------------------------------------
@dataclass
class Layer:
    mode: str
    names: list[str]
    lat: np.ndarray
    lon: np.ndarray
    W: csr_matrix          # directed generalised costs [s]
    p: np.ndarray          # demand share, sums to 1
    c_per_m: float         # median C_ij / haversine_ij over existing edges

    @property
    def n(self) -> int:
        return len(self.names)


def _layer_from_edges(mode, names, lat, lon, i, j, c, p) -> Layer:
    n = len(names)
    df = pd.DataFrame({"i": i, "j": j, "c": c})
    df = df[df.i != df.j].groupby(["i", "j"], as_index=False)["c"].median()
    W = csr_matrix((df.c.values, (df.i.values, df.j.values)), shape=(n, n))
    hav = haversine_m(lat[df.i], lon[df.i], lat[df.j], lon[df.j])
    ok = hav > 50
    c_per_m = float(np.median(df.c.values[ok] / hav[ok]))
    p = np.asarray(p, float)
    return Layer(mode, list(names), np.asarray(lat), np.asarray(lon), W,
                 p / p.sum(), c_per_m)


def metro_layer(net, N0m, stops) -> Layer:
    mg = net.modal[1]
    s = stops.set_index(stops.stop_id.astype(str))
    ids = list(mg.stop_ids)
    return _layer_from_edges(
        "metro", s.loc[ids, "stop_name"].tolist(),
        s.loc[ids, "stop_lat"].astype(float).values,
        s.loc[ids, "stop_lon"].astype(float).values,
        mg.i_arr, mg.j_arr, mg.cost_arr, N0m[1])


def _area_key(name: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*", "", str(name)).strip()


def tram_layer(net, N0m, stops, merge_m: float = 400.0) -> Layer:
    """Merge tram platforms that share a stop name (parentheses stripped)
    and lie within ``merge_m`` of each other into one stop area."""
    mg = net.modal[0]
    s = stops.set_index(stops.stop_id.astype(str))
    ids = list(mg.stop_ids)
    nm = s.loc[ids, "stop_name"].map(_area_key).values
    la = s.loc[ids, "stop_lat"].astype(float).values
    lo = s.loc[ids, "stop_lon"].astype(float).values
    area = -np.ones(len(ids), int)
    areas: list[list[int]] = []
    for k in range(len(ids)):
        for a, mem in enumerate(areas):
            r = mem[0]
            if nm[r] == nm[k] and haversine_m(la[r], lo[r], la[k], lo[k]) < merge_m:
                mem.append(k)
                area[k] = a
                break
        else:
            areas.append([k])
            area[k] = len(areas) - 1
    names = [nm[m[0]] for m in areas]
    alat = np.array([la[m].mean() for m in areas])
    alon = np.array([lo[m].mean() for m in areas])
    p = np.bincount(area, weights=np.asarray(N0m[0], float), minlength=len(areas))
    return _layer_from_edges("tram", names, alat, alon,
                             area[mg.i_arr], area[mg.j_arr], mg.cost_arr, p)


# --------------------------------------------------------------------------
# Distance tensors and the vulnerability functional
# --------------------------------------------------------------------------
class Evaluator:
    """Holds D (intact) and D^{(-k)} for k in the failure set, and evaluates
    J(A) for tentative links through the exact shortcut update."""

    def __init__(self, L: Layer, failures: np.ndarray, alpha: float = 0.5,
                 omega: np.ndarray | None = None):
        self.L, self.F, self.alpha = L, np.asarray(failures), alpha
        n = L.n
        if omega is None:
            omega = np.outer(L.p, L.p)
        omega = np.array(omega, float)
        np.fill_diagonal(omega, 0.0)
        self.omega = omega / omega.sum()
        self.D = dijkstra(L.W, directed=True)
        Dk = np.empty((len(self.F), n, n))
        for r, k in enumerate(self.F):
            W = L.W.tolil(copy=True)
            W[k, :] = 0
            W[:, k] = 0
            Dk[r] = dijkstra(W.tocsr(), directed=True)
            Dk[r, k, :] = np.inf
            Dk[r, :, k] = np.inf
        self.Dk = Dk
        # masks: pairs not involving k
        self.keep = np.ones((len(self.F), n, n), bool)
        for r, k in enumerate(self.F):
            self.keep[r, k, :] = False
            self.keep[r, :, k] = False

    @staticmethod
    def _inv(D):
        with np.errstate(divide="ignore"):
            X = 1.0 / D
        X[~np.isfinite(X)] = 0.0
        return X

    def _eff_k(self, D, Dk):
        """E_k(G) and E_k(G-k) for all failures (vectorised)."""
        w = self.omega
        Ei = self._inv(D) * w                       # (n, n)
        tot = Ei.sum()
        # E_k(G): drop row/col k from intact efficiency
        Ek_int = np.array([tot - Ei[k, :].sum() - Ei[:, k].sum() + Ei[k, k]
                           for k in self.F])
        Ek_rem = (self._inv(Dk) * w).sum(axis=(1, 2))
        return Ek_int, Ek_rem, tot

    def vuln(self, D=None, Dk=None) -> np.ndarray:
        D = self.D if D is None else D
        Dk = self.Dk if Dk is None else Dk
        Ek_int, Ek_rem, _ = self._eff_k(D, Dk)
        return 1.0 - Ek_rem / np.where(Ek_int > 0, Ek_int, 1.0)

    def J(self, V: np.ndarray) -> float:
        return float(self.alpha * V.mean() + (1 - self.alpha) * V.max())

    @staticmethod
    def add_edge(D, a, b, c):
        """Exact update of a cost matrix after adding a <-> b with cost c."""
        via_ab = D[..., :, a, None] + c + D[..., None, b, :]
        via_ba = D[..., :, b, None] + c + D[..., None, a, :]
        return np.minimum(D, np.minimum(via_ab, via_ba))

    def with_edges(self, edges):
        D, Dk = self.D, self.Dk
        for a, b, c in edges:
            D = self.add_edge(D, a, b, c)
            Dk2 = self.add_edge(Dk, a, b, c)
            # a failed endpoint cannot use the new link
            for r, k in enumerate(self.F):
                if k in (a, b):
                    Dk2[r] = Dk[r]
            Dk = Dk2
        return D, Dk

    def score(self, edges) -> dict:
        D, Dk = self.with_edges(edges)
        V = self.vuln(D, Dk)
        _, _, tot = self._eff_k(D, Dk)
        return {"J": self.J(V), "Vmean": float(V.mean()), "Vmax": float(V.max()),
                "argmax": int(self.F[int(np.argmax(V))]), "E": float(tot), "V": V}


def candidates(L: Layer, cap_m: float, no_river: bool, min_detour: float = 1.5):
    """Non-adjacent station pairs within cap_m whose current network cost is at
    least ``min_detour`` times the cost of a direct link (a genuine shortcut)."""
    D = dijkstra(L.W, directed=True)
    A = (L.W + L.W.T).toarray() > 0
    side = pest_side(L.lat, L.lon)
    out = []
    for a, b in itertools.combinations(range(L.n), 2):
        if A[a, b]:
            continue
        d = haversine_m(L.lat[a], L.lon[a], L.lat[b], L.lon[b])
        if d > cap_m or d < 150:
            continue
        if no_river and side[a] != side[b]:
            continue
        c = L.c_per_m * d
        cur = min(D[a, b], D[b, a])
        if cur < min_detour * c:
            continue
        out.append((a, b, c, d))
    return out


def greedy(ev: Evaluator, cands, B: int, prescreen: int | None = None,
           fixed=()):
    """Greedy maximisation of the J reduction.  Optionally pre-screens the
    candidate list to the best ``prescreen`` single links.  ``fixed`` links
    (a committed project) are always present."""
    fixed = [tuple(e[:3]) for e in fixed]
    fx = {frozenset(e[:2]) for e in fixed}
    cands = [e for e in cands if frozenset(e[:2]) not in fx]
    base = ev.score(fixed)
    if prescreen and len(cands) > prescreen:
        s1 = [(ev.score(fixed + [(a, b, c)])["J"], idx)
              for idx, (a, b, c, _) in enumerate(cands)]
        s1.sort()
        cands = [cands[i] for _, i in s1[:prescreen]]
    chosen, steps = [], []
    for _ in range(B):
        best = None
        for a, b, c, d in cands:
            if any((a, b) == (x[0], x[1]) for x in chosen):
                continue
            sc = ev.score(fixed + [(x[0], x[1], x[2]) for x in chosen] + [(a, b, c)])
            if best is None or sc["J"] < best[0]["J"] - 1e-15:
                best = (sc, (a, b, c, d))
        if best is None:
            break
        chosen.append(best[1])
        steps.append(best[0])
    return base, chosen, steps, cands


# --------------------------------------------------------------------------
# Markov-chain diagnostics of the augmented metro chain
# --------------------------------------------------------------------------
def chain_diagnostics(L: Layer, edges, lam: float = 1.0 / 300.0) -> dict:
    W = L.W.tolil(copy=True)
    for a, b, c, *_ in edges:
        W[a, b] = c
        W[b, a] = c
    W = W.tocsr()
    n = L.n
    P = np.zeros((n, n))
    for i in range(n):
        row = W.getrow(i)
        if row.nnz == 0:
            P[i, i] = 1.0
            continue
        lw = -lam * row.data
        lw -= lw.max()
        w = np.exp(lw)
        P[i, row.indices] = w / w.sum()
    ev = np.linalg.eigvals(P)
    ev = ev[np.argsort(-np.abs(ev))]
    k1 = int(np.argmin(np.abs(ev - 1)))
    rest = np.delete(ev, k1)
    ncomp, _ = connected_components(csr_matrix(P), directed=True, connection="strong")
    K = float(np.real(np.sum(1.0 / (1.0 - rest)))) if ncomp == 1 else float("nan")
    gap = float(1.0 - np.abs(rest).max())
    frag = 0
    for k in range(n):
        keep = np.arange(n) != k
        nc, _ = connected_components(csr_matrix(P[keep][:, keep]), directed=True,
                                     connection="strong")
        frag += nc > 1
    # bipartite <=> -1 is an eigenvalue
    return {"kemeny": K, "gap": gap, "periodic": bool(np.min(np.abs(ev + 1)) < 1e-8),
            "n_frag": int(frag)}


# --------------------------------------------------------------------------
def find(L: Layer, name: str) -> int:
    hits = [k for k, s in enumerate(L.names) if s == name]
    if not hits:
        hits = [k for k, s in enumerate(L.names) if s.startswith(name)]
    if not hits:
        raise KeyError(f"{name!r} not in {L.mode} layer")
    return hits[0]


def link(L: Layer, a: str, b: str):
    ia, ib = find(L, a), find(L, b)
    d = haversine_m(L.lat[ia], L.lon[ia], L.lat[ib], L.lon[ib])
    return (ia, ib, L.c_per_m * d, d)


def describe(L, e):
    return f"{L.names[e[0]]} -- {L.names[e[1]]}"


def run_layer(L: Layer, failures, cap_m, no_river, B, prescreen, planned,
              alpha=0.5, exhaustive=False, diagnostics=False):
    t0 = time.perf_counter()
    ev = Evaluator(L, failures, alpha=alpha)
    cands = candidates(L, cap_m, no_river)
    log.info("%s: n=%d |F|=%d candidates=%d (%.1fs setup)", L.mode, L.n,
             len(failures), len(cands), time.perf_counter() - t0)
    base, chosen, steps, pool = greedy(ev, cands, B, prescreen)
    res = {
        "n": L.n, "n_fail": int(len(failures)), "n_cand": len(cands),
        "cap_m": cap_m, "c_per_m": L.c_per_m, "alpha": alpha,
        "base": {k: v for k, v in base.items() if k != "V"},
        "base_worst": L.names[base["argmax"]],
        "base_top_vuln": [(L.names[failures[r]], float(base["V"][r]))
                          for r in np.argsort(-base["V"])[:8]],
        "greedy": [], "planned": {},
    }
    for step, (e, sc) in enumerate(zip(chosen, steps), 1):
        res["greedy"].append({
            "step": step, "a": L.names[e[0]], "b": L.names[e[1]],
            "len_m": e[3], "J": sc["J"], "Vmean": sc["Vmean"], "Vmax": sc["Vmax"],
            "worst": L.names[sc["argmax"]], "E": sc["E"]})
    for pname, links in planned.items():
        es = [link(L, a, b) for a, b in links]
        sc = ev.score([(a, b, c) for a, b, c, _ in es])
        res["planned"][pname] = {
            "links": [describe(L, e) for e in es], "len_m": float(sum(e[3] for e in es)),
            "J": sc["J"], "Vmean": sc["Vmean"], "Vmax": sc["Vmax"],
            "worst": L.names[sc["argmax"]], "E": sc["E"], "n_links": len(es)}
        # greedy with the same number of links
        k = len(es)
        if k <= len(steps):
            res["planned"][pname]["greedy_same_B_J"] = steps[k - 1]["J"]
        if diagnostics:
            res["planned"][pname]["chain"] = chain_diagnostics(L, es)
        # next links, given that the project is built
        _, nxt, nsteps, _ = greedy(ev, cands, 3, prescreen, fixed=es)
        res["planned"][pname]["next"] = [
            {"a": L.names[e[0]], "b": L.names[e[1]], "len_m": e[3], "J": sc2["J"],
             "Vmean": sc2["Vmean"], "Vmax": sc2["Vmax"],
             "worst": L.names[sc2["argmax"]]}
            for e, sc2 in zip(nxt, nsteps)]
        if diagnostics:
            res["planned"][pname]["chain_next"] = chain_diagnostics(L, es + nxt)
    if exhaustive:
        t1 = time.perf_counter()
        best = None
        for e1, e2 in itertools.combinations(pool, 2):
            sc = ev.score([e1[:3], e2[:3]])
            if best is None or sc["J"] < best[0]:
                best = (sc["J"], e1, e2)
        res["exhaustive_B2"] = {"J": best[0], "links": [describe(L, best[1]),
                                                         describe(L, best[2])],
                                "greedy_J": steps[1]["J"],
                                "seconds": time.perf_counter() - t1}
    if diagnostics:
        res["chain_base"] = chain_diagnostics(L, [])
        res["chain_greedy"] = [chain_diagnostics(L, chosen[:k]) for k in range(1, len(chosen) + 1)]
    res["_chosen"] = chosen
    res["_ev"] = ev
    return res


def sensitivity(L, failures, cap_m, no_river, B, prescreen, chosen_ref):
    """Top-B greedy picks under alternative alpha and a uniform demand weight."""
    ref = {frozenset(e[:2]) for e in chosen_ref}
    out = {}
    for tag, kw in (("alpha0", {"alpha": 0.0}), ("alpha1", {"alpha": 1.0}),
                    ("uniform", {"omega": np.ones((L.n, L.n))})):
        ev = Evaluator(L, failures, **kw)
        _, ch, _, _ = greedy(ev, candidates(L, cap_m, no_river), B, prescreen)
        s = {frozenset(e[:2]) for e in ch}
        out[tag] = {"links": [describe(L, e) for e in ch], "overlap": len(s & ref)}
    return out


# --------------------------------------------------------------------------
def write_tables(res):
    TAB.mkdir(parents=True, exist_ok=True)

    def table(name, cols, header, rows, caption, label, notes=None):
        out = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
               rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header + r" \\", r"\midrule"]
        out += [r if r == r"\midrule" else r + r" \\" for r in rows]
        out += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}",
                rf"\label{{{label}}}"]
        if notes:
            out.append(r"\par\smallskip{\footnotesize " + notes + "}")
        out += [rf"\forras{{{SRC}}}", r"\end{table}", ""]
        (TAB / name).write_text("\n".join(out), encoding="utf-8")

    for mode, lab in (("metro", "Metro"), ("tram", "Tram")):
        r = res[mode]
        b = r["base"]
        rows = [" & ".join(["0", r"\emph{current network}", "---",
                            f"{b['Vmean']:.3f}", f"{b['Vmax']:.3f}",
                            tex(r["base_worst"]), f"{b['J']:.3f}", "---"])]
        for g in r["greedy"]:
            rows.append(" & ".join([
                str(g["step"]), tex(g["a"]) + r" -- " + tex(g["b"]),
                f"{g['len_m']/1000:.2f}", f"{g['Vmean']:.3f}", f"{g['Vmax']:.3f}",
                tex(g["worst"]), f"{g['J']:.3f}",
                f"{100*(1 - g['J']/b['J']):.1f}"]))
        table(f"t_aug_{mode}.tex", r"r>{\raggedright\arraybackslash}p{5.0cm}rrr>{\raggedright\arraybackslash}p{3.0cm}rr",
              r"$b$ & Added link & km & $\bar V$ & $V_{\max}$ & Worst failure & $J$ & $-\Delta J$ (\%)",
              rows,
              rf"{lab}: greedy robustness-driven link additions. Each row adds "
              r"one bidirectional link to all previous ones. $\bar V$ and "
              r"$V_{\max}$ are the mean and maximum demand-weighted "
              r"vulnerability over single-station failures, "
              r"$J=\tfrac12(\bar V+V_{\max})$, and the last column is the "
              r"cumulative reduction of $J$.",
              f"tab:aug-{mode}")

    rows = []
    for mode, lab in (("metro", "Metro"), ("tram", "Tram")):
        r = res[mode]
        b = r["base"]
        for pname, p in r["planned"].items():
            gJ = p.get("greedy_same_B_J")
            rows.append(" & ".join([
                lab, tex(pname), str(p["n_links"]), f"{p['len_m']/1000:.1f}",
                f"{p['Vmean']:.3f}", f"{p['Vmax']:.3f}",
                f"${100*(1 - p['J']/b['J']):.1f}$",
                "---" if gJ is None else f"{100*(1 - gJ/b['J']):.1f}"]))
    table("t_aug_planned.tex", r"l>{\raggedright\arraybackslash}p{4.6cm}rrrrrr",
          r"Layer & Project & Links & km & $\bar V$ & $V_{\max}$ & $-\Delta J$ (\%) & greedy, same $b$ (\%)",
          rows,
          r"Planned and discussed projects evaluated on the same robustness "
          r"functional. The last column is the reduction achieved by the greedy "
          r"plan with the same number of links.",
          "tab:aug-planned",
          notes=r"Pure extensions (M3 to Rákospalota-Újpest, M2--Gödöllő HÉV, "
                r"M4 to Bosnyák tér) are not listed: by "
                r"Proposition~\ref{prop:pendant} they leave every $V_k$ unchanged.")


def write_conditional(res):
    rows = []
    for mode, lab in (("metro", "Metro"), ("tram", "Tram")):
        r = res[mode]
        b = r["base"]["J"]
        for pname, p in r["planned"].items():
            if "Thököly" in pname:
                continue
            rows.append(" & ".join([lab, r"\emph{" + tex(pname.split(" (")[0]) + " only}",
                                    "---", f"{p['Vmean']:.3f}", f"{p['Vmax']:.3f}",
                                    tex(p["worst"]), f"${100*(1-p['J']/b):.1f}$"]))
            for k, x in enumerate(p["next"], 1):
                rows.append(" & ".join(["", f"+{k}: " + tex(x["a"]) + " -- " + tex(x["b"]),
                                        f"{x['len_m']/1000:.2f}", f"{x['Vmean']:.3f}",
                                        f"{x['Vmax']:.3f}", tex(x["worst"]),
                                        f"${100*(1-x['J']/b):.1f}$"]))
            rows.append(r"\midrule")
    rows = rows[:-1]
    out = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
           r"\begin{tabular}{l>{\raggedright\arraybackslash}p{5.4cm}rrr>{\raggedright\arraybackslash}p{3.0cm}r}", r"\toprule",
           r"Layer & Project / next link & km & $\bar V$ & $V_{\max}$ & Worst failure & $-\Delta J$ (\%) \\",
           r"\midrule"]
    out += [x if x == r"\midrule" else x + r" \\" for x in rows]
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Suggested next links once the planned project is built: "
            r"greedy additions conditional on the project's links. $-\Delta J$ "
            r"is measured against the current network.}",
            r"\label{tab:aug-next}", rf"\forras{{{SRC}}}", r"\end{table}", ""]
    (TAB / "t_aug_next.tex").write_text("\n".join(out), encoding="utf-8")


def write_chain(res):
    r = res["metro"]
    rows = []

    def row(label, c):
        K = "undef." if not np.isfinite(c["kemeny"]) else f"{c['kemeny']:.1f}"
        return " & ".join([label, K, f"{max(c['gap'], 0.0):.4f}",
                           "yes" if c["periodic"] else "no", str(c["n_frag"])])
    rows.append(row(r"current network", r["chain_base"]))
    for k, c in enumerate(r["chain_greedy"], 1):
        rows.append(row(f"greedy, $b={k}$", c))
    for pname, p in r["planned"].items():
        rows.append(row(tex(pname.split(" (")[0]), p["chain"]))
        if "chain_next" in p:
            rows.append(row(tex(pname.split(" (")[0]) + " + 3 next links", p["chain_next"]))
    out = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
           r"\begin{tabular}{lrrcr}", r"\toprule",
           r"Metro network & $K(P)$ & $g(P)$ & periodic & fragmenting stations \\",
           r"\midrule"] + [x + r" \\" for x in rows] + [
           r"\bottomrule", r"\end{tabular}",
           r"\caption{Markov-chain diagnostics of the augmented metro chain "
           r"($\lambda=1/300$\,s$^{-1}$). The last column counts the stations "
           r"whose removal leaves a chain that is not irreducible.}",
           r"\label{tab:aug-chain}", rf"\forras{{{SRC}}}", r"\end{table}", ""]
    (TAB / "t_aug_chain.tex").write_text("\n".join(out), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=str(DATA / "budapest_gtfs.zip"))
    ap.add_argument("--cache", default=None, help="pickle with net/gens/N0m/stops")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--tables-only", action="store_true",
                    help="rewrite the LaTeX tables from data/augment_results.json")
    ap.add_argument("--metro-cap", type=float, default=3000.0)
    ap.add_argument("--tram-cap", type=float, default=1500.0)
    ap.add_argument("--tram-failures", type=int, default=100)
    ap.add_argument("--tram-prescreen", type=int, default=60)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    if args.tables_only:
        res = json.loads((DATA / "augment_results.json").read_text(encoding="utf-8"))
        write_tables(res)
        write_conditional(res)
        write_chain(res)
        return 0

    if args.cache:
        import pickle
        d = pickle.load(open(args.cache, "rb"))
        net, N0m, stops = d["net"], d["N0m"], d["stops"]
    else:
        from bkk import (DemandPrior, GeneratorBuilder, GTFSLoader, NetworkBuilder,
                         allocate_modal_population)
        loader = GTFSLoader(cache_dir=str(DATA))
        loader.load(args.zip)
        feed = loader.parse(weekday="monday")
        net = NetworkBuilder().build(feed, peak_window=("07:00:00", "09:00:00"))
        gens = GeneratorBuilder(lambda_v=1.0 / 300.0).build_all(net)
        N0 = DemandPrior().e1_service_proxy(net)
        N0m = allocate_modal_population(gens, N0, net.all_stop_ids)
        stops = feed.stops

    res = {}
    # ------------------------------------------------------------- metro
    M = metro_layer(net, N0m, stops)
    planned_metro = {
        "M5 inner section (Kálvin tér--Astoria--Oktogon--Lehel tér)": [
            ("Kálvin tér", "Astoria"), ("Astoria", "Oktogon"), ("Oktogon", "Lehel tér")],
    }
    res["metro"] = run_layer(M, np.arange(M.n), args.metro_cap, False, args.budget,
                             None, planned_metro, exhaustive=True, diagnostics=True)
    res["metro"]["sensitivity"] = sensitivity(M, np.arange(M.n), args.metro_cap,
                                              False, 3, None, res["metro"]["_chosen"][:3])
    res["metro"]["sensitivity_cap"] = {}
    for cap in (2000.0, 4000.0):
        ev = Evaluator(M, np.arange(M.n))
        _, ch, st, _ = greedy(ev, candidates(M, cap, False), 3)
        res["metro"]["sensitivity_cap"][str(int(cap))] = {
            "links": [describe(M, e) for e in ch], "J3": st[-1]["J"]}

    # ------------------------------------------------------------- tram
    T = tram_layer(net, N0m, stops)
    fail = np.argsort(-T.p)[:args.tram_failures]
    planned_tram = {
        "Pesti fonódó (Deák tér--Nyugati tér--Lehel tér)": [
            ("Deák Ferenc tér M", "Nyugati pályaudvar M"),
            ("Nyugati pályaudvar M", "Lehel tér M")],
        "Thököly út tram (Keleti pu.--Bosnyák tér), discussed": [
            ("Keleti pályaudvar M", "Bosnyák tér")],
    }
    res["tram"] = run_layer(T, fail, args.tram_cap, True, args.budget,
                            args.tram_prescreen, planned_tram)
    ncomp0, lab0 = connected_components(T.W, directed=True, connection="strong")
    res["tram"]["scc_base"] = int(ncomp0)
    Wg = T.W.tolil(copy=True)
    for a, b, c, _ in res["tram"]["_chosen"]:
        Wg[a, b] = c
        Wg[b, a] = c
    res["tram"]["scc_greedy"] = int(connected_components(Wg.tocsr(), directed=True,
                                                         connection="strong")[0])
    res["tram"]["sensitivity"] = sensitivity(T, fail, args.tram_cap, True, 3,
                                             args.tram_prescreen,
                                             res["tram"]["_chosen"][:3])

    for r in res.values():
        r.pop("_chosen"), r.pop("_ev")
    write_tables(res)
    write_conditional(res)
    write_chain(res)
    (DATA / "augment_results.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log.info("wrote data/augment_results.json and paper/tables/t_aug_*.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
