#!/usr/bin/env python3
"""
scripts/tdk_figures.py
======================
Figures of the TDK paper from data/results/ (run tdk_analysis.py first).

    fig_flow_map.pdf      steady-state passenger flow on the real network
    fig_critical_map.pdf  boardings and the most critical interchange hubs
    fig_transient.pdf     fill-up of the network from an empty state
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from budapest_basemap import draw_basemap, eov, load_districts, load_shapes, scale_bar  # noqa: E402

RES = Path(__import__("os").environ.get("TDK_RES", "data/results"))
OUT = Path("tdk/figures")

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8e8c86"
BLUE_RAMP = ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
ACCENT = "#eb6834"          # categorical slot 2: critical hubs

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "pdf.fonttype": 42,
})


def undirected_flows(segs: pd.DataFrame) -> pd.DataFrame:
    """Sum both directions of every hub pair (load on the corridor)."""
    a = np.minimum(segs.from_hub, segs.to_hub)
    b = np.maximum(segs.from_hub, segs.to_hub)
    first = segs.from_hub <= segs.to_hub
    g = segs.assign(a=a, b=b,
                    la=np.where(first, segs.lat_i, segs.lat_j),
                    lo=np.where(first, segs.lon_i, segs.lon_j),
                    lb=np.where(first, segs.lat_j, segs.lat_i),
                    lp=np.where(first, segs.lon_j, segs.lon_i))
    return g.groupby(["a", "b"]).agg(flow=("flow_ph", "sum"), lat_i=("la", "first"),
                                     lon_i=("lo", "first"), lat_j=("lb", "first"),
                                     lon_j=("lp", "first")).reset_index()


def fig_flow(districts, shapes):
    segs = pd.read_csv(RES / "segments.csv")
    u = undirected_flows(segs).sort_values("flow")
    u = u[u.flow > 100]
    xi, yi = eov(u.lon_i.values, u.lat_i.values)
    xj, yj = eov(u.lon_j.values, u.lat_j.values)
    lines = np.stack([np.column_stack([xi, yi]), np.column_stack([xj, yj])], axis=1)
    fmax = float(u.flow.max())
    norm = LogNorm(vmin=100, vmax=fmax)
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb"] + BLUE_RAMP[1:])
    widths = 0.2 + 3.8 * (u.flow.values / fmax) ** 0.8

    fig, ax = plt.subplots(figsize=(6.3, 6.6))
    draw_basemap(ax, districts, network_lines=shapes, labels=True)
    ax.add_collection(LineCollection(lines, colors=cmap(norm(u.flow.values)),
                                     linewidths=widths, capstyle="round", zorder=3))
    scale_bar(ax)
    cax = fig.add_axes([0.70, 0.10, 0.22, 0.014])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("utas/óra (két irány összesen)", fontsize=6.5, color=INK2)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_visible(False)
    fig.savefig(OUT / "fig_flow_map.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_flow_map.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_critical(districts, shapes, k: int = 10):
    hubs = pd.read_csv(RES / "hubs.csv")
    act = hubs[hubs.boardings_ph > 0]
    x, y = eov(act.lon.values, act.lat.values)
    size = 0.4 + act.boardings_ph.values / 25.0

    crit = hubs.dropna(subset=["dE_net_closure"]).sort_values("dE_net_closure",
                                                              ascending=False).head(k)
    cx, cy = eov(crit.lon.values, crit.lat.values)

    fig, ax = plt.subplots(figsize=(6.3, 6.6))
    draw_basemap(ax, districts, network_lines=shapes, labels=False)
    big = act.boardings_ph.values > 150
    ax.scatter(x[big], y[big], s=size[big], color="#6da7ec", alpha=0.45, linewidths=0, zorder=3)
    ax.scatter(cx, cy, s=22 + 1500 * crit.dE_net_closure.values, facecolor=ACCENT,
               edgecolor="white", linewidth=0.8, zorder=4)
    # rank numbers beside the markers; greedy placement among 8 directions,
    # names are given in the paper's table (fewer words on the map)
    placed = [(xx, yy) for xx, yy in zip(cx, cy)]
    r_off = 900.0
    dirs = [(np.cos(a), np.sin(a)) for a in np.deg2rad([45, 135, 315, 225, 0, 180, 90, 270])]
    for k in np.argsort(-crit.dE_net_closure.values):
        best, best_d = None, -1.0
        for dx, dy in dirs:
            px, py = cx[k] + r_off * dx, cy[k] + r_off * dy
            dmin = min(np.hypot(px - qx, py - qy) for qx, qy in placed)
            if dmin > best_d:
                best, best_d = (px, py), dmin
        placed.append(best)
        ax.plot([cx[k], best[0]], [cy[k], best[1]], color=MUTED, lw=0.4, zorder=3.6)
        ax.text(best[0], best[1], str(k + 1), fontsize=7, fontweight="bold", color=INK,
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=MUTED, lw=0.4))
    scale_bar(ax)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], ls="", marker="o", ms=6, mfc="#6da7ec", mec="none", alpha=0.6,
               label="felszállás (terület ∝ utas/óra)"),
        Line2D([], [], ls="", marker="o", ms=7, mfc=ACCENT, mec="white",
               label="kritikus csomópont (terület ∝ hálózati ΔE/E)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.5,
              labelcolor=INK2, handletextpad=0.3)
    fig.savefig(OUT / "fig_critical_map.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_critical_map.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_transient():
    tr = pd.read_csv(RES / "transient.csv")
    R = json.load(open(RES / "results.json"))
    t95 = R["transient"]["t95_min"]
    fig, ax = plt.subplots(figsize=(4.2, 2.1))
    ax.plot(tr.t_s / 60, tr.frac * 100, color="#2a78d6", lw=2)
    ax.axhline(95, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.axvline(t95, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.text(t95 + 2, 40, f"95%-os feltöltődés: {t95:.0f} perc", fontsize=7, color=INK2)
    ax.set_xlabel("idő 07:00 óta [perc]")
    ax.set_ylabel(r"$\sum_x m_x(t)\,/\,\sum_x L_x$ [%]")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 102)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", lw=0.5)
    fig.savefig(OUT / "fig_transient.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_transient.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    districts = load_districts()
    shapes = load_shapes()
    fig_flow(districts, shapes)
    fig_critical(districts, shapes)
    fig_transient()


if __name__ == "__main__":
    main()
