#!/usr/bin/env python3
"""
scripts/tdk_figures.py
======================
Figures of the TDK paper from data/results/ (run tdk_analysis.py first).

    fig_flow_map.pdf      steady-state passenger flow on the real network
    fig_critical_map.pdf  boardings and the most critical interchange hubs
    fig_new_lines.pdf     proposed new lines and the planned M5
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
from budapest_basemap import (draw_basemap, eov, load_districts, load_shapes,  # noqa: E402
                              scale_bar, shape_ids_by_mode)

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


def fig_new_lines(districts, shapes):
    """Proposed lines (U1..U5) and the planned M5 on top of the rail network."""
    nl = pd.read_csv(RES / "new_lines.csv")
    sc = json.load(open(RES / "scenarios.json"))
    by_mode = shape_ids_by_mode()
    metro = load_shapes(shape_ids=by_mode.get(1, set()))
    hev = load_shapes(shape_ids=by_mode.get(109, set()))
    tram = load_shapes(shape_ids=by_mode.get(0, set()))
    fig, ax = plt.subplots(figsize=(6.3, 6.6))
    draw_basemap(ax, districts, network_lines=shapes, labels=False)
    ax.add_collection(LineCollection(tram, colors="#8e8c86", linewidths=0.6, zorder=2.2))
    ax.add_collection(LineCollection(hev, colors="#52514e", linewidths=1.0,
                                     linestyles=(0, (4, 2)), zorder=2.3))
    ax.add_collection(LineCollection(metro, colors="#52514e", linewidths=1.6, zorder=2.4))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    allx, ally = [], []
    geoms = []
    for k, rec in enumerate(sc["selected"]):
        g = nl[nl.line == rec["name"]].sort_values("order")
        x, y = eov(g.lon.values, g.lat.values)
        geoms.append((rec, x, y, cols[k]))
        allx += list(x); ally += list(y)
    g = nl[nl.line == "M5A"].sort_values("order")
    mx, my = eov(g.lon.values, g.lat.values)
    allx += list(mx); ally += list(my)
    cx0, cy0 = np.mean(allx), np.mean(ally)
    for rec, x, y, col in geoms:
        lw = 3.2 if rec["mode"] == "metró" else 2.2
        ax.plot(x, y, color="white", lw=lw + 1.8, solid_capstyle="round", zorder=4)
        ax.plot(x, y, color=col, lw=lw, solid_capstyle="round", zorder=4.1)
        ax.scatter(x, y, s=14, facecolor="white", edgecolor=col, lw=1.0, zorder=4.2)
        # label at the end farther from the centre, pushed outwards
        e = 0 if np.hypot(x[0] - cx0, y[0] - cy0) > np.hypot(x[-1] - cx0, y[-1] - cy0) else -1
        dx, dy = x[e] - cx0, y[e] - cy0
        nrm = max(np.hypot(dx, dy), 1.0)
        ax.annotate(rec["name"], xy=(x[e], y[e]),
                    xytext=(x[e] + 550 * dx / nrm, y[e] + 550 * dy / nrm),
                    fontsize=7.5, fontweight="bold", color=INK, ha="center", va="center",
                    zorder=5, arrowprops=dict(arrowstyle="-", color=col, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=col, lw=1.0))
    ax.plot(mx, my, color=INK, lw=1.8, ls=(0, (2.5, 1.5)), zorder=4.3)
    ax.scatter(mx, my, s=12, facecolor="white", edgecolor=INK, lw=0.8, zorder=4.4)
    ax.annotate("M5", xy=(mx[0], my[0]), xytext=(mx[0] - 700, my[0] + 300), fontsize=7.5,
                fontweight="bold", color="white", ha="center", va="center", zorder=5,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.18", fc=INK, ec="none"))
    pad = 1800.0
    x0, x1 = min(allx) - pad, max(allx) + pad
    y0, y1 = min(ally) - pad, max(ally) + pad
    half = max(x1 - x0, y1 - y0) / 2           # square window
    ax.set_xlim((x0 + x1) / 2 - half, (x0 + x1) / 2 + half)
    ax.set_ylim((y0 + y1) / 2 - half, (y0 + y1) / 2 + half)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#52514e", lw=1.6, label="meglévő metró"),
               Line2D([], [], color="#52514e", lw=1.0, ls=(0, (4, 2)), label="meglévő HÉV"),
               Line2D([], [], color="#8e8c86", lw=0.6, label="meglévő villamos"),
               Line2D([], [], color=INK, lw=1.8, ls=(0, (2.5, 1.5)), label="tervezett M5 (A)"),
               Line2D([], [], color="#2a78d6", lw=2.2, label="javasolt villamos (U1–U5)")]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=6.3,
              labelcolor=INK2, handlelength=2.4, facecolor="white", edgecolor="none",
              framealpha=0.9)
    scale_bar(ax, km=2)
    fig.savefig(OUT / "fig_new_lines.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_new_lines.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    districts = load_districts()
    shapes = load_shapes()
    fig_flow(districts, shapes)
    fig_critical(districts, shapes)
    if (RES / "scenarios.json").exists():
        fig_new_lines(districts, shapes)


if __name__ == "__main__":
    main()
