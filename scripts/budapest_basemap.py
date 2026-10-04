"""
scripts/budapest_basemap.py
===========================
Vector basemap of Budapest in the Hungarian national grid (EOV, EPSG:23700).

Layers, all from open data downloaded at runtime (see fetch_geodata):
  * the 23 districts            geoBoundaries HUN ADM2 (CC BY 4.0, OSM-derived)
  * the Danube                  Buda/Pest/Csepel district shared boundaries
                                (the legal district limits follow the river axis)
  * the real BKK route network  GTFS shapes.txt of the pinned feed

Tile servers are not used, so the figures print cleanly in black and white.
"""
from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import unary_union

GEO = Path("data/geo")
GEOBOUNDARIES = ("https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/"
                 "HUN/ADM2/geoBoundaries-HUN-ADM2.geojson")
BUDA = {"I. kerület", "II. kerület", "III. kerület", "XI. kerület", "XII. kerület",
        "XXII. kerület"}
CSEPEL = {"XXI. kerület"}

_to_eov = Transformer.from_crs("EPSG:4326", "EPSG:23700", always_xy=True)


def eov(lon, lat):
    """WGS84 lon/lat -> EOV metres (x east, y north)."""
    return _to_eov.transform(np.asarray(lon), np.asarray(lat))


def fetch_geodata() -> Path:
    GEO.mkdir(parents=True, exist_ok=True)
    path = GEO / "hun_adm2.geojson"
    if not path.exists():
        urllib.request.urlretrieve(GEOBOUNDARIES, path)
    return path


def load_districts():
    """Return {district name: shapely polygon in EOV}."""
    feats = json.load(open(fetch_geodata()))["features"]
    out = {}
    for f in feats:
        name = f["properties"]["shapeName"]
        if name.endswith("kerület"):
            g = shape(f["geometry"])
            out[name] = _project_geom(g)
    if len(out) != 23:
        raise RuntimeError(f"expected 23 Budapest districts, found {len(out)}")
    return out


def _project_geom(g):
    from shapely.ops import transform
    return transform(lambda x, y, z=None: _to_eov.transform(x, y), g)


def danube_lines(districts):
    """River axis = shared boundary of Buda vs Pest and Csepel vs the rest."""
    buda = unary_union([g for n, g in districts.items() if n in BUDA])
    csepel = unary_union([g for n, g in districts.items() if n in CSEPEL])
    pest = unary_union([g for n, g in districts.items() if n not in BUDA | CSEPEL])
    main = buda.buffer(30).intersection(pest.union(csepel).buffer(30)).boundary
    # The main arm is the Buda boundary facing Pest/Csepel; the Ráckeve (Soroksár)
    # arm separates Csepel from Pest.
    arm = csepel.buffer(30).intersection(pest.buffer(30)).boundary
    return buda.boundary.intersection(pest.union(csepel).buffer(60)), \
        csepel.boundary.intersection(pest.buffer(60)), arm, main


def load_shapes(zip_path="data/budapest_gtfs.zip", shape_ids=None):
    """GTFS shapes as a list of EOV polylines (optionally a subset of shape_ids)."""
    with zipfile.ZipFile(zip_path) as z:
        sh = pd.read_csv(z.open("shapes.txt"), dtype={"shape_id": str})
    if shape_ids is not None:
        sh = sh[sh["shape_id"].isin(set(shape_ids))]
    sh = sh.sort_values(["shape_id", "shape_pt_sequence"])
    x, y = eov(sh["shape_pt_lon"].to_numpy(), sh["shape_pt_lat"].to_numpy())
    sh = sh.assign(x=x, y=y)
    return [g[["x", "y"]].to_numpy() for _, g in sh.groupby("shape_id", sort=False)]


def shape_ids_by_mode(zip_path="data/budapest_gtfs.zip"):
    with zipfile.ZipFile(zip_path) as z:
        trips = pd.read_csv(z.open("trips.txt"), dtype=str, usecols=["route_id", "shape_id"])
        routes = pd.read_csv(z.open("routes.txt"), dtype=str, usecols=["route_id", "route_type"])
    t = trips.drop_duplicates().merge(routes, on="route_id")
    return {int(rt): set(g["shape_id"]) for rt, g in t.groupby("route_type")}


def draw_basemap(ax, districts, network_lines=None, river=True, labels=True):
    """District fills, district borders, the Danube and (optionally) the network."""
    from matplotlib.collections import LineCollection
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    def patch(poly, **kw):
        polys = getattr(poly, "geoms", [poly])
        verts, codes = [], []
        for p in polys:
            for ring in [p.exterior, *p.interiors]:
                c = np.asarray(ring.coords)
                verts.append(c)
                codes.append([MPath.MOVETO] + [MPath.LINETO] * (len(c) - 2) + [MPath.CLOSEPOLY])
        return PathPatch(MPath(np.vstack(verts), np.concatenate(codes)), **kw)

    for name, g in districts.items():
        ax.add_patch(patch(g, facecolor="#f4f3f0", edgecolor="#c9c7c1", lw=0.4, zorder=0))
    city = unary_union(list(districts.values()))
    ax.add_patch(patch(city, facecolor="none", edgecolor="#8e8c86", lw=0.8, zorder=1))
    if river:
        buda_edge, csepel_edge, _, _ = danube_lines(districts)
        for geom, lw in ((buda_edge, 5.0), (csepel_edge, 2.2)):
            parts = getattr(geom, "geoms", [geom])
            segs = [np.asarray(p.coords) for p in parts if p.geom_type == "LineString"]
            ax.add_collection(LineCollection(segs, colors="#b7d3f6", linewidths=lw,
                                             capstyle="round", zorder=1.5))
    if network_lines is not None:
        ax.add_collection(LineCollection(network_lines, colors="#a9a7a1", linewidths=0.25,
                                         zorder=2))
    if labels:
        for name, g in districts.items():
            p = g.representative_point()
            ax.text(p.x, p.y, name.split(".")[0], fontsize=5.5, color="#8e8c86",
                    ha="center", va="center", zorder=1.8)
    minx, miny, maxx, maxy = city.bounds
    ax.set_xlim(minx - 800, maxx + 800)
    ax.set_ylim(miny - 800, maxy + 800)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return city


def scale_bar(ax, km=5, loc=(0.05, 0.05)):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + loc[0] * (x1 - x0)
    y = y0 + loc[1] * (y1 - y0)
    ax.plot([x, x + km * 1000], [y, y], color="#52514e", lw=1.2, solid_capstyle="butt")
    ax.text(x + km * 500, y + 400, f"{km} km", ha="center", va="bottom", fontsize=6.5,
            color="#52514e")
    ax.annotate("É", xy=(x + km * 1000 + 1500, y + 2200), xytext=(x + km * 1000 + 1500, y),
                ha="center", fontsize=7, color="#52514e",
                arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.8))
