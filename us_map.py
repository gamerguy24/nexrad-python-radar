#!/usr/bin/env python3
"""
us_map.py -- map projections, raster grids and the base map for the 50 US states.

Each region (Lower 48, Alaska, Hawaii) is drawn in its own Lambert conformal
conic projection, the projection the NWS uses for its national maps, so state
shapes look the way people expect instead of being stretched east-west.

State and county outlines come from Natural Earth (10m) through cartopy's
downloader (cached under ~/.local/share/cartopy after the first run).
"""
from __future__ import annotations

import functools
import math
import threading
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer

DOMAINS = ("CONUS", "ALASKA", "HAWAII")
DOMAIN_NAMES = {"CONUS": "Lower 48", "ALASKA": "Alaska", "HAWAII": "Hawaii"}
_R = "+R=6371229 +units=m +no_defs"
PROJ = {
    "CONUS": f"+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 {_R}",
    "ALASKA": f"+proj=lcc +lat_1=55 +lat_2=65 +lat_0=60 +lon_0=-154 {_R}",
    "HAWAII": f"+proj=lcc +lat_1=19 +lat_2=22 +lat_0=20.5 +lon_0=-157 {_R}",
}
# lon/lat box each region's radar grid covers (W, E, S, N)
COVER = {"CONUS": (-127.5, -64.5, 21.5, 51.5), "ALASKA": (-180.0, -128.0, 50.0, 72.5),
         "HAWAII": (-161.5, -153.5, 17.5, 23.5)}
# lon/lat box the map opens on
VIEW = {"CONUS": (-124.8, -66.9, 24.5, 49.4), "ALASKA": (-170.5, -129.5, 51.5, 71.5),
        "HAWAII": (-160.3, -154.8, 18.9, 22.3)}
GRID_RES_M = {"CONUS": 2000.0, "ALASKA": 3000.0, "HAWAII": 1000.0}


def domain_of_state(postal: str) -> str:
    return {"AK": "ALASKA", "HI": "HAWAII"}.get(postal, "CONUS")


_local = threading.local()          # pyproj transformers are per-thread


def _tf(domain: str, inverse: bool = False) -> Transformer:
    cache = _local.__dict__.setdefault("tf", {})
    t = cache.get((domain, inverse))
    if t is None:
        a, b = (PROJ[domain], "EPSG:4326") if inverse else ("EPSG:4326", PROJ[domain])
        t = cache[(domain, inverse)] = Transformer.from_crs(a, b, always_xy=True)
    return t


def project(domain, lon, lat):
    """lon/lat degrees -> projected metres (arrays or scalars)."""
    return _tf(domain).transform(np.asarray(lon, float), np.asarray(lat, float))


def unproject(domain, x, y):
    return _tf(domain, True).transform(np.asarray(x, float), np.asarray(y, float))


def box_bounds(domain, box) -> Tuple[float, float, float, float]:
    """Projected (xmin, xmax, ymin, ymax) of a lon/lat box."""
    w, e, s, n = box
    t = np.linspace(0.0, 1.0, 200)
    lon = np.concatenate([w + (e - w) * t, np.full_like(t, e), e - (e - w) * t, np.full_like(t, w)])
    lat = np.concatenate([np.full_like(t, s), s + (n - s) * t, np.full_like(t, n), n - (n - s) * t])
    x, y = project(domain, lon, lat)
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


# --------------------------------------------------------------------------
# Raster grids in a region's projection (row 0 = north)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RasterGrid:
    domain: str
    x0: float        # west edge, m
    y1: float        # north edge, m
    res: float       # m per pixel
    nx: int
    ny: int

    @property
    def shape(self):
        return self.ny, self.nx


def domain_grid(domain: str) -> RasterGrid:
    xmin, xmax, ymin, ymax = box_bounds(domain, COVER[domain])
    res = GRID_RES_M[domain]
    return RasterGrid(domain, xmin, ymax, res, int(math.ceil((xmax - xmin) / res)),
                      int(math.ceil((ymax - ymin) / res)))


def site_grid(domain: str, lon: float, lat: float, range_km: float, res_m: float) -> RasterGrid:
    """Square grid centred on a radar, big enough for its full range."""
    x, y = project(domain, lon, lat)
    half = range_km * 1000.0 * 1.04
    n = int(math.ceil(2 * half / res_m))
    return RasterGrid(domain, float(x) - half, float(y) + half, res_m, n, n)


@functools.lru_cache(maxsize=6)
def grid_lonlat(g: RasterGrid):
    """(lon, lat) float32 arrays of every pixel centre."""
    xs = g.x0 + (np.arange(g.nx) + 0.5) * g.res
    ys = g.y1 - (np.arange(g.ny) + 0.5) * g.res
    X, Y = np.meshgrid(xs, ys)
    lon, lat = unproject(g.domain, X, Y)
    return lon.astype(np.float32), lat.astype(np.float32)


# --------------------------------------------------------------------------
# Base map
# --------------------------------------------------------------------------
@dataclass
class BaseMap:
    """Projected outlines per region.  Each ring is an (n, 2) array in metres."""
    land: Dict[str, Dict[str, List[np.ndarray]]]       # dom -> {"coarse"/"fine": rings}
    counties: Dict[str, List[Tuple[Tuple[float, float, float, float], np.ndarray]]]
    labels: Dict[str, List[Tuple[str, float, float]]]   # dom -> (postal, x, y)
    mask_rings: Dict[str, List[np.ndarray]]             # other countries' land (hidden)


def _ne(name):
    import cartopy.io.shapereader as sr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sr.Reader(sr.natural_earth("10m", "cultural", name)).records()


def _rings(geom):
    polys = getattr(geom, "geoms", [geom])
    return [np.asarray(p.exterior.coords) for p in polys if not p.is_empty]


def _simplify(ring: np.ndarray, tol: float) -> np.ndarray:
    import shapely
    return np.asarray(shapely.LineString(ring).simplify(tol, preserve_topology=False).coords)


def load_basemap() -> BaseMap:
    land = {d: {"coarse": [], "fine": []} for d in DOMAINS}
    labels = {d: [] for d in DOMAINS}
    mask_rings = {d: [] for d in DOMAINS}
    counties = {d: [] for d in DOMAINS}
    for rec in _ne("admin_1_states_provinces_lakes"):
        a = rec.attributes
        if a["admin"] != "United States of America":
            continue
        dom = domain_of_state(a["postal"])
        for ring in _rings(rec.geometry):
            x, y = project(dom, ring[:, 0], ring[:, 1])
            r = np.column_stack([x, y])
            for lvl, tol in (("coarse", 2500.0), ("fine", 300.0)):
                s = _simplify(r, tol)
                if len(s) >= 4:
                    land[dom][lvl].append(s)
        if a["postal"] != "DC":
            x, y = project(dom, a["longitude"], a["latitude"])
            labels[dom].append((a["postal"], float(x), float(y)))
    import shapely
    for rec in _ne("admin_0_countries"):                        # other countries' land
        if rec.attributes.get("ADM0_A3") == "USA":
            continue
        for dom in DOMAINS:
            w, e, s_, n = COVER[dom]
            part = shapely.clip_by_rect(rec.geometry, w - 3, s_ - 3, e + 3, n + 3)
            if part.is_empty:
                continue
            for ring in _rings(part):
                x, y = project(dom, ring[:, 0], ring[:, 1])
                s = _simplify(np.column_stack([x, y]), 1000.0)
                if len(s) >= 4:
                    mask_rings[dom].append(s)
    for rec in _ne("admin_2_counties"):
        a = rec.attributes
        if a.get("ADM0_A3") != "USA":
            continue
        dom = domain_of_state(a.get("REGION") or "")
        for ring in _rings(rec.geometry):
            x, y = project(dom, ring[:, 0], ring[:, 1])
            s = _simplify(np.column_stack([x, y]), 250.0)
            if len(s) >= 4:
                bb = (s[:, 0].min(), s[:, 0].max(), s[:, 1].min(), s[:, 1].max())
                counties[dom].append((bb, s))
    return BaseMap(land, counties, labels, mask_rings)


def us_mask(g: RasterGrid, foreign_rings: List[np.ndarray]) -> np.ndarray:
    """Boolean (ny, nx): False over other countries' land, True elsewhere
    (the US, its coastal waters, the Gulf and the open ocean)."""
    img = Image.new("L", (g.nx, g.ny), 0)
    d = ImageDraw.Draw(img)
    for r in foreign_rings:
        px = (r[:, 0] - g.x0) / g.res
        py = (g.y1 - r[:, 1]) / g.res
        if px.max() < 0 or px.min() > g.nx or py.max() < 0 or py.min() > g.ny:
            continue
        d.polygon(list(zip(px.tolist(), py.tolist())), fill=1)
    return ~np.asarray(img, dtype=bool)
