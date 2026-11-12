#!/usr/bin/env python3
"""
storm_tracker.py -- MRMS storm-cell tracker with lightning-threat and MESH
progression guidance, attributed to every National Weather Service office.

WHAT IT DOES
  1. Ingests NOAA MRMS grids (public, no key): composite reflectivity,
     reflectivity at -10 C, MESH, NLDN cloud-to-ground flash density,
     lightning-jump grid, and the MRMS 0-60 min lightning probability.
     Optional extra lightning: generic stroke CSV (any NLDN-type vendor
     feed) or Lightning Mapping Array (LMA) ASCII .dat/.dat.gz files.
  2. Segments convective cells (35+ dBZ) and tracks them scan to scan with
     Hungarian assignment + a constant-velocity Kalman filter, seeded by
     phase-correlation (TREC-style) motion vectors.
  3. Forecasts the position of each cell 55-60 min ahead and scores the
     lightning threat there (first-flash "initiation" and any-lightning).
  4. Tracks MESH over rolling 45-min windows per cell and flags severe-hail
     crossings and intensifying / weakening trends.
  5. Attributes everything to its WFO (county warning area), both where the
     cell is now and where it is forecast to be in 55-60 min.

USAGE  (runs live and continuously until Ctrl+C)
  python storm_tracker.py                    # all domains, refresh every 2 min
  python storm_tracker.py --office FFC       # only cells over/into one WFO
  python storm_tracker.py --loop 6           # refresh every 6 min instead
  python storm_tracker.py --strokes file.csv --lma lma_dir/

DEPENDENCIES: numpy scipy requests pillow
  (MRMS GRIB2 is decoded natively; ecCodes is not needed)

IMPORTANT LIMITS (read before operational use)
  * The threat model is an interpretable heuristic, NOT a calibrated or
    NWS-endorsed product.  Every run appends its forecasts to
    forecast_log.jsonl so you can verify and fit coefficients later.
  * Extrapolation cannot anticipate NEW cell initiation; it only forecasts
    the evolution of cells that already exist at the time of the scan.
  * Raw NLDN is a commercial (Vaisala) feed.  Out of the box this uses the
    public NLDN-derived MRMS flash-density grid (CONUS only); regional MRMS
    domains (AK/HI/Caribbean/Guam) have no lightning grids, so those
    offices get radar-only lightning guidance.
  * MESH is a radar-derived estimate, not a measured hail size.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import dataclasses
import datetime as dt
import gzip
import io
import json
import logging
import math
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = ImageDraw = None

log = logging.getLogger("storm_tracker")
UTC = dt.timezone.utc

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MRMS_BASE = "https://mrms.ncep.noaa.gov/2D"
WFO_URL = ("https://opengeo.ncep.noaa.gov/geoserver/wfs?service=WFS&request="
           "GetFeature&typenames=nws:cwa_boundary&outputFormat=application/json"
           "&srsName=EPSG:4326")


def _from_webmercator(lonm, latm):
    """Inverse spherical Web Mercator (EPSG:3857) -> lon/lat degrees."""
    lon = np.degrees(lonm / 6378137.0)
    lat = np.degrees(2 * np.arctan(np.exp(latm / 6378137.0)) - np.pi / 2.0)
    return lon, lat

# key -> (directory name, block reducer, kind)
PRODUCTS = {
    "cref": ("MergedReflectivityQCComposite", "max", "refl"),
    "z10":  ("Reflectivity_-10C", "max", "refl"),
    "mesh": ("MESH", "max", "nonneg"),
    "nldn": ("NLDN_CG_005min_AvgDensity", "mean", "nonneg"),   # flashes/km2/min
    "jump": ("LtgJumpGrid", "max", "nonneg"),
    "prob": ("LightningProbabilityNext60min", "max", "nonneg"),  # percent
}
LIGHTNING_KEYS = ("nldn", "jump", "prob")
DOMAINS = ("CONUS", "ALASKA", "HAWAII", "CARIB", "GUAM")
KM_LAT = 111.0
SEVERE_MESH_MM = 25.4          # 1.00 in, NWS severe-hail criterion
SIG_MESH_MM = 50.8             # 2.00 in, significant hail


@dataclass
class Config:
    step_min: float = 6.0            # spacing of scans used for tracking
    history_min: float = 90.0        # scans back from newest
    coarsen: int = 2                 # 0.01 deg -> 0.02 deg (~2 km)
    detect_dbz: float = 35.0
    min_area_km2: float = 20.0
    split_sep_km: float = 12.0       # minimum separation of cell maxima
    max_miss: int = 2                # scans a track may coast
    lead_min: Tuple[float, float] = (55.0, 60.0)
    mesh_window_min: float = 45.0
    mesh_stride_min: float = 15.0
    cache_dir: Path = Path("mrms_cache")
    out_dir: Path = Path("storm_out")
    domains: Tuple[str, ...] = DOMAINS
    office: Optional[str] = None


# --------------------------------------------------------------------------
# Grid geometry
# --------------------------------------------------------------------------
@dataclass
class Grid:
    lat0: float      # latitude of row 0 centre (north edge)
    lon0: float      # longitude of col 0 centre, -180..180
    dlat: float      # degrees per row (positive; rows run north->south)
    dlon: float
    ny: int
    nx: int

    def coarsen(self, k: int) -> "Grid":
        if k <= 1:
            return self
        return Grid(self.lat0 - 0.5 * (k - 1) * self.dlat,
                    self.lon0 + 0.5 * (k - 1) * self.dlon,
                    self.dlat * k, self.dlon * k, self.ny // k, self.nx // k)

    def rc(self, lat, lon):
        lon = (np.asarray(lon) + 180.0) % 360.0 - 180.0
        dl = (lon - self.lon0 + 180.0) % 360.0 - 180.0
        return (self.lat0 - np.asarray(lat)) / self.dlat, dl / self.dlon

    def latlon(self, r, c):
        lon = self.lon0 + np.asarray(c) * self.dlon
        return self.lat0 - np.asarray(r) * self.dlat, (lon + 180.0) % 360.0 - 180.0

    @property
    def row_lat(self):
        return self.lat0 - np.arange(self.ny) * self.dlat

    @property
    def px_area_km2(self):
        """Pixel area per row (ny,) in km^2."""
        return (self.dlat * KM_LAT) * (self.dlon * 111.32 *
                                       np.cos(np.radians(self.row_lat)))

    @property
    def km_per_px(self):
        return float(self.dlat * KM_LAT)


def block_reduce(a: np.ndarray, k: int, how: str) -> np.ndarray:
    if k <= 1:
        return a
    ny, nx = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:ny, :nx].reshape(ny // k, k, nx // k, k)
    return b.max(axis=(1, 3)) if how == "max" else b.mean(axis=(1, 3))


# --------------------------------------------------------------------------
# MRMS ingest
# --------------------------------------------------------------------------
_TS = re.compile(r"_(\d{8})-(\d{6})\.grib2\.gz$")


def _grib_int(b: bytes) -> int:
    """GRIB2 signed integer (sign-magnitude, not two's complement)."""
    v = int.from_bytes(b, "big")
    top = 1 << (8 * len(b) - 1)
    return -(v & (top - 1)) if v & top else v


def _decode_grib2(msg: bytes):
    """Minimal GRIB2 reader for MRMS: grid template 3.0 (regular lat/lon) with
    data template 5.41 (PNG packing), the only layout MRMS 2D products use.
    -> (values (nj, ni) float32, lat1, lon1, di, dj, north_first)."""
    if msg[:4] != b"GRIB" or msg[7] != 2:
        raise ValueError("not a GRIB2 message")
    p, sec = 16, {}
    while msg[p:p + 4] != b"7777":
        ln = struct.unpack(">I", msg[p:p + 4])[0]
        sec[msg[p + 4]] = msg[p:p + ln]
        p += ln
    s3, s5, s7 = sec[3], sec[5], sec[7]
    gdt, drt = struct.unpack(">H", s3[12:14])[0], struct.unpack(">H", s5[9:11])[0]
    if gdt != 0 or drt != 41:
        raise ValueError(f"unsupported GRIB2 templates 3.{gdt} / 5.{drt}")
    ni, nj = struct.unpack(">II", s3[30:38])
    lat1, lon1 = _grib_int(s3[46:50]) * 1e-6, _grib_int(s3[50:54]) * 1e-6
    di, dj = _grib_int(s3[63:67]) * 1e-6, _grib_int(s3[67:71]) * 1e-6
    north_first = not (s3[71] & 0x40)
    ref = struct.unpack(">f", s5[11:15])[0]
    e, d = _grib_int(s5[15:17]), _grib_int(s5[17:19])
    x = np.asarray(Image.open(io.BytesIO(s7[5:])))
    if x.ndim == 3:                                  # 24/32-bit packed as RGB(A)
        x = x.astype(np.uint32)
        x = np.bitwise_or.reduce([x[..., i] << (8 * (x.shape[2] - 1 - i))
                                  for i in range(x.shape[2])])
    v = ((ref + x.astype(np.float32) * np.float32(2.0 ** e)) / np.float32(10.0 ** d))
    return v.reshape(nj, ni), lat1, lon1, di, dj, north_first


def decode_grib2_gz(raw: bytes, kind: str, how: str, k: int):
    """Decode one MRMS GRIB2 (gzip) message -> (coarsened float16, Grid)."""
    if Image is None:
        raise RuntimeError("pillow required to decode MRMS (pip install pillow)")
    v, lat1, lon1, di, dj, north_first = _decode_grib2(gzip.decompress(raw))
    nj, ni = v.shape
    if not north_first:
        v = v[::-1]
        lat1 = lat1 + (nj - 1) * dj
    if kind == "refl":
        v[(v <= -90) | ~np.isfinite(v)] = -30.0
    else:
        v[(v < 0) | ~np.isfinite(v)] = 0.0
    lon1 = (lon1 + 180.0) % 360.0 - 180.0
    g = Grid(lat1, lon1, dj, di, nj, ni)
    return block_reduce(v, k, how).astype(np.float16), g.coarsen(k)


class MrmsFeeder:
    """Lists, downloads, decodes and caches MRMS products by time."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sess = requests.Session()
        self.sess.headers["User-Agent"] = "storm_tracker/1.0 (research use)"
        self._listing: Dict[Tuple[str, str], List[Tuple[dt.datetime, str]]] = {}
        for sub in ("raw", "dec"):
            (cfg.cache_dir / sub).mkdir(parents=True, exist_ok=True)

    def _url(self, domain: str, product: str) -> str:
        d = "" if domain == "CONUS" else f"/{domain}"
        return f"{MRMS_BASE}{d}/{product}/"

    def _get(self, url: str, tries: int = 3) -> Optional[bytes]:
        for i in range(tries):
            try:
                r = self.sess.get(url, timeout=40)
                if r.status_code == 200:
                    return r.content
                if r.status_code == 404:
                    return None
            except requests.RequestException as e:
                log.debug("GET %s failed: %s", url, e)
            time.sleep(1.5 * (i + 1))
        return None

    def listing(self, domain: str, key: str, refresh=False):
        ck = (domain, key)
        if ck in self._listing and not refresh:
            return self._listing[ck]
        body = self._get(self._url(domain, PRODUCTS[key][0]))
        out = []
        if body:
            for fn in re.findall(r'href="(MRMS_[^"]+\.grib2\.gz)"',
                                 body.decode("utf8", "ignore")):
                m = _TS.search(fn)
                if m:
                    t = dt.datetime.strptime(m.group(1) + m.group(2),
                                             "%Y%m%d%H%M%S").replace(tzinfo=UTC)
                    out.append((t, fn))
        out.sort()
        self._listing[ck] = out
        return out

    def nearest(self, domain, key, t, tol_s=200):
        best = None
        for ft, fn in self.listing(domain, key):
            d = abs((ft - t).total_seconds())
            if d <= tol_s and (best is None or d < best[0]):
                best = (d, ft, fn)
        return best[1:] if best else None

    def timeline(self, domain: str) -> List[dt.datetime]:
        """Scan times to use, newest first, spaced ~step_min apart."""
        lst = self.listing(domain, "cref", refresh=True)
        if not lst:
            return []
        newest = lst[-1][0]
        n = int(self.cfg.history_min // self.cfg.step_min) + 1
        out = []
        for i in range(n):
            hit = self.nearest(domain, "cref",
                               newest - dt.timedelta(minutes=i * self.cfg.step_min))
            if hit and (not out or hit[0] != out[-1]):
                out.append(hit[0])
        return out

    def load(self, domain: str, key: str, t: dt.datetime):
        """-> (array float16, Grid) or None if unavailable."""
        if key in LIGHTNING_KEYS and domain != "CONUS":
            return None
        hit = self.nearest(domain, key, t)
        if not hit:
            return None
        ft, fn = hit
        stem = f"{domain}__{key}__{ft:%Y%m%d%H%M%S}__k{self.cfg.coarsen}"
        npz = self.cfg.cache_dir / "dec" / (stem + ".npz")
        if npz.exists():
            z = np.load(npz)
            return z["a"], Grid(*[float(x) for x in z["g"][:4]],
                                int(z["g"][4]), int(z["g"][5]))
        rawp = self.cfg.cache_dir / "raw" / f"{domain}__{fn}"
        if rawp.exists():
            raw = rawp.read_bytes()
        else:
            raw = self._get(self._url(domain, PRODUCTS[key][0]) + fn)
            if raw is None:
                return None
            rawp.write_bytes(raw)
        _, how, kind = PRODUCTS[key]
        a, g = decode_grib2_gz(raw, kind, how, self.cfg.coarsen)
        np.savez_compressed(npz, a=a, g=np.array(
            [g.lat0, g.lon0, g.dlat, g.dlon, g.ny, g.nx], dtype=np.float64))
        rawp.unlink(missing_ok=True)      # decoded copy is kept instead
        return a, g

    def prefetch(self, domain: str, times: Sequence[dt.datetime]):
        """Download raw files in parallel (decode happens later)."""
        jobs = []
        for t in times:
            for key in PRODUCTS:
                if key in LIGHTNING_KEYS and domain != "CONUS":
                    continue
                hit = self.nearest(domain, key, t)
                if not hit:
                    continue
                ft, fn = hit
                stem = f"{domain}__{key}__{ft:%Y%m%d%H%M%S}__k{self.cfg.coarsen}"
                rawp = self.cfg.cache_dir / "raw" / f"{domain}__{fn}"
                if not (self.cfg.cache_dir / "dec" / (stem + ".npz")).exists() \
                        and not rawp.exists():
                    jobs.append((self._url(domain, PRODUCTS[key][0]) + fn, rawp))

        def one(j):
            b = self._get(j[0])
            if b:
                j[1].write_bytes(b)
        with cf.ThreadPoolExecutor(8) as ex:
            list(ex.map(one, jobs))

    def prune(self, hours=8):
        cut = time.time() - hours * 3600
        for sub in ("raw", "dec"):
            for p in (self.cfg.cache_dir / sub).iterdir():
                if p.stat().st_mtime < cut:
                    p.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Optional supplementary lightning: stroke CSV and LMA ASCII
# --------------------------------------------------------------------------
def _parse_time(s: str) -> dt.datetime:
    s = s.strip()
    try:
        return dt.datetime.fromtimestamp(float(s), UTC)
    except ValueError:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC) \
            if "+" in s or s.endswith("Z") else \
            dt.datetime.fromisoformat(s).replace(tzinfo=UTC)


def read_stroke_csv(path: Path):
    """CSV with header containing time, lat, lon (ISO-8601 or epoch seconds).
    Works for NLDN/GLM/any vendor export.  -> list[(datetime, lat, lon)]."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            r = {k.strip().lower(): v for k, v in row.items()}
            try:
                out.append((_parse_time(r.get("time") or r["datetime"]),
                            float(r["lat"] if "lat" in r else r["latitude"]),
                            float(r["lon"] if "lon" in r else r["longitude"])))
            except (KeyError, ValueError):
                continue
    return out


def read_lma_dat(path: Path, min_stations=6, max_chi2=2.0):
    """Parse an NM Tech LMA ASCII file (.dat or .dat.gz) into flash proxies.

    Sources (time-of-day seconds, lat, lon, alt, chi2, power, station mask)
    are quality-filtered then clustered into flashes by (0.3 s, 0.05 deg)
    bins -- a deliberately simple stand-in for the full LMA flash sorter.
    """
    op = gzip.open if str(path).endswith(".gz") else open
    day0, rows, in_data = None, [], False
    with op(path, "rt", errors="ignore") as f:
        for line in f:
            if not in_data:
                if "Data start time" in line:
                    m = re.search(r"(\d{2})/(\d{2})/(\d{2,4})", line)
                    if m:
                        mo, d, y = int(m[1]), int(m[2]), int(m[3])
                        day0 = dt.datetime(y + (2000 if y < 100 else 0), mo, d,
                                           tzinfo=UTC)
                if line.strip().startswith("*** data ***"):
                    in_data = True
                continue
            p = line.split()
            if len(p) >= 7:
                try:
                    t, la, lo, _, chi, _, mask = (float(p[0]), float(p[1]), float(p[2]),
                                                  p[3], float(p[4]), p[5], p[6])
                    if chi <= max_chi2 and bin(int(mask, 16)).count("1") >= min_stations:
                        rows.append((t, la, lo))
                except ValueError:
                    continue
    if day0 is None or not rows:
        return []
    a = np.array(rows)
    key = np.stack([np.floor(a[:, 0] / 0.3), np.floor(a[:, 1] / 0.05),
                    np.floor(a[:, 2] / 0.05)], 1)
    _, idx = np.unique(key, axis=0, return_index=True)
    return [(day0 + dt.timedelta(seconds=float(a[i, 0])), a[i, 1], a[i, 2])
            for i in idx]


def points_to_density(points, grid: Grid, t: dt.datetime, window_min=5.0):
    """Point lightning in [t-window, t] -> flashes/km2/min on the grid."""
    t0 = t - dt.timedelta(minutes=window_min)
    sel = [(la, lo) for (pt, la, lo) in points if t0 < pt <= t]
    out = np.zeros((grid.ny, grid.nx), np.float32)
    if not sel:
        return out
    la, lo = np.array(sel).T
    r, c = grid.rc(la, lo)
    r, c = np.round(r).astype(int), np.round(c).astype(int)
    ok = (r >= 0) & (r < grid.ny) & (c >= 0) & (c < grid.nx)
    np.add.at(out, (r[ok], c[ok]), 1.0)
    out /= (grid.px_area_km2[:, None] * window_min)
    return out


# --------------------------------------------------------------------------
# NWS office (WFO / CWA) catalogue
# --------------------------------------------------------------------------
class WfoCatalog:
    def __init__(self, geojson: dict):
        self.codes: List[str] = []
        self.info: Dict[str, dict] = {}
        self.polys: Dict[str, List[np.ndarray]] = {}
        for f in geojson["features"]:
            p = f["properties"]
            code = p.get("cwa") or p.get("wfo")
            if not code:
                continue
            g = f["geometry"]
            if g is None:
                continue
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            self.polys.setdefault(code, [])
            for poly in polys:
                a = np.array(poly[0], dtype=float)[:, :2]
                if np.abs(a).max() > 360.0:            # EPSG:3857 meters, not degrees
                    a = np.column_stack(_from_webmercator(a[:, 0], a[:, 1]))
                self.polys[code].append(a)
            if code not in self.info:
                self.codes.append(code)
                self.info[code] = {"name": p.get("citystate") or p.get("city", code),
                                   "region": p.get("region", ""),
                                   "state": p.get("st", "")}
        self._rasters: Dict[tuple, np.ndarray] = {}

    @classmethod
    def fetch(cls, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / "cwa_boundary.geojson"
        if not p.exists() or time.time() - p.stat().st_mtime > 30 * 86400:
            try:
                r = requests.get(WFO_URL, timeout=90)
                r.raise_for_status()
                p.write_bytes(r.content)
            except requests.RequestException:
                if not p.exists():
                    raise
                log.warning("WFO boundary refresh failed; using cached copy")
        return cls(json.loads(p.read_text()))

    def raster(self, grid: Grid) -> np.ndarray:
        """uint8 label raster: 0 = outside any office, i+1 = codes[i]."""
        key = (grid.lat0, grid.lon0, grid.dlat, grid.dlon, grid.ny, grid.nx)
        if key in self._rasters:
            return self._rasters[key]
        if Image is None:
            raise RuntimeError("pillow required for WFO rasterisation")
        img = Image.new("L", (grid.nx, grid.ny), 0)
        d = ImageDraw.Draw(img)
        for i, code in enumerate(self.codes):
            for ring in self.polys[code]:
                lon = ring[:, 0]
                r = (grid.lat0 - ring[:, 1]) / grid.dlat      # rows run north -> south
                c = (lon - grid.lon0) / grid.dlon
                lon2 = (lon + 180.0) % 360.0 - 180.0          # keep -180..180
                c = np.where(lon2 <= lon + 180.0, c, c)
                d.polygon(list(zip(c.tolist(), r.tolist())), fill=i + 1)
        arr = np.asarray(img)
        self._rasters[key] = arr
        return arr

    def lookup(self, grid: Grid, lat, lon) -> str:
        r, c = grid.rc(lat, lon)
        r, c = int(round(float(r))), int(round(float(c)))
        if 0 <= r < grid.ny and 0 <= c < grid.nx:
            v = int(self.raster(grid)[r, c])
            return self.codes[v - 1] if v else ""
        return ""


# --------------------------------------------------------------------------
# Cell segmentation
# --------------------------------------------------------------------------
@dataclass
class CellObs:
    t: dt.datetime
    lat: float
    lon: float
    area_km2: float
    cref: float
    z10: float
    mesh: float
    mesh_area_km2: float
    flash_per_min: float
    jump: float
    prob60: float
    wfo: str = ""
    domain: str = "CONUS"


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def segment_cells(fr: Dict[str, np.ndarray], grid: Grid, cfg: Config,
                  t: dt.datetime, domain: str, wfos: Optional[WfoCatalog]):
    """Find convective cells in one scan and compute per-cell features."""
    cref = fr["cref"].astype(np.float32)
    mask = cref >= cfg.detect_dbz
    if not mask.any():
        return []
    sm = ndimage.gaussian_filter(cref, 1.0)
    comp, ncomp = ndimage.label(mask, structure=np.ones((3, 3)))

    # split merged echoes at distinct reflectivity maxima
    rad = max(1, int(round(cfg.split_sep_km / grid.km_per_px)))
    peak = (ndimage.maximum_filter(sm, footprint=_disk(rad)) == sm) & mask
    seeds, nseed = ndimage.label(peak, structure=np.ones((3, 3)))
    lab = comp.copy()
    if nseed > ncomp:
        sc = ndimage.maximum(comp, seeds, index=np.arange(1, nseed + 1)).astype(int)
        nper = np.bincount(sc, minlength=ncomp + 1)
        nxt = ncomp + 1
        for ci in np.nonzero(nper > 1)[0]:
            if ci == 0:
                continue
            sl = ndimage.find_objects((comp == ci).astype(np.uint8))[0]
            sub_c, sub_s = comp[sl] == ci, seeds[sl] * (comp[sl] == ci)
            _, (iy, ix) = ndimage.distance_transform_edt(sub_s == 0, return_indices=True)
            near = sub_s[iy, ix]
            ids = np.unique(near[sub_c])
            new = np.zeros_like(sub_c, dtype=lab.dtype)
            for j, sid in enumerate(ids):
                new[sub_c & (near == sid)] = nxt + j
            lab[sl][sub_c] = new[sub_c]
            nxt += len(ids)

    ids = np.unique(lab[lab > 0])
    slices = ndimage.find_objects(lab)
    area_row = grid.px_area_km2
    z10, mesh = fr.get("z10"), fr.get("mesh")
    nld, jmp, prb = fr.get("nldn"), fr.get("jump"), fr.get("prob")
    out: List[CellObs] = []
    for cid in ids:
        sl = slices[cid - 1] if cid - 1 < len(slices) else None
        if sl is None:
            continue
        r0, r1 = max(sl[0].start - 3, 0), min(sl[0].stop + 3, grid.ny)
        c0, c1 = max(sl[1].start - 3, 0), min(sl[1].stop + 3, grid.nx)
        win = (slice(r0, r1), slice(c0, c1))
        m = lab[win] == cid
        ar = area_row[r0:r1][:, None] * np.ones((1, c1 - c0), np.float32)
        area = float(ar[m].sum())
        if area < cfg.min_area_km2:
            continue
        md = ndimage.binary_dilation(m, iterations=2)       # ~4 km footprint
        rr, cc = np.nonzero(m)
        lat, lon = grid.latlon(rr.mean() + r0, cc.mean() + c0)

        def mx(a, mm=md, default=np.nan):
            return float(a[win][mm].max()) if a is not None else default
        mesh_v = mx(mesh, default=0.0)
        mesh_a = float(ar[md & (mesh[win] >= SEVERE_MESH_MM)].sum()) \
            if mesh is not None else 0.0
        fl = float((nld[win][md] * ar[md]).sum()) if nld is not None else np.nan
        obs = CellObs(t, float(lat), float(lon), area, float(cref[win][m].max()),
                      mx(z10), mesh_v, mesh_a, fl, mx(jmp), mx(prb), domain=domain)
        if wfos is not None:
            obs.wfo = wfos.lookup(grid, obs.lat, obs.lon)
        out.append(obs)
    return out


# --------------------------------------------------------------------------
# Motion: phase-correlation (TREC-style) tile flow
# --------------------------------------------------------------------------
class FlowField:
    def __init__(self, grid: Grid, vx, vy, stride):
        self.grid, self.vx, self.vy, self.stride = grid, vx, vy, stride

    def sample(self, lat, lon):
        r, c = self.grid.rc(lat, lon)
        i, j = int(round(float(r) / self.stride)), int(round(float(c) / self.stride))
        for rad in (0, 1, 2):
            sl = (slice(max(i - rad, 0), i + rad + 1), slice(max(j - rad, 0), j + rad + 1))
            vx, vy = self.vx[sl], self.vy[sl]
            ok = np.isfinite(vx)
            if ok.any():
                return float(np.median(vx[ok])), float(np.median(vy[ok]))
        return None


def estimate_flow(prev: np.ndarray, cur: np.ndarray, grid: Grid, dt_min: float,
                  tile=48, vmax_kmmin=2.4):
    """Per-tile advection (km/min east, north) via phase correlation."""
    stride = tile // 2
    ny, nx = (grid.ny - tile) // stride + 1, (grid.nx - tile) // stride + 1
    vx = np.full((max(ny, 1), max(nx, 1)), np.nan, np.float32)
    vy = vx.copy()
    if ny < 1 or nx < 1:
        return FlowField(grid, vx, vy, stride)
    win = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    a = np.clip(prev.astype(np.float32), 0, None)
    b = np.clip(cur.astype(np.float32), 0, None)
    kmy = grid.km_per_px
    kmx_row = grid.dlon * 111.32 * np.cos(np.radians(grid.row_lat))
    maxpx = max(1, int(vmax_kmmin * dt_min / kmy))
    act = ((a >= 20).astype(np.float32), (b >= 20).astype(np.float32))
    for i in range(ny):
        r0 = i * stride
        for j in range(nx):
            c0 = j * stride
            if act[0][r0:r0 + tile, c0:c0 + tile].mean() < 0.06 or \
               act[1][r0:r0 + tile, c0:c0 + tile].mean() < 0.06:
                continue
            ta = a[r0:r0 + tile, c0:c0 + tile]
            tb = b[r0:r0 + tile, c0:c0 + tile]
            fa = np.fft.rfft2((ta - ta.mean()) * win)
            fb = np.fft.rfft2((tb - tb.mean()) * win)
            cross = fb * np.conj(fa)
            cross /= np.abs(cross) + 1e-6
            corr = np.fft.irfft2(cross, s=(tile, tile))
            corr = np.fft.fftshift(corr)
            ctr = tile // 2
            sub = corr[ctr - maxpx:ctr + maxpx + 1, ctr - maxpx:ctr + maxpx + 1]
            if sub.size == 0:
                continue
            py, px = np.unravel_index(np.argmax(sub), sub.shape)
            if sub[py, px] < 0.12:
                continue
            dy = py - maxpx + _parab(sub, py, px, 0)
            dx = px - maxpx + _parab(sub, py, px, 1)
            rowc = min(r0 + tile // 2, grid.ny - 1)
            vx[i, j] = dx * kmx_row[rowc] / dt_min
            vy[i, j] = -dy * kmy / dt_min
    return FlowField(grid, vx, vy, stride)


def _parab(sub, py, px, axis):
    try:
        if axis == 0:
            l, c, r = sub[py - 1, px], sub[py, px], sub[py + 1, px]
        else:
            l, c, r = sub[py, px - 1], sub[py, px], sub[py, px + 1]
        if py - 1 < 0 or px - 1 < 0:
            return 0.0
        den = l - 2 * c + r
        return float(0.5 * (l - r) / den) if abs(den) > 1e-9 else 0.0
    except IndexError:
        return 0.0


# --------------------------------------------------------------------------
# Tracking (Hungarian assignment + Kalman filter)
# --------------------------------------------------------------------------
@dataclass
class Track:
    tid: int
    ref_lat: float
    ref_lon: float
    x: np.ndarray                  # [x, y, vx, vy] km, km/min
    P: np.ndarray
    t_last: dt.datetime
    obs: List[CellObs] = field(default_factory=list)
    miss: int = 0

    @property
    def cos_ref(self):
        return math.cos(math.radians(self.ref_lat))

    def to_xy(self, lat, lon):
        return ((lon - self.ref_lon) * 111.32 * self.cos_ref, (lat - self.ref_lat) * KM_LAT)

    def to_ll(self, x, y):
        return self.ref_lat + y / KM_LAT, self.ref_lon + x / (111.32 * self.cos_ref)

    @property
    def age_min(self):
        return (self.obs[-1].t - self.obs[0].t).total_seconds() / 60.0 if self.obs else 0.0

    @property
    def speed_ms(self):
        return float(np.hypot(self.x[2], self.x[3]) * 1000.0 / 60.0)

    @property
    def heading_deg(self):
        """Direction the cell moves TOWARD, deg clockwise from north."""
        return float((math.degrees(math.atan2(self.x[2], self.x[3])) + 360) % 360)


SIGMA_A = 0.008      # km/min^2  process (storm acceleration) noise
SIGMA_M = 3.5        # km        centroid measurement noise


def _F(dt_m):
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt_m
    return F


def _Q(dt_m):
    q = SIGMA_A ** 2
    G = np.array([[dt_m ** 2 / 2, 0], [0, dt_m ** 2 / 2], [dt_m, 0], [0, dt_m]])
    return q * G @ G.T


class CellTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracks: List[Track] = []
        self.dead: List[Track] = []
        self._next = 1

    def _new(self, o: CellObs, flow: Optional[FlowField]) -> Track:
        v = flow.sample(o.lat, o.lon) if flow else None
        x = np.array([0.0, 0.0, v[0] if v else 0.0, v[1] if v else 0.0])
        sv = 0.5 if v else 1.2
        tr = Track(self._next, o.lat, o.lon, x,
                   np.diag([SIGMA_M ** 2, SIGMA_M ** 2, sv ** 2, sv ** 2]), o.t, [o])
        self._next += 1
        return tr

    def step(self, obs: List[CellObs], t: dt.datetime, flow: Optional[FlowField]):
        live = [tr for tr in self.tracks if tr.miss <= self.cfg.max_miss]
        preds = []
        for tr in live:
            dtm = (t - tr.t_last).total_seconds() / 60.0
            F = _F(dtm)
            preds.append((F @ tr.x, F @ tr.P @ F.T + _Q(dtm)))
        n, m = len(live), len(obs)
        matched_o, matched_t = set(), set()
        if n and m:
            plat = np.empty(n)
            plon = np.empty(n)
            gate = np.empty(n)
            for i, (tr, (xp, Pp)) in enumerate(zip(live, preds)):
                plat[i], plon[i] = tr.to_ll(xp[0], xp[1])
                gate[i] = min(60.0, 10.0 + 2.5 * math.sqrt(max(Pp[0, 0], Pp[1, 1])))
            olat = np.array([o.lat for o in obs])
            olon = np.array([o.lon for o in obs])
            ar_o = np.array([max(o.area_km2, 1.0) for o in obs])
            ar_t = np.array([max(tr.obs[-1].area_km2, 1.0) for tr in live])
            cosl = np.cos(np.radians((plat[:, None] + olat[None, :]) / 2))
            dx = (olon[None, :] - plon[:, None]) * 111.32 * cosl
            dy = (olat[None, :] - plat[:, None]) * KM_LAT
            dist = np.hypot(dx, dy)
            cost = dist + 6.0 * np.abs(np.log(ar_o[None, :] / ar_t[:, None]))
            cost = np.where(dist <= gate[:, None], cost, 1e6)
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] >= 1e6:
                    continue
                tr, (xp, Pp) = live[i], preds[i]
                z = np.array(tr.to_xy(obs[j].lat, obs[j].lon))
                H = np.zeros((2, 4))
                H[0, 0] = H[1, 1] = 1
                S = H @ Pp @ H.T + np.eye(2) * SIGMA_M ** 2
                K = Pp @ H.T @ np.linalg.inv(S)
                x = xp + K @ (z - H[:, :] @ xp)
                P = (np.eye(4) - K @ H) @ Pp
                if flow is not None and len(tr.obs) < 3:       # young: fuse flow prior
                    v = flow.sample(obs[j].lat, obs[j].lon)
                    if v:
                        Hv = np.zeros((2, 4))
                        Hv[0, 2] = Hv[1, 3] = 1
                        Sv = Hv @ P @ Hv.T + np.eye(2) * 0.5 ** 2
                        Kv = P @ Hv.T @ np.linalg.inv(Sv)
                        x = x + Kv @ (np.array(v) - Hv @ x)
                        P = (np.eye(4) - Kv @ Hv) @ P
                spd = np.hypot(x[2], x[3])
                if spd > 2.4:                                   # cap at ~40 m/s
                    x[2:] *= 2.4 / spd
                tr.x, tr.P, tr.t_last, tr.miss = x, P, t, 0
                tr.obs.append(obs[j])
                matched_o.add(j)
                matched_t.add(i)
        for i, tr in enumerate(live):
            if i not in matched_t:
                tr.miss += 1
                if i < len(preds):
                    tr.x, tr.P, tr.t_last = preds[i][0], preds[i][1], t
        for j, o in enumerate(obs):
            if j not in matched_o:
                self.tracks.append(self._new(o, flow))
        keep = []
        for tr in self.tracks:
            (keep if tr.miss <= self.cfg.max_miss else self.dead).append(tr)
        self.tracks = keep


# --------------------------------------------------------------------------
# Lightning threat 55-60 minutes ahead
# --------------------------------------------------------------------------
def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def _trend(track: Track, attr: str, span_min=30.0):
    """OLS slope per minute of a cell attribute over the last span."""
    t_end = track.obs[-1].t
    pts = [((o.t - t_end).total_seconds() / 60.0, getattr(o, attr))
           for o in track.obs if (t_end - o.t).total_seconds() / 60.0 <= span_min]
    pts = [(a, b) for a, b in pts if np.isfinite(b)]
    if len(pts) < 3 or pts[-1][0] - pts[0][0] < 10:
        return 0.0
    a, b = np.array(pts).T
    return float(np.polyfit(a, b, 1)[0])


def forecast_position(tr: Track, lead_min: float):
    F = _F(lead_min)
    xf = F @ tr.x
    Pf = F @ tr.P @ F.T + _Q(lead_min) * 1.0
    return xf, Pf


def lightning_forecast(tr: Track, cfg: Config, wfos: Optional[WfoCatalog],
                       grids: Dict[str, Grid]) -> dict:
    o = tr.obs[-1]
    lo, hi = cfg.lead_min
    mid = 0.5 * (lo + hi)
    xm, Pm = forecast_position(tr, mid)
    lat_m, lon_m = tr.to_ll(xm[0], xm[1])
    xlo, _ = forecast_position(tr, lo)
    xhi, _ = forecast_position(tr, hi)
    lat_lo, lon_lo = tr.to_ll(xlo[0], xlo[1])
    lat_hi, lon_hi = tr.to_ll(xhi[0], xhi[1])

    # recent lightning in the cell footprint (any of last ~18 min)
    recent = [x.flash_per_min for x in tr.obs[-3:] if np.isfinite(x.flash_per_min)]
    has_ltg_data = bool(recent) or np.isfinite(o.prob60)
    active = bool(recent) and max(recent) > 0.0

    z_now = o.z10 if np.isfinite(o.z10) and o.z10 > -20 else o.cref - 10.0
    attr = "z10" if np.isfinite(o.z10) else "cref"
    dz = max(-0.4, min(0.4, _trend(tr, attr)))                 # dB/min, bounded
    tau_d = 20.0
    z_proj = z_now + dz * tau_d * (1 - math.exp(-mid / tau_d))
    p_elec = _sig(0.20 * (z_proj - 33.0))
    p_elec *= min(1.0, math.sqrt(max(o.area_km2, 1.0) / 40.0))   # small-cell penalty
    p_prob = o.prob60 / 100.0 if np.isfinite(o.prob60) else None
    if p_prob is not None:
        p_elec = 0.65 * p_elec + 0.35 * p_prob
    if active:
        p_elec = max(p_elec, 0.90)

    boost = 1.2 if (np.isfinite(o.jump) and o.jump > 0) else 1.0
    # Persistence: extrapolate the observed core-dBZ trend.  A cell whose
    # 35 dBZ core is forecast to collapse before the window gets a survival
    # term that decays with a ~15 min lightning lag after core death
    # (dissipating cells keep producing flashes for a short while).
    # Steady or growing cells use a slow exponential decay instead.
    dcore = _trend(tr, "cref", span_min=30.0)            # dBZ per minute
    if dcore < -0.02 and o.cref > 30.0:
        t_death = (o.cref - 35.0) / (-dcore)             # minutes to 35 dBZ core
        if t_death < hi + 30.0:                          # alive (1.0) until t_death
            surv = 0.5 * (math.exp(-max(lo - t_death, 0.0) / 15.0)
                          + math.exp(-max(hi - t_death, 0.0) / 15.0))
        else:
            surv = 1.0
    else:
        tau = boost * (180.0 + 1.5 * max(o.cref - 40.0, 0.0)
                       + 60.0 * max(dcore, 0.0))
        surv = 0.5 * (math.exp(-lo / tau) + math.exp(-hi / tau))
    p = float(min(1.0, surv * p_elec))

    # threat ellipse around the 55-60 min position
    hd = math.radians(tr.heading_deg)
    ux, uy = math.sin(hd), math.cos(hd)
    s_al = math.sqrt(max(ux * ux * Pm[0, 0] + 2 * ux * uy * Pm[0, 1] + uy * uy * Pm[1, 1], 0))
    s_cr = math.sqrt(max(uy * uy * Pm[0, 0] - 2 * ux * uy * Pm[0, 1] + ux * ux * Pm[1, 1], 0))
    r_cell = min(25.0, math.sqrt(o.area_km2 / math.pi))
    a = r_cell + 1.5 * s_al + 0.5 * (hi - lo) * tr.speed_ms * 0.06
    b = r_cell + 1.5 * s_cr
    ring = []
    for ang in np.linspace(0, 2 * math.pi, 25):
        e, n_ = a * math.cos(ang), b * math.sin(ang)
        ring.append(tr.to_ll(xm[0] + e * ux + n_ * uy, xm[1] + e * uy - n_ * ux))

    wfo_now = o.wfo
    wfo_fut = ""
    if wfos is not None and o.domain in grids:
        wfo_fut = wfos.lookup(grids[o.domain], lat_m, lon_m)
    level = ("VERY HIGH" if p >= 0.75 else "HIGH" if p >= 0.5
             else "MODERATE" if p >= 0.2 else "LOW")
    conf = ("high" if tr.age_min >= 24 and len(tr.obs) >= 4 else
            "medium" if len(tr.obs) >= 3 else "low")
    return {
        "p_lightning_55_60": round(p, 3),
        "p_first_flash_55_60": 0.0 if active else round(p, 3),
        "lightning_active_now": active,
        "radar_only": not has_ltg_data,
        "threat_level": level,
        "forecast_lat_57": round(lat_m, 4), "forecast_lon_57": round(lon_m, 4),
        "forecast_lat_55": round(lat_lo, 4), "forecast_lon_55": round(lon_lo, 4),
        "forecast_lat_60": round(lat_hi, 4), "forecast_lon_60": round(lon_hi, 4),
        "wfo_now": wfo_now, "wfo_at_55_60": wfo_fut,
        "pos_sigma_km": round(float(math.hypot(s_al, s_cr)), 1),
        "track_confidence": conf,
        "threat_polygon": [[round(lo_, 4), round(la_, 4)] for la_, lo_ in ring],
        "survival": round(surv, 3), "p_electrify": round(p_elec, 3),
    }


# --------------------------------------------------------------------------
# MESH progression in rolling 45-minute windows
# --------------------------------------------------------------------------
def hail_class(mm: float) -> str:
    if mm >= 76.2:
        return "baseball+ (>=3 in)"
    if mm >= SIG_MESH_MM:
        return "significant (>=2 in)"
    if mm >= 44.5:
        return "golf ball (>=1.75 in)"
    if mm >= SEVERE_MESH_MM:
        return "severe (>=1 in)"
    if mm >= 12.7:
        return "sub-severe (>=0.5 in)"
    return "none/small"


def mesh_windows(tr: Track, cfg: Config) -> List[dict]:
    """Rolling 45-min MESH windows ending now, -stride, -2*stride, ..."""
    w, stride = cfg.mesh_window_min, cfg.mesh_stride_min
    t_end = tr.obs[-1].t
    out = []
    k = 0
    while True:
        we = t_end - dt.timedelta(minutes=k * stride)
        ws = we - dt.timedelta(minutes=w)
        pts = [(o.t, o.mesh, o.mesh_area_km2) for o in tr.obs if ws <= o.t <= we]
        if len(pts) < 3 or (k > 0 and tr.obs[0].t > we):
            break
        ts = np.array([(p[0] - ws).total_seconds() / 3600.0 for p in pts])
        ms = np.array([p[1] for p in pts], float)
        slope = float(np.polyfit(ts, ms, 1)[0]) if np.ptp(ts) > 0.15 else 0.0
        third = max(1, len(ms) // 3)
        change = float(ms[-third:].mean() - ms[:third].mean())
        ip = int(np.argmax(ms))
        cat = ("intensifying" if slope >= 6.0 and change > 2 else
               "weakening" if slope <= -6.0 and change < -2 else "steady")
        out.append({
            "window_start": ws.isoformat(), "window_end": we.isoformat(),
            "n_scans": len(pts),
            "covers_full_window": bool(tr.obs[0].t <= ws + dt.timedelta(minutes=cfg.step_min)),
            "mesh_start_mm": round(float(ms[0]), 1),
            "mesh_end_mm": round(float(ms[-1]), 1),
            "mesh_peak_mm": round(float(ms.max()), 1),
            "peak_time": pts[ip][0].isoformat(),
            "trend_mm_per_h": round(slope, 1),
            "change_mm": round(change, 1),
            "trend": cat,
            "peak_hail_class": hail_class(float(ms.max())),
            "severe_in_window": bool(ms.max() >= SEVERE_MESH_MM),
            "crossed_severe": bool(ms[0] < SEVERE_MESH_MM <= ms.max()),
            "max_severe_core_km2": round(float(max(p[2] for p in pts)), 1),
        })
        k += 1
        if k > int(180 // stride):
            break
    return out


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
class FrameSource:
    """Interface: timeline(domain) -> [datetimes newest first];
    frame(domain, t) -> (dict[str, ndarray], Grid)."""


class LiveSource(FrameSource):
    def __init__(self, cfg: Config, extra_points=None):
        self.f = MrmsFeeder(cfg)
        self.cfg = cfg
        self.points = extra_points or []

    def timeline(self, domain):
        return self.f.timeline(domain)

    def prepare(self, domain, times):
        self.f.prefetch(domain, times)

    def frame(self, domain, t):
        fr, grid = {}, None
        for key in PRODUCTS:
            res = self.f.load(domain, key, t)
            if res is None:
                continue
            fr[key], grid = res[0], (grid or res[1])
        if "cref" not in fr:
            return None
        if self.points and domain == "CONUS":
            extra = points_to_density(self.points, grid, t)
            if "nldn" in fr:
                fr["nldn"] = np.maximum(fr["nldn"].astype(np.float32), extra)
            else:
                fr["nldn"] = extra
        return fr, grid


class StormPipeline:
    def __init__(self, cfg: Config, source: FrameSource, wfos: Optional[WfoCatalog]):
        self.cfg, self.src, self.wfos = cfg, source, wfos
        self.grids: Dict[str, Grid] = {}
        self.trackers: Dict[str, CellTracker] = {}
        self.latest: Dict[str, dt.datetime] = {}
        self.status: Dict[str, str] = {}

    def run_domain(self, domain: str):
        times = self.src.timeline(domain)
        if not times:
            self.status[domain] = "no data"
            return
        times = sorted(times)                                      # oldest -> newest
        if hasattr(self.src, "prepare"):
            self.src.prepare(domain, times)
        trk = CellTracker(self.cfg)
        prev, prev_t = None, None
        n_ok = 0
        for t in times:
            got = self.src.frame(domain, t)
            if got is None:
                continue
            fr, grid = got
            self.grids[domain] = grid
            obs = segment_cells(fr, grid, self.cfg, t, domain, self.wfos)
            flow = None
            if prev is not None:
                flow = estimate_flow(prev, fr["cref"], grid,
                                     (t - prev_t).total_seconds() / 60.0)
            trk.step(obs, t, flow)
            prev, prev_t = fr["cref"], t
            n_ok += 1
        self.trackers[domain] = trk
        self.latest[domain] = times[-1]
        self.status[domain] = f"ok ({n_ok} scans, newest {times[-1]:%H:%MZ})"

    def run(self):
        for d in self.cfg.domains:
            try:
                self.run_domain(d)
            except Exception as e:                       # one domain must not stop the rest
                log.exception("domain %s failed", d)
                self.status[d] = f"error: {e}"
        return self.products()

    def products(self):
        rows = []
        for domain, trk in self.trackers.items():
            t_new = self.latest[domain]
            for tr in trk.tracks:
                if tr.miss > 0 or tr.obs[-1].t != t_new:      # only cells seen in newest scan
                    continue
                o = tr.obs[-1]
                lf = lightning_forecast(tr, self.cfg, self.wfos, self.grids)
                mw = mesh_windows(tr, self.cfg)
                rows.append({
                    "track_id": f"{domain[:2]}{tr.tid}", "domain": domain,
                    "time": o.t.isoformat(), "lat": round(o.lat, 4), "lon": round(o.lon, 4),
                    "wfo": o.wfo, "area_km2": round(o.area_km2, 1),
                    "cref_dbz": round(o.cref, 1),
                    "z_minus10C_dbz": None if not np.isfinite(o.z10) else round(o.z10, 1),
                    "mesh_mm": round(o.mesh, 1), "hail_class": hail_class(o.mesh),
                    "speed_ms": round(tr.speed_ms, 1), "heading_deg": round(tr.heading_deg),
                    "age_min": round(tr.age_min), "n_scans": len(tr.obs),
                    "lightning": lf, "mesh_windows": mw,
                    "current_window": mw[0] if mw else None,
                })
        if self.cfg.office:
            off = self.cfg.office.upper()
            rows = [r for r in rows if off in (r["wfo"], r["lightning"]["wfo_at_55_60"])]
        return rows

    def wfo_summary(self, rows):
        by = {}
        codes = self.wfos.codes if self.wfos else sorted({r["wfo"] for r in rows})
        for c in codes:
            by[c] = {"wfo": c, "name": (self.wfos.info[c]["name"] if self.wfos else c),
                     "n_cells": 0, "n_severe_mesh_cells": 0, "max_mesh_mm": 0.0,
                     "n_intensifying_hail_cells": 0, "n_active_lightning_cells": 0,
                     "n_cells_arriving_55_60": 0, "max_p_lightning_arriving": 0.0,
                     "n_first_flash_candidates_arriving": 0}
        for r in rows:
            L = r["lightning"]
            s = by.get(r["wfo"])
            if s:
                s["n_cells"] += 1
                s["max_mesh_mm"] = max(s["max_mesh_mm"], r["mesh_mm"])
                s["n_severe_mesh_cells"] += r["mesh_mm"] >= SEVERE_MESH_MM
                cw = r["current_window"]
                s["n_intensifying_hail_cells"] += bool(cw and cw["trend"] == "intensifying"
                                                        and cw["mesh_peak_mm"] >= 12.7)
                s["n_active_lightning_cells"] += L["lightning_active_now"]
            f = by.get(L["wfo_at_55_60"])
            if f and L["p_lightning_55_60"] >= 0.2:
                f["n_cells_arriving_55_60"] += 1
                f["max_p_lightning_arriving"] = max(f["max_p_lightning_arriving"],
                                                    L["p_lightning_55_60"])
                f["n_first_flash_candidates_arriving"] += (
                    not L["lightning_active_now"] and L["p_lightning_55_60"] >= 0.5)
        return list(by.values())


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def write_outputs(rows, summary, pipe: StormPipeline, out_dir: Path):
    stamp = max(pipe.latest.values()).strftime("%Y%m%dT%H%MZ") if pipe.latest else "none"
    d = out_dir / stamp
    d.mkdir(parents=True, exist_ok=True)

    cells, swaths = [], []
    for r in rows:
        L = r["lightning"]
        props = {k: v for k, v in r.items() if k not in ("lightning", "mesh_windows",
                                                          "current_window")}
        props.update({k: v for k, v in L.items() if k != "threat_polygon"})
        if r["current_window"]:
            props.update({"mesh45_" + k: v for k, v in r["current_window"].items()})
        cells.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]}})
        swaths.append({"type": "Feature", "properties": {
            "track_id": r["track_id"], "wfo_now": r["wfo"], "wfo_at_55_60": L["wfo_at_55_60"],
            "p_lightning_55_60": L["p_lightning_55_60"], "threat_level": L["threat_level"]},
            "geometry": {"type": "Polygon", "coordinates": [L["threat_polygon"]]}})
        swaths.append({"type": "Feature", "properties": {"track_id": r["track_id"], "kind": "path"},
                       "geometry": {"type": "LineString", "coordinates": [
                           [r["lon"], r["lat"]], [L["forecast_lon_55"], L["forecast_lat_55"]],
                           [L["forecast_lon_60"], L["forecast_lat_60"]]]}})
    (d / "cells.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": cells}))
    (d / "lightning_threat_55_60.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": swaths}))
    (d / "wfo_summary.json").write_text(json.dumps(summary, indent=1))
    with open(d / "wfo_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["wfo"])
        w.writeheader()
        w.writerows(summary)
    wcols = ["window_start", "window_end", "n_scans", "covers_full_window", "mesh_start_mm",
             "mesh_end_mm", "mesh_peak_mm", "peak_time", "trend_mm_per_h", "change_mm", "trend",
             "peak_hail_class", "severe_in_window", "crossed_severe", "max_severe_core_km2"]
    with open(d / "mesh_windows_45min.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "wfo", "lat", "lon"] + wcols)
        for r in rows:
            for mw in r["mesh_windows"]:
                w.writerow([r["track_id"], r["wfo"], r["lat"], r["lon"]] + [mw[c] for c in wcols])
    (d / "tracks.json").write_text(json.dumps(rows, indent=1, default=float))
    with open(out_dir / "forecast_log.jsonl", "a") as f:        # for later verification
        for r in rows:
            L = r["lightning"]
            f.write(json.dumps({"issued": r["time"], "track": r["track_id"],
                                "lat": r["lat"], "lon": r["lon"], "cref": r["cref_dbz"],
                                "z10": r["z_minus10C_dbz"], "mesh": r["mesh_mm"],
                                "area": r["area_km2"], "age": r["age_min"],
                                "active": L["lightning_active_now"],
                                "p": L["p_lightning_55_60"],
                                "fc_lat": L["forecast_lat_57"], "fc_lon": L["forecast_lon_57"],
                                "sigma_km": L["pos_sigma_km"]}) + "\n")
    return d


def print_report(rows, summary, pipe: StormPipeline, top=15):
    print("\nDomain status:")
    for k, v in pipe.status.items():
        print(f"  {k:7s} {v}")
    print(f"\n{len(rows)} tracked cells.  Top lightning threats 55-60 min ahead:")
    print(f"{'ID':8s}{'WFO now':8s}{'->WFO':7s}{'P(ltg)':8s}{'level':10s}{'1st flash':10s}"
          f"{'dBZ':6s}{'MESH':6s}{'v m/s':7s}{'dir':5s}{'conf':7s}")
    for r in sorted(rows, key=lambda r: -r["lightning"]["p_lightning_55_60"])[:top]:
        L = r["lightning"]
        print(f"{r['track_id']:8s}{r['wfo']:8s}{L['wfo_at_55_60']:7s}"
              f"{L['p_lightning_55_60']:<8.2f}{L['threat_level']:10s}"
              f"{'yes' if not L['lightning_active_now'] and L['p_lightning_55_60'] >= .5 else '-':10s}"
              f"{r['cref_dbz']:<6.0f}{r['mesh_mm']:<6.1f}{r['speed_ms']:<7.1f}"
              f"{r['heading_deg']:<5d}{L['track_confidence']:7s}")
    print("\nSevere hail (MESH 45-min windows, peak >= 1 in):")
    print(f"{'ID':8s}{'WFO':6s}{'start':7s}{'end':7s}{'peak':7s}{'trend':14s}{'mm/h':7s}{'class'}")
    sev = [r for r in rows if r["current_window"] and r["current_window"]["severe_in_window"]]
    for r in sorted(sev, key=lambda r: -r["current_window"]["mesh_peak_mm"])[:top]:
        w = r["current_window"]
        print(f"{r['track_id']:8s}{r['wfo']:6s}{w['mesh_start_mm']:<7.1f}{w['mesh_end_mm']:<7.1f}"
              f"{w['mesh_peak_mm']:<7.1f}{w['trend']:14s}{w['trend_mm_per_h']:<7.1f}"
              f"{w['peak_hail_class']}")
    if not sev:
        print("  none")
    act = [s for s in summary if s["n_cells"] or s["n_cells_arriving_55_60"]]
    print(f"\nOffices with activity now or arriving ({len(act)} of {len(summary)}):")
    for s in sorted(act, key=lambda s: -(s["n_severe_mesh_cells"] * 100 + s["n_cells"]))[:25]:
        print(f"  {s['wfo']:4s} cells {s['n_cells']:<4d} severe-MESH {s['n_severe_mesh_cells']:<3d} "
              f"maxMESH {s['max_mesh_mm']:<5.1f} ltg-active {s['n_active_lightning_cells']:<3d} "
              f"arriving55-60 {s['n_cells_arriving_55_60']:<3d} "
              f"first-flash {s['n_first_flash_candidates_arriving']:<2d} {s['name']}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--office", help="3-letter WFO id (cells over it, or forecast into it)")
    ap.add_argument("--domains", default="all", help="comma list of CONUS,ALASKA,HAWAII,CARIB,GUAM")
    ap.add_argument("--history", type=float, default=90.0, help="minutes of scans to track")
    ap.add_argument("--step", type=float, default=6.0, help="minutes between scans used")
    ap.add_argument("--coarsen", type=int, default=2, help="block size (2 => ~2 km)")
    ap.add_argument("--out", default="storm_out")
    ap.add_argument("--cache", default="mrms_cache")
    ap.add_argument("--strokes", help="CSV of lightning strokes/flashes (time,lat,lon)")
    ap.add_argument("--lma", help="directory of LMA .dat/.dat.gz files")
    ap.add_argument("--loop", type=float, default=2.0, metavar="MIN",
                    help="refresh every MIN minutes (default 2, the MRMS update rate)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(step_min=a.step, history_min=a.history, coarsen=a.coarsen,
                 cache_dir=Path(a.cache), out_dir=Path(a.out), office=a.office,
                 domains=DOMAINS if a.domains == "all"
                 else tuple(x.strip().upper() for x in a.domains.split(",")))

    points = []
    if a.strokes:
        points += read_stroke_csv(Path(a.strokes))
    if a.lma:
        for p in sorted(Path(a.lma).glob("*.dat*")):
            points += read_lma_dat(p)
    if points:
        print(f"Loaded {len(points)} supplementary lightning points")
    wfos = None
    print(f"Live mode: refreshing every {a.loop:g} min (Ctrl+C to stop)")
    try:
        while True:
            t0 = time.time()
            try:                      # a network/data hiccup must not stop the live loop
                if wfos is None:
                    wfos = WfoCatalog.fetch(cfg.cache_dir)
                    print(f"Loaded {len(wfos.codes)} NWS office boundaries")
                src = LiveSource(cfg, points)
                pipe = StormPipeline(cfg, src, wfos)
                rows = pipe.run()
                summary = pipe.wfo_summary(rows)
                if pipe.latest:
                    d = write_outputs(rows, summary, pipe, cfg.out_dir)
                    print(f"\n===== {dt.datetime.now(UTC):%Y-%m-%d %H:%M:%SZ} =====")
                    print_report(rows, summary, pipe, a.top)
                    print(f"\nProducts written to {d}  ({time.time() - t0:.0f}s)")
                else:
                    print("No MRMS data retrieved; will retry.", file=sys.stderr)
                src.f.prune()
            except Exception as e:
                log.exception("live update failed")
                print(f"Update failed ({e}); retrying next cycle.", file=sys.stderr)
            time.sleep(max(5.0, a.loop * 60 - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
