#!/usr/bin/env python3
"""
nexrad_radar.py -- live NEXRAD Level III radar from the NWS tgftp server.

Products come from
    https://tgftp.nws.noaa.gov/SL.us008001/DF.of/DC.radar/
  DS.p94r0  N0Q  base reflectivity, 0.5 deg tilt, 1 km bins, 460 km range
  DS.p99v0  N0U  base velocity,     0.5 deg tilt, 0.25 km bins, 300 km range
Each site folder (SI.kxxx) keeps a rolling set of recent scans (sn.0000 ...)
plus sn.last, which is how the radar loop gets about an hour of history.

Decoding needs nothing beyond numpy / requests.  Scans are kept as 8-bit
codes (0 = no echo, 1 = range folded, c >= 2 -> DMIN + (c - 2) * 0.5) so the
GUI can recolour them with any colour table (see color_tables.py).

Only radars in the 50 states are used (Guam, Puerto Rico and the overseas
DoD sites are skipped).
"""
from __future__ import annotations

import bz2
import concurrent.futures as cf
import datetime as dt
import logging
import math
import re
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

import us_map as um

log = logging.getLogger("storm_tracker.nexrad")
UTC = dt.timezone.utc

TGFTP_RADAR = "https://tgftp.nws.noaa.gov/SL.us008001/DF.of/DC.radar/"
# key -> (tgftp folder, product code, bin size km, display name)
PRODUCTS = {
    "refl": ("DS.p94r0", 94, 1.0, "Base reflectivity"),
    "vel": ("DS.p99v0", 99, 0.25, "Base velocity"),
}
NON_US_SITES = {"pgua", "tjua", "rkjk", "rksg", "rodn"}
MAX_AGE_MIN = 20.0          # a site whose newest scan is older is treated as down


@dataclass
class Level3Scan:
    site: str
    product: str             # "refl" / "vel"
    lat: float
    lon: float
    time: dt.datetime
    codes: np.ndarray        # (n_radials, n_bins) uint8
    az_start: np.ndarray     # (n_radials,) degrees
    az_width: np.ndarray
    bin_km: float

    @property
    def range_km(self):
        return self.codes.shape[1] * self.bin_km


def decode_level3(raw: bytes, site: str = "") -> Level3Scan:
    """Decode a Level III radial product (packet 16, digital data array)."""
    i = raw.find(b"\r\r\n")
    i = raw.find(b"\r\r\n", i + 3) + 3          # skip WMO header and AWIPS id lines
    m = raw[i:]
    code = struct.unpack(">h", m[:2])[0]
    prod = next((k for k, v in PRODUCTS.items() if v[1] == code), None)
    if prod is None:
        raise ValueError(f"{site}: unsupported product {code}")
    pdb = m[18:120]
    _, lat, lon = struct.unpack(">hii", pdb[:10])
    vdate, vtime = struct.unpack(">hi", pdb[22:28])
    sym = m[120:]
    if sym[:2] == b"BZ":
        sym = bz2.decompress(sym)
    pk = sym[16:]                                # past block + layer headers
    pcode, _, nbins, _, _, _, nrad = struct.unpack(">7h", pk[:14])
    if pcode != 16:
        raise ValueError(f"{site}: packet {pcode}, expected 16")
    codes = np.zeros((nrad, nbins), np.uint8)
    az0 = np.zeros(nrad, np.float32)
    daz = np.zeros(nrad, np.float32)
    o = 14
    for k in range(nrad):
        nbytes, start, width = struct.unpack(">3h", pk[o:o + 6])
        o += 6
        codes[k] = np.frombuffer(pk[o:o + nbins], np.uint8)
        o += nbytes
        az0[k], daz[k] = start / 10.0, width / 10.0
    t = dt.datetime(1970, 1, 1, tzinfo=UTC) + dt.timedelta(days=vdate - 1, seconds=vtime)
    return Level3Scan(site, prod, lat / 1000.0, lon / 1000.0, t, codes, az0, daz,
                      PRODUCTS[prod][2])


def site_domain(lat: float, lon: float) -> Optional[str]:
    for d in um.DOMAINS:
        w, e, s, n = um.COVER[d]
        if w <= lon <= e and s <= lat <= n:
            return d
    return None


# --------------------------------------------------------------------------
# Polar -> grid
# --------------------------------------------------------------------------
def _radial_index(sc: Level3Scan, az: np.ndarray) -> np.ndarray:
    """Radial index for azimuths (degrees); -1 where no radial covers it."""
    lut = np.full(3600, -1, np.int32)
    for k, (a0, da) in enumerate(zip(sc.az_start, sc.az_width)):
        i0 = int(round(a0 * 10))
        lut[np.arange(i0, i0 + max(int(round(da * 10)), 1)) % 3600] = k
    return lut[(az * 10).astype(np.int32) % 3600]


def _polar_geometry(lon, lat, slon, slat):
    """Azimuth (deg) and range (km) of points from a radar (local tangent plane)."""
    coslat = math.cos(math.radians(slat))
    dx = (lon - slon) * (111.32 * coslat)
    dy = (lat - slat) * 111.0
    return (np.degrees(np.arctan2(dx, dy)) % 360.0).astype(np.float32), \
        np.hypot(dx, dy).astype(np.float32)


def sample(sc: Level3Scan, az: np.ndarray, rng: np.ndarray) -> np.ndarray:
    """Codes of the scan at points given by azimuth / range."""
    b = (rng / sc.bin_km).astype(np.int32)
    ri = _radial_index(sc, az)
    ok = (b < sc.codes.shape[1]) & (ri >= 0)
    out = np.zeros(az.shape, np.uint8)
    out[ok] = sc.codes[ri[ok], b[ok]]
    return out


def mosaic(scans: List[Level3Scan], g: um.RasterGrid) -> np.ndarray:
    """Max-composite reflectivity codes of the scans on a projected grid."""
    lon_all, lat_all = um.grid_lonlat(g)
    out = np.zeros(g.shape, np.uint8)
    for sc in scans:
        x, y = um.project(g.domain, sc.lon, sc.lat)
        half = sc.range_km * 1000.0 * 1.08
        c0, c1 = max(0, int((x - half - g.x0) / g.res)), min(g.nx, int((x + half - g.x0) / g.res) + 1)
        r0, r1 = max(0, int((g.y1 - y - half) / g.res)), min(g.ny, int((g.y1 - y + half) / g.res) + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        az, rng = _polar_geometry(lon_all[r0:r1, c0:c1], lat_all[r0:r1, c0:c1], sc.lon, sc.lat)
        np.maximum(out[r0:r1, c0:c1], sample(sc, az, rng), out=out[r0:r1, c0:c1])
    return out


class SiteRenderer:
    """Caches one radar's polar geometry on its own fine grid."""

    def __init__(self, domain: str, lat: float, lon: float, product: str):
        rng_km = 460.0 if product == "refl" else 300.0
        res = 500.0 if product == "refl" else 350.0
        self.grid = um.site_grid(domain, lon, lat, rng_km, res)
        glon, glat = um.grid_lonlat(self.grid)
        self.az, self.rng = _polar_geometry(glon, glat, lon, lat)
        self.rng[self.rng > rng_km] = 1e5            # beyond range -> no data

    def render(self, sc: Level3Scan) -> np.ndarray:
        return sample(sc, self.az, self.rng)


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
_LIST_RE = re.compile(r'href="(sn\.\d{4})">[^<]*</a></td><td[^>]*>\s*(\d{2}-\w{3}-\d{4} \d{2}:\d{2})')


class NexradFeeder:
    """Lists and downloads scans for every site from tgftp."""

    def __init__(self, workers: int = 16):
        self.sess = requests.Session()
        self.sess.headers["User-Agent"] = "storm_tracker/1.0 (research use)"
        self.sess.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=workers))
        self.workers = workers
        self._sites: Dict[str, List[str]] = {}
        self._raw: Dict[Tuple[str, str, str, dt.datetime], bytes] = {}

    def _base(self, product):
        return TGFTP_RADAR + PRODUCTS[product][0] + "/"

    def sites(self, product: str = "refl") -> List[str]:
        if product not in self._sites:
            r = self.sess.get(self._base(product), timeout=30)
            r.raise_for_status()
            found = set(re.findall(r'href="SI\.(\w{4})/?"', r.text))
            self._sites[product] = sorted(found - NON_US_SITES)
        return self._sites[product]

    def listing(self, site: str, product: str = "refl") -> List[Tuple[dt.datetime, str]]:
        """[(file time UTC, file name)] oldest first."""
        try:
            r = self.sess.get(f"{self._base(product)}SI.{site}/", timeout=30)
            if r.status_code != 200:
                return []
        except requests.RequestException:
            return []
        out = []
        for fn, ts in _LIST_RE.findall(r.text):
            try:
                out.append((dt.datetime.strptime(ts, "%d-%b-%Y %H:%M").replace(tzinfo=UTC), fn))
            except ValueError:
                pass
        out.sort()
        return out

    def scan(self, site: str, product: str, fn: str, ft: dt.datetime) -> Optional[Level3Scan]:
        key = (site, product, fn, ft)
        raw = self._raw.get(key)
        try:
            if raw is None:
                r = self.sess.get(f"{self._base(product)}SI.{site}/{fn}", timeout=30)
                if r.status_code != 200:
                    return None
                raw = self._raw[key] = r.content
            return decode_level3(raw, site)
        except Exception as e:                   # one bad file must not stop the rest
            log.debug("radar %s %s failed: %s", site, fn, e)
            self._raw.pop(key, None)
            return None

    def prune(self, before: dt.datetime):
        for k in [k for k in self._raw if k[3] < before]:
            del self._raw[k]

    def history(self, frame_times: List[dt.datetime], product: str = "refl",
                max_gap_min: float = 9.0):
        """For each frame time, the newest scan of every site at that time.

        frame_times may include None meaning 'latest available'.
        -> {frame_time: [Level3Scan]}, {site: (lat, lon)}"""
        sites = self.sites(product)
        with cf.ThreadPoolExecutor(self.workers) as ex:
            listings = dict(zip(sites, ex.map(lambda s: self.listing(s, product), sites)))
        now = dt.datetime.now(UTC)
        wanted: Dict[Tuple[str, str, dt.datetime], List] = {}
        for ft in frame_times:
            ref = ft or now
            for s, lst in listings.items():
                # files appear ~1-3 min after the scan; take the newest one published by ref+2min
                best = None
                for t, fn in lst:
                    if t <= ref + dt.timedelta(minutes=2):
                        best = (t, fn)
                if best and (ref - best[0]).total_seconds() / 60.0 <= (
                        MAX_AGE_MIN if ft is None else max_gap_min):
                    wanted.setdefault((s, best[1], best[0]), []).append(ft)
        keys = list(wanted)
        with cf.ThreadPoolExecutor(self.workers) as ex:
            scans = list(ex.map(lambda k: self.scan(k[0], product, k[1], k[2]), keys))
        out: Dict = {ft: [] for ft in frame_times}
        locs = {}
        for k, sc in zip(keys, scans):
            if sc is None or site_domain(sc.lat, sc.lon) is None:
                continue
            locs[sc.site] = (sc.lat, sc.lon)
            for ft in wanted[k]:
                out[ft].append(sc)
        return out, locs
