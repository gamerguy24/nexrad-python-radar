#!/usr/bin/env python3
"""
level3.py -- decoder for NEXRAD / TDWR Level III products from the NWS tgftp server
(https://tgftp.nws.noaa.gov/SL.us008001/DF.of/DC.radar/), covering the products
GRLevel3 displays.

Every image product is reduced to 8-bit data codes plus a 256-entry table that
turns a code into a physical value (NaN = nothing to draw), so any colour
table can be applied later.  Three image layouts exist:
  * digital radial  (packet 16)       e.g. N0Q, N0U, N0B, ZDR, CC, KDP, HC, DVL, EET, DAA
  * run-length radial (packet AF1F)   e.g. SRV (N0S), NSW, N1P, NTP, OHA
  * raster (packets BA07 / BA0F)      e.g. NCR, NET, NVL, layer composites
Overlay products (storm tracks, hail, mesocyclones, TVS, melting layer) and the
VAD wind profile are vector graphics; their features are returned in km east /
north of the radar (or in screen units for the wind profile).
"""
from __future__ import annotations

import bz2
import datetime as dt
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

UTC = dt.timezone.utc
KTS = 1.0 / 1.943844          # knots -> m/s


# --------------------------------------------------------------------------
# Product catalogue
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Product:
    key: str                  # NWS product id shown in the menu, e.g. "N0Q"
    name: str
    group: str
    family: str               # colour-table family: BR BV SW ZDR CC KDP HC ET VIL PRECIP DIFF
    code: int
    tilts: Tuple[Tuple[str, str], ...]     # ((label, tgftp folder), ...)
    bin_km: float = 1.0       # radial products: gate spacing
    cell_km: float = 0.0      # raster products: cell size
    tdwr: bool = False        # TDWR (airport radar) product

    @property
    def raster(self):
        return self.cell_km > 0


def _tilts(fmt, sfx="0a1b23"):
    ang = {"0": "0.5", "a": "0.9", "1": "1.5", "b": "1.8", "2": "2.4", "3": "3.4"}
    return tuple((f"{ang[s]}°", fmt.format(s)) for s in sfx)


def _one(folder):
    return (("-", folder),)


PRODUCTS: List[Product] = [
    # reflectivity
    Product("N0Q", "Base Reflectivity", "Reflectivity", "BR", 94, _tilts("DS.p94r{}"), 1.0),
    Product("N0B", "Super-Res Reflectivity", "Reflectivity", "BR", 153, _tilts("DS.00n{}b"), 0.25),
    Product("N0Z", "Long-Range Reflectivity (248 nm)", "Reflectivity", "BR", 20, _one("DS.p20-r"), 2.0),
    Product("NCR", "Composite Reflectivity", "Reflectivity", "BR", 37, _one("DS.p37cr"), cell_km=1.0),
    Product("NCZ", "Composite Reflectivity (248 nm)", "Reflectivity", "BR", 38, _one("DS.p38cr"), cell_km=4.0),
    Product("DHR", "Hybrid Scan Reflectivity", "Reflectivity", "BR", 32, _one("DS.32dhr"), 1.0),
    Product("NLA", "Layer Max Reflectivity, low (AP removed)", "Reflectivity", "BR", 67,
            _one("DS.67apr"), cell_km=4.0),
    Product("NML", "Layer Max Reflectivity, mid (24-33 kft)", "Reflectivity", "BR", 66,
            _one("DS.66lrm"), cell_km=4.0),
    Product("NHL", "Layer Max Reflectivity, high (33-60 kft)", "Reflectivity", "BR", 90,
            _one("DS.90lrm"), cell_km=4.0),
    # velocity
    Product("N0U", "Base Velocity", "Velocity", "BV", 99, _tilts("DS.p99v{}"), 0.25),
    Product("N0G", "Super-Res Velocity", "Velocity", "BV", 154, _tilts("DS.00n{}g", "0a1"), 0.25),
    Product("N0S", "Storm Relative Velocity", "Velocity", "BV", 56, _tilts("DS.56rm{}", "0123"), 1.0),
    Product("NSW", "Spectrum Width", "Velocity", "SW", 30, _one("DS.p30sw"), 0.25),
    # dual polarisation
    Product("N0X", "Differential Reflectivity (ZDR)", "Dual-Pol", "ZDR", 159, _tilts("DS.159x{}"), 0.25),
    Product("N0C", "Correlation Coefficient (CC)", "Dual-Pol", "CC", 161, _tilts("DS.161c{}"), 0.25),
    Product("N0K", "Specific Differential Phase (KDP)", "Dual-Pol", "KDP", 163, _tilts("DS.163k{}"), 0.25),
    Product("N0H", "Hydrometeor Classification", "Dual-Pol", "HC", 165, _tilts("DS.165h{}"), 0.25),
    Product("HHC", "Hybrid Hydrometeor Classification", "Dual-Pol", "HC", 177, _one("DS.177hh"), 0.25),
    # storm structure
    Product("NET", "Echo Tops", "Derived", "ET", 41, _one("DS.p41et"), cell_km=4.0),
    Product("EET", "Enhanced Echo Tops", "Derived", "ET", 135, _one("DS.135et"), 1.0),
    Product("NVL", "Vertically Integrated Liquid", "Derived", "VIL", 57, _one("DS.57vil"), cell_km=4.0),
    Product("DVL", "Digital VIL", "Derived", "VIL", 134, _one("DS.134il"), 1.0),
    # precipitation
    Product("N1P", "One-Hour Precipitation", "Precipitation", "PRECIP", 78, _one("DS.78ohp"), 2.0),
    Product("N3P", "Three-Hour Precipitation", "Precipitation", "PRECIP", 79, _one("DS.79thp"), 2.0),
    Product("NTP", "Storm Total Precipitation", "Precipitation", "PRECIP", 80, _one("DS.80stp"), 2.0),
    Product("DSP", "Digital Storm Total Precipitation", "Precipitation", "PRECIP", 138,
            _one("DS.138dp"), 2.0),
    Product("OHA", "One-Hour Accumulation (dual-pol)", "Precipitation", "PRECIP", 169,
            _one("DS.169oh"), 2.0),
    Product("DAA", "Digital One-Hour Accumulation", "Precipitation", "PRECIP", 170, _one("DS.170aa"), 0.25),
    Product("DTA", "Digital Storm Total Accumulation", "Precipitation", "PRECIP", 172,
            _one("DS.172dt"), 0.25),
    Product("DU3", "Digital 3-Hour Accumulation", "Precipitation", "PRECIP", 173, _one("DS.173u1"), 0.25),
    Product("DU6", "Digital 24-Hour Accumulation", "Precipitation", "PRECIP", 173, _one("DS.173u3"), 0.25),
    Product("DOD", "One-Hour Difference (dual-pol - legacy)", "Precipitation", "DIFF", 174,
            _one("DS.174od"), 0.25),
    Product("DSD", "Storm Total Difference (dual-pol - legacy)", "Precipitation", "DIFF", 175,
            _one("DS.175sd"), 0.25),
    # TDWR airport radars
    Product("TZ0", "TDWR Reflectivity", "TDWR", "BR", 180,
            (("Tilt 1", "DS.180z0"), ("Tilt 2", "DS.180z1"), ("Tilt 3", "DS.180z2")), 0.15, tdwr=True),
    Product("TV0", "TDWR Velocity", "TDWR", "BV", 182,
            (("Tilt 1", "DS.182v0"), ("Tilt 2", "DS.182v1"), ("Tilt 3", "DS.182v2")), 0.15, tdwr=True),
    Product("TZL", "TDWR Long-Range Reflectivity", "TDWR", "BR", 186, _one("DS.186zl"), 0.3, tdwr=True),
]
BY_KEY = {p.key: p for p in PRODUCTS}

OVERLAYS = {                 # key -> (menu name, folder; {t} = tilt suffix)
    "sti": ("Storm tracks", "DS.58sti"),
    "hail": ("Hail", "DS.p59hi"),
    "meso": ("Mesocyclones", "DS.141md"),
    "tvs": ("TVS", "DS.61tvs"),
    "ml": ("Melting layer", "DS.166m{t}"),
}
VWP_FOLDER = "DS.48vwp"

HC_CLASSES = {1: ("BI", "Biological"), 2: ("GC", "Ground clutter"), 3: ("IC", "Ice crystals"),
              4: ("DS", "Dry snow"), 5: ("WS", "Wet snow"), 6: ("RA", "Rain"),
              7: ("HR", "Heavy rain"), 8: ("BD", "Big drops"), 9: ("GR", "Graupel"),
              10: ("HA", "Hail / rain"), 11: ("LH", "Large hail"), 12: ("GH", "Giant hail"),
              14: ("UK", "Unknown")}


# --------------------------------------------------------------------------
# Decoded product
# --------------------------------------------------------------------------
@dataclass
class Level3:
    code: int
    lat: float
    lon: float
    time: dt.datetime
    elev: Optional[float]                   # elevation angle (deg) for tilt products
    values: Optional[np.ndarray] = None     # (256,) code -> value, NaN = transparent
    rf_code: Optional[int] = None           # code meaning "range folded"
    # radial image
    codes: Optional[np.ndarray] = None      # (n_radials, n_bins) or (rows, cols) for rasters
    az_start: Optional[np.ndarray] = None
    az_width: Optional[np.ndarray] = None
    bin_km: float = 1.0
    first_bin: int = 0
    cell_km: float = 0.0                    # raster products
    features: Dict[str, list] = field(default_factory=dict)
    graphic: list = field(default_factory=list)      # VWP: drawing primitives

    @property
    def range_km(self):
        if self.codes is None:
            return 0.0
        if self.cell_km:
            return self.codes.shape[0] * self.cell_km / 2.0
        return (self.first_bin + self.codes.shape[1]) * self.bin_km


def _f32(a: int, b: int) -> float:
    return struct.unpack(">f", struct.pack(">HH", a, b))[0]


def _f16(v: int) -> float:
    """NEXRAD 16-bit float (sign, 5-bit exponent, 10-bit fraction) used by DVL."""
    frac, exp, sign = v & 0x3FF, (v >> 10) & 0x1F, v >> 15
    x = 2.0 ** (exp - 16) * (1 + frac / 1024.0) if exp else frac / 512.0
    return -x if sign else x


def _thresholds16(hw, code):
    """16 data-level thresholds (legacy 16-level products) -> (values, rf_code)."""
    vals = np.full(256, np.nan, np.float32)
    rf = None
    for c, h in enumerate(hw):
        fl, v = h >> 8, h & 0xFF
        if fl & 0x80:
            if v == 3:
                rf = c
            continue
        x = float(v)
        if fl & 0x40:
            x /= 100.0
        if fl & 0x20:
            x /= 20.0
        if fl & 0x10:
            x /= 10.0
        if fl & 0x01:
            x = -x
        vals[c] = x
    if code in (56, 30):                    # velocity / width thresholds are in knots
        vals *= KTS
    return vals, rf


def code_values(code: int, hw: Tuple[int, ...]):
    """(values[256], rf_code) for a product's data codes.  hw = unsigned halfwords 31-46."""
    s = [h - 65536 if h >= 32768 else h for h in hw]
    vals = np.full(256, np.nan, np.float32)
    c = np.arange(256, dtype=np.float32)
    rf = None
    if code in (94, 153, 180, 186, 32, 99, 154, 182):        # linear dBZ or m/s
        vals[2:] = s[0] / 10.0 + (c[2:] - 2) * (s[1] / 10.0)
        if code in (99, 154, 182):
            rf = 1
    elif code in (159, 161, 163):                           # ZDR / CC / KDP
        sc, of = _f32(hw[0], hw[1]), _f32(hw[2], hw[3])
        vals[2:] = (c[2:] - of) / sc
        rf = 1
    elif code in (170, 172, 173, 174, 175):                 # dual-pol accumulations, 0.01 in
        sc, of = _f32(hw[0], hw[1]), _f32(hw[2], hw[3])
        lead = max(1, hw[6])
        vals[lead:] = (c[lead:] - of) / sc / 100.0
    elif code in (165, 177):                                # hydrometeor class index
        for k in HC_CLASSES:
            vals[k * 10:k * 10 + 10] = k
        rf = 150
    elif code == 134:                                       # digital VIL
        lin_sc, lin_of, log_start = _f16(hw[0]), _f16(hw[1]), hw[2]
        log_sc, log_of = _f16(hw[3]), _f16(hw[4])
        lo = np.arange(2, log_start)
        vals[2:log_start] = (lo - lin_of) / lin_sc
        hi = np.arange(log_start, 256)
        vals[log_start:] = np.exp((hi - log_of) / log_sc)
    elif code == 135:                                       # enhanced echo tops, kft
        mask, scale, offset = hw[0], hw[1] or 1, hw[2]
        ci = np.arange(2, 256)
        vals[2:] = (ci & mask) / scale - offset
    elif code == 138:                                       # digital storm total, 0.01 in
        vals[1:] = (s[0] + c[1:] * s[1]) / 100.0
    else:                                                   # 16-level products
        vals, rf = _thresholds16(hw, code)
    if rf is not None:
        vals[rf] = np.nan
    return vals.astype(np.float32), rf


# --------------------------------------------------------------------------
# Symbology parsing
# --------------------------------------------------------------------------
def _iter_packets(buf: bytes):
    """Yield (packet code, payload start, payload bytes) for every packet in a layer."""
    k, n = 0, len(buf)
    while k + 4 <= n:
        pc = struct.unpack(">H", buf[k:k + 2])[0]
        if pc in (16, 0xAF1F):
            nrad = struct.unpack(">h", buf[k + 12:k + 14])[0]
            e = k + 14
            for _ in range(nrad):
                cnt = struct.unpack(">h", buf[e:e + 2])[0]
                e += 6 + (cnt if pc == 16 else cnt * 2)
            yield pc, buf[k:e]
            k = e
        elif pc in (0xBA07, 0xBA0F):
            nrows = struct.unpack(">h", buf[k + 18:k + 20])[0]
            e = k + 22
            for _ in range(nrows):
                e += 2 + struct.unpack(">h", buf[e:e + 2])[0]
            yield pc, buf[k:e]
            k = e
        elif pc == 28:
            ln = struct.unpack(">i", buf[k + 4:k + 8])[0]
            yield pc, buf[k:k + 8 + ln]
            k += 8 + ln
        elif pc == 0x0802:
            yield pc, buf[k:k + 6]
            k += 6
        elif pc == 0x0E03:
            ln = struct.unpack(">h", buf[k + 8:k + 10])[0]
            yield pc, buf[k:k + 10 + ln]
            k += 10 + ln
        else:                                   # code, length, payload
            ln = struct.unpack(">h", buf[k + 2:k + 4])[0]
            if ln < 0:
                return
            yield pc, buf[k:k + 4 + ln]
            k += 4 + ln


def _layers(full: bytes, off_halfwords: int):
    if not off_halfwords:
        return
    blk = full[off_halfwords * 2:]
    nl = struct.unpack(">h", blk[8:10])[0]
    q = 10
    for _ in range(nl):
        ll = struct.unpack(">i", blk[q + 2:q + 6])[0]
        yield blk[q + 6:q + 6 + ll]
        q += 6 + ll


def _radial16(p: bytes):
    first, nbins, _, _, _, nrad = struct.unpack(">6h", p[2:14])
    codes = np.zeros((nrad, nbins), np.uint8)
    az0, daz = np.zeros(nrad, np.float32), np.zeros(nrad, np.float32)
    k = 14
    for r in range(nrad):
        cnt, a0, da = struct.unpack(">3h", p[k:k + 6])
        row = np.frombuffer(p[k + 6:k + 6 + min(cnt, nbins)], np.uint8)
        codes[r, :len(row)] = row
        az0[r], daz[r] = a0 / 10.0, da / 10.0
        k += 6 + cnt
    return codes, az0, daz, first


def _radial_rle(p: bytes):
    first, nbins, _, _, _, nrad = struct.unpack(">6h", p[2:14])
    codes = np.zeros((nrad, nbins), np.uint8)
    az0, daz = np.zeros(nrad, np.float32), np.zeros(nrad, np.float32)
    k = 14
    for r in range(nrad):
        cnt, a0, da = struct.unpack(">3h", p[k:k + 6])
        b = np.frombuffer(p[k + 6:k + 6 + 2 * cnt], np.uint8)
        row = np.repeat(b & 0x0F, b >> 4)[:nbins]
        codes[r, :len(row)] = row
        az0[r], daz[r] = a0 / 10.0, da / 10.0
        k += 6 + 2 * cnt
    return codes, az0, daz, first


def _raster(p: bytes):
    nrows = struct.unpack(">h", p[18:20])[0]
    rows, k = [], 22
    for _ in range(nrows):
        n = struct.unpack(">h", p[k:k + 2])[0]
        b = np.frombuffer(p[k + 2:k + 2 + n], np.uint8)
        rows.append(np.repeat(b & 0x0F, b >> 4))
        k += 2 + n
    ncols = max((len(r) for r in rows), default=0)
    out = np.zeros((nrows, ncols), np.uint8)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def _ij(p, off):
    i, j = struct.unpack(">hh", p[off:off + 4])
    return i / 4.0, j / 4.0                  # 1/4 km units, i east, j north


def _features(code: int, layers) -> Dict[str, list]:
    """Overlay features in km east / north of the radar."""
    f = {"ids": [], "past": [], "fcst": [], "hail": [], "meso": [], "tvs": [], "etvs": [],
         "ml": []}
    color = 0
    for lay in layers:
        for pc, p in _iter_packets(lay):
            if pc == 15:                                     # storm id
                for q in range(4, len(p) - 5, 6):
                    x, y = _ij(p, q)
                    f["ids"].append((x, y, p[q + 4:q + 6].decode("latin1").strip()))
            elif pc in (23, 24):                             # SCIT past / forecast
                key = "past" if pc == 23 else "fcst"
                for spc, sp in _iter_packets(p[4:]):
                    if spc == 6:
                        pts = [_ij(sp, q) for q in range(4, len(sp) - 3, 4)]
                        if pts:
                            f[key].append(pts)
                    elif spc == 2:
                        f[key + "_pts"] = f.get(key + "_pts", []) + [_ij(sp, 4)]
            elif pc == 19:                                   # hail (HDA)
                for q in range(4, len(p) - 9, 10):
                    x, y = _ij(p, q)
                    poh, posh, size = struct.unpack(">3h", p[q + 4:q + 10])
                    f["hail"].append((x, y, poh, posh, size))
            elif pc == 20:                                   # mesocyclone / circulation
                for q in range(4, len(p) - 7, 8):
                    x, y = _ij(p, q)
                    typ, rad = struct.unpack(">hh", p[q + 4:q + 8])
                    f["meso"].append((x, y, typ, rad / 4.0))
            elif pc in (12, 26):                             # TVS / elevated TVS
                for q in range(4, len(p) - 3, 4):
                    f["tvs" if pc == 12 else "etvs"].append(_ij(p, q))
            elif pc == 0x0802:
                color = struct.unpack(">h", p[4:6])[0]
            elif pc == 0x0E03:                               # linked contour (melting layer)
                x0, y0 = _ij(p, 4)
                pts = [(x0, y0)] + [_ij(p, q) for q in range(10, len(p) - 3, 4)]
                f["ml"].append((color, pts))
    return f


def _graphic(layers):
    """VWP-style screen graphics: barbs, text and vectors (screen units, y down)."""
    out = []
    for lay in layers:
        for pc, p in _iter_packets(lay):
            if pc == 4:                                      # wind barb
                for q in range(4, len(p) - 9, 10):
                    col, x, y, d, s = struct.unpack(">5h", p[q:q + 10])
                    out.append(("barb", col, x, y, d, s))
            elif pc in (8, 1):                               # text (8 = with colour)
                if pc == 8:
                    col, x, y = struct.unpack(">3h", p[4:10])
                    txt = p[10:].decode("latin1", "ignore")
                else:
                    col = -1
                    x, y = struct.unpack(">hh", p[4:8])
                    txt = p[8:].decode("latin1", "ignore")
                out.append(("text", col, x, y, txt.rstrip()))
            elif pc in (10, 7):                              # unlinked vectors
                q0 = 6 if pc == 10 else 4
                col = struct.unpack(">h", p[4:6])[0] if pc == 10 else -1
                for q in range(q0, len(p) - 7, 8):
                    x1, y1, x2, y2 = struct.unpack(">4h", p[q:q + 8])
                    out.append(("line", col, x1, y1, x2, y2))
            elif pc in (9, 6):                               # linked vectors
                q0 = 6 if pc == 9 else 4
                col = struct.unpack(">h", p[4:6])[0] if pc == 9 else -1
                pts = [struct.unpack(">hh", p[q:q + 4]) for q in range(q0, len(p) - 3, 4)]
                for a, b in zip(pts, pts[1:]):
                    out.append(("line", col, *a, *b))
    return out


def decode(raw: bytes, product: Optional[Product] = None) -> Level3:
    """Decode a tgftp Level III file (WMO header + message)."""
    i = raw.find(b"\r\r\n")
    i = raw.find(b"\r\r\n", i + 3) + 3          # skip WMO header and AWIPS id lines
    m = raw[i:]
    code = struct.unpack(">h", m[:2])[0]
    if m[120:122] == b"BZ":
        m = m[:120] + bz2.decompress(m[120:])
    pdb = m[18:120]
    lat, lon = struct.unpack(">ii", pdb[2:10])
    vdate, vtime = struct.unpack(">hi", pdb[22:28])
    elnum, p3 = struct.unpack(">hh", pdb[38:42])
    hw = struct.unpack(">16H", pdb[42:74])
    sym_off, gra_off, tab_off = struct.unpack(">3i", pdb[90:102])
    t = dt.datetime(1970, 1, 1, tzinfo=UTC) + dt.timedelta(days=vdate - 1, seconds=vtime)
    out = Level3(code, lat / 1000.0, lon / 1000.0, t, p3 / 10.0 if elnum > 0 else None)
    layers = list(_layers(m, sym_off))
    if code in (58, 59, 141, 61, 166):
        out.features = _features(code, layers)
        return out
    if code == 48:
        out.graphic = _graphic(layers)
        return out
    out.values, out.rf_code = code_values(code, hw)
    for lay in layers:
        for pc, p in _iter_packets(lay):
            if pc == 16:
                out.codes, out.az_start, out.az_width, out.first_bin = _radial16(p)
            elif pc == 0xAF1F:
                out.codes, out.az_start, out.az_width, out.first_bin = _radial_rle(p)
            elif pc in (0xBA07, 0xBA0F):
                out.codes = _raster(p)
                out.cell_km = product.cell_km if product else 4.0
            if out.codes is not None:
                break
        if out.codes is not None:
            break
    if out.codes is not None and not out.cell_km:
        out.bin_km = product.bin_km if product else 1.0
    return out


def km_to_lonlat(lat0, lon0, x_km, y_km):
    """Local tangent plane (km east / north of the radar) -> lon, lat."""
    lat = lat0 + np.asarray(y_km) / 111.0
    lon = lon0 + np.asarray(x_km) / (111.32 * math.cos(math.radians(lat0)))
    return lon, lat
