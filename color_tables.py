#!/usr/bin/env python3
"""
color_tables.py -- radar colour tables, including GRLevelX / GR2Analyst .pal files.

Each product belongs to a colour-table family (see level3.Product.family):
    BR  reflectivity (dBZ)          BV  velocity (data in m/s)
    SW  spectrum width (m/s)        ZDR differential reflectivity (dB)
    CC  correlation coefficient     KDP specific differential phase (deg/km)
    HC  hydrometeor class (index)   ET  echo tops (kft)
    VIL VIL (kg/m2)                 PRECIP precipitation (in)
    DIFF precipitation difference (in)
A decoded product carries a 256-entry table code -> value (level3.code_values);
a colour table turns that into a 256-entry RGBA lookup, so recolouring a frame
is a single numpy index operation.

Supported .pal lines (case-insensitive, ';' or '#' starts a comment):
    Product: BR | BV | SRV | SW | ZDR | CC | KDP | HC | ET | VIL | PRECIP ...
    Units: DBZ | KTS | MPH | MPS | IN | MM | KFT | FT | KM
    Scale: <factor>   Offset: <value>       data * Scale + Offset -> table units
    RF: r g b                               range-folded colour
    Color:  value r g b [r2 g2 b2]          blend to r2 g2 b2 (or the next entry)
    Color4: value r g b a [r2 g2 b2 a2]
    SolidColor: value r g b   SolidColor4: value r g b a
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_ALPHA = 235
FAMILIES = {          # family -> (name, data units)
    "BR": ("Reflectivity", "dBZ"), "BV": ("Velocity", "m/s"), "SW": ("Spectrum width", "m/s"),
    "ZDR": ("Differential reflectivity", "dB"), "CC": ("Correlation coefficient", ""),
    "KDP": ("Specific differential phase", "deg/km"), "HC": ("Hydrometeor class", ""),
    "ET": ("Echo tops", "kft"), "VIL": ("VIL", "kg/m²"), "PRECIP": ("Precipitation", "in"),
    "DIFF": ("Precipitation difference", "in"),
}
# .pal "Product:" names -> family
PAL_PRODUCT = {
    "BR": "BR", "BZ": "BR", "REF": "BR", "CR": "BR", "DHR": "BR", "N0Q": "BR", "N0R": "BR",
    "BV": "BV", "VEL": "BV", "V": "BV", "SRV": "BV", "SRM": "BV", "N0U": "BV", "N0V": "BV",
    "SW": "SW", "ZDR": "ZDR", "DR": "ZDR", "CC": "CC", "RHO": "CC", "KDP": "KDP", "HC": "HC",
    "HCA": "HC", "HHC": "HC", "ET": "ET", "EET": "ET", "NET": "ET", "VIL": "VIL", "DVL": "VIL",
    "PRECIP": "PRECIP", "PR": "PRECIP", "OHP": "PRECIP", "STP": "PRECIP", "DSP": "PRECIP",
    "THP": "PRECIP", "OHA": "PRECIP", "DTA": "PRECIP", "DAA": "PRECIP", "DIFF": "DIFF",
    "DOD": "DIFF", "DSD": "DIFF",
}
# data units -> table units, by the table's Units line
UNIT_SCALE = {
    "BV": {"KTS": 1.943844, "KT": 1.943844, "KNOTS": 1.943844, "MPH": 2.236936, "MPS": 1.0,
           "M/S": 1.0, "KPH": 3.6, "KM/H": 3.6},
    "PRECIP": {"MM": 25.4, "CM": 2.54},
    "DIFF": {"MM": 25.4, "CM": 2.54},
    "ET": {"FT": 1000.0, "KM": 0.3048, "M": 304.8},
}
UNIT_SCALE["SW"] = UNIT_SCALE["BV"]

# reflectivity code encoding used by the mosaic and MRMS layers
DMIN_REFL, INC = -32.0, 0.5
REFL_VALUES = np.full(256, np.nan, np.float32)
REFL_VALUES[2:] = DMIN_REFL + (np.arange(2, 256) - 2) * INC

RGBA = Tuple[int, int, int, int]


@dataclass
class Stop:
    value: float
    c1: RGBA
    c2: Optional[RGBA] = None
    solid: bool = False


@dataclass
class ColorTable:
    name: str
    family: str
    stops: List[Stop]
    units: str = ""
    scale: float = 1.0               # data units -> table units
    offset: float = 0.0
    rf: RGBA = (119, 0, 119, 255)
    source: Optional[str] = None     # file path for loaded tables
    labels: Optional[Dict[int, str]] = None     # categorical legend (HC)
    _luts: Dict = field(default_factory=dict, repr=False)

    def color(self, v: float) -> Optional[RGBA]:
        """Colour for a value in table units (None = transparent)."""
        s = self.stops
        i = -1
        for k, st in enumerate(s):
            if v >= st.value:
                i = k
            else:
                break
        if i < 0:
            return None
        st = s[i]
        if st.solid:
            col = st.c1
        elif i == len(s) - 1:
            col = st.c2 or st.c1
        else:
            nxt = s[i + 1]
            end = st.c2 or nxt.c1
            span = nxt.value - st.value
            f = 0.0 if span <= 0 else (v - st.value) / span
            col = tuple(int(round(a + (b - a) * f)) for a, b in zip(st.c1, end))
        return col if col[3] > 0 else None

    def lut_for(self, values: np.ndarray, rf_code: Optional[int] = None) -> np.ndarray:
        """(256, 4) uint8 RGBA lookup for a product's code -> value table."""
        key = (values.tobytes(), rf_code)
        lut = self._luts.get(key)
        if lut is None:
            lut = np.zeros((256, 4), np.uint8)
            for c in range(256):
                v = values[c]
                if np.isfinite(v):
                    col = self.color(float(v) * self.scale + self.offset)
                    if col:
                        lut[c] = col
            if rf_code is not None:
                lut[rf_code] = self.rf
            if len(self._luts) > 64:
                self._luts.clear()
            self._luts[key] = lut
        return lut

    def lut(self) -> np.ndarray:
        """Lookup for the standard reflectivity encoding (mosaic / MRMS)."""
        return self.lut_for(REFL_VALUES)

    def value_range(self) -> Tuple[float, float]:
        return self.stops[0].value, self.stops[-1].value

    @property
    def kind(self):                  # backwards compatible name
        return self.family


def _rgba(nums, n):
    vals = [int(round(float(x))) for x in nums[:n]]
    if n == 3:
        vals.append(DEFAULT_ALPHA)
    return tuple(max(0, min(255, v)) for v in vals)


def parse_pal(text: str, name: str = "custom", source: Optional[str] = None,
              family: Optional[str] = None) -> ColorTable:
    stops: List[Stop] = []
    product, units, scale, offset = "", "", None, 0.0
    rf = (119, 0, 119, 255)
    for line in text.splitlines():
        line = re.split(r"[;#]", line, 1)[0].strip()
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, parts = key.strip().lower(), val.replace(",", " ").split()
        try:
            if key == "product":
                product = val.strip().upper()
            elif key == "units":
                units = val.strip().upper()
            elif key == "scale":
                scale = float(parts[0])
            elif key == "offset":
                offset = float(parts[0])
            elif key == "rf":
                rf = _rgba(parts, 3)[:3] + (255,)
            elif key in ("color", "color4", "solidcolor", "solidcolor4"):
                n = 4 if key.endswith("4") else 3
                v = float(parts[0])
                c1 = _rgba(parts[1:], n)
                c2 = _rgba(parts[1 + n:], n) if len(parts) >= 1 + 2 * n else None
                stops.append(Stop(v, c1, c2, key.startswith("solid")))
        except (ValueError, IndexError):
            continue                              # skip a malformed line, keep the rest
    if not stops:
        raise ValueError("no Color: lines found")
    stops.sort(key=lambda s: s.value)
    fam = family or PAL_PRODUCT.get(product)
    if fam is None:                               # guess from the units
        fam = {"KTS": "BV", "KT": "BV", "MPH": "BV", "MPS": "BV", "M/S": "BV", "IN": "PRECIP",
               "KFT": "ET", "DB": "ZDR"}.get(units, "BR")
    if scale is None:
        scale = UNIT_SCALE.get(fam, {}).get(units, 1.0)
    return ColorTable(name, fam, stops, units or FAMILIES[fam][1].upper(), scale, offset, rf, source)


def load_pal(path) -> ColorTable:
    p = Path(path)
    return parse_pal(p.read_text(encoding="utf8", errors="ignore"), p.stem, str(p))


# --------------------------------------------------------------------------
# Built-in tables
# --------------------------------------------------------------------------
def _solid(name, family, pairs, units=""):
    return ColorTable(name, family, [Stop(v, (*c, DEFAULT_ALPHA), None, True) for v, c in pairs],
                      units)


NWS_REFL = _solid("NWS (classic)", "BR", [
    (10, (1, 159, 244)), (15, (3, 0, 244)), (20, (2, 253, 2)), (25, (1, 197, 1)),
    (30, (0, 142, 0)), (35, (253, 248, 2)), (40, (229, 188, 0)), (45, (253, 149, 0)),
    (50, (253, 0, 0)), (55, (212, 0, 0)), (60, (188, 0, 0)), (65, (248, 0, 253)),
    (70, (152, 84, 198)), (75, (253, 253, 253))], "DBZ")

SMOOTH_REFL = parse_pal("""
Product: BR
Units: DBZ
Color: 10  80 150 200    40 90 200
Color: 20  20 200 60     10 140 30
Color: 30  10 120 20     240 240 50
Color: 40  250 200 20    250 120 10
Color: 50  240 30 30     160 0 0
Color: 60  255 50 230    150 60 210
Color: 70  220 200 255   255 255 255
Color: 80  255 255 255
""", "Smooth gradient")

DARK_REFL = parse_pal("""
Product: BR
Units: DBZ
Color: 10  30 60 80     20 110 130
Color: 20  30 160 110   20 110 40
Color: 30  140 170 30   230 210 40
Color: 40  240 150 30   230 60 30
Color: 50  200 20 40    120 10 60
Color: 60  220 90 220   250 220 255
Color: 75  255 255 255
""", "Muted / night")

CLEAR_AIR = parse_pal("""
Product: BR
Units: DBZ
Color: -28 60 60 60     110 110 110
Color: -10 110 110 110  160 160 160
Color: 0   0 90 160     0 160 230
Color: 10  0 160 230    0 200 80
Color: 20  0 200 80     240 240 0
Color: 30  240 240 0    240 120 0
Color: 40  240 120 0    220 0 0
Color: 50  220 0 0      255 0 255
Color: 60  255 255 255
""", "Clear air (shows weak echoes)")

VEL_DEFAULT = parse_pal("""
Product: BV
Units: KTS
RF: 119 0 119
Color: -140 255 255 255
Color: -100 150 255 150
Color: -64  0 235 0
Color: -36  0 150 0
Color: -10  0 80 10      70 100 70
Color: 0    100 80 80
Color: 1    90 40 40     110 0 0
Color: 10   140 0 0
Color: 36   210 0 0
Color: 64   255 60 60
Color: 100  255 200 190
Color: 140  255 255 255
""", "Velocity (green/red)")

VEL_BRIGHT = parse_pal("""
Product: BV
Units: KTS
RF: 145 70 200
Color: -120 255 0 255
Color: -80  0 255 255
Color: -50  0 200 0
Color: -20  0 120 0
Color: -5   60 90 60
Color: 0    120 120 120
Color: 5    100 60 60
Color: 20   160 0 0
Color: 50   255 70 0
Color: 80   255 255 0
Color: 120  255 255 255
""", "Velocity (bright)")

SW_DEFAULT = parse_pal("""
Product: SW
Units: KTS
Color: 0   50 50 50      110 110 110
Color: 5   110 110 110   0 150 0
Color: 10  0 150 0       230 230 0
Color: 15  230 230 0     240 120 0
Color: 20  240 120 0     220 0 0
Color: 25  220 0 0       255 255 255
Color: 30  255 255 255
""", "Spectrum width")

ZDR_DEFAULT = parse_pal("""
Product: ZDR
Units: DB
Color: -4  40 0 60       90 30 130
Color: -2  130 100 170   180 180 180
Color: 0   200 200 200   60 120 230
Color: 1   60 120 230    0 200 230
Color: 1.5 0 200 230     0 200 80
Color: 2   0 200 80      240 240 0
Color: 3   240 240 0     250 130 0
Color: 4   250 130 0     220 0 0
Color: 5   220 0 0       250 80 200
Color: 6   250 80 200    255 255 255
Color: 8   255 255 255
""", "ZDR")

CC_DEFAULT = parse_pal("""
Product: CC
Color: 0.2  20 20 60     40 40 160
Color: 0.45 40 40 160    100 100 240
Color: 0.65 100 100 240  60 200 60
Color: 0.75 60 200 60    200 230 50
Color: 0.85 200 230 50   250 200 0
Color: 0.9  250 200 0    240 100 0
Color: 0.95 240 100 0    220 0 0
Color: 0.97 220 0 0      150 0 60
Color: 1.0  150 0 60     255 200 255
Color: 1.05 255 230 255
""", "Correlation coefficient")

KDP_DEFAULT = parse_pal("""
Product: KDP
Color: -2  60 60 60      120 120 120
Color: 0   140 140 140   100 200 255
Color: 0.5 100 200 255   0 200 60
Color: 1.5 0 200 60      240 240 0
Color: 2.5 240 240 0     240 100 0
Color: 4   240 100 0     220 0 0
Color: 6   220 0 0       255 100 255
Color: 8   255 100 255   255 255 255
Color: 10  255 255 255
""", "KDP")

_HC_COLORS = {1: (156, 156, 156), 2: (118, 90, 60), 3: (240, 150, 230), 4: (130, 170, 255),
              5: (30, 70, 220), 6: (110, 220, 110), 7: (20, 140, 20), 8: (240, 240, 60),
              9: (250, 170, 40), 10: (240, 30, 30), 11: (170, 0, 170), 12: (255, 255, 255),
              14: (200, 190, 120)}
HC_DEFAULT = ColorTable(
    "Hydrometeor classes", "HC",
    [Stop(k, (*c, 240), None, True) for k, c in sorted(_HC_COLORS.items())] +
    [Stop(13, (0, 0, 0, 0), None, True), Stop(15, (0, 0, 0, 0), None, True)])
HC_DEFAULT.stops.sort(key=lambda s: s.value)
HC_DEFAULT.labels = {1: "BI", 2: "GC", 3: "IC", 4: "DS", 5: "WS", 6: "RA", 7: "HR", 8: "BD",
                     9: "GR", 10: "HA", 11: "LH", 12: "GH", 14: "UK"}

ET_DEFAULT = parse_pal("""
Product: ET
Units: KFT
Color: 0  40 70 110     0 110 200
Color: 10 0 110 200     0 190 255
Color: 20 0 190 255     0 200 0
Color: 30 0 200 0       240 240 0
Color: 40 240 240 0     240 130 0
Color: 50 240 130 0     220 0 0
Color: 60 220 0 0       240 0 240
Color: 70 240 0 240     255 255 255
Color: 80 255 255 255
""", "Echo tops")

VIL_DEFAULT = parse_pal("""
Product: VIL
Color: 1  60 60 120     0 120 220
Color: 5  0 120 220     0 200 200
Color: 10 0 200 200     0 200 0
Color: 20 0 200 0       240 240 0
Color: 30 240 240 0     250 140 0
Color: 40 250 140 0     220 0 0
Color: 55 220 0 0       240 0 240
Color: 70 240 0 240     255 255 255
Color: 80 255 255 255
""", "VIL")

PRECIP_DEFAULT = parse_pal("""
Product: PRECIP
Units: IN
Color: 0.0   170 200 230   100 160 230
Color: 0.1   100 160 230   0 110 200
Color: 0.25  0 110 200     0 180 80
Color: 0.5   0 180 80      0 120 0
Color: 1.0   0 120 0       240 240 0
Color: 1.5   240 240 0     250 150 0
Color: 2.0   250 150 0     230 0 0
Color: 3.0   230 0 0       150 0 0
Color: 4.0   150 0 0       240 0 240
Color: 6.0   240 0 240     150 80 200
Color: 8.0   150 80 200    255 255 255
Color: 12.0  255 255 255
""", "Precipitation")

DIFF_DEFAULT = parse_pal("""
Product: DIFF
Units: IN
Color: -3     80 0 120     0 0 200
Color: -1     0 0 200      100 150 255
Color: -0.25  100 150 255  190 210 255
SolidColor4: -0.03  0 0 0 0
Color: 0.03   255 210 190  255 150 100
Color: 0.25   255 150 100  220 0 0
Color: 1      220 0 0      120 0 0
Color: 3      120 0 0
""", "Difference (blue = dual-pol less)")

BUILTIN: Dict[str, List[ColorTable]] = {
    "BR": [NWS_REFL, SMOOTH_REFL, DARK_REFL, CLEAR_AIR],
    "BV": [VEL_DEFAULT, VEL_BRIGHT],
    "SW": [SW_DEFAULT], "ZDR": [ZDR_DEFAULT], "CC": [CC_DEFAULT], "KDP": [KDP_DEFAULT],
    "HC": [HC_DEFAULT], "ET": [ET_DEFAULT], "VIL": [VIL_DEFAULT], "PRECIP": [PRECIP_DEFAULT],
    "DIFF": [DIFF_DEFAULT],
}


def load_folder(folder: Path) -> List[ColorTable]:
    out = []
    if folder.is_dir():
        for p in sorted(folder.glob("*.pal")):
            try:
                out.append(load_pal(p))
            except Exception:
                continue
    return out
