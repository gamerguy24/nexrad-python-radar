#!/usr/bin/env python3
"""
color_tables.py -- radar colour tables, including GRLevelX / GR2Analyst .pal files.

Radar data is kept as 8-bit codes (the NEXRAD Level III encoding):
    0 = no data / below threshold, 1 = range folded,
    code c >= 2 -> value = DMIN + (c - 2) * 0.5
with DMIN = -32 dBZ for reflectivity and -63.5 m/s for velocity.  A colour
table turns that into a 256-entry RGBA lookup table, so recolouring a frame
is a single numpy index operation.

Supported .pal lines (case-insensitive, ';' or '#' starts a comment):
    Product: BR | BV | ...     (velocity tables: BV, VEL, SRV, SRM, V)
    Units: DBZ | KTS | MPH | MPS
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

DMIN = {"refl": -32.0, "vel": -63.5}
INC = 0.5
UNIT_SCALE = {"KTS": 1.943844, "KT": 1.943844, "KNOTS": 1.943844, "MPH": 2.236936,
              "MPS": 1.0, "M/S": 1.0, "DBZ": 1.0}
VEL_PRODUCTS = {"BV", "VEL", "V", "SRV", "SRM", "N0U", "N0V"}
DEFAULT_ALPHA = 235

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
    kind: str                        # "refl" or "vel"
    stops: List[Stop]
    units: str = "DBZ"
    scale: float = 1.0               # data units -> table units
    offset: float = 0.0
    rf: RGBA = (119, 0, 119, 255)
    source: Optional[str] = None     # file path for loaded tables
    _lut: Optional[np.ndarray] = field(default=None, repr=False)

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
            return st.c1
        if i == len(s) - 1:
            return st.c2 or st.c1
        nxt = s[i + 1]
        end = st.c2 or nxt.c1
        span = nxt.value - st.value
        f = 0.0 if span <= 0 else (v - st.value) / span
        return tuple(int(round(a + (b - a) * f)) for a, b in zip(st.c1, end))

    def lut(self) -> np.ndarray:
        """(256, 4) uint8 RGBA lookup indexed by the 8-bit data code."""
        if self._lut is None:
            lut = np.zeros((256, 4), np.uint8)
            dmin = DMIN[self.kind]
            for c in range(2, 256):
                col = self.color((dmin + (c - 2) * INC) * self.scale + self.offset)
                if col:
                    lut[c] = col
            if self.kind == "vel":
                lut[1] = self.rf
            self._lut = lut
        return self._lut

    def value_range(self) -> Tuple[float, float]:
        return self.stops[0].value, self.stops[-1].value


def _rgba(nums, n):
    vals = [int(round(float(x))) for x in nums[:n]]
    if n == 3:
        vals.append(DEFAULT_ALPHA)
    return tuple(max(0, min(255, v)) for v in vals)


def parse_pal(text: str, name: str = "custom", source: Optional[str] = None) -> ColorTable:
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
                rf = _rgba(parts, 4 if len(parts) >= 4 else 3)
                rf = rf[:3] + (255,)
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
    kind = "vel" if (product in VEL_PRODUCTS or units in ("KTS", "KT", "KNOTS", "MPH", "MPS", "M/S")) \
        else "refl"
    if scale is None:
        scale = UNIT_SCALE.get(units, 1.0) if kind == "vel" else 1.0
    return ColorTable(name, kind, stops, units or ("KTS" if kind == "vel" else "DBZ"),
                      scale, offset, rf, source)


def load_pal(path) -> ColorTable:
    p = Path(path)
    return parse_pal(p.read_text(encoding="utf8", errors="ignore"), p.stem, str(p))


# --------------------------------------------------------------------------
# Built-in tables
# --------------------------------------------------------------------------
def _solid(name, pairs, kind="refl"):
    return ColorTable(name, kind, [Stop(v, (*c, DEFAULT_ALPHA), None, True) for v, c in pairs])


NWS_REFL = _solid("NWS (classic)", [
    (10, (1, 159, 244)), (15, (3, 0, 244)), (20, (2, 253, 2)), (25, (1, 197, 1)),
    (30, (0, 142, 0)), (35, (253, 248, 2)), (40, (229, 188, 0)), (45, (253, 149, 0)),
    (50, (253, 0, 0)), (55, (212, 0, 0)), (60, (188, 0, 0)), (65, (248, 0, 253)),
    (70, (152, 84, 198)), (75, (253, 253, 253))])

SMOOTH_REFL = parse_pal("""
Product: BR
Units: DBZ
Color: 5   40 70 90      60 110 130
Color: 15  80 150 200    40 90 200
Color: 20  20 200 60     10 140 30
Color: 30  10 120 20     240 240 50
Color: 40  250 200 20    250 120 10
Color: 50  240 30 30     160 0 0
Color: 60  255 50 230    150 60 210
Color: 70  220 200 255   255 255 255
Color: 80  255 255 255
""", "Smooth gradient")
SMOOTH_REFL.stops[0].value = 10.0                   # hide light clutter below 10 dBZ

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

BUILTIN: Dict[str, List[ColorTable]] = {
    "refl": [NWS_REFL, SMOOTH_REFL, DARK_REFL],
    "vel": [VEL_DEFAULT, VEL_BRIGHT],
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
