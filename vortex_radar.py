#!/usr/bin/env python3
"""
Vortex Radar -- desktop radar viewer + storm tracker for the 50 US states.

Two data sources, each with its own on/off switch:
  * NEXRAD radar sites  (NWS tgftp, see nexrad_radar.py / level3.py): a mosaic of
    every US radar, or pick one radar (NEXRAD or TDWR) to see any of its Level III
    products -- reflectivity, velocity, storm-relative velocity, dual-pol, echo
    tops, VIL, precipitation -- with storm-track, hail, mesocyclone, TVS and
    melting-layer overlays and the VAD wind profile
  * MRMS                (NOAA MRMS, see storm_tracker.py): the MRMS composite
    reflectivity layer and the tracked storm cells with their 55-60 min
    forecast paths, lightning threat and hail

The map uses a Lambert conformal projection per region (Lower 48, Alaska,
Hawaii) with state and county lines, and hides data over other countries
(toggle "US only").  The radar loop has play / pause / step and a time
slider.  Colour tables: built-in ones, or load any GRLevelX / GR2Analyst
.pal file (copied into color_tables/, which is loaded at start-up).

USAGE
  python vortex_radar.py
  python vortex_radar.py --office FFC --loop 4

Keys: space = play / pause, left / right = step, Home = latest frame.
Products are still written to storm_out/ exactly as the command-line tool does.
"""
from __future__ import annotations

import os

# numpy's and scipy's OpenBLAS reserve ~30 MB per CPU thread each as soon as they are
# imported (about 1 GB per process on a 16-thread PC, three processes here).  Nothing
# in this program uses multi-threaded linear algebra, so keep them to one thread.
# Must run before numpy is imported; the worker processes inherit it.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import concurrent.futures as cf
import contextlib
import datetime as dt
import io
import logging
import math
import multiprocessing as mp
import os
import queue
import re
import shutil
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
import requests
import shapely
from PIL import Image, ImageTk

import color_tables as ct
import level3 as l3
import nexrad_radar as nr
import storm_tracker as st
import us_map as um

log = logging.getLogger("storm_tracker.gui")
UTC = dt.timezone.utc
APP_NAME = "Vortex Radar"
APP_VERSION = "1.1.0"     # bump this, rebuild (build.bat) and publish a GitHub release
GITHUB_REPO = "gamerguy24/nexrad-python-radar"
FROZEN = getattr(sys, "frozen", False)      # running as the PyInstaller exe


def _data_dir() -> Path:
    """Where downloads, outputs, colour tables and the log live: next to the exe
    (or this script); %LOCALAPPDATA%\\VortexRadar if that folder is read-only."""
    here = Path(sys.executable if FROZEN else __file__).resolve().parent
    try:
        probe = here / ".write_test"
        probe.touch()
        probe.unlink()
        return here
    except OSError:
        alt = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VortexRadar"
        alt.mkdir(parents=True, exist_ok=True)
        return alt


DATA_DIR = _data_dir()
PAL_DIR = DATA_DIR / "color_tables"


def _setup_logging():
    """The exe has no console, so log to a file there; the script logs to the terminal."""
    kw = dict(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    if FROZEN:
        kw["filename"] = str(DATA_DIR / "vortex_radar.log")
    logging.basicConfig(**kw)


def _version_tuple(v: str):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])

NEX_FRAMES, NEX_STEP_MIN = 12, 5      # radar-site loop: 12 frames, 5 min apart
MRMS_FRAMES = 10                      # MRMS loop frames (6 min apart, from the tracker)
SITE_FRAMES = 15                      # single-site loop: newest 15 scans
SITE_REFRESH_S = 120
MATCH_MIN = 6.0                       # other layers must be this close to the loop time
SPEEDS = {"Slow": 900, "Normal": 450, "Fast": 200}

LEVEL_COLORS = {"LOW": "#6f8597", "MODERATE": "#e8c547", "HIGH": "#f08a3c",
                "VERY HIGH": "#ef5359"}
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

BG, PANEL, PANEL2, LINE = "#0e1620", "#15202b", "#1b2835", "#2a3a4a"
WATER, LAND, COUNTY, BORDER = "#0a1118", "#1a2836", "#3a536a", "#8aa4ba"
FG, MUTED, ACCENT = "#dbe4ec", "#8c9fb1", "#f2b84b"


def compass(deg):
    return COMPASS[int(round((deg % 360) / 22.5)) % 16]


def inches(mm):
    return "-" if mm < 0.5 else f"{mm / 25.4:.2f} in"


def fmt_time(t):
    if t is None:
        return "--"
    return f"{t:%H:%M} UTC  ({t.astimezone():%I:%M %p} local)"


def fmt_value(v, fam):
    """Readout text for a data value of a colour-table family."""
    if fam == "BR":
        return f"{v:.1f} dBZ"
    if fam in ("BV", "SW"):
        return f"{v * 1.943844:.0f} kts ({v:.1f} m/s)"
    if fam == "ZDR":
        return f"{v:.2f} dB"
    if fam == "CC":
        return f"{v:.3f}"
    if fam == "KDP":
        return f"{v:.2f} deg/km"
    if fam == "HC":
        return l3.HC_CLASSES.get(int(round(v)), ("?", "unknown"))[1]
    if fam == "ET":
        return f"{v:.0f} kft"
    if fam == "VIL":
        return f"{v:.1f} kg/m\u00b2"
    if fam == "PRECIP":
        return f"{v:.2f} in"
    if fam == "DIFF":
        return f"{v:+.2f} in"
    return f"{v:g}"


# --------------------------------------------------------------------------
# Background workers (separate processes, so number crunching never stalls
# the window).  Each idles while its data source is switched off.
# --------------------------------------------------------------------------
def _wait_enabled(enabled, stop, q, tag) -> bool:
    if not enabled.is_set():
        q.put((tag, None))
        while not enabled.is_set() and not stop.is_set():
            enabled.wait(1.0)
    return not stop.is_set()


def _mrms_codes(arr, mg: st.Grid, g: um.RasterGrid, cache) -> np.ndarray:
    """MRMS composite reflectivity -> 8-bit codes on the map's projected grid."""
    key = (mg.lat0, mg.lon0, mg.dlat, mg.dlon, mg.ny, mg.nx, g)
    idx = cache.get(key)
    if idx is None:
        lon, lat = um.grid_lonlat(g)
        r, c = mg.rc(lat, lon)
        r, c = np.rint(r).astype(np.int32), np.rint(c).astype(np.int32)
        ok = (r >= 0) & (r < mg.ny) & (c >= 0) & (c < mg.nx)
        idx = cache[key] = np.where(ok, r * mg.nx + c, -1)
    v = arr.astype(np.float32).ravel()[np.maximum(idx, 0)]
    codes = np.clip(np.rint((v + 32.0) * 2.0) + 2, 2, 255).astype(np.uint8)
    codes[(v < -29.0) | (idx < 0)] = 0
    return codes


def run_mrms_worker(cfg: st.Config, loop_min: float, top: int, q, wake, stop, enabled, refl_on):
    _setup_logging()
    wfos = None
    grids = {d: um.domain_grid(d) for d in cfg.domains}
    sent = {d: set() for d in cfg.domains}
    idx_cache = {}
    while not stop.is_set():
        if not _wait_enabled(enabled, stop, q, "mrms_off"):
            break
        t0 = time.time()
        try:
            if wfos is None:
                q.put(("status", "Loading NWS office boundaries..."))
                wfos = st.WfoCatalog.fetch(cfg.cache_dir)
            q.put(("status", "Downloading and tracking MRMS storm data..."))
            src = st.LiveSource(cfg)
            pipe = st.StormPipeline(cfg, src, wfos)
            rows = pipe.run()
            summary = pipe.wfo_summary(rows)
            if pipe.latest:
                out_dir = st.write_outputs(rows, summary, pipe, cfg.out_dir)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    st.print_report(rows, summary, pipe, top)
                q.put(("result", {
                    "rows": rows, "summary": summary, "status": dict(pipe.status),
                    "newest": max(pipe.latest.values()), "report": buf.getvalue(),
                    "out_dir": str(out_dir), "secs": time.time() - t0}))
                for dom in (pipe.latest if refl_on.is_set() else ()):   # MRMS reflectivity loop
                    times = sorted(src.f.timeline(dom))[-MRMS_FRAMES:]
                    for t in reversed(times):
                        if t in sent[dom]:
                            continue
                        res = src.f.load(dom, "cref", t)
                        if res is not None:
                            q.put(("mrms_frame", (dom, t, _mrms_codes(res[0], res[1], grids[dom],
                                                                       idx_cache))))
                    sent[dom] = set(times)
                    q.put(("mrms_keep", (dom, times)))
                if not refl_on.is_set():                     # layer off: window keeps nothing
                    sent = {d: set() for d in cfg.domains}
            else:
                q.put(("error", "No MRMS data retrieved; will retry."))
            src.f.prune()
        except Exception as e:                  # keep the live loop alive
            log.exception("live update failed")
            q.put(("error", f"MRMS update failed ({e}); will retry."))
        next_at = t0 + loop_min * 60
        q.put(("next", next_at))
        wake.clear()
        wake.wait(max(5.0, next_at - time.time()))


def run_nexrad_worker(loop_min: float, q, wake, stop, enabled):
    _setup_logging()
    feeder = nr.NexradFeeder()
    grids = {d: um.domain_grid(d) for d in um.DOMAINS}
    have = set()
    tdwr_sent = False
    while not stop.is_set():
        if not _wait_enabled(enabled, stop, q, "nexrad_off"):
            break
        t0 = time.time()
        try:
            now = dt.datetime.now(UTC).replace(second=0, microsecond=0) - dt.timedelta(minutes=3)
            last = now - dt.timedelta(minutes=now.minute % NEX_STEP_MIN)
            marks = [last - dt.timedelta(minutes=NEX_STEP_MIN * k) for k in range(NEX_FRAMES - 1)]
            todo = [m for m in marks if m not in have]
            q.put(("radar_status", f"Radar sites: downloading {len(todo) + 1} frames..."))
            hist, locs = feeder.history([None] + todo, max_gap_min=11.0)
            q.put(("sites", locs))
            if not tdwr_sent:
                td = feeder.tdwr_sites()
                if td:
                    q.put(("tdwr_sites", td))
                    tdwr_sent = True
            for ft in [None] + todo:                       # newest first: map fills quickly
                scans = hist.get(ft) or []
                if not scans:
                    continue
                key = ft or max(s.time for s in scans)
                by_dom = {}
                for d, g in grids.items():
                    ds = [s for s in scans if nr.site_domain(s.lat, s.lon) == d]
                    if ds:
                        by_dom[d] = nr.mosaic(ds, g)
                q.put(("nexrad_frame", (key, by_dom, len(scans), ft is None)))
                if ft is not None:
                    have.add(ft)
            have &= set(marks)
            q.put(("nexrad_keep", {"marks": marks, "secs": time.time() - t0}))
            feeder.prune(marks[-1] - dt.timedelta(minutes=20))
        except Exception as e:                  # radar must not kill the loop
            log.exception("NEXRAD update failed")
            q.put(("radar_error", str(e)))
        next_at = t0 + loop_min * 60
        q.put(("radar_next", next_at))
        wake.clear()
        wake.wait(max(5.0, next_at - time.time()))


class SiteLoader(threading.Thread):
    """Downloads and renders one radar site's recent scans of any Level III product,
    plus its overlay products (storm tracks, hail, ...).  Runs in the GUI process."""

    def __init__(self, out: queue.Queue):
        super().__init__(daemon=True)
        self.out, self.req = out, None
        self.wake = threading.Event()
        self.feeder = nr.NexradFeeder(workers=8)
        self.rend = {}
        self.frames = {}

    def request(self, req):
        """req = (site, product key, tilt index, domain, lat, lon) or None."""
        self.req = req
        self.wake.set()

    def _renderer(self, site, dom, lat, lon, d):
        key = (site, dom) + nr.SiteRenderer.key_for(d)
        r = self.rend.get(key)
        if r is None:
            if len(self.rend) >= 2:
                self.rend.clear()
            r = self.rend[key] = nr.SiteRenderer(dom, lat, lon, *nr.SiteRenderer.key_for(d))
        return r

    def run(self):
        while True:
            self.wake.wait(SITE_REFRESH_S)
            self.wake.clear()
            req = self.req
            if req is None:
                continue
            site, key, tilt, dom, lat, lon = req
            prod = l3.BY_KEY[key]
            folder = prod.tilts[min(tilt, len(prod.tilts) - 1)][1]
            try:
                self.out.put(("site_status", f"Radar site {site.upper()}: loading {prod.key} "
                                             f"{prod.name.lower()}..."))
                now = dt.datetime.now(UTC)
                full = self.feeder.list_folder(site, folder)
                lst = [x for x in full if x[0] >= now - dt.timedelta(minutes=80)][-SITE_FRAMES:]
                if not lst and full and now - full[-1][0] < dt.timedelta(hours=26):
                    lst = full[-1:]                       # slow products (e.g. 24-h totals)
                stale = full[-1][0] if full and not lst else None
                keys = [(site, folder, fn, ft) for ft, fn in lst]
                todo = [k for k in keys if k not in self.frames]
                with cf.ThreadPoolExecutor(8) as ex:
                    raws = list(ex.map(lambda k: self.feeder.fetch(k[0], k[1], k[2]), todo))
                for k, raw in zip(todo, raws):
                    if raw is None:
                        continue
                    try:
                        d = l3.decode(raw, prod)
                    except Exception:
                        log.debug("decode %s failed", k, exc_info=True)
                        continue
                    if d.codes is None:                   # e.g. a "no accumulation" product
                        self.frames[k] = None
                        continue
                    r = self._renderer(site, dom, lat, lon, d)
                    self.frames[k] = (d.time, r.render(d), d.values, d.rf_code, d.elev, r.grid)
                keep = set(keys)
                self.frames = {k: v for k, v in self.frames.items() if k in keep}
                if self.req != req:
                    continue
                frames = sorted({v[0]: v for v in self.frames.values() if v}.values(),
                                key=lambda v: v[0])
                self.out.put(("site_frames", {"req": req, "frames": frames, "listed": len(lst),
                                              "stale": stale}))
                if not prod.tdwr:                         # overlay products (NEXRAD only)
                    f0 = prod.tilts[0][1]
                    diff = [b for a, b in zip(f0, folder) if a != b]
                    sfx = diff[0] if diff else "0"
                    ovs = {}
                    for ok, (_, ofolder) in l3.OVERLAYS.items():
                        raw = self.feeder.fetch(site, ofolder.format(t=sfx))
                        if raw:
                            try:
                                d = l3.decode(raw)
                                ovs[ok] = (d.time, d.features)
                            except Exception:
                                log.debug("overlay %s failed", ok, exc_info=True)
                    if self.req == req:
                        self.out.put(("site_overlays", {"site": site, "data": ovs}))
            except Exception as e:
                log.exception("site load failed")
                self.out.put(("site_error", f"{site.upper()}: {e}"))


class VwpWindow(tk.Toplevel):
    """VAD wind profile (product 48), drawn the way the radar product lays it out."""

    COLORS = {0: "#8c9fb1", 1: "#46d246", 2: "#00b4ff", 3: "#f0e040", 4: "#ff5050", 5: "#ff50ff"}

    def __init__(self, app, site):
        super().__init__(app)
        self.title(f"{site.upper()} VAD Wind Profile")
        self.configure(bg=BG)
        self.geometry("760x640")
        self.site = site
        self.cv = tk.Canvas(self, bg="#0a1017", highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.cv.create_text(20, 20, anchor="nw", fill=MUTED, text="Loading wind profile...",
                            font=("Segoe UI", 11))
        self.graphic = None
        self.cv.bind("<Configure>", lambda e: self.draw())
        self._result = None
        threading.Thread(target=self._load, daemon=True).start()
        self.after(200, self._check)

    def _load(self):
        try:
            raw = nr.NexradFeeder(workers=1).fetch(self.site, l3.VWP_FOLDER)
            g = l3.decode(raw).graphic if raw else None
            msg = None if g else f"No wind profile available for {self.site.upper()}."
        except Exception as e:
            g, msg = None, f"Wind profile failed: {e}"
        self._result = (g, msg)

    def _check(self):                                    # Tk calls stay on the GUI thread
        if self._result is None:
            self.after(200, self._check)
            return
        g, msg = self._result
        self.graphic = g
        if msg:
            self.cv.delete("all")
            self.cv.create_text(20, 20, anchor="nw", fill=MUTED, text=msg, font=("Segoe UI", 11))
        else:
            self.draw()

    def draw(self):
        g = self.graphic
        if not g:
            return
        cv = self.cv
        cv.delete("all")
        xs = [v for e in g for v in ((e[2], e[4]) if e[0] == "line" else (e[2],))]
        ys = [v for e in g for v in ((e[3], e[5]) if e[0] == "line" else (e[3],))]
        x0, x1, y0, y1 = min(xs), max(xs) + 20, min(ys), max(ys) + 10
        W, H = max(cv.winfo_width(), 200), max(cv.winfo_height(), 200)
        k = min((W - 20) / max(x1 - x0, 1), (H - 20) / max(y1 - y0, 1))

        def X(x):
            return 10 + (x - x0) * k

        def Y(y):
            return 10 + (y - y0) * k
        for e in g:
            col = self.COLORS.get(e[1], FG)
            if e[0] == "line":
                cv.create_line(X(e[2]), Y(e[3]), X(e[4]), Y(e[5]),
                               fill="#3a536a" if e[1] not in (1, 2, 3, 4, 5) else col)
            elif e[0] == "text":
                cv.create_text(X(e[2]), Y(e[3]), anchor="nw", text=e[4],
                               fill=FG if e[1] not in self.COLORS else col,
                               font=("Consolas", max(7, int(8 * k))))
            elif e[0] == "barb":
                self._barb(X(e[2]), Y(e[3]), e[4], e[5], col, 22 * k)

    def _barb(self, x, y, direction, speed, col, L):
        """Standard wind barb; the staff points toward where the wind comes from."""
        a = math.radians(direction)
        ux, uy = math.sin(a), -math.cos(a)
        px, py = -uy, ux                                  # perpendicular side for the barbs
        ex, ey = x + ux * L, y + uy * L
        self.cv.create_line(x, y, ex, ey, fill=col, width=1.5)
        self.cv.create_oval(x - 2, y - 2, x + 2, y + 2, fill=col, outline=col)
        s = int(round(speed / 5.0)) * 5
        pos, step, bl = 0.0, L * 0.16, L * 0.45
        while s >= 50:                                    # pennants
            bx, by = ex - ux * pos, ey - uy * pos
            self.cv.create_polygon(bx, by, bx + px * bl, by + py * bl,
                                   bx - ux * step, by - uy * step, fill=col, outline=col)
            pos += step * 1.3
            s -= 50
        while s >= 10:                                    # full barbs
            bx, by = ex - ux * pos, ey - uy * pos
            self.cv.create_line(bx, by, bx + px * bl + ux * step * 0.6,
                                by + py * bl + uy * step * 0.6, fill=col, width=1.5)
            pos += step
            s -= 10
        if s >= 5:                                        # half barb
            pos = max(pos, step)
            bx, by = ex - ux * pos, ey - uy * pos
            self.cv.create_line(bx, by, bx + px * bl * 0.5 + ux * step * 0.3,
                                by + py * bl * 0.5 + uy * step * 0.3, fill=col, width=1.5)


# --------------------------------------------------------------------------
# Map canvas
# --------------------------------------------------------------------------
class MapView(tk.Canvas):
    def __init__(self, master, app):
        super().__init__(master, bg=WATER, highlightthickness=0, cursor="fleur")
        self.app = app
        self.domain = "CONUS"
        self.geo = None                    # dom -> shapely geometry arrays
        self.base = None
        self.scale = None                  # pixels per metre
        self.ox = self.oy = 0.0
        self._drag = None
        self._pending = None
        self._photos = []
        self._ov_tips = {}                 # overlay hover texts
        self._fitted = None                # view box to keep fitted until the user zooms / pans
        self.bind("<Configure>", self._resized)
        self.bind("<MouseWheel>", self._wheel)                       # Windows / macOS
        self.bind("<Button-4>", lambda e: self._zoom(1.25, e.x, e.y))  # Linux
        self.bind("<Button-5>", lambda e: self._zoom(0.8, e.x, e.y))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Motion>", self._hover)
        self.bind("<Leave>", lambda e: self.delete("tip"))

    def set_basemap(self, bm: um.BaseMap):
        self.base = bm
        geo = {}
        for d in um.DOMAINS:
            geo[d] = {lvl: np.array([shapely.Polygon(r) for r in bm.land[d][lvl]], dtype=object)
                      for lvl in ("coarse", "fine")}
            geo[d]["counties"] = np.array([shapely.LineString(r) for _, r in bm.counties[d]],
                                          dtype=object)
        self.geo = geo
        self.redraw()

    # ---------- projection: lon/lat -> projected metres -> pixels ----------
    def to_px(self, lon, lat):
        x, y = um.project(self.domain, lon, lat)
        return self.ox + x * self.scale, self.oy - y * self.scale

    def xy_px(self, x, y):
        return self.ox + np.asarray(x) * self.scale, self.oy - np.asarray(y) * self.scale

    def view_xy(self, margin=0.0):
        cw, ch = max(self.winfo_width(), 50), max(self.winfo_height(), 50)
        m = margin * max(cw, ch)
        return ((-m - self.ox) / self.scale, (cw + m - self.ox) / self.scale,
                (self.oy - ch - m) / self.scale, (self.oy + m) / self.scale)

    def px_per_km(self):
        return self.scale * 1000.0

    def set_domain(self, dom):
        self.domain = dom
        self.fit()

    def fit(self, box=None):
        self.fit_xy(*um.box_bounds(self.domain, box or um.VIEW[self.domain]))

    def _resized(self, _e):
        if self.scale is None:
            self.fit()
        elif self._fitted:                 # untouched view: re-fit to the new size
            self._schedule_refit()
        else:
            self._schedule_redraw(100)

    def _schedule_refit(self):
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(100, lambda: (setattr(self, "_pending", None),
                                                 self.fit_xy(*self._fitted)))

    def fit_xy(self, xmin, xmax, ymin, ymax):
        self._fitted = (xmin, xmax, ymin, ymax)
        cw, ch = max(self.winfo_width(), 50), max(self.winfo_height(), 50)
        self.scale = min(cw / (xmax - xmin), ch / (ymax - ymin)) * 0.97
        self.ox = cw / 2 - (xmin + xmax) / 2 * self.scale
        self.oy = ch / 2 + (ymin + ymax) / 2 * self.scale
        self.redraw()

    def zoom_to(self, lon, lat, half_km):
        x, y = um.project(self.domain, lon, lat)
        h = half_km * 1000.0
        self.fit_xy(float(x) - h, float(x) + h, float(y) - h, float(y) + h)

    def center_on(self, lon, lat):
        self._fitted = None
        x, y = self.to_px(lon, lat)
        self.ox += self.winfo_width() / 2 - float(x)
        self.oy += self.winfo_height() / 2 - float(y)
        self.redraw()

    def _fit_scale(self):
        xmin, xmax, ymin, ymax = um.box_bounds(self.domain, um.VIEW[self.domain])
        cw, ch = max(self.winfo_width(), 50), max(self.winfo_height(), 50)
        return min(cw / (xmax - xmin), ch / (ymax - ymin))

    def _zoom(self, f, x, y):
        self._fitted = None
        fs = self._fit_scale()
        new = min(max(self.scale * f, fs * 0.5), fs * 150)
        f = new / self.scale
        if f == 1.0:
            return
        self.ox = x - (x - self.ox) * f
        self.oy = y - (y - self.oy) * f
        self.scale = new
        # instant feedback: stretch what is already drawn, then redraw properly
        self.delete("tip")
        tk.Canvas.scale(self, "all", x, y, f, f)
        self._schedule_redraw(150)

    def _schedule_redraw(self, ms):
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(ms, self._do_pending)

    def _do_pending(self):
        self._pending = None
        self.redraw()

    def _wheel(self, e):
        self._zoom(1.25 if e.delta > 0 else 0.8, e.x, e.y)

    def _press(self, e):
        self._drag = [e.x, e.y, e.x, e.y, False]          # start x/y, last x/y, moved

    def _motion(self, e):
        if not self._drag:
            return
        x0, y0, lx, ly, moved = self._drag
        if moved or abs(e.x - x0) + abs(e.y - y0) > 4:
            if not moved:
                self.delete("tip")
            dx, dy = e.x - lx, e.y - ly
            self.move("all", dx, dy)                       # slide, don't redraw
            self.ox += dx
            self.oy += dy
            self._drag = [x0, y0, e.x, e.y, True]
            self._fitted = None

    def _release(self, e):
        if self._drag and not self._drag[4]:
            kind, val = self._item_at(e.x, e.y)
            if kind == "id":
                self.app.select(val)
            elif kind == "site":
                self.app.choose_site(val)
        elif self._drag:
            self.redraw()                                  # fill in newly exposed areas
        self._drag = None

    def _item_at(self, x, y):
        for item in reversed(self.find_overlapping(x - 3, y - 3, x + 3, y + 3)):
            for tag in self.gettags(item):
                if tag.startswith(("id:", "site:", "ov:")):
                    k, v = tag.split(":", 1)
                    return k, v
        return None, None

    def _hover(self, e):
        self.delete("tip")
        if self.scale is not None:
            self.app.show_readout(e.x, e.y)
        kind, val = self._item_at(e.x, e.y)
        if kind == "ov":
            txt = self._ov_tips.get(val)
            if not txt:
                return
        elif kind == "id":
            r = self.app.by_id.get(val)
            if not r:
                return
            L = r["lightning"]
            txt = (f"{r['track_id']}  {r['cref_dbz']:.0f} dBZ  lightning {L['p_lightning_55_60'] * 100:.0f}%\n"
                   f"{r['wfo'] or 'over water'} -> {L['wfo_at_55_60'] or 'over water'}")
        elif kind == "site":
            txt = (f"{'TDWR' if val in self.app.tdwr else 'Radar'} {val.upper()}"
                   " - click to view this site")
        else:
            return
        t = self.create_text(e.x + 14, e.y - 14, text=txt, anchor="sw", fill=FG,
                             font=("Consolas", 9), tags="tip")
        x0, y0, x1, y1 = self.bbox(t)
        self.tag_lower(self.create_rectangle(x0 - 6, y0 - 4, x1 + 6, y1 + 4, fill=PANEL2,
                                             outline=LINE, tags="tip"), t)

    # ---------- drawing ----------
    def _draw_raster(self, codes, g: um.RasterGrid, lut, mask, kind):
        xmin, xmax, ymin, ymax = self.view_xy()
        c0, c1 = max(0, int((xmin - g.x0) / g.res)), min(g.nx, math.ceil((xmax - g.x0) / g.res))
        r0, r1 = max(0, int((g.y1 - ymax) / g.res)), min(g.ny, math.ceil((g.y1 - ymin) / g.res))
        if c1 <= c0 or r1 <= r0:
            return
        # zoomed out: several grid cells per screen pixel -> shrink before colouring,
        # keeping the strongest echo of each block (velocity: plain subsample)
        k = max(1, int(1.0 / (g.res * self.scale)))
        if k >= 2:
            c1, r1 = c0 + (c1 - c0) // k * k, r0 + (r1 - r0) // k * k
            if c1 <= c0 or r1 <= r0:
                return
        sub = codes[r0:r1, c0:c1]
        if mask is not None:
            sub = np.where(mask[r0:r1, c0:c1], sub, 0)
        if k >= 2:
            h, w = sub.shape
            sub = (sub.reshape(h // k, k, w // k, k).max(axis=(1, 3)) if kind == "max"
                   else sub[::k, ::k])
        if not sub.any():
            return
        img = Image.fromarray(lut[sub], "RGBA")
        x0, y0 = self.xy_px(g.x0 + c0 * g.res, g.y1 - r0 * g.res)
        x1, y1 = self.xy_px(g.x0 + c1 * g.res, g.y1 - r1 * g.res)
        tw, th = int(round(float(x1 - x0))), int(round(float(y1 - y0)))
        if tw < 1 or th < 1:
            return
        img = img.resize((tw, th), Image.NEAREST if tw >= img.width else Image.BOX)
        photo = ImageTk.PhotoImage(img)
        self._photos.append(photo)                         # keep a reference for Tk
        self.create_image(float(x0), float(y0), image=photo, anchor="nw", tags="radar")

    def redraw_radar(self):
        if self.scale is None:
            return
        self.delete("radar")
        self._photos = []
        for codes, g, lut, kind, _ in self.app.layers():
            self._draw_raster(codes, g, lut, self.app.mask_for(g), kind)
        self.tag_lower("radar")
        self.tag_lower("land")
        self._draw_frame_label()

    def _draw_frame_label(self):
        self.delete("framelbl")
        t = self.app.current_time()
        if t is None:
            return
        old = not self.app.at_latest()
        lbl = self.create_text(12, 12, anchor="nw", fill=ACCENT if old else FG,
                               font=("Segoe UI Semibold", 11), tags="framelbl",
                               text=f"{self.app.layer_title()}   {t:%Y-%m-%d %H:%M} UTC")
        x0, y0, x1, y1 = self.bbox(lbl)
        self.tag_lower(self.create_rectangle(x0 - 6, y0 - 3, x1 + 6, y1 + 3, fill=PANEL,
                                             outline=LINE, tags="framelbl"), lbl)

    def _clipped(self, geoms, rect):
        if geoms is None or len(geoms) == 0:
            return []
        out = []
        for gm in shapely.clip_by_rect(geoms, *rect):
            if gm is None or gm.is_empty:
                continue
            for part in getattr(gm, "geoms", [gm]):
                c = part.exterior.coords if part.geom_type == "Polygon" else part.coords
                a = np.asarray(c)
                if len(a) >= 2:
                    x, y = self.xy_px(a[:, 0], a[:, 1])
                    out.append(np.column_stack([x, y]).ravel().tolist())
        return out

    def redraw(self):
        if self.scale is None:
            return
        self.delete("all")
        self._photos = []
        app = self.app
        xmin, xmax, ymin, ymax = self.view_xy(margin=0.6)
        rect = (xmin, ymin, xmax, ymax)
        ppk = self.px_per_km()
        if self.geo:
            g = self.geo[self.domain]
            states = self._clipped(g["fine" if ppk > 0.4 else "coarse"], rect)
            for pts in states:
                if len(pts) >= 6:
                    self.create_polygon(pts, fill=LAND, outline="", tags="land")
        for codes, gr, lut, kind, _ in app.layers():
            self._draw_raster(codes, gr, lut, app.mask_for(gr), kind)
        if self.geo:
            if app.show_counties.get() and ppk >= 0.6:
                for pts in self._clipped(self.geo[self.domain]["counties"], rect):
                    self.create_line(pts, fill=COUNTY)
            for pts in states:
                self.create_line(pts, fill=BORDER, width=1.3 if ppk > 1.5 else 1)
            if 0.15 <= ppk <= 4:
                for postal, x, y in self.base.labels[self.domain]:
                    px, py = self.xy_px(x, y)
                    self.create_text(float(px), float(py), text=postal, fill="#5d7489",
                                     font=("Segoe UI Semibold", 8 if ppk < 0.6 else 10))
        self._draw_sites()
        self._draw_overlays()
        self._draw_storms()
        cw, ch = self.winfo_width(), self.winfo_height()
        if not self.geo:
            self.create_text(cw / 2, ch / 2, fill=MUTED, font=("Segoe UI", 12),
                             text="Loading map...")
        self._draw_frame_label()
        self.create_text(10, ch - 8, anchor="sw", fill=MUTED, font=("Segoe UI", 8),
                         text="Scroll to zoom  |  drag to pan  |  click a storm or radar site  |"
                              "  space = play/pause, arrows = step  |  hover = value")
        self.tag_lower("radar")
        self.tag_lower("land")

    def _draw_sites(self):
        app = self.app
        ppk = self.px_per_km()
        ring = app.site_range_km()
        if app.site and app.show_nexrad.get() and app.site in app.site_locs and ring:
            lat, lon = app.site_locs[app.site]
            a = np.radians(np.arange(0, 361, 3))
            lats = lat + ring / 111.0 * np.cos(a)
            lons = lon + ring / (111.32 * math.cos(math.radians(lat))) * np.sin(a)
            x, y = self.to_px(lons, lats)
            self.create_line(np.column_stack([x, y]).ravel().tolist(), fill=ACCENT, dash=(5, 4))
        if not app.show_sites.get() or ppk < 0.22:
            return
        for s, (lat, lon) in app.site_locs.items():
            if nr.site_domain(lat, lon) != self.domain:
                continue
            tdwr = s in app.tdwr
            sel = s == app.site
            if tdwr and ppk < 0.45 and not sel:          # keep the national view uncluttered
                continue
            x, y = (float(v) for v in self.to_px(lon, lat))
            if not (-50 < x < self.winfo_width() + 50 and -50 < y < self.winfo_height() + 50):
                continue
            tags = ("site:" + s,)
            if tdwr:
                self.create_polygon(x, y - 4, x + 4, y, x, y + 4, x - 4, y,
                                    fill=ACCENT if sel else "#b59ce0", outline="#000000", tags=tags)
            else:
                self.create_rectangle(x - 3, y - 3, x + 3, y + 3, fill=ACCENT if sel else "#c8d4de",
                                      outline="#000000", tags=tags)
            if ppk >= 0.45 or sel:
                self.create_text(x + 6, y, anchor="w", text=s.upper(), tags=tags,
                                 fill=ACCENT if sel else ("#b59ce0" if tdwr else "#a9bccc"),
                                 font=("Segoe UI Semibold" if sel else "Segoe UI", 8))

    def _ov_tag(self, text):
        k = str(len(self._ov_tips))
        self._ov_tips[k] = text
        return ("ov:" + k,)

    def _draw_overlays(self):
        """Storm tracks, hail, mesocyclones, TVS and melting layer of the selected radar."""
        self._ov_tips = {}
        app = self.app
        ovs = app.site_overlays
        if not (app.site and app.show_nexrad.get() and ovs and app.at_latest()
                and app.site in app.site_locs):
            return
        lat0, lon0 = app.site_locs[app.site]

        def px(pts):
            a = np.asarray(pts, float).reshape(-1, 2)
            lon, lat = l3.km_to_lonlat(lat0, lon0, a[:, 0], a[:, 1])
            x, y = self.to_px(lon, lat)
            return np.column_stack([x, y])
        show = {k: v.get() for k, v in app.ov_vars.items()}
        if show["ml"] and "ml" in ovs:
            dashes = {1: (2, 4), 2: (6, 3), 3: (6, 3), 4: (2, 4)}
            for color, pts in ovs["ml"][1].get("ml", []):
                xy = px(pts)
                self.create_line(xy.ravel().tolist(), fill="#7fd4ff" if color in (2, 3) else "#3f8fc0",
                                 dash=dashes.get(color, (4, 4)), width=1.5)
        if show["sti"] and "sti" in ovs:
            f = ovs["sti"][1]
            for pts in f.get("past", []):
                xy = px(pts)
                if len(xy) > 1:
                    self.create_line(xy.ravel().tolist(), fill="#c8d4de", width=1.5)
            for x, y in px(f.get("past_pts", [])) if f.get("past_pts") else []:
                self.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5, outline="#c8d4de", fill=WATER)
            for pts in f.get("fcst", []):
                xy = px(pts)
                if len(xy) > 1:
                    self.create_line(xy.ravel().tolist(), fill="#ffffff", width=1.5, dash=(5, 3))
            for x, y in px(f.get("fcst_pts", [])) if f.get("fcst_pts") else []:
                self.create_line(x - 3, y, x + 3, y, fill="#ffffff")
                self.create_line(x, y - 3, x, y + 3, fill="#ffffff")
            for (x, y), (_, _, sid) in zip(px([(a, b) for a, b, _ in f.get("ids", [])]) if f.get("ids") else [],
                                           f.get("ids", [])):
                self.create_text(x + 7, y - 7, text=sid, fill="#ffffff", anchor="sw",
                                 font=("Segoe UI Semibold", 9),
                                 tags=self._ov_tag(f"Storm {sid} (radar storm tracking)"))
        if show["hail"] and "hail" in ovs:
            for x_km, y_km, poh, posh, size in ovs["hail"][1].get("hail", []):
                if poh < 30 and posh < 30:
                    continue
                (x, y), = px([(x_km, y_km)])
                fill = "#ff3c3c" if posh >= 50 else ("#ffd23c" if posh >= 30 or poh >= 50 else "")
                r = 6 if posh >= 50 else 5
                txt = f"Hail: {poh}% chance of hail, {posh}% chance of severe hail"
                if size > 0:
                    txt += f", max size {size} in"
                self.create_polygon(x, y - r, x + r, y + r * 0.8, x - r, y + r * 0.8,
                                    fill=fill, outline="#46d246" if not fill else "#000000",
                                    width=1.5, tags=self._ov_tag(txt))
        if show["meso"] and "meso" in ovs:
            for x_km, y_km, typ, rad in ovs["meso"][1].get("meso", []):
                (x, y), = px([(x_km, y_km)])
                r = max(rad * self.px_per_km(), 6)
                self.create_oval(x - r, y - r, x + r, y + r, outline="#ffe040",
                                 width=3 if typ >= 5 else 2,
                                 tags=self._ov_tag(f"Mesocyclone / circulation (type {typ}), "
                                                   f"radius {rad:.0f} km"))
        if show["tvs"] and "tvs" in ovs:
            f = ovs["tvs"][1]
            for kind, pts in (("TVS", f.get("tvs", [])), ("Elevated TVS", f.get("etvs", []))):
                for x, y in px(pts) if pts else []:
                    self.create_polygon(x - 7, y - 6, x + 7, y - 6, x, y + 8,
                                        fill="#ff2020" if kind == "TVS" else "",
                                        outline="#ff2020", width=2,
                                        tags=self._ov_tag(f"{kind} (tornado vortex signature)"))

    def _draw_storms(self):
        app = self.app
        if not (app.show_tracks.get() and app.at_latest()):
            return
        rows = [r for r in app.rows if r["domain"] == self.domain]
        cw, ch = self.winfo_width(), self.winfo_height()
        if app.show_areas.get():
            for r in rows:
                L = r["lightning"]
                poly = np.array(L["threat_polygon"])
                x, y = self.to_px(poly[:, 0], poly[:, 1])
                if x.max() < 0 or x.min() > cw or y.max() < 0 or y.min() > ch:
                    continue
                self.create_polygon(np.column_stack([x, y]).ravel().tolist(), fill="",
                                    outline=LEVEL_COLORS[L["threat_level"]], dash=(4, 3))
        for r in rows:
            L = r["lightning"]
            x, y = self.to_px([r["lon"], L["forecast_lon_55"], L["forecast_lon_60"]],
                              [r["lat"], L["forecast_lat_55"], L["forecast_lat_60"]])
            if x.max() < 0 or x.min() > cw or y.max() < 0 or y.min() > ch:
                continue
            self.create_line(np.column_stack([x, y]).ravel().tolist(), fill=FG,
                             arrow="last", arrowshape=(7, 8, 3))
        sel = app.selected
        refl = app.table("BR")
        ppk = self.px_per_km()
        for r in sorted(rows, key=lambda r: r["cref_dbz"]):
            x, y = (float(v) for v in self.to_px(r["lon"], r["lat"]))
            if not (-30 < x < cw + 30 and -30 < y < ch + 30):
                continue
            rad = max(math.sqrt(r["area_km2"] / math.pi) * ppk, 4.5)
            is_sel = r["track_id"] == sel
            tags = ("cell", "id:" + r["track_id"])
            if is_sel:
                self.create_oval(x - rad - 5, y - rad - 5, x + rad + 5, y + rad + 5,
                                 outline=ACCENT, width=2)
            c = refl.color(r["cref_dbz"]) or (200, 200, 200, 255)
            self.create_oval(x - rad, y - rad, x + rad, y + rad,
                             outline=ACCENT if is_sel else "#ffffff",
                             width=2.5 if is_sel else 1.5, tags=tags)
            self.create_oval(x - 3.5, y - 3.5, x + 3.5, y + 3.5, fill="#%02x%02x%02x" % c[:3],
                             outline="#000000", tags=tags)


class Legend(tk.Canvas):
    """Colour bar (or class key) for the active colour table."""

    def __init__(self, master):
        super().__init__(master, height=34, bg=BG, highlightthickness=0)
        self.table = None
        self._photo = None
        self.bind("<Configure>", lambda e: self.draw(self.table))

    def draw(self, table):
        self.table = table
        self.delete("all")
        if table is None:
            return
        w = max(self.winfo_width(), 200)
        if table.labels:                                  # categorical (hydrometeor classes)
            n = len(table.labels)
            bw = min(48, (w - 10) / n)
            for i, (k, lab) in enumerate(sorted(table.labels.items())):
                c = table.color(k) or (0, 0, 0, 0)
                x = 4 + i * bw
                self.create_rectangle(x, 2, x + bw - 2, 15, fill="#%02x%02x%02x" % c[:3], outline="")
                self.create_text(x + bw / 2 - 1, 21, anchor="n", fill=MUTED, font=("Consolas", 8),
                                 text=lab)
            return
        x0, x1 = 70, w - 20
        lo, hi = table.value_range()
        if table.stops[-1].solid or len(table.stops) < 2:
            hi += (hi - lo) * 0.07 or 1.0
        vals = lo + (hi - lo) * (np.arange(x1 - x0) + 0.5) / (x1 - x0)
        row = np.zeros((len(vals), 4), np.uint8)
        for i, v in enumerate(vals):
            c = table.color(v)
            if c:
                row[i] = (*c[:3], 255)
        img = Image.fromarray(np.repeat(row[None], 12, axis=0), "RGBA")
        self._photo = ImageTk.PhotoImage(img)
        units = ({"DBZ": "dBZ", "DB": "dB", "MPS": "m/s", "M/S": "m/s"}.get(table.units, table.units.lower())
                 if table.units else ct.FAMILIES[table.family][1])
        self.create_text(4, 9, anchor="w", fill=MUTED, font=("Segoe UI", 9), text=units)
        self.create_image(x0, 3, image=self._photo, anchor="nw")
        span = hi - lo
        raw = span / 10.0
        mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
        step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
        v = math.ceil(lo / step) * step
        while v <= hi + 1e-9:
            x = x0 + (v - lo) / span * (x1 - x0)
            self.create_line(x, 15, x, 19, fill=MUTED)
            lab = f"{v:g}" if abs(v) >= 1 or v == 0 else f"{v:.2f}".rstrip("0")
            self.create_text(x, 21, anchor="n", fill=MUTED, font=("Consolas", 8), text=lab)
            v += step


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self, cfg: st.Config, loop_min: float, top: int):
        super().__init__()
        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("1400x900")
        self.minsize(1000, 650)
        def maximise():
            try:
                self.state("zoomed")          # Windows / macOS
            except tk.TclError:
                pass
        self.after(50, maximise)
        self.configure(bg=BG)
        if FROZEN:                          # no console in the exe: keep errors in the log
            self.report_callback_exception = lambda *exc: log.error("window error", exc_info=exc)
        self.cfg = cfg
        self.all_rows, self.rows, self.by_id, self.selected = [], [], {}, None
        self.last_result = None
        self.next_at = self.radar_next_at = None
        # radar data
        self.nex = {}                     # time -> {domain: codes}
        self.nex_latest = None
        self.nex_sites_n = 0
        self.mrms = {d: {} for d in um.DOMAINS}
        self.site_locs = {}               # site -> (lat, lon)
        self.site = None                  # selected radar site (lower case) or None = mosaic
        self.tdwr = set()                 # TDWR (airport radar) site ids
        self.product = "N0Q"              # level3 product key
        self.tilt = 0
        self.site_data = None
        self.site_overlays = {}
        self.dom_grids = {d: um.domain_grid(d) for d in um.DOMAINS}
        self._masks = {}
        # loop state
        self.fi = None                    # frame index, None = follow the latest
        self.playing = False
        self._dwell = 0
        self._anim_job = None
        self._slider_guard = False
        # colour tables
        self.tables = {k: list(v) for k, v in ct.BUILTIN.items()}
        for t in ct.load_folder(PAL_DIR):
            self.tables[t.family].append(t)
        self.table_sel = {f: 0 for f in self.tables}
        # layer switches
        self.show_nexrad = tk.BooleanVar(value=True)
        self.show_mrms = tk.BooleanVar(value=False)
        self.show_tracks = tk.BooleanVar(value=True)
        self.show_areas = tk.BooleanVar(value=True)
        self.show_counties = tk.BooleanVar(value=True)
        self.show_sites = tk.BooleanVar(value=True)
        self.us_only = tk.BooleanVar(value=True)
        self.ov_vars = {k: tk.BooleanVar(value=k != "ml") for k in l3.OVERLAYS}
        self.region = tk.StringVar(value="CONUS")
        # processes / threads
        self.q = mp.Queue()
        self.lq = queue.Queue()
        self.stop = mp.Event()
        self.mrms_wake, self.nex_wake = mp.Event(), mp.Event()
        self.mrms_on, self.nex_on = mp.Event(), mp.Event()
        self.mrms_refl_on = mp.Event()
        self._style()
        self._build()
        self._sync_workers()
        self.mrms_proc = mp.Process(target=run_mrms_worker, daemon=True,
                                    args=(cfg, loop_min, top, self.q, self.mrms_wake,
                                          self.stop, self.mrms_on, self.mrms_refl_on))
        self.nex_proc = mp.Process(target=run_nexrad_worker, daemon=True,
                                   args=(loop_min, self.q, self.nex_wake, self.stop, self.nex_on))
        self.mrms_proc.start()
        self.nex_proc.start()
        self.site_loader = SiteLoader(self.lq)
        self.site_loader.start()
        threading.Thread(target=lambda: self.lq.put(("basemap", um.load_basemap())),
                         daemon=True).start()
        threading.Thread(target=self._check_for_update, daemon=True).start()
        self.after(200, self._poll)
        self.after(1000, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind_all("<<ComboboxSelected>>", lambda e: self.after_idle(self.map.focus_set), add="+")
        self.bind("<space>", lambda e: self.toggle_play())
        self.bind("<Left>", lambda e: self.step(-1))
        self.bind("<Right>", lambda e: self.step(1))
        self.bind("<Home>", lambda e: self.go_latest())

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=PANEL,
                    bordercolor=LINE, font=("Segoe UI", 10))
        s.configure("TFrame", background=BG)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("Muted.TLabel", foreground=MUTED)
        s.configure("Title.TLabel", font=("Segoe UI Semibold", 18))
        s.configure("Panel.TLabel", background=PANEL)
        s.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED)
        s.configure("PanelBig.TLabel", background=PANEL, font=("Consolas", 22))
        s.configure("PanelHead.TLabel", background=PANEL, font=("Segoe UI Semibold", 15))
        s.configure("Toolbutton", background=BG, foreground=MUTED, padding=(10, 4))
        s.map("Toolbutton", background=[("selected", ACCENT), ("active", PANEL2)],
              foreground=[("selected", BG), ("active", FG)])
        s.configure("TCheckbutton", background=BG, foreground=MUTED)
        s.map("TCheckbutton", background=[("active", BG)], foreground=[("selected", FG)])
        s.configure("TButton", background=PANEL, foreground=FG, padding=(10, 3))
        s.map("TButton", background=[("active", PANEL2)])
        s.configure("Play.TButton", background=ACCENT, foreground=BG, padding=(12, 3),
                    font=("Segoe UI Semibold", 10))
        s.map("Play.TButton", background=[("active", "#f5c86e")])
        s.configure("TCombobox", fieldbackground=PANEL, background=PANEL2, foreground=FG,
                    arrowcolor=FG, selectbackground=PANEL, selectforeground=FG, padding=(6, 2))
        s.map("TCombobox",
              fieldbackground=[("disabled", BG), ("readonly", PANEL)],
              background=[("disabled", BG), ("readonly", PANEL2)],
              foreground=[("disabled", MUTED), ("readonly", FG)],
              selectbackground=[("readonly", PANEL)], selectforeground=[("readonly", FG)],
              arrowcolor=[("disabled", MUTED)])
        # keep keyboard focus off buttons / checkboxes so space and arrows drive the loop
        for cls in ("TButton", "TCheckbutton", "TRadiobutton", "TScale"):
            self.option_add(f"*{cls}.takeFocus", 0)
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", "#2b3f52")
        s.configure("Horizontal.TScale", background=BG, troughcolor=PANEL2)
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(12, 4))
        s.map("TNotebook.Tab", background=[("selected", PANEL2)], foreground=[("selected", FG)])
        s.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG,
                    rowheight=24, borderwidth=0)
        s.map("Treeview", background=[("selected", "#2b3f52")])
        s.configure("Treeview.Heading", background=PANEL2, foreground=MUTED,
                    font=("Segoe UI Semibold", 9), relief="flat")
        s.map("Treeview.Heading", background=[("active", LINE)])

    def _build(self):
        # header
        top = ttk.Frame(self, padding=(14, 10, 14, 4))
        top.pack(fill="x")
        ttk.Label(top, text=APP_NAME, style="Title.TLabel").pack(side="left")
        self.status_lbl = ttk.Label(top, text="Starting...", style="Muted.TLabel")
        self.status_lbl.pack(side="left", padx=18)
        self.next_lbl = ttk.Label(top, text="", style="Muted.TLabel")
        self.next_lbl.pack(side="right")
        ttk.Button(top, text="Update now", command=self._update_now).pack(side="right", padx=10)
        # shown only when GitHub has a newer release (see _check_for_update)
        self.upgrade_btn = ttk.Button(top, style="Play.TButton")
        info = ttk.Frame(self, padding=(14, 0, 14, 6))
        info.pack(fill="x")
        self.summary_lbl = ttk.Label(info, text="", style="Muted.TLabel")
        self.summary_lbl.pack(side="left")
        self.radar_lbl = ttk.Label(info, text="Radar sites: starting...", style="Muted.TLabel")
        self.radar_lbl.pack(side="right")

        panes = ttk.PanedWindow(self, orient="vertical")
        panes.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self.after(300, lambda: panes.sashpos(0, int(panes.winfo_height() * 0.70)))

        upper = ttk.Frame(panes)
        panes.add(upper, weight=3)

        # row 1: region, radar site, product, colour table
        bar = ttk.Frame(upper)
        bar.pack(fill="x", pady=(0, 4))
        self.region_btns = {}
        for dom in um.DOMAINS:
            b = ttk.Radiobutton(bar, text=um.DOMAIN_NAMES[dom], value=dom, variable=self.region,
                                style="Toolbutton", command=self._region_changed)
            b.pack(side="left", padx=(0, 4))
            self.region_btns[dom] = b
        ttk.Label(bar, text="Radar site", style="Muted.TLabel").pack(side="left", padx=(18, 4))
        self.site_var = tk.StringVar(value="All sites (mosaic)")
        self.site_cb = ttk.Combobox(bar, textvariable=self.site_var, state="readonly", width=18,
                                    values=["All sites (mosaic)"])
        self.site_cb.pack(side="left")
        self.site_cb.bind("<<ComboboxSelected>>", lambda e: self.choose_site(
            None if self.site_var.get().startswith("All") else self.site_var.get().split()[0]))
        ttk.Label(bar, text="Product", style="Muted.TLabel").pack(side="left", padx=(12, 4))
        self.prod_var = tk.StringVar()
        self.prod_cb = ttk.Combobox(bar, textvariable=self.prod_var, state="readonly", width=34,
                                    height=40)
        self.prod_cb.pack(side="left")
        self.prod_cb.bind("<<ComboboxSelected>>", lambda e: self._product_changed())
        ttk.Label(bar, text="Tilt", style="Muted.TLabel").pack(side="left", padx=(8, 4))
        self.tilt_var = tk.StringVar()
        self.tilt_cb = ttk.Combobox(bar, textvariable=self.tilt_var, state="readonly", width=7)
        self.tilt_cb.pack(side="left")
        self.tilt_cb.bind("<<ComboboxSelected>>", lambda e: self._tilt_changed())
        ttk.Label(bar, text="Color table", style="Muted.TLabel").pack(side="left", padx=(12, 4))
        self.table_var = tk.StringVar()
        self.table_cb = ttk.Combobox(bar, textvariable=self.table_var, state="readonly", width=22)
        self.table_cb.pack(side="left")
        self.table_cb.bind("<<ComboboxSelected>>", lambda e: self._table_changed())
        ttk.Button(bar, text="Load .pal...", command=self._load_pal).pack(side="left", padx=6)

        # row 2: layers
        lay = ttk.Frame(upper)
        lay.pack(fill="x", pady=(0, 6))
        ttk.Label(lay, text="Layers:", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        for text, var, cmd in [
                ("NEXRAD radar sites", self.show_nexrad, self._sources_changed),
                ("MRMS reflectivity", self.show_mrms, self._sources_changed),
                ("MRMS storm tracks", self.show_tracks, self._sources_changed),
                ("Forecast areas", self.show_areas, None),
                ("Counties", self.show_counties, None),
                ("Site markers", self.show_sites, None),
                ("US only (hide other countries)", self.us_only, None)]:
            ttk.Checkbutton(lay, text=text, variable=var,
                            command=cmd or (lambda: self.map.redraw())).pack(side="left", padx=(0, 12))

        # row 3: single-radar overlays, wind profile, value under the cursor
        ovr = ttk.Frame(upper)
        ovr.pack(fill="x", pady=(0, 6))
        ttk.Label(ovr, text="Radar-site overlays:", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        for k, (name, _) in l3.OVERLAYS.items():
            ttk.Checkbutton(ovr, text=name, variable=self.ov_vars[k],
                            command=lambda: self.map.redraw()).pack(side="left", padx=(0, 12))
        self.vwp_btn = ttk.Button(ovr, text="Wind profile (VWP)...", command=self._open_vwp)
        self.vwp_btn.pack(side="left", padx=(6, 0))
        ttk.Button(ovr, text="Reset view", command=self._reset_view).pack(side="right")
        self.readout_lbl = ttk.Label(ovr, text="", style="Muted.TLabel", font=("Consolas", 9))
        self.readout_lbl.pack(side="right", padx=(0, 12))

        body = ttk.Frame(upper)
        body.pack(fill="both", expand=True)
        self.map = MapView(body, self)
        self.map.pack(side="left", fill="both", expand=True)
        self.panel = ttk.Frame(body, style="Panel.TFrame", padding=14, width=300)
        self.panel.pack(side="right", fill="y", padx=(10, 0))
        self.panel.pack_propagate(False)
        self._build_panel()

        # loop controls + colour legend
        ctl = ttk.Frame(upper)
        ctl.pack(fill="x", pady=(6, 0))
        ttk.Button(ctl, text="|<", width=3, command=lambda: self.go_frame(0)).pack(side="left")
        ttk.Button(ctl, text="<", width=3, command=lambda: self.step(-1)).pack(side="left", padx=2)
        self.play_btn = ttk.Button(ctl, text="▶  Play", width=9, style="Play.TButton",
                                   command=self.toggle_play)
        self.play_btn.pack(side="left", padx=2)
        ttk.Button(ctl, text=">", width=3, command=lambda: self.step(1)).pack(side="left", padx=2)
        ttk.Button(ctl, text=">|", width=3, command=self.go_latest).pack(side="left")
        self.slider = ttk.Scale(ctl, from_=0, to=0, orient="horizontal", length=260,
                                command=self._slider_moved)
        self.slider.pack(side="left", padx=10)
        self.frame_lbl = ttk.Label(ctl, text="Waiting for radar...", style="Muted.TLabel", width=46)
        self.frame_lbl.pack(side="left")
        ttk.Label(ctl, text="Speed", style="Muted.TLabel").pack(side="left", padx=(8, 4))
        self.speed_var = tk.StringVar(value="Normal")
        ttk.Combobox(ctl, textvariable=self.speed_var, state="readonly", width=7,
                     values=list(SPEEDS)).pack(side="left")
        self.legend = Legend(ctl)
        self.legend.pack(side="left", fill="x", expand=True, padx=(16, 0))
        self._set_product_choices()
        self._refresh_table_choices()

        # tables
        nb = ttk.Notebook(panes)
        panes.add(nb, weight=2)
        self.ltg_tree = self._tree(nb, "Lightning threat (next hour)", [
            ("id", "Storm", 70), ("region", "Region", 80), ("now", "Office now", 85),
            ("fut", "Office in 1 hr", 95), ("p", "Lightning %", 85), ("lvl", "Threat", 85),
            ("act", "Lightning now", 95), ("ff", "First flash?", 85), ("dbz", "dBZ", 50),
            ("hail", "Hail", 70), ("move", "Moving", 100), ("conf", "Track", 70)])
        self.off_tree = self._tree(nb, "NWS offices", [
            ("wfo", "Office", 60), ("name", "Area", 200), ("n", "Storms now", 85),
            ("ltg", "With lightning", 100), ("hail", "Largest hail", 90),
            ("arr", "Arriving 55-60 min", 120), ("ff", "First-flash arriving", 130)])
        self.hail_tree = self._tree(nb, "Severe hail", [
            ("id", "Storm", 70), ("wfo", "Office", 60), ("start", "45 min ago", 90),
            ("end", "Now", 80), ("peak", "Peak", 80), ("trend", "Trend", 110),
            ("rate", "Change / hr", 90), ("cls", "Size", 160)])
        logf = ttk.Frame(nb)
        nb.add(logf, text="Program output")
        self.log_txt = tk.Text(logf, bg="#0a1017", fg="#c9d6e2", font=("Consolas", 9),
                               relief="flat", wrap="none", padx=10, pady=8)
        sb = ttk.Scrollbar(logf, command=self.log_txt.yview)
        self.log_txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_txt.pack(fill="both", expand=True)
        self.ltg_tree.bind("<<TreeviewSelect>>", self._tree_select)

    def _tree(self, nb, title, cols):
        f = ttk.Frame(nb)
        nb.add(f, text=title)
        tree = ttk.Treeview(f, columns=[c[0] for c in cols], show="headings", selectmode="browse")
        for key, head, w in cols:
            tree.heading(key, text=head, command=lambda k=key, t=tree: self._sort(t, k))
            tree.column(key, width=w, anchor="w", stretch=key == cols[-1][0])
        for lvl, c in LEVEL_COLORS.items():
            tree.tag_configure(lvl, foreground=c)
        sb = ttk.Scrollbar(f, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)
        return tree

    def _sort(self, tree, key):
        items = [(tree.set(i, key), i) for i in tree.get_children("")]

        def num(v):
            try:
                return (0, float(v.split()[0].rstrip("%")))
            except (ValueError, IndexError):
                return (1, v)
        desc = getattr(tree, "_sort_desc", {}).get(key, True)
        items.sort(key=lambda t: num(t[0]), reverse=desc)
        for n, (_, i) in enumerate(items):
            tree.move(i, "", n)
        tree._sort_desc = {key: not desc}

    def _build_panel(self):
        p = self.panel
        head = ttk.Frame(p, style="Panel.TFrame")
        head.pack(fill="x")
        self.p_id = ttk.Label(head, text="No storm selected", style="PanelHead.TLabel")
        self.p_id.pack(side="left")
        self.p_lvl = tk.Label(head, text="", bg=PANEL, fg=BG, font=("Segoe UI Semibold", 9), padx=6)
        self.p_lvl.pack(side="right")
        ttk.Label(p, text="LIGHTNING CHANCE IN 55-60 MIN", style="PanelMuted.TLabel",
                  font=("Segoe UI", 8)).pack(anchor="w", pady=(12, 0))
        self.p_prob = ttk.Label(p, text="--", style="PanelBig.TLabel")
        self.p_prob.pack(anchor="w")
        self.p_bar = tk.Canvas(p, height=6, bg=PANEL2, highlightthickness=0)
        self.p_bar.pack(fill="x", pady=(2, 12))
        grid = ttk.Frame(p, style="Panel.TFrame")
        grid.pack(fill="x")
        self.p_fields = {}
        for i, name in enumerate(["Office now", "Office in 1 hr", "Lightning now",
                                  "First flash expected", "Moving", "Strongest return",
                                  "At -10 C level", "Hail estimate", "Hail trend (45 min)",
                                  "Size", "Tracked for", "Position error (1 hr)", "Location"]):
            ttk.Label(grid, text=name, style="PanelMuted.TLabel",
                      font=("Segoe UI", 9)).grid(row=i, column=0, sticky="w", pady=1)
            v = ttk.Label(grid, text="", style="Panel.TLabel", font=("Consolas", 9))
            v.grid(row=i, column=1, sticky="e", pady=1)
            self.p_fields[name] = v
        grid.columnconfigure(1, weight=1)

    # ---------------- layers / frames ----------------
    def table(self, kind):
        return self.tables[kind][self.table_sel[kind]]

    def mask_for(self, g: um.RasterGrid):
        if not self.us_only.get() or self.map.base is None:
            return None
        m = self._masks.get(g)
        if m is None:
            if len(self._masks) > 6:
                self._masks = {k: v for k, v in self._masks.items() if k in self.dom_grids.values()}
            m = self._masks[g] = um.us_mask(g, self.map.base.mask_rings[g.domain])
        return m

    @property
    def family(self):
        """Colour-table family of the radar layer being shown."""
        return l3.BY_KEY[self.product].family if self.site else "BR"

    def _site_frames(self):
        """{time: (time, codes, values, rf, elev, grid)} for the selected site/product/tilt."""
        sd = self.site_data
        if (sd and self.site and sd["req"][:3] == (self.site, self.product, self.tilt)
                and sd["req"][3] == self.map.domain):
            return {f[0]: f for f in sd["frames"]}
        return {}

    def _primary(self):
        """{time: frame} of the layer that drives the loop."""
        dom = self.map.domain
        if self.show_nexrad.get():
            if self.site:
                return self._site_frames(), None
            return {t: v[dom] for t, v in self.nex.items() if dom in v}, self.dom_grids[dom]
        if self.show_mrms.get():
            return self.mrms[dom], self.dom_grids[dom]
        return {}, None

    def _current_site_frame(self):
        fr = self._site_frames()
        if not fr:
            return None
        return fr.get(self.current_time()) or fr[max(fr)]

    def site_range_km(self):
        f = self._current_site_frame()
        if f is None:
            return None
        g = f[5]
        return g.nx * g.res / 2 / 1.04 / 1000.0

    def frame_times(self):
        return sorted(self._primary()[0])

    def current_time(self):
        ts = self.frame_times()
        if not ts:
            return None
        return ts[-1] if self.fi is None else ts[min(self.fi, len(ts) - 1)]

    def at_latest(self):
        ts = self.frame_times()
        return self.fi is None or not ts or self.fi >= len(ts) - 1

    def layer_title(self):
        if self.show_nexrad.get():
            if self.site:
                p = l3.BY_KEY[self.product]
                f = self._current_site_frame()
                el = f"  {f[4]:.1f}\u00b0" if f and f[4] is not None else ""
                return f"{self.site.upper()} {p.key} {p.name}{el}"
            return "NEXRAD mosaic" + (" + MRMS" if self.show_mrms.get() else "")
        return "MRMS composite reflectivity" if self.show_mrms.get() else "No radar layer"

    def layers(self):
        """[(codes, grid, lut, reduce, (values, rf, family, name))] to draw for the
        current loop time, bottom first.  reduce = "max" or "sub" (see _draw_raster)."""
        T = self.current_time()
        dom = self.map.domain
        out = []

        def pick(d):
            if not d:
                return None
            if T is None:
                return d[max(d)]
            t = min(d, key=lambda t: abs((t - T).total_seconds()))
            return d[t] if abs((t - T).total_seconds()) <= MATCH_MIN * 60 else None

        refl = self.table("BR").lut()
        if self.show_mrms.get():
            c = pick(self.mrms[dom])
            if c is not None:
                out.append((c, self.dom_grids[dom], refl, "max",
                            (ct.REFL_VALUES, None, "BR", "MRMS")))
        if self.show_nexrad.get():
            if self.site:
                f = pick(self._site_frames())
                if f is not None:
                    fam = self.family
                    _, codes, values, rf, _, grid = f
                    out.append((codes, grid, self.table(fam).lut_for(values, rf),
                                "max" if fam in ("BR", "ET", "VIL", "PRECIP") else "sub",
                                (values, rf, fam, f"{self.site.upper()} {self.product}")))
            else:
                c = pick({t: v[dom] for t, v in self.nex.items() if dom in v})
                if c is not None:
                    out.append((c, self.dom_grids[dom], refl, "max",
                                (ct.REFL_VALUES, None, "BR", "Mosaic")))
        return out

    def show_readout(self, px, py):
        """Lat/lon and the data value under the mouse pointer."""
        m = self.map
        x, y = (px - m.ox) / m.scale, (m.oy - py) / m.scale
        lon, lat = (float(v) for v in um.unproject(m.domain, x, y))
        txt = f"{abs(lat):.2f}\u00b0{'N' if lat >= 0 else 'S'} {abs(lon):.2f}\u00b0{'W' if lon < 0 else 'E'}"
        for codes, g, _, _, (values, rf, fam, name) in reversed(self.layers()):
            r, c = int((g.y1 - y) / g.res), int((x - g.x0) / g.res)
            if not (0 <= r < g.ny and 0 <= c < g.nx):
                continue
            mask = self.mask_for(g)
            if mask is not None and not mask[r, c]:
                continue
            code = int(codes[r, c])
            if rf is not None and code == rf:
                txt += f"   {name}: range folded"
                break
            v = values[code]
            if np.isfinite(v):
                txt += f"   {name}: {fmt_value(float(v), fam)}"
                break
        self.readout_lbl.config(text=txt)

    # ---------------- loop controls ----------------
    def _sync_loop(self):
        ts = self.frame_times()
        n = len(ts)
        if self.fi is not None and self.fi >= n:
            self.fi = None
        i = (n - 1) if self.fi is None else self.fi
        self._slider_guard = True
        self.slider.configure(to=max(n - 1, 0))
        self.slider.set(max(i, 0))
        self._slider_guard = False
        if not n:
            self.frame_lbl.config(text="No radar frames yet" if (self.show_nexrad.get() or
                                                                 self.show_mrms.get())
                                  else "Radar layers are off")
            return
        tag = "LATEST" if i == n - 1 else f"{i + 1} of {n}"
        self.frame_lbl.config(text=f"{fmt_time(ts[i])}   [{tag}]",
                              foreground=FG if i == n - 1 else ACCENT)

    def _show_frame(self, full=False):
        self._sync_loop()
        if full:
            self.map.redraw()
        else:
            self.map.redraw_radar()

    def go_frame(self, i):
        ts = self.frame_times()
        if not ts:
            return
        i = max(0, min(i, len(ts) - 1))
        self.fi = None if (i == len(ts) - 1 and not self.playing) else i
        self._show_frame(full=True)

    def go_latest(self):
        self.fi = None
        self._show_frame(full=True)

    def step(self, d):
        if self.playing:
            self.toggle_play()
        ts = self.frame_times()
        if not ts:
            return
        i = (len(ts) - 1) if self.fi is None else self.fi
        self.go_frame((i + d) % len(ts))

    def _slider_moved(self, val):
        if self._slider_guard:
            return
        i = int(round(float(val)))
        ts = self.frame_times()
        cur = (len(ts) - 1) if self.fi is None else self.fi
        if ts and i != cur:
            self.go_frame(i)

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_btn.config(text="❚❚  Pause")
            ts = self.frame_times()
            if self.fi is None and ts:
                self.fi = 0                          # start the loop from the oldest frame
            self._dwell = 0
            self._show_frame(full=True)
            self._anim_job = self.after(SPEEDS[self.speed_var.get()], self._animate)
        else:
            self.play_btn.config(text="▶  Play")
            if self._anim_job:
                self.after_cancel(self._anim_job)
                self._anim_job = None
            if self.at_latest():
                self.fi = None
            self._show_frame(full=True)

    def _animate(self):
        self._anim_job = None
        if not self.playing:
            return
        ts = self.frame_times()
        delay = SPEEDS[self.speed_var.get()]
        if len(ts) < 2:
            self.fi = None
            self._show_frame(full=True)
        else:
            i = 0 if self.fi is None else self.fi
            was_latest = i >= len(ts) - 1
            if was_latest and self._dwell < 3:      # pause on the newest frame
                self._dwell += 1
            else:
                self._dwell = 0
                self.fi = 0 if was_latest else i + 1
                self._show_frame(full=was_latest or self.fi >= len(ts) - 1)
        self._anim_job = self.after(delay, self._animate)

    # ---------------- user choices ----------------
    def _reset_view(self):
        if self.site and self.site in self.site_locs:
            lat, lon = self.site_locs[self.site]
            self.map.zoom_to(lon, lat, min(self.site_range_km() or 300, 330))
        else:
            self.map.fit()

    def _sync_workers(self):
        if self.show_nexrad.get() and not self.site:
            self.nex_on.set()
        else:
            self.nex_on.clear()
        if self.show_mrms.get() or self.show_tracks.get():
            self.mrms_on.set()
        else:
            self.mrms_on.clear()
        if self.show_mrms.get():
            if not self.mrms_refl_on.is_set():
                self.mrms_refl_on.set()
                if not any(self.mrms.values()):
                    self.mrms_wake.set()             # fetch the MRMS layer now, not in 2 min
        else:
            self.mrms_refl_on.clear()
            self.mrms = {d: {} for d in um.DOMAINS}  # free the frames while hidden

    def _sources_changed(self):
        self._sync_workers()
        if self.show_nexrad.get() and self.site:
            self._request_site()
        self.fi = None if not self.playing else self.fi
        self._sync_loop()
        self.map.redraw()
        self._show_rows()

    def _site_label(self, site):
        return f"{site.upper()} (TDWR)" if site in self.tdwr else site.upper()

    def _update_site_list(self):
        nex = sorted(s for s in self.site_locs if s not in self.tdwr)
        self.site_cb.config(values=["All sites (mosaic)"] + [s.upper() for s in nex] +
                            [self._site_label(s) for s in sorted(self.tdwr)])

    def choose_site(self, site):
        site = site.lower() if site else None
        if site and site not in self.site_locs:
            return
        self.site = site
        self.site_overlays = {}
        self.site_data = None
        if site:
            tdwr = site in self.tdwr
            if l3.BY_KEY[self.product].tdwr != tdwr:      # TDWR sites have their own products
                self.product, self.tilt = ("TZ0" if tdwr else "N0Q"), 0
            self.site_var.set(self._site_label(site))
            if not self.show_nexrad.get():
                self.show_nexrad.set(True)
            lat, lon = self.site_locs[site]
            dom = nr.site_domain(lat, lon)
            if dom != self.map.domain:
                self.region.set(dom)
                self.map.domain = dom
            self.fi = None
            self.map.zoom_to(lon, lat, 120 if tdwr else 330)
            self._request_site()
        else:
            self.site_loader.request(None)
            self.product, self.tilt = "N0Q", 0
            self.site_var.set("All sites (mosaic)")
            self.fi = None
            self.radar_lbl.config(text=self._mosaic_status(), foreground=MUTED)
            self.map.fit()
        self._sync_workers()
        self._set_product_choices()
        self._refresh_table_choices()
        self._sync_loop()

    def _request_site(self):
        lat, lon = self.site_locs[self.site]
        self.site_loader.request((self.site, self.product, self.tilt,
                                  nr.site_domain(lat, lon), lat, lon))

    def _set_product_choices(self):
        tdwr = bool(self.site and self.site in self.tdwr)
        prods = [p for p in l3.PRODUCTS if p.tdwr == tdwr]
        self.prod_cb.config(values=[f"{p.key}  {p.name}" for p in prods])
        p = l3.BY_KEY[self.product]
        self.prod_var.set(f"{p.key}  {p.name}")
        labels = [t[0] for t in p.tilts]
        self.tilt = min(self.tilt, len(labels) - 1)
        self.tilt_cb.config(values=labels, state="readonly" if len(labels) > 1 else "disabled")
        self.tilt_var.set(labels[self.tilt])
        self.vwp_btn.config(state="normal" if self.site and not tdwr else "disabled")

    def _nearest_site(self):
        """NEXRAD site nearest the middle of the map."""
        m = self.map
        cx = (m.winfo_width() / 2 - m.ox) / m.scale
        cy = (m.oy - m.winfo_height() / 2) / m.scale
        lon, lat = (float(v) for v in um.unproject(m.domain, cx, cy))
        k = math.cos(math.radians(lat))
        cands = [(((la - lat) ** 2 + ((lo - lon) * k) ** 2), s) for s, (la, lo) in self.site_locs.items()
                 if s not in self.tdwr and nr.site_domain(la, lo) == m.domain]
        return min(cands)[1] if cands else None

    def _product_changed(self):
        key = self.prod_var.get().split()[0]
        if key not in l3.BY_KEY:
            return
        self.product = key
        self.fi = None
        self.site_data = None
        if self.site:
            self._request_site()
        elif key != "N0Q":                       # the mosaic is base reflectivity only
            site = self._nearest_site()
            if site:
                self.choose_site(site)
                return
            self.product = "N0Q"
            self.radar_lbl.config(text="Radar site list is still loading - try again in a moment",
                                  foreground=ACCENT)
        self._set_product_choices()
        self._refresh_table_choices()
        self._sync_loop()
        self.map.redraw()

    def _tilt_changed(self):
        labels = [t[0] for t in l3.BY_KEY[self.product].tilts]
        if self.tilt_var.get() in labels:
            self.tilt = labels.index(self.tilt_var.get())
        self.fi = None
        self.site_data = None
        if self.site:
            self._request_site()
        self._sync_loop()
        self.map.redraw()

    def _open_vwp(self):
        if self.site and self.site not in self.tdwr:
            VwpWindow(self, self.site)

    def _refresh_table_choices(self):
        fam = self.family
        names = [t.name for t in self.tables[fam]]
        self.table_cb.config(values=names)
        self.table_var.set(names[self.table_sel[fam]])
        self.legend.draw(self.table(fam))

    def _table_changed(self):
        fam = self.family
        names = [t.name for t in self.tables[fam]]
        if self.table_var.get() in names:
            self.table_sel[fam] = names.index(self.table_var.get())
        self.legend.draw(self.table(fam))
        self.map.redraw()

    def _load_pal(self):
        path = filedialog.askopenfilename(
            parent=self, title="Load colour table",
            filetypes=[("Colour tables", "*.pal"), ("All files", "*.*")])
        if not path:
            return
        try:
            t = ct.load_pal(path)
        except Exception as e:
            self.radar_lbl.config(text=f"Could not read colour table: {e}",
                                  foreground=LEVEL_COLORS["VERY HIGH"])
            return
        try:                                     # keep a copy so it loads next time too
            PAL_DIR.mkdir(exist_ok=True)
            dst = PAL_DIR / Path(path).name
            if Path(path).resolve() != dst.resolve():
                shutil.copyfile(path, dst)
        except OSError:
            pass
        lst = self.tables[t.family]
        lst[:] = [x for x in lst if x.name != t.name] + [t]
        self.table_sel[t.family] = len(lst) - 1
        if t.family != self.family:
            self.radar_lbl.config(text=f"Loaded {ct.FAMILIES[t.family][0].lower()} table "
                                       f"'{t.name}' - it is used for those products",
                                  foreground=ACCENT)
        else:
            self.radar_lbl.config(text=f"Loaded colour table '{t.name}'", foreground=MUTED)
        self._refresh_table_choices()
        self.map.redraw()

    # ---------------- data flow ----------------
    def _mosaic_status(self):
        if not self.nex:
            return "Radar sites: loading..."
        return (f"Radar sites: {self.nex_sites_n} US radars, newest "
                f"{self.nex_latest:%H:%MZ}" if self.nex_latest else "Radar sites: no recent scans") + \
            f", {len(self.nex)} loop frames"

    def _poll(self):
        dirty = loop_dirty = False
        t_end = time.time() + 0.15
        try:
            while time.time() < t_end:
                try:
                    kind, val = self.lq.get_nowait()
                except queue.Empty:
                    kind, val = self.q.get_nowait()
                if kind == "status":
                    self.status_lbl.config(text=val, foreground=MUTED)
                    self.next_at = None
                elif kind == "error":
                    self.status_lbl.config(text=val, foreground=LEVEL_COLORS["VERY HIGH"])
                elif kind == "mrms_off":
                    self.status_lbl.config(text="MRMS is off (turn on MRMS reflectivity or "
                                                "storm tracks to resume)", foreground=MUTED)
                    self.next_at = None
                elif kind == "next":
                    self.next_at = val
                elif kind == "result":
                    self._show(val)
                elif kind == "mrms_frame":
                    dom, t, codes = val
                    self.mrms[dom][t] = codes
                    loop_dirty = True
                    dirty |= dom == self.map.domain and self.show_mrms.get()
                elif kind == "mrms_keep":
                    dom, times = val
                    keep = set(times)
                    self.mrms[dom] = {t: c for t, c in self.mrms[dom].items() if t in keep}
                    loop_dirty = True
                elif kind == "nexrad_frame":
                    t, by_dom, n_sites, is_latest = val
                    if is_latest:
                        if self.nex_latest is not None and self.nex_latest != t:
                            self.nex.pop(self.nex_latest, None)
                        self.nex_latest, self.nex_sites_n = t, n_sites
                    self.nex[t] = by_dom
                    loop_dirty = True
                    dirty |= self.show_nexrad.get() and not self.site
                    if not self.site:
                        self.radar_lbl.config(text=self._mosaic_status(), foreground=MUTED)
                elif kind == "nexrad_keep":
                    keep = set(val["marks"]) | {self.nex_latest}
                    self.nex = {t: v for t, v in self.nex.items() if t in keep}
                    loop_dirty = True
                    if not self.site:
                        self.radar_lbl.config(text=self._mosaic_status(), foreground=MUTED)
                elif kind == "sites":
                    if val:
                        self.site_locs.update(val)
                        self._update_site_list()
                        dirty |= self.show_sites.get()
                elif kind == "tdwr_sites":
                    self.tdwr |= set(val)
                    self.site_locs.update(val)
                    self._update_site_list()
                    dirty |= self.show_sites.get()
                elif kind == "radar_status":
                    if not self.site:
                        self.radar_lbl.config(text=val, foreground=MUTED)
                elif kind == "radar_error":
                    self.radar_lbl.config(text=f"Radar sites failed ({val}); will retry",
                                          foreground=LEVEL_COLORS["VERY HIGH"])
                elif kind == "radar_next":
                    self.radar_next_at = val
                elif kind == "nexrad_off":
                    self.radar_next_at = None
                elif kind == "site_status":
                    self.radar_lbl.config(text=val, foreground=MUTED)
                elif kind == "site_error":
                    self.radar_lbl.config(text=f"Radar site {val}; will retry",
                                          foreground=LEVEL_COLORS["VERY HIGH"])
                elif kind == "site_frames":
                    req = val["req"]
                    if req[:3] == (self.site, self.product, self.tilt):
                        self.site_data = val
                        fr = val["frames"]
                        p = l3.BY_KEY[req[1]]
                        name = f"{req[0].upper()} {p.key} {p.name}"
                        if fr:
                            msg, col = (f"{name}: {len(fr)} scans, newest {fr[-1][0]:%H:%MZ}", MUTED)
                        elif val.get("stale"):
                            msg, col = (f"{name}: this radar has not produced it since "
                                        f"{val['stale']:%b %d, %Y}", ACCENT)
                        elif val["listed"]:
                            msg, col = (f"{name}: nothing to show right now "
                                        "(e.g. no precipitation)", ACCENT)
                        else:
                            msg, col = (f"{name} is not available from this radar right now",
                                        LEVEL_COLORS["VERY HIGH"])
                        self.radar_lbl.config(text=msg, foreground=col)
                        self._refresh_table_choices()
                        dirty = loop_dirty = True
                elif kind == "site_overlays":
                    if val["site"] == self.site:
                        self.site_overlays = val["data"]
                        dirty = True
                elif kind == "basemap":
                    self.map.set_basemap(val)
                    self._show_rows()
                elif kind == "new_version":
                    tag, url = val
                    self.upgrade_btn.config(text=f"New version {tag} available - download",
                                            command=lambda: webbrowser.open(url))
                    self.upgrade_btn.pack(side="right", padx=(0, 6))
        except queue.Empty:
            pass
        if loop_dirty:
            self._sync_loop()
        if dirty and not self.playing:
            self.map.redraw()
        self.after(200, self._poll)

    def _tick(self):
        parts = []
        if self.mrms_on.is_set():
            parts.append(f"MRMS in {max(0, int(self.next_at - time.time()))} s" if self.next_at
                         else "MRMS updating...")
        if self.site and self.show_nexrad.get():
            parts.append(f"{self.site.upper()} refreshes every {SITE_REFRESH_S // 60} min")
        elif self.nex_on.is_set():
            parts.append(f"radar in {max(0, int(self.radar_next_at - time.time()))} s"
                         if self.radar_next_at else "radar updating...")
        self.next_lbl.config(text="Next: " + ",  ".join(parts) if parts else "All data paused")
        self.after(1000, self._tick)

    def _check_for_update(self):
        """Background: ask GitHub for the newest release; offer it if newer than us."""
        try:
            r = requests.get(f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
                             timeout=15, headers={"Accept": "application/vnd.github+json"})
            if r.status_code != 200:
                return
            rel = r.json()
            tag = rel.get("tag_name", "")
            if _version_tuple(tag) > _version_tuple(APP_VERSION):
                self.lq.put(("new_version", (tag, rel.get("html_url") or
                                             f"https://github.com/{GITHUB_REPO}/releases/latest")))
        except Exception:                        # offline etc.: just skip the check
            log.debug("update check failed", exc_info=True)

    def _update_now(self):
        self.mrms_wake.set()
        self.nex_wake.set()
        if self.site:
            self._request_site()

    def _in_us(self, dom, lon, lat):
        m = self.mask_for(self.dom_grids[dom]) if self.map.base is not None else None
        if m is None:
            return True
        g = self.dom_grids[dom]
        x, y = um.project(dom, lon, lat)
        c, r = int((float(x) - g.x0) / g.res), int((g.y1 - float(y)) / g.res)
        return 0 <= r < g.ny and 0 <= c < g.nx and bool(m[r, c])

    def _show(self, res):
        self.last_result = res
        ok = sum(1 for s in res["status"].values() if s.startswith("ok"))
        self.status_lbl.config(
            text=f"MRMS newest scan {res['newest']:%Y-%m-%d %H:%MZ}   |   {ok} of {len(res['status'])}"
                 f" regions ok   |   update took {res['secs']:.0f} s   |   saved to {res['out_dir']}",
            foreground=MUTED)
        self.log_txt.delete("1.0", "end")
        self.log_txt.insert("1.0", res["report"])
        self._show_rows()

    def _show_rows(self):
        """Apply the US-only filter to the latest storm results and refresh the views."""
        res = self.last_result
        if res is None:
            self.map.redraw()
            return
        old = self.by_id.get(self.selected)
        rows = res["rows"]
        if self.us_only.get():
            rows = [r for r in rows if self._in_us(r["domain"], r["lon"], r["lat"]) or self._in_us(
                r["domain"], r["lightning"]["forecast_lon_60"], r["lightning"]["forecast_lat_60"])]
        self.rows = rows
        self.by_id = {r["track_id"]: r for r in self.rows}
        # track ids are rebuilt each update; keep the selection on the nearest storm
        if old and self.selected not in self.by_id:
            near = [r for r in self.rows if r["domain"] == old["domain"]]
            near.sort(key=lambda r: (r["lat"] - old["lat"]) ** 2 + (r["lon"] - old["lon"]) ** 2)
            self.selected = near[0]["track_id"] if near and abs(near[0]["lat"] - old["lat"]) < 0.3 else None
        if self.selected not in self.by_id:
            self._select_top()
        n_ltg = sum(r["lightning"]["lightning_active_now"] for r in self.rows)
        n_ff = sum(1 for r in self.rows if not r["lightning"]["lightning_active_now"]
                   and r["lightning"]["p_lightning_55_60"] >= 0.5)
        n_sev = sum(r["mesh_mm"] >= st.SEVERE_MESH_MM for r in self.rows)
        big = max((r["mesh_mm"] for r in self.rows), default=0.0)
        self.summary_lbl.config(
            text=f"{len(self.rows)} storms tracked    {n_ltg} with lightning now    "
                 f"{n_ff} first-flash candidates    {n_sev} with severe hail    "
                 f"largest hail {inches(big) if big >= 0.5 else 'none'}")
        for dom, b in self.region_btns.items():
            n = sum(r["domain"] == dom for r in self.rows)
            b.config(text=f"{um.DOMAIN_NAMES[dom]}  {n}")
        self._fill_tables(res)
        self.map.redraw()
        self._update_panel()

    def _fill_tables(self, res):
        t = self.ltg_tree
        t.delete(*t.get_children())
        for r in sorted(self.rows, key=lambda r: -r["lightning"]["p_lightning_55_60"]):
            L = r["lightning"]
            ff = not L["lightning_active_now"] and L["p_lightning_55_60"] >= 0.5
            t.insert("", "end", iid=r["track_id"], tags=(L["threat_level"],), values=(
                r["track_id"], um.DOMAIN_NAMES.get(r["domain"], r["domain"]), r["wfo"] or "water",
                L["wfo_at_55_60"] or "water", f"{L['p_lightning_55_60'] * 100:.0f}%",
                L["threat_level"].title(), "yes" if L["lightning_active_now"] else "no",
                "yes" if ff else "-", f"{r['cref_dbz']:.0f}", inches(r["mesh_mm"]),
                f"{compass(r['heading_deg'])} {r['speed_ms'] * 2.23694:.0f} mph",
                L["track_confidence"]))
        t = self.off_tree
        t.delete(*t.get_children())
        act = [s for s in res["summary"] if s["n_cells"] or s["n_cells_arriving_55_60"]]
        for s in sorted(act, key=lambda s: -(s["n_severe_mesh_cells"] * 100 + s["n_cells"])):
            t.insert("", "end", values=(
                s["wfo"], s["name"], s["n_cells"], s["n_active_lightning_cells"],
                inches(s["max_mesh_mm"]), s["n_cells_arriving_55_60"],
                s["n_first_flash_candidates_arriving"]))
        t = self.hail_tree
        t.delete(*t.get_children())
        sev = [r for r in self.rows if r["current_window"] and r["current_window"]["severe_in_window"]]
        for r in sorted(sev, key=lambda r: -r["current_window"]["mesh_peak_mm"]):
            w = r["current_window"]
            t.insert("", "end", values=(
                r["track_id"], r["wfo"] or "water", inches(w["mesh_start_mm"]),
                inches(w["mesh_end_mm"]), inches(w["mesh_peak_mm"]), w["trend"],
                f"{w['trend_mm_per_h'] / 25.4:+.2f} in", w["peak_hail_class"]))
        if not sev:
            t.insert("", "end", values=("", "", "No storms with hail of 1 inch or larger right now"))

    # ---------------- selection ----------------
    def _select_top(self):
        rows = [r for r in self.rows if r["domain"] == self.region.get()]
        top = max(rows, key=lambda r: r["lightning"]["p_lightning_55_60"], default=None)
        self.selected = top["track_id"] if top else None

    def _region_changed(self):
        dom = self.region.get()
        if self.site and nr.site_domain(*self.site_locs[self.site]) != dom:
            self.choose_site(None)
        if not self.playing:
            self.fi = None                       # each region opens on its latest frame
        sel = self.by_id.get(self.selected)
        if not sel or sel["domain"] != dom:
            self._select_top()
        self.map.set_domain(dom)
        self._sync_loop()
        self._update_panel()

    def select(self, tid, center=False):
        r = self.by_id.get(tid)
        if not r:
            return
        self.selected = tid
        if r["domain"] != self.region.get():
            self.region.set(r["domain"])
            self._region_changed()
            self.selected = tid
        if center:
            self.map.center_on(r["lon"], r["lat"])
        else:
            self.map.redraw()
        self._update_panel()

    def _tree_select(self, _):
        sel = self.ltg_tree.selection()
        if sel and sel[0] != self.selected:
            self.select(sel[0], center=True)

    def _update_panel(self):
        r = self.by_id.get(self.selected)
        f = self.p_fields
        if not r:
            self.p_id.config(text="No storm selected")
            self.p_lvl.config(text="", bg=PANEL)
            self.p_prob.config(text="--")
            self.p_bar.delete("all")
            for v in f.values():
                v.config(text="")
            return
        L, w = r["lightning"], r["current_window"]
        col = LEVEL_COLORS[L["threat_level"]]
        p = L["p_lightning_55_60"]
        self.p_id.config(text=r["track_id"])
        self.p_lvl.config(text=L["threat_level"], bg=col)
        self.p_prob.config(text=f"{p * 100:.0f}%")
        self.p_bar.delete("all")
        self.p_bar.update_idletasks()
        self.p_bar.create_rectangle(0, 0, self.p_bar.winfo_width() * p, 6, fill=col, width=0)
        ff = not L["lightning_active_now"] and p >= 0.5
        vals = {
            "Office now": r["wfo"] or "over water",
            "Office in 1 hr": L["wfo_at_55_60"] or "over water",
            "Lightning now": "yes" if L["lightning_active_now"] else "no",
            "First flash expected": "yes" if ff else "no",
            "Moving": f"{compass(r['heading_deg'])} {r['speed_ms'] * 2.23694:.0f} mph",
            "Strongest return": f"{r['cref_dbz']} dBZ",
            "At -10 C level": "n/a" if r["z_minus10C_dbz"] is None else f"{r['z_minus10C_dbz']} dBZ",
            "Hail estimate": f"{inches(r['mesh_mm'])} ({r['mesh_mm']} mm)",
            "Hail trend (45 min)": w["trend"] if w else "n/a",
            "Size": f"{r['area_km2']:.0f} km2",
            "Tracked for": f"{r['age_min']} min, {L['track_confidence']} conf.",
            "Position error (1 hr)": f"+/- {L['pos_sigma_km']} km",
            "Location": f"{r['lat']:.2f}, {r['lon']:.2f}",
        }
        for k, v in vals.items():
            f[k].config(text=v, foreground=ACCENT if (k == "First flash expected" and ff) else FG)
        if self.ltg_tree.exists(r["track_id"]) and self.ltg_tree.selection() != (r["track_id"],):
            self.ltg_tree.selection_set(r["track_id"])
            self.ltg_tree.see(r["track_id"])

    def _close(self):
        self.stop.set()
        self.mrms_wake.set()
        self.nex_wake.set()
        self.mrms_on.set()
        self.nex_on.set()
        self.destroy()
        for p in (self.mrms_proc, self.nex_proc):
            p.join(timeout=3)
            if p.is_alive():                     # mid-download: don't wait for it
                p.terminate()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--office", help="3-letter WFO id (cells over it, or forecast into it)")
    ap.add_argument("--domains", default="all", help="comma list of CONUS,ALASKA,HAWAII")
    ap.add_argument("--loop", type=float, default=2.0, metavar="MIN",
                    help="refresh every MIN minutes (default 2)")
    ap.add_argument("--out", default="storm_out")
    ap.add_argument("--cache", default="mrms_cache")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)
    _setup_logging()
    doms = um.DOMAINS if a.domains == "all" else tuple(
        d for d in (x.strip().upper() for x in a.domains.split(",")) if d in um.DOMAINS)
    cfg = st.Config(cache_dir=DATA_DIR / a.cache, out_dir=DATA_DIR / a.out, office=a.office,
                    domains=doms or um.DOMAINS)
    App(cfg, a.loop, a.top).mainloop()
    return 0


if __name__ == "__main__":
    mp.freeze_support()          # required for the worker processes in the exe
    raise SystemExit(main())
