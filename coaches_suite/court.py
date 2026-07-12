"""
court.py — matplotlib 14-wedge "hot-zone" shot-chart, matching the geometry of
site/js/zone-chart.js so the Coaches Suite reads the same as the website.

The court is drawn hoop-at-origin, x = lateral (right +), y = distance toward
halfcourt (feet).  Each of the 14 zones is a radial wedge filled by a value:

    scheme="sequential"  magnitude  (e.g. share of shots) — one hue, light->dark
    scheme="diverging"   polarity   (e.g. FG% vs baseline) — blue(-) / red(+)

Zones with too few attempts are drawn hollow (hatched grey) instead of colored.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Arc, Circle
from matplotlib.colors import TwoSlopeNorm, Normalize
import matplotlib.cm as cm

# Ring radii (feet) — RA / close / mid-to-arc / three.  Match zone-chart.js.
RA_R, CLOSE_R, ARC_R, THREE_OUT = 4.0, 11.0, 22.146, 27.5
CORNER_X, Y_MEET = 21.65, np.sqrt(22.146**2 - 21.65**2)   # ~4.658

# (inner_r, outer_r, theta1, theta2) per zone name — angles 0=right .. 180=left.
ZONE_WEDGES = {
    "Restricted Area":    (0.0,    RA_R,    0,   180),
    "Close Mid - Right":  (RA_R,   CLOSE_R, 0,   60),
    "Close Mid - Center": (RA_R,   CLOSE_R, 60,  120),
    "Close Mid - Left":   (RA_R,   CLOSE_R, 120, 180),
    "Mid - Right":        (CLOSE_R, ARC_R,  0,   36),
    "Mid - Right Center": (CLOSE_R, ARC_R,  36,  72),
    "Mid - Center":       (CLOSE_R, ARC_R,  72,  108),
    "Mid - Left Center":  (CLOSE_R, ARC_R,  108, 144),
    "Mid - Left":         (CLOSE_R, ARC_R,  144, 180),
    "3PT - Right":        (ARC_R,  THREE_OUT, 0,   36),
    "3PT - Right Center": (ARC_R,  THREE_OUT, 36,  72),
    "3PT - Center":       (ARC_R,  THREE_OUT, 72,  108),
    "3PT - Left Center":  (ARC_R,  THREE_OUT, 108, 144),
    "3PT - Left":         (ARC_R,  THREE_OUT, 144, 180),
}

# Coarse 3-band version: paint/restricted (inside 11ft), mid-range, three.
MACRO_WEDGES = {
    "Paint / Restricted": (0.0,     CLOSE_R,   0, 180),
    "Mid-Range":          (CLOSE_R, ARC_R,     0, 180),
    "Three":              (ARC_R,   THREE_OUT, 0, 180),
}

MUTED = "#6b7280"


def _label_xy(inner, outer, t1, t2):
    a = np.radians((t1 + t2) / 2)
    r = (inner + outer) / 2
    return r * np.cos(a), r * np.sin(a)


def _text_color(rgba):
    r, g, b = rgba[:3]
    return "#111111" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "#ffffff"


def draw_zone_court(ax, values, *, scheme, cmap, vmin=None, vcenter=None,
                    vmax=None, title="", note="", cbar_label="", wedges=None,
                    label_size=9.5, cbar=True):
    """Render one zone court.

    values : {zone_name: {"c": color_value or None, "top": str, "sub": str}}
             c=None  -> hollow / unreliable zone.
    wedges : which wedge set to draw (default the 14-zone ZONE_WEDGES; pass
             MACRO_WEDGES for the coarse paint/mid/three view).
    cbar   : draw a colorbar (set False for small-multiple grids on one scale).
    """
    if wedges is None:
        wedges = ZONE_WEDGES
    finite = [v["c"] for v in values.values() if v["c"] is not None]
    if scheme == "diverging":
        vmin = vmin if vmin is not None else -0.15
        vmax = vmax if vmax is not None else 0.15
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter or 0.0, vmax=vmax)
    else:
        vmin = vmin if vmin is not None else 0.0
        vmax = vmax if vmax is not None else (max(finite) if finite else 1.0)
        norm = Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    for zone, (inner, outer, t1, t2) in wedges.items():
        v = values.get(zone, {"c": None, "top": "", "sub": ""})
        if v["c"] is None:
            face, edge, hatch = "#eceef1", "#c3c7cc", "////"
            tcol = MUTED
        else:
            rgba = mapper.to_rgba(np.clip(v["c"], vmin, vmax))
            face, edge, hatch = rgba, "white", None
            tcol = _text_color(rgba)
        ax.add_patch(Wedge((0, 0), outer, t1, t2, width=outer - inner,
                           facecolor=face, edgecolor=edge, lw=1.4, hatch=hatch,
                           zorder=2))
        lx, ly = _label_xy(inner, outer, t1, t2)
        if v.get("top"):
            ax.text(lx, ly + 0.5, v["top"], ha="center", va="center",
                    fontsize=label_size, weight="bold", color=tcol, zorder=4)
        if v.get("sub"):
            ax.text(lx, ly - 1.1, v["sub"], ha="center", va="center",
                    fontsize=label_size * 0.8, color=tcol, zorder=4)

    _draw_lines(ax)
    ax.set_xlim(-26, 26); ax.set_ylim(-3, THREE_OUT + 1.5)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, weight="bold", color="#1a1d21",
                     loc="center", pad=8)
    if note:
        ax.text(0, -2.4, note, ha="center", va="top", fontsize=8.5, color=MUTED)

    if not cbar:
        return None
    cb = plt.colorbar(mapper, ax=ax, fraction=0.04, pad=0.02, shrink=0.7)
    cb.set_label(cbar_label, fontsize=8.5, color=MUTED)
    cb.ax.tick_params(labelsize=7.5, colors=MUTED)
    return cb


def _draw_lines(ax):
    """Court markings: baseline, three-point arc + corners, key, hoop."""
    ax.plot([-25, 25], [0, 0], color=MUTED, lw=1.5, zorder=1)          # baseline
    ax.add_patch(Circle((0, 0), 0.75, fill=False, color=MUTED, lw=1.3, zorder=3))
    # three-point line: straight corners then arc
    ax.plot([CORNER_X, CORNER_X], [0, Y_MEET], color=MUTED, lw=1.5, zorder=3)
    ax.plot([-CORNER_X, -CORNER_X], [0, Y_MEET], color=MUTED, lw=1.5, zorder=3)
    a0 = np.degrees(np.arctan2(Y_MEET, CORNER_X))
    ax.add_patch(Arc((0, 0), 2 * ARC_R, 2 * ARC_R, theta1=a0, theta2=180 - a0,
                     color=MUTED, lw=1.5, zorder=3))
    # free-throw lane (schematic, 12ft wide, 19ft deep)
    ax.plot([-6, -6], [0, 19], color=MUTED, lw=1, alpha=0.6, zorder=1)
    ax.plot([6, 6], [0, 19], color=MUTED, lw=1, alpha=0.6, zorder=1)
    ax.plot([-6, 6], [19, 19], color=MUTED, lw=1, alpha=0.6, zorder=1)
