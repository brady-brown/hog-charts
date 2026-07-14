"""Shot-zone classification — one copy for the whole pipeline.

The 14-wedge "2K-style" scheme: a restricted-area disk, concentric rings
(close-mid / mid / three) sliced into angular wedges. Index order and the
boundary geometry MUST stay identical to classifyZone() in
site/js/zone-chart.js (the client-side mirror) — the browser stores shots by
ZONE_NAMES index, so any drift here silently mislabels every chart.

Lifted verbatim from the copies build_site.py and coaches_suite/scout_lib.py
each carried; build_shot_diet.py imports the boundary constants from here too.
"""
import numpy as np
import pandas as pd

# Ring boundaries (feet from the basket) and the college three-point geometry.
RESTRICTED_RADIUS = 4.0     # RA: inside this is the restricted area / rim
CLOSE_RADIUS      = 11.0    # close-mid extends from RA out to here
THREE_POINT_RADIUS = 22.146  # college 3PT arc radius
CORNER_X          = 21.65   # corner-three straight sections at |lateral| = this

ZONE_NAMES = [
    "Restricted Area",                                              # 0
    "Close Mid - Right", "Close Mid - Center", "Close Mid - Left",  # 1-3
    "Mid - Right", "Mid - Right Center", "Mid - Center",            # 4-6
    "Mid - Left Center", "Mid - Left",                             # 7-8
    "3PT - Right", "3PT - Right Center", "3PT - Center",            # 9-11
    "3PT - Left Center", "3PT - Left",                             # 12-13
]
ZONE_NAME_TO_INDEX = {zone_name: idx for idx, zone_name in enumerate(ZONE_NAMES)}
THREE_POINT_ZONES  = {"3PT - Right", "3PT - Right Center", "3PT - Center",
                      "3PT - Left Center", "3PT - Left"}


def classify_shot_zones(shot_df):
    """Assign each shot to a zone based on its court coordinates.

    Coordinate convention (ESPN/sportsdataverse):
        coordinate_x: distance from center-court baseline (0 = baseline, 47 = halfcourt)
        coordinate_y: distance from court centerline (positive = left side, negative = right)

    14-wedge "2K-style" scheme.  Boundaries MUST match the SVG polygons and
    classifyZone() in site/js/zone-chart.js:
        rings    RA<4ft · close 4-11ft · mid 11ft-arc · three beyond arc
        3PT line college arc r=22.146ft, corner straights at |lat|=21.65ft
        angle    0deg = right baseline, 90deg = straight out, 180deg = left
    Returns a Series of zone name strings.
    """
    RA, CLOSE, THREE, CORNER = (RESTRICTED_RADIUS, CLOSE_RADIUS,
                                THREE_POINT_RADIUS, CORNER_X)
    Y_MEET = np.sqrt(THREE**2 - CORNER**2)            # ~4.658 ft, corner/arc junction

    valid_coords_mask = shot_df["coordinate_x"].notna() & shot_df["coordinate_y"].notna()
    zone_series = pd.Series("Unknown", index=shot_df.index, dtype="object")

    x_abs   = shot_df.loc[valid_coords_mask, "coordinate_x"].abs()
    y_coord = shot_df.loc[valid_coords_mask, "coordinate_y"]

    lateral  = -y_coord            # right side positive (matches screen x / svg +x)
    toward   = 41.75 - x_abs       # distance from basket toward halfcourt
    distance = np.sqrt(lateral**2 + toward**2)
    angle    = np.degrees(np.arctan2(toward, lateral))
    angle    = angle.where(angle >= -90, angle + 360)   # unwrap left-baseline corner

    is_heave = distance >= 40
    is_rim   = ~is_heave & (distance < RA)
    is_three = ~is_heave & ~is_rim & (
        (distance >= THREE) | ((lateral.abs() >= CORNER) & (toward <= Y_MEET))
    )
    is_close = ~is_heave & ~is_rim & ~is_three & (distance < CLOSE)
    is_mid   = ~is_heave & ~is_rim & ~is_three & ~is_close

    m = valid_coords_mask
    zone_series[m & is_heave] = "Heave"
    zone_series[m & is_rim]   = "Restricted Area"

    zone_series[m & is_close & (angle <  60)]                  = "Close Mid - Right"
    zone_series[m & is_close & (angle >= 60) & (angle < 120)]  = "Close Mid - Center"
    zone_series[m & is_close & (angle >= 120)]                 = "Close Mid - Left"

    zone_series[m & is_mid & (angle <  36)]                    = "Mid - Right"
    zone_series[m & is_mid & (angle >= 36)  & (angle < 72)]    = "Mid - Right Center"
    zone_series[m & is_mid & (angle >= 72)  & (angle < 108)]   = "Mid - Center"
    zone_series[m & is_mid & (angle >= 108) & (angle < 144)]   = "Mid - Left Center"
    zone_series[m & is_mid & (angle >= 144)]                   = "Mid - Left"

    zone_series[m & is_three & (angle <  36)]                  = "3PT - Right"
    zone_series[m & is_three & (angle >= 36)  & (angle < 72)]  = "3PT - Right Center"
    zone_series[m & is_three & (angle >= 72)  & (angle < 108)] = "3PT - Center"
    zone_series[m & is_three & (angle >= 108) & (angle < 144)] = "3PT - Left Center"
    zone_series[m & is_three & (angle >= 144)]                 = "3PT - Left"
    return zone_series
