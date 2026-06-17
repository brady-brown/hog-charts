"""Interactive Plotly shot charts for the Hog Charts app.

Reuses the data layer in `MBBZoneEfficiencyVisualizer` (loading, team/game
resolution, zone classification) and replaces the matplotlib rendering with
compact, responsive, interactive Plotly figures.

Coordinate convention used here (a clean half-court, hoop near the bottom):
    px = -coordinate_y            left/right across the floor (-25..25)
    py = 47 - |coordinate_x|      distance up from the baseline (0 = baseline)
The hoop sits at (0, 5.25).  Left-side zones (positive coordinate_y) render on
the screen-left so the labels match their position.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── theme (matches ui.py / the dark site) ────────────────────────────────────
BG = "#0B0E14"
PANEL = "#11151F"
LINE = "#3C4456"
MAKE = "#22C55E"
MISS = "#EF4444"
TEXT = "#E6E9EF"
MUTED = "#9AA4B2"

HOOP_Y = 5.25
THREE_R = 22.146
CORNER_X = 21.65

# View window — landscape half-court, ~1.6:1, fits a screen with no scroll.
X_RANGE = (-26, 26)
Y_RANGE = (-2.5, 31)

# Where each zone's stat label is anchored (screen coordinates).
ZONE_XY = {
    "At Rim": (0, 3.0),
    "Paint (Non-Rim)": (0, 12.0),
    "Center Mid-Range": (0, 20.5),
    "Left Mid-Range": (-12, 16.0),
    "Right Mid-Range": (12, 16.0),
    "Left Baseline Mid": (-17.5, 7.0),
    "Right Baseline Mid": (17.5, 7.0),
    "Center 3PT": (0, 28.0),
    "Left Wing 3PT": (-20, 21.0),
    "Right Wing 3PT": (20, 21.0),
    "Left Corner 3PT": (-23.5, 4.0),
    "Right Corner 3PT": (23.5, 4.0),
}


# ── geometry helpers ─────────────────────────────────────────────────────────
def _arc(cx, cy, r, t0, t1, n=80):
    t = np.radians(np.linspace(t0, t1, n))
    return cx + r * np.cos(t), cy + r * np.sin(t)


def _line(x0, y0, x1, y1):
    return np.array([x0, x1]), np.array([y0, y1])


def _court_polylines():
    """Return a list of (x, y) polylines that draw an NCAA half-court."""
    lines = []
    # baseline + sidelines
    lines.append(_line(-25, 0, 25, 0))
    lines.append(_line(-25, 0, -25, Y_RANGE[1]))
    lines.append(_line(25, 0, 25, Y_RANGE[1]))
    # backboard + rim
    lines.append(_line(-3, 4, 3, 4))
    lines.append(_arc(0, HOOP_Y, 0.75, 0, 360))            # hoop
    lines.append(_line(0, 4, 0, 4.5))                       # rim mount
    # restricted-area arc (4 ft)
    lines.append(_arc(0, HOOP_Y, 4, 0, 180))
    # lane (12 ft wide, 19 ft to the FT line) + FT circle
    lines.append(_line(-6, 0, -6, 19))
    lines.append(_line(6, 0, 6, 19))
    lines.append(_line(-6, 19, 6, 19))
    lines.append(_arc(0, 19, 6, 0, 360))
    # three-point line: straight corners + top arc
    junc = HOOP_Y + np.sqrt(THREE_R**2 - CORNER_X**2)       # y where corner meets arc
    ang = np.degrees(np.arctan2(junc - HOOP_Y, CORNER_X))
    lines.append(_line(-CORNER_X, 0, -CORNER_X, junc))
    lines.append(_line(CORNER_X, 0, CORNER_X, junc))
    lines.append(_arc(0, HOOP_Y, THREE_R, ang, 180 - ang))
    return lines


_COURT = _court_polylines()


# ── figure scaffolding ───────────────────────────────────────────────────────
def _new_fig(title, subtitle):
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:13px;color:{MUTED}'>{subtitle}</span>",
            x=0.5, xanchor="center", y=0.97, font=dict(color=TEXT, size=20),
        ),
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=600, margin=dict(l=10, r=10, t=70, b=10),
        showlegend=False,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=0.02,
                    bgcolor="rgba(17,21,31,0.7)", font=dict(color=TEXT)),
        dragmode="pan",
    )
    fig.update_xaxes(range=list(X_RANGE), visible=False,
                     scaleanchor="y", scaleratio=1, constrain="domain")
    fig.update_yaxes(range=list(Y_RANGE), visible=False, constrain="domain")
    return fig


def _add_court(fig, color=LINE, width=1.6):
    for x, y in _COURT:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip", showlegend=False,
        ))
    return fig


# ── data prep ────────────────────────────────────────────────────────────────
def _field_goals(shots):
    """Drop free throws + invalid coordinates, add plot coords + shot distance."""
    if shots is None or len(shots) == 0:
        return shots.iloc[0:0] if shots is not None else None
    is_ft = (
        shots["type_text"].astype(str).str.contains("Free", case=False, na=False)
        | shots["text"].astype(str).str.contains("free throw", case=False, na=False)
    )
    fgs = shots[~is_ft].dropna(subset=["coordinate_x", "coordinate_y"]).copy()
    fgs = fgs[(fgs["coordinate_x"].abs() <= 50) & (fgs["coordinate_y"].abs() <= 30)]
    fgs["px"] = -fgs["coordinate_y"]
    fgs["py"] = 47 - fgs["coordinate_x"].abs()
    fgs["dist"] = np.sqrt(fgs["px"] ** 2 + (fgs["py"] - HOOP_Y) ** 2)
    return fgs


def _splits_subtitle(base, stats):
    valid = stats[~stats["zone"].isin(["Heave", "Unknown"])]
    tm, ta = int(valid["makes"].sum()), int(valid["attempts"].sum())
    is3 = valid["zone"].astype(str).str.contains("3PT", na=False)
    t3m, t3a = int(valid.loc[is3, "makes"].sum()), int(valid.loc[is3, "attempts"].sum())
    t2m, t2a = int(valid.loc[~is3, "makes"].sum()), int(valid.loc[~is3, "attempts"].sum())

    def pct(m, a):
        return f"{m / a * 100:.0f}%" if a else "—"

    return (f"{base}  ·  FG {tm}/{ta} ({pct(tm, ta)})  ·  "
            f"2PT {pct(t2m, t2a)}  ·  3PT {pct(t3m, t3a)}")


# ── renderers ────────────────────────────────────────────────────────────────
def _zone_fig(stats, title, subtitle, label_fmt):
    fig = _new_fig(title, subtitle)
    _add_court(fig)
    for _, row in stats.iterrows():
        zone = row["zone"]
        if zone not in ZONE_XY:
            continue
        x, y = ZONE_XY[zone]
        pct = row.get("fg_pct", np.nan)
        if pd.notna(pct) and pct >= 45:
            bg = "#C8E6C9"
        elif pd.notna(pct) and pct <= 35:
            bg = "#FFCDD2"
        else:
            bg = "#E7ECF2"
        fig.add_annotation(
            x=x, y=y, text=label_fmt(row), showarrow=False, align="center",
            font=dict(size=11, color="#111827", family="Arial"),
            bgcolor=bg, bordercolor="#0B0E14", borderwidth=1, borderpad=4,
            opacity=0.95,
        )
    return fig


def zone_efficiency(stats, title, subtitle):
    def fmt(row):
        return f"<b>{row['zone']}</b><br>{int(row['makes'])}/{int(row['attempts'])} · {row['fg_pct']:.0f}%"
    return _zone_fig(stats, title, subtitle, fmt)


def territory(leaders, title, subtitle):
    def fmt(row):
        return f"<b>{row['zone']}</b><br>{row['athlete_display_name']}<br>{int(row['pts'])} pts"
    return _zone_fig(leaders, title, subtitle, fmt)


def scatter(fgs, title, subtitle):
    fig = _new_fig(title, subtitle)
    _add_court(fig)
    fig.update_layout(showlegend=True)
    makes = fgs[fgs["scoring_play"] == True]
    misses = fgs[fgs["scoring_play"] == False]
    has_player = "athlete_display_name" in fgs.columns

    def _trace(d, name, color, symbol):
        cd = d["athlete_display_name"] if has_player else pd.Series([""] * len(d))
        hover = ("%{customdata}<br>" if has_player else "") + \
                name + " · %{text:.0f} ft<extra></extra>"
        return go.Scatter(
            x=d["px"], y=d["py"], mode="markers", name=f"{name} ({len(d)})",
            marker=dict(color=color, size=8, symbol=symbol, line=dict(color="white", width=0.6),
                        opacity=0.85),
            customdata=cd, text=d["dist"], hovertemplate=hover,
        )

    fig.add_trace(_trace(misses, "Miss", MISS, "x"))
    fig.add_trace(_trace(makes, "Make", MAKE, "circle"))
    return fig


def density(fgs, title, subtitle):
    from scipy.stats import gaussian_kde
    fig = _new_fig(title, subtitle)
    x, y = fgs["px"].values, fgs["py"].values
    xi = np.linspace(X_RANGE[0], X_RANGE[1], 140)
    yi = np.linspace(Y_RANGE[0], Y_RANGE[1], 140)
    xx, yy = np.meshgrid(xi, yi)
    kde = gaussian_kde(np.vstack([x, y]), bw_method=0.18)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    fig.add_trace(go.Contour(
        x=xi, y=yi, z=zz, showscale=False, ncontours=18,
        colorscale="YlOrRd", opacity=0.85, hoverinfo="skip",
        contours=dict(coloring="fill", showlines=False),
    ))
    _add_court(fig, color="#E6E9EF", width=1.4)
    return fig


# ── public wrapper that mirrors the page's call surface ──────────────────────
class PlotlyShotCharts:
    def __init__(self, viz):
        self.viz = viz
        self.season = viz.season

    # -- shot pulls ----------------------------------------------------------
    def _team_shots(self, team_name):
        team_id, team_full = self.viz._get_team_id(team_name)
        return self.viz._get_shots(team_id), team_full

    def _team_game_shots(self, team_name, opponent_name, date):
        team_id, team_full = self.viz._get_team_id(team_name)
        game_id, is_home = self.viz._get_game_id_from_matchup(team_name, opponent_name, date)
        return self.viz._get_shots(team_id, game_id), team_full, ("vs" if is_home else "@")

    @staticmethod
    def _only(shots, player_name):
        return shots[shots["athlete_display_name"].str.contains(player_name, case=False, na=False)]

    # -- zone efficiency -----------------------------------------------------
    def player_season_chart(self, player_name, team_name):
        shots, team_full = self._team_shots(team_name)
        shots = self._only(shots, player_name)
        stats = self.viz._calculate_zone_stats(shots)
        return zone_efficiency(stats, player_name,
                               _splits_subtitle(f"{team_full} · {self.season} season", stats))

    def team_season_chart(self, team_name):
        shots, team_full = self._team_shots(team_name)
        stats = self.viz._calculate_zone_stats(shots)
        return zone_efficiency(stats, team_full,
                               _splits_subtitle(f"{self.season} season", stats))

    def player_game_chart(self, player_name, team_name, opponent_name, date):
        shots, team_full, loc = self._team_game_shots(team_name, opponent_name, date)
        shots = self._only(shots, player_name)
        stats = self.viz._calculate_zone_stats(shots)
        return zone_efficiency(stats, player_name,
                               _splits_subtitle(f"{team_full} {loc} {opponent_name} · {date}", stats))

    def team_game_chart(self, team_name, opponent_name, date):
        shots, team_full, loc = self._team_game_shots(team_name, opponent_name, date)
        stats = self.viz._calculate_zone_stats(shots)
        return zone_efficiency(stats, team_full,
                               _splits_subtitle(f"{loc} {opponent_name} · {date}", stats))

    # -- territory -----------------------------------------------------------
    def plot_team_zone_leaders(self, team_name):
        team_full, leaders = self.viz.get_team_zone_leaders(team_name)
        if leaders is None or leaders.empty:
            return None
        return territory(leaders, f"{team_full} — Territory Map",
                         f"Top scorer in each zone · {self.season} season")

    # -- scatter (single game) ----------------------------------------------
    def player_game_scatter(self, player_name, team_name, opponent_name, date):
        shots, team_full, loc = self._team_game_shots(team_name, opponent_name, date)
        fgs = _field_goals(self._only(shots, player_name))
        if fgs is None or fgs.empty:
            return None
        m = int(fgs["scoring_play"].sum())
        return scatter(fgs, player_name,
                       f"{team_full} {loc} {opponent_name} · {date}  ·  {m}/{len(fgs)} FG")

    def team_game_scatter(self, team_name, opponent_name, date):
        shots, team_full, loc = self._team_game_shots(team_name, opponent_name, date)
        fgs = _field_goals(shots)
        if fgs is None or fgs.empty:
            return None
        m = int(fgs["scoring_play"].sum())
        return scatter(fgs, team_full, f"{loc} {opponent_name} · {date}  ·  {m}/{len(fgs)} FG")

    # -- density (full season) ----------------------------------------------
    def player_season_density(self, player_name, team_name):
        shots, team_full = self._team_shots(team_name)
        fgs = _field_goals(self._only(shots, player_name))
        if fgs is None or len(fgs) < 5:
            return None
        return density(fgs, player_name, f"{team_full} · {self.season} season · shot density")

    def team_season_density(self, team_name):
        shots, team_full = self._team_shots(team_name)
        fgs = _field_goals(shots)
        if fgs is None or len(fgs) < 5:
            return None
        return density(fgs, team_full, f"{self.season} season · shot density")
