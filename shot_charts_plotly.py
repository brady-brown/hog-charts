"""
shot_charts_plotly.py — Interactive Plotly shot charts for the Hog Charts Streamlit app.

WHY THIS FILE EXISTS
--------------------
mbb_viz.py uses matplotlib to produce static PNG shot charts.  The Streamlit
app needs interactive charts that users can hover over, so this module
replaces the matplotlib rendering with Plotly's go.Figure.

The data layer (loading, team/game resolution, zone classification) lives in
MBBZoneEfficiencyVisualizer and is reused here.  PlotlyShotCharts is a thin
wrapper that calls the data methods from mbb_viz.py and passes the results
to the module-level rendering functions below.

Coordinate convention used in Plotly (clean half-court, hoop near the bottom):
    screen_x = −coordinate_y            left/right across the floor (range −25…25)
    screen_y = 47 − |coordinate_x|      distance up from the baseline (0 = baseline)
    hoop sits at (screen_x=0, screen_y=5.25)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Color theme (matches ui.py / the dark site)
# ---------------------------------------------------------------------------
BACKGROUND_COLOR    = "#0B0E14"
PANEL_COLOR         = "#11151F"
COURT_LINE_COLOR    = "#3C4456"
MADE_SHOT_COLOR     = "#22C55E"    # green
MISSED_SHOT_COLOR   = "#EF4444"    # red
TEXT_COLOR          = "#E6E9EF"
MUTED_TEXT_COLOR    = "#9AA4B2"

# ---------------------------------------------------------------------------
# Court geometry constants (all in feet, same projection as screen_x/screen_y)
# ---------------------------------------------------------------------------
HOOP_SCREEN_Y    = 5.25       # how far up from baseline the hoop sits
THREE_POINT_RADIUS  = 22.146  # NCAA 3PT arc radius from the basket
CORNER_THREE_X      = 21.65   # x distance where corner 3PT lines end

# Plotly display window for a landscape half-court view.
SCREEN_X_RANGE = (-26, 26)
SCREEN_Y_RANGE = (-2.5, 31)

# Where each zone's annotation label is anchored (screen coordinates).
ZONE_LABEL_SCREEN_XY = {
    "At Rim":            (0,     3.0),
    "Paint (Non-Rim)":   (0,    12.0),
    "Center Mid-Range":  (0,    20.5),
    "Left Mid-Range":    (-12,  16.0),
    "Right Mid-Range":   (12,   16.0),
    "Left Baseline Mid": (-17.5, 7.0),
    "Right Baseline Mid":(17.5,  7.0),
    "Center 3PT":        (0,    28.0),
    "Left Wing 3PT":     (-20,  21.0),
    "Right Wing 3PT":    (20,   21.0),
    "Left Corner 3PT":   (-23.5, 4.0),
    "Right Corner 3PT":  (23.5,  4.0),
}


# ---------------------------------------------------------------------------
# Court geometry helpers
# ---------------------------------------------------------------------------

def _arc_coords(center_x, center_y, radius, angle_start_degrees, angle_end_degrees, num_points=80):
    """Return (x_coords, y_coords) for an arc from angle_start to angle_end."""
    angles = np.radians(np.linspace(angle_start_degrees, angle_end_degrees, num_points))
    x_coords = center_x + radius * np.cos(angles)
    y_coords = center_y + radius * np.sin(angles)
    return x_coords, y_coords


def _line_coords(x_start, y_start, x_end, y_end):
    """Return (x_coords, y_coords) for a straight line segment."""
    return np.array([x_start, x_end]), np.array([y_start, y_end])


def _build_ncaa_half_court_lines():
    """Return a list of (x_coords, y_coords) tuples that draw an NCAA half-court.

    All coordinates are in Plotly screen space:
        x = 0 at center of the court (left-right)
        y = 0 at the baseline (increases toward half-court)
    """
    court_polylines = []

    # Outer boundary: baseline and sidelines
    court_polylines.append(_line_coords(-25, 0, 25, 0))                       # baseline
    court_polylines.append(_line_coords(-25, 0, -25, SCREEN_Y_RANGE[1]))      # left sideline
    court_polylines.append(_line_coords(25, 0, 25, SCREEN_Y_RANGE[1]))        # right sideline

    # Backboard + rim
    court_polylines.append(_line_coords(-3, 4, 3, 4))                         # backboard
    court_polylines.append(_arc_coords(0, HOOP_SCREEN_Y, 0.75, 0, 360))       # hoop circle
    court_polylines.append(_line_coords(0, 4, 0, 4.5))                        # rim mount

    # Restricted-area arc (4 ft radius)
    court_polylines.append(_arc_coords(0, HOOP_SCREEN_Y, 4, 0, 180))

    # Paint lane (12 ft wide = ±6 ft, extends 19 ft from baseline to FT line)
    court_polylines.append(_line_coords(-6, 0, -6, 19))                       # left lane line
    court_polylines.append(_line_coords(6, 0, 6, 19))                         # right lane line
    court_polylines.append(_line_coords(-6, 19, 6, 19))                       # free-throw line

    # Free-throw circle (6 ft radius)
    court_polylines.append(_arc_coords(0, 19, 6, 0, 360))

    # Three-point line: straight corner segments + top arc
    corner_arc_junction_y = HOOP_SCREEN_Y + np.sqrt(THREE_POINT_RADIUS**2 - CORNER_THREE_X**2)
    corner_arc_angle = np.degrees(np.arctan2(corner_arc_junction_y - HOOP_SCREEN_Y, CORNER_THREE_X))
    court_polylines.append(_line_coords(-CORNER_THREE_X, 0, -CORNER_THREE_X, corner_arc_junction_y))
    court_polylines.append(_line_coords(CORNER_THREE_X, 0, CORNER_THREE_X, corner_arc_junction_y))
    court_polylines.append(_arc_coords(0, HOOP_SCREEN_Y, THREE_POINT_RADIUS,
                                        corner_arc_angle, 180 - corner_arc_angle))

    return court_polylines


# Pre-compute court lines once at module load time (they never change).
_NCAA_HALF_COURT_LINES = _build_ncaa_half_court_lines()


# ---------------------------------------------------------------------------
# Figure scaffolding
# ---------------------------------------------------------------------------

def _new_figure(chart_title, chart_subtitle):
    """Create an empty Plotly figure with consistent dark-theme styling."""
    figure = go.Figure()
    figure.update_layout(
        title=dict(
            text=f"<b>{chart_title}</b><br>"
                 f"<span style='font-size:13px;color:{MUTED_TEXT_COLOR}'>{chart_subtitle}</span>",
            x=0.5, xanchor="center", y=0.97,
            font=dict(color=TEXT_COLOR, size=20),
        ),
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        height=600,
        margin=dict(l=10, r=10, t=70, b=10),
        showlegend=False,
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=0.02,
            bgcolor="rgba(17,21,31,0.7)", font=dict(color=TEXT_COLOR)
        ),
        dragmode="pan",
    )
    figure.update_xaxes(
        range=list(SCREEN_X_RANGE), visible=False,
        scaleanchor="y", scaleratio=1, constrain="domain"
    )
    figure.update_yaxes(
        range=list(SCREEN_Y_RANGE), visible=False, constrain="domain"
    )
    return figure


def _add_court_lines(figure, line_color=COURT_LINE_COLOR, line_width=1.6):
    """Draw the NCAA half-court lines onto an existing Plotly figure."""
    for x_coords, y_coords in _NCAA_HALF_COURT_LINES:
        figure.add_trace(go.Scatter(
            x=x_coords, y=y_coords, mode="lines",
            line=dict(color=line_color, width=line_width),
            hoverinfo="skip", showlegend=False,
        ))
    return figure


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _prepare_field_goals(raw_shots_df):
    """Filter and project shot data to Plotly screen coordinates.

    Removes:
      - Free throws (type_text or text contains "Free"/"free throw")
      - Shots with missing or out-of-bounds coordinates

    Adds:
      - screen_x = −coordinate_y   (left/right on screen)
      - screen_y = 47 − |coordinate_x|  (distance from baseline)
      - shot_distance = Euclidean distance from the hoop
    """
    if raw_shots_df is None or len(raw_shots_df) == 0:
        return raw_shots_df.iloc[0:0] if raw_shots_df is not None else None

    is_free_throw_mask = (
        raw_shots_df["type_text"].astype(str).str.contains("Free", case=False, na=False)
        | raw_shots_df["text"].astype(str).str.contains("free throw", case=False, na=False)
    )
    field_goals_df = raw_shots_df[~is_free_throw_mask].dropna(subset=["coordinate_x", "coordinate_y"]).copy()
    field_goals_df = field_goals_df[
        (field_goals_df["coordinate_x"].abs() <= 50)
        & (field_goals_df["coordinate_y"].abs() <= 30)
    ]
    field_goals_df["screen_x"]     = -field_goals_df["coordinate_y"]
    field_goals_df["screen_y"]     = 47 - field_goals_df["coordinate_x"].abs()
    field_goals_df["shot_distance"] = np.sqrt(
        field_goals_df["screen_x"] ** 2 + (field_goals_df["screen_y"] - HOOP_SCREEN_Y) ** 2
    )
    return field_goals_df


def _build_splits_subtitle(base_subtitle, zone_stats_df):
    """Append shooting-split percentages to the chart subtitle string.

    Format: "Base subtitle  ·  FG 47/100 (47%)  ·  2PT 55%  ·  3PT 36%"
    """
    valid_zones_df = zone_stats_df[~zone_stats_df["zone"].isin(["Heave", "Unknown"])]
    total_makes    = int(valid_zones_df["makes"].sum())
    total_attempts = int(valid_zones_df["attempts"].sum())

    is_three_pt_zone_mask = valid_zones_df["zone"].astype(str).str.contains("3PT", na=False)
    three_pt_makes     = int(valid_zones_df.loc[is_three_pt_zone_mask,  "makes"].sum())
    three_pt_attempts  = int(valid_zones_df.loc[is_three_pt_zone_mask,  "attempts"].sum())
    two_pt_makes       = int(valid_zones_df.loc[~is_three_pt_zone_mask, "makes"].sum())
    two_pt_attempts    = int(valid_zones_df.loc[~is_three_pt_zone_mask, "attempts"].sum())

    def format_pct(makes, attempts):
        return f"{makes / attempts * 100:.0f}%" if attempts else "—"

    return (
        f"{base_subtitle}  ·  "
        f"FG {total_makes}/{total_attempts} ({format_pct(total_makes, total_attempts)})  ·  "
        f"2PT {format_pct(two_pt_makes, two_pt_attempts)}  ·  "
        f"3PT {format_pct(three_pt_makes, three_pt_attempts)}"
    )


# ---------------------------------------------------------------------------
# Renderers (module-level functions, called by PlotlyShotCharts)
# ---------------------------------------------------------------------------

def zone_efficiency_chart(zone_stats_df, chart_title, chart_subtitle):
    """Render a zone-efficiency chart with FG% annotation labels per zone.

    Label background is green if FG% ≥ 45%, red if ≤ 35%, neutral otherwise.
    """
    figure = _new_figure(chart_title, chart_subtitle)
    _add_court_lines(figure)

    for _, zone_row in zone_stats_df.iterrows():
        zone_name = zone_row["zone"]
        if zone_name not in ZONE_LABEL_SCREEN_XY:
            continue

        label_x, label_y = ZONE_LABEL_SCREEN_XY[zone_name]
        zone_fg_pct = zone_row.get("fg_pct", np.nan)

        if pd.notna(zone_fg_pct) and zone_fg_pct >= 45:
            label_bg_color = "#C8E6C9"    # green: above average shooting zone
        elif pd.notna(zone_fg_pct) and zone_fg_pct <= 35:
            label_bg_color = "#FFCDD2"    # red: below average shooting zone
        else:
            label_bg_color = "#E7ECF2"    # neutral

        annotation_text = (
            f"<b>{zone_name}</b><br>"
            f"{int(zone_row['makes'])}/{int(zone_row['attempts'])} · {zone_fg_pct:.0f}%"
        )
        figure.add_annotation(
            x=label_x, y=label_y, text=annotation_text, showarrow=False, align="center",
            font=dict(size=11, color="#111827", family="Arial"),
            bgcolor=label_bg_color, bordercolor="#0B0E14",
            borderwidth=1, borderpad=4, opacity=0.95,
        )
    return figure


def territory_chart(zone_leaders_df, chart_title, chart_subtitle):
    """Render a territory map showing the top scorer in each zone per team."""
    def format_leader_label(leader_row):
        return (
            f"<b>{leader_row['zone']}</b><br>"
            f"{leader_row['athlete_display_name']}<br>"
            f"{int(leader_row['pts'])} pts"
        )
    return _zone_annotation_figure(zone_leaders_df, chart_title, chart_subtitle, format_leader_label)


def _zone_annotation_figure(data_df, chart_title, chart_subtitle, label_formatter_fn):
    """Shared renderer for zone charts: draws court + one annotation per zone."""
    figure = _new_figure(chart_title, chart_subtitle)
    _add_court_lines(figure)
    for _, data_row in data_df.iterrows():
        zone_name = data_row["zone"]
        if zone_name not in ZONE_LABEL_SCREEN_XY:
            continue
        label_x, label_y = ZONE_LABEL_SCREEN_XY[zone_name]
        figure.add_annotation(
            x=label_x, y=label_y, text=label_formatter_fn(data_row),
            showarrow=False, align="center",
            font=dict(size=11, color="#111827", family="Arial"),
            bgcolor="#E7ECF2", bordercolor="#0B0E14",
            borderwidth=1, borderpad=4, opacity=0.95,
        )
    return figure


def scatter_chart(field_goals_df, chart_title, chart_subtitle):
    """Render an interactive scatter plot: circles for makes, X's for misses."""
    figure = _new_figure(chart_title, chart_subtitle)
    _add_court_lines(figure)
    figure.update_layout(showlegend=True)

    made_shots_df   = field_goals_df[field_goals_df["scoring_play"] == True]
    missed_shots_df = field_goals_df[field_goals_df["scoring_play"] == False]
    has_player_names = "athlete_display_name" in field_goals_df.columns

    def build_scatter_trace(shot_subset_df, trace_name, dot_color, dot_symbol):
        """Build one Plotly Scatter trace with hover data."""
        player_names  = (shot_subset_df["athlete_display_name"]
                         if has_player_names else pd.Series([""] * len(shot_subset_df)))
        hover_template = (
            ("%{customdata}<br>" if has_player_names else "")
            + trace_name + " · %{text:.0f} ft<extra></extra>"
        )
        return go.Scatter(
            x=shot_subset_df["screen_x"], y=shot_subset_df["screen_y"],
            mode="markers",
            name=f"{trace_name} ({len(shot_subset_df)})",
            marker=dict(
                color=dot_color, size=8, symbol=dot_symbol,
                line=dict(color="white", width=0.6), opacity=0.85
            ),
            customdata=player_names,
            text=shot_subset_df["shot_distance"],
            hovertemplate=hover_template,
        )

    figure.add_trace(build_scatter_trace(missed_shots_df, "Miss", MISSED_SHOT_COLOR, "x"))
    figure.add_trace(build_scatter_trace(made_shots_df,   "Make", MADE_SHOT_COLOR,   "circle"))
    return figure


def density_chart(field_goals_df, chart_title, chart_subtitle):
    """Render a KDE shot-density heatmap using a Plotly Contour trace."""
    from scipy.stats import gaussian_kde

    figure = _new_figure(chart_title, chart_subtitle)

    grid_x = np.linspace(SCREEN_X_RANGE[0], SCREEN_X_RANGE[1], 140)
    grid_y = np.linspace(SCREEN_Y_RANGE[0], SCREEN_Y_RANGE[1], 140)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    kde_estimator = gaussian_kde(
        np.vstack([field_goals_df["screen_x"].values, field_goals_df["screen_y"].values]),
        bw_method=0.18
    )
    density_values = kde_estimator(
        np.vstack([grid_xx.ravel(), grid_yy.ravel()])
    ).reshape(grid_xx.shape)

    figure.add_trace(go.Contour(
        x=grid_x, y=grid_y, z=density_values,
        showscale=False, ncontours=18,
        colorscale="YlOrRd", opacity=0.85, hoverinfo="skip",
        contours=dict(coloring="fill", showlines=False),
    ))
    _add_court_lines(figure, line_color=TEXT_COLOR, line_width=1.4)
    return figure


# ---------------------------------------------------------------------------
# PlotlyShotCharts — public API wrapper
# ---------------------------------------------------------------------------

class PlotlyShotCharts:
    """Thin wrapper around PlotlyShotCharts rendering functions.

    Delegates all data access to an existing MBBZoneEfficiencyVisualizer
    instance, which handles team/game resolution and zone classification.
    PlotlyShotCharts only handles chart rendering.

    Attributes:
        viz      MBBZoneEfficiencyVisualizer  The data layer.
        season   int                          Forwarded from viz.season.
    """

    def __init__(self, mbb_visualizer):
        self.viz    = mbb_visualizer
        self.season = mbb_visualizer.season

    # ── Internal shot-pull helpers ────────────────────────────────────────────

    def _get_full_season_shots(self, team_name):
        """Return (filtered_shots_df, team_full_display_name) for the full season."""
        team_id, team_full = self.viz._get_team_id(team_name)
        return self.viz._get_shots(team_id), team_full

    def _get_single_game_shots(self, team_name, opponent_name, game_date_string):
        """Return (filtered_shots_df, team_full_name, location_label) for one game."""
        team_id, team_full = self.viz._get_team_id(team_name)
        game_id, is_home   = self.viz._get_game_id_from_matchup(team_name, opponent_name, game_date_string)
        location_label     = "vs" if is_home else "@"
        return self.viz._get_shots(team_id, game_id), team_full, location_label

    @staticmethod
    def _filter_to_player(shots_df, player_name_substring):
        """Filter shots to rows where the player name contains the given substring."""
        return shots_df[
            shots_df["athlete_display_name"].str.contains(player_name_substring, case=False, na=False)
        ]

    # ── Zone efficiency charts ────────────────────────────────────────────────

    def player_season_chart(self, player_name, team_name):
        """Zone efficiency chart for one player's full season."""
        shots_df, team_full   = self._get_full_season_shots(team_name)
        player_shots_df       = self._filter_to_player(shots_df, player_name)
        zone_stats_df         = self.viz._calculate_zone_stats(player_shots_df)
        return zone_efficiency_chart(
            zone_stats_df, player_name,
            _build_splits_subtitle(f"{team_full} · {self.season} season", zone_stats_df)
        )

    def team_season_chart(self, team_name):
        """Zone efficiency chart for one team's full season."""
        shots_df, team_full = self._get_full_season_shots(team_name)
        zone_stats_df       = self.viz._calculate_zone_stats(shots_df)
        return zone_efficiency_chart(
            zone_stats_df, team_full,
            _build_splits_subtitle(f"{self.season} season", zone_stats_df)
        )

    def player_game_chart(self, player_name, team_name, opponent_name, game_date):
        """Zone efficiency chart for one player in a single game."""
        shots_df, team_full, loc = self._get_single_game_shots(team_name, opponent_name, game_date)
        player_shots_df          = self._filter_to_player(shots_df, player_name)
        zone_stats_df            = self.viz._calculate_zone_stats(player_shots_df)
        return zone_efficiency_chart(
            zone_stats_df, player_name,
            _build_splits_subtitle(f"{team_full} {loc} {opponent_name} · {game_date}", zone_stats_df)
        )

    def team_game_chart(self, team_name, opponent_name, game_date):
        """Zone efficiency chart for one team in a single game."""
        shots_df, team_full, loc = self._get_single_game_shots(team_name, opponent_name, game_date)
        zone_stats_df            = self.viz._calculate_zone_stats(shots_df)
        return zone_efficiency_chart(
            zone_stats_df, team_full,
            _build_splits_subtitle(f"{loc} {opponent_name} · {game_date}", zone_stats_df)
        )

    # ── Territory maps ────────────────────────────────────────────────────────

    def plot_team_zone_leaders(self, team_name):
        """Territory map: one annotation per zone showing the top scorer."""
        team_full, zone_leaders_df = self.viz.get_team_zone_leaders(team_name)
        if zone_leaders_df is None or zone_leaders_df.empty:
            return None
        return territory_chart(
            zone_leaders_df,
            f"{team_full} — Territory Map",
            f"Top scorer in each zone · {self.season} season"
        )

    # ── Scatter charts (usually single-game) ─────────────────────────────────

    def player_game_scatter(self, player_name, team_name, opponent_name, game_date):
        """Scatter plot for one player's shots in a single game."""
        shots_df, team_full, loc = self._get_single_game_shots(team_name, opponent_name, game_date)
        field_goals_df           = _prepare_field_goals(self._filter_to_player(shots_df, player_name))
        if field_goals_df is None or field_goals_df.empty:
            return None
        total_makes = int(field_goals_df["scoring_play"].sum())
        return scatter_chart(
            field_goals_df, player_name,
            f"{team_full} {loc} {opponent_name} · {game_date}  ·  {total_makes}/{len(field_goals_df)} FG"
        )

    def team_game_scatter(self, team_name, opponent_name, game_date):
        """Scatter plot for one team's shots in a single game."""
        shots_df, team_full, loc = self._get_single_game_shots(team_name, opponent_name, game_date)
        field_goals_df           = _prepare_field_goals(shots_df)
        if field_goals_df is None or field_goals_df.empty:
            return None
        total_makes = int(field_goals_df["scoring_play"].sum())
        return scatter_chart(
            field_goals_df, team_full,
            f"{loc} {opponent_name} · {game_date}  ·  {total_makes}/{len(field_goals_df)} FG"
        )

    # ── Density charts (full season) ─────────────────────────────────────────

    def player_season_density(self, player_name, team_name):
        """Shot-density heatmap for one player's full season."""
        shots_df, team_full = self._get_full_season_shots(team_name)
        field_goals_df      = _prepare_field_goals(self._filter_to_player(shots_df, player_name))
        if field_goals_df is None or len(field_goals_df) < 5:
            return None
        return density_chart(
            field_goals_df, player_name,
            f"{team_full} · {self.season} season · shot density"
        )

    def team_season_density(self, team_name):
        """Shot-density heatmap for one team's full season."""
        shots_df, team_full = self._get_full_season_shots(team_name)
        field_goals_df      = _prepare_field_goals(shots_df)
        if field_goals_df is None or len(field_goals_df) < 5:
            return None
        return density_chart(
            field_goals_df, team_full,
            f"{self.season} season · shot density"
        )


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
#
# --- Module-level constants ---
# BACKGROUND_COLOR / PANEL_COLOR   str     Dark theme colors matching ui.py.
# MADE_SHOT_COLOR / MISSED_SHOT_COLOR str  Green / red for scatter plots.
# HOOP_SCREEN_Y                    float   Screen y-coordinate of the hoop (5.25 ft from baseline).
# THREE_POINT_RADIUS               float   NCAA 3PT arc radius in feet (22.146).
# CORNER_THREE_X                   float   x-distance where corner 3PT lines end (21.65).
# SCREEN_X_RANGE / SCREEN_Y_RANGE  tuple   Plotly axis display limits.
# ZONE_LABEL_SCREEN_XY             dict    {zone_name: (screen_x, screen_y)} annotation anchors.
# _NCAA_HALF_COURT_LINES           list    Pre-computed (x_coords, y_coords) tuples for court lines.
#
# --- _arc_coords() ---
# center_x / center_y             float   Center of the arc.
# radius                          float   Arc radius.
# angle_start/end_degrees         float   Angular range for the arc.
# angles                          ndarray Linearly spaced angle values in radians.
#
# --- _prepare_field_goals() ---
# raw_shots_df                    DataFrame  Input shot data (may include free throws).
# is_free_throw_mask              Series   True for rows identified as free throws.
# field_goals_df                  DataFrame  Non-FT shots with screen_x, screen_y, shot_distance added.
# screen_x                        Series   = −coordinate_y (left-right on screen).
# screen_y                        Series   = 47 − |coordinate_x| (distance from baseline).
# shot_distance                   Series   Euclidean distance from hoop position.
#
# --- _build_splits_subtitle() ---
# base_subtitle                   str     The base label text (team + season).
# zone_stats_df                   DataFrame  Output of _calculate_zone_stats.
# is_three_pt_zone_mask           Series   True for zone names containing "3PT".
# return value                    str     Full subtitle with shooting split percentages appended.
#
# --- zone_efficiency_chart() ---
# zone_stats_df                   DataFrame  [zone, makes, attempts, fg_pct].
# label_bg_color                  str     Background color for annotation box (green/red/neutral).
# annotation_text                 str     HTML string for the zone label.
#
# --- scatter_chart() ---
# field_goals_df                  DataFrame  Output of _prepare_field_goals.
# made_shots_df / missed_shots_df DataFrame  Filtered subsets.
# has_player_names                bool    True if "athlete_display_name" column is present.
# hover_template                  str     Plotly hover text format string.
#
# --- density_chart() ---
# kde_estimator                   gaussian_kde  SciPy KDE object fitted to screen_x / screen_y.
# density_values                  ndarray  KDE output reshaped to the grid dimensions.
#
# --- PlotlyShotCharts ---
# viz                             MBBZoneEfficiencyVisualizer  The data layer (mbb_viz.py).
# season                          int     Forwarded from viz.season.
