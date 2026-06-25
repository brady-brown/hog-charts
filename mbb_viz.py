"""
mbb_viz.py — Matplotlib zone-efficiency shot charts for NCAA Men's Basketball.

WHY THIS FILE EXISTS
--------------------
This class was the original visualization layer before the site switched to
interactive Plotly charts.  It is still used in two ways:

  1. The Streamlit app calls the zone-classification methods (_classify_zone,
     _calculate_zone_stats, get_team_zone_leaders) which are shared with the
     Plotly layer (shot_charts_plotly.py).

  2. Batch chart generation: team_season_roster_batch() saves PNG zone charts
     for every player on a team roster.

The class loads data from the pre-built parquet files (fast path) or
fetches live from sportsdataverse (slow path for local dev).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplbasketball import Court
from scipy.stats import gaussian_kde


class MBBZoneEfficiencyVisualizer:
    """NCAA Men's Basketball zone-efficiency shot chart visualizer.

    Attributes:
        season           int     Calendar year the season ends (e.g. 2026).
        output_folder    str     Directory where PNG files are saved.
        box_df           DataFrame  Player identity: athlete_id + name + team info.
        pbp_df           DataFrame  Shooting plays (pre-filtered from the parquet).
        game_index       DataFrame  One row per game: game_id, home/away team IDs, date.
        player_map       DataFrame  Deduplicated athlete_id → display_name lookup.
        _date_col        str     Name of the game-date column in the data.
    """

    def __init__(self, season=2026, output_folder="charts/mbb_zone_efficiency", team_filter=None):
        print(f"Loading {season} MBB shot data{f' for {team_filter}' if team_filter else ''}...")
        self.season = season

        # Write charts to /tmp on read-only filesystems (cloud deployments).
        write_root = "/tmp" if not os.access(".", os.W_OK) else "."
        self.output_folder = os.path.join(write_root, f"{output_folder}_{self.season}")

        shots_parquet_path    = os.path.join(os.path.dirname(__file__), "shots_2026.parquet")
        boxscore_parquet_path = os.path.join(os.path.dirname(__file__), "box_2026.parquet")

        if os.path.exists(shots_parquet_path) and os.path.exists(boxscore_parquet_path):
            # ── Fast path: pre-built parquet files (nightly build output) ──────
            self.box_df = pd.read_parquet(boxscore_parquet_path)

            raw_shot_parquet = pd.read_parquet(shots_parquet_path)
            self._date_col = (
                raw_shot_parquet["_date_col"].iloc[0]
                if "_date_col" in raw_shot_parquet.columns else "game_date"
            )

            game_metadata_columns = ["game_id", "home_team_id", "away_team_id", self._date_col]
            self.game_index = (
                raw_shot_parquet[[c for c in game_metadata_columns if c in raw_shot_parquet.columns]]
                .drop_duplicates(subset=["game_id"])
                .copy()
            )

            if team_filter is not None:
                resolved_team_id = self._resolve_team_id(team_filter)
                self.pbp_df = (
                    raw_shot_parquet[raw_shot_parquet["team_id"] == resolved_team_id].copy()
                    if resolved_team_id else raw_shot_parquet.copy()
                )
            else:
                self.pbp_df = raw_shot_parquet.copy()
            del raw_shot_parquet

        else:
            # ── Slow path: live sportsdataverse fetch (local dev, no parquet) ──
            import sportsdataverse.mbb as mbb
            raw_boxscore_df = mbb.load_mbb_player_boxscore(seasons=[self.season], return_as_pandas=True)
            player_box_columns = ["athlete_id", "athlete_display_name", "team_id", "team_display_name"]
            self.box_df = raw_boxscore_df[
                [c for c in player_box_columns if c in raw_boxscore_df.columns]
            ].copy()
            del raw_boxscore_df

            raw_pbp_df = mbb.load_mbb_pbp(seasons=[self.season], return_as_pandas=True)
            self._date_col = "game_date" if "game_date" in raw_pbp_df.columns else "date"

            game_metadata_columns = ["game_id", "home_team_id", "away_team_id", self._date_col]
            self.game_index = (
                raw_pbp_df[[c for c in game_metadata_columns if c in raw_pbp_df.columns]]
                .drop_duplicates(subset=["game_id"])
                .copy()
            )

            shot_columns_to_keep = [
                "game_id", "team_id", "athlete_id_1",
                "coordinate_x", "coordinate_y",
                "scoring_play", "type_text", "text"
            ]
            is_shooting_play_mask = raw_pbp_df["shooting_play"] == True
            if team_filter is not None:
                team_box_match = self.box_df[
                    self.box_df["team_display_name"].str.contains(team_filter, case=False, na=False)
                ]
                if not team_box_match.empty:
                    is_shooting_play_mask = is_shooting_play_mask & (
                        raw_pbp_df["team_id"] == team_box_match.iloc[0]["team_id"]
                    )
            self.pbp_df = raw_pbp_df.loc[
                is_shooting_play_mask,
                [c for c in shot_columns_to_keep if c in raw_pbp_df.columns]
            ].copy()
            del raw_pbp_df

        self.player_map = (
            self.box_df[["athlete_id", "athlete_display_name"]]
            .drop_duplicates(subset=["athlete_id"])
            .copy()
        )
        self.player_map["athlete_id"] = self.player_map["athlete_id"].astype(float)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        print("Data loaded successfully!\n")

    # ── Team resolution helpers ───────────────────────────────────────────────

    def _resolve_team_id(self, team_name_substring):
        """Find a team_id from a partial team name string.  Returns None if not found."""
        matching_rows = self.box_df[
            self.box_df["team_display_name"].str.contains(team_name_substring, case=False, na=False)
        ]
        return matching_rows.iloc[0]["team_id"] if not matching_rows.empty else None

    def _get_team_id(self, team_name_substring):
        """Look up team_id + full display name from a partial name.

        Raises ValueError if the team is not in the dataset.
        """
        matching_box_rows = self.box_df[
            self.box_df["team_display_name"].str.contains(team_name_substring, case=False, na=False)
        ]
        if len(matching_box_rows) == 0:
            raise ValueError(f"Team '{team_name_substring}' not found.")
        return matching_box_rows.iloc[0]["team_id"], matching_box_rows.iloc[0]["team_display_name"]

    def _get_game_id_from_matchup(self, team_name, opponent_name, date_string):
        """Find the game_id for a specific matchup on a specific date.

        Returns (game_id, is_team1_home_bool).
        Raises ValueError with helpful messaging if the game is not found.
        """
        team_id, team_full_name = self._get_team_id(team_name)
        opponent_id, opponent_full_name = self._get_team_id(opponent_name)

        matchup_games_df = self.game_index[
            ((self.game_index["home_team_id"] == team_id) & (self.game_index["away_team_id"] == opponent_id))
            | ((self.game_index["away_team_id"] == team_id) & (self.game_index["home_team_id"] == opponent_id))
        ]

        if matchup_games_df.empty:
            raise ValueError(f"No games found between {team_full_name} and {opponent_full_name}.")

        games_on_date_df = matchup_games_df[
            matchup_games_df[self._date_col].astype(str).str.contains(date_string, na=False)
        ]

        if games_on_date_df.empty:
            available_dates = matchup_games_df[self._date_col].astype(str).str[:10].unique()
            raise ValueError(
                f"No game found on {date_string}.  "
                f"{team_full_name} and {opponent_full_name} played on: {', '.join(available_dates)}"
            )

        target_game_row = games_on_date_df.iloc[0]
        team_was_home   = (target_game_row["home_team_id"] == team_id)
        return target_game_row["game_id"], team_was_home

    def _get_shots(self, team_id, game_id=None):
        """Filter the shot data to one team (and optionally one game), merge player names."""
        if "shooting_play" in self.pbp_df.columns:
            filtered_shots_df = self.pbp_df[
                (self.pbp_df["team_id"] == team_id) & (self.pbp_df["shooting_play"] == True)
            ].copy()
        else:
            filtered_shots_df = self.pbp_df[self.pbp_df["team_id"] == team_id].copy()

        if game_id:
            filtered_shots_df = filtered_shots_df[filtered_shots_df["game_id"] == game_id]

        filtered_shots_df = filtered_shots_df.merge(
            self.player_map, left_on="athlete_id_1", right_on="athlete_id", how="left"
        )
        return filtered_shots_df

    # ── Zone classification ───────────────────────────────────────────────────

    @staticmethod
    def _classify_zone(shot_row):
        """Assign a shot to one of 12 named zones using court coordinates.

        Coordinate convention (ESPN / sportsdataverse):
            coordinate_x: distance from center-court baseline (0 = baseline, 47 = halfcourt)
            coordinate_y: lateral position (positive = left, negative = right)

        Zone boundaries match the site's interactive shot chart and build_site.py.
        Returns a zone name string.
        """
        raw_x = abs(shot_row["coordinate_x"])
        lateral_y = shot_row["coordinate_y"]

        if pd.isna(raw_x) or pd.isna(lateral_y):
            return "Unknown"

        # Shift x so that it measures distance from the basket (basket sits at x=41.75).
        x_from_basket = 41.75 - raw_x
        shot_distance = np.sqrt(x_from_basket ** 2 + lateral_y ** 2)
        shot_angle    = np.degrees(np.arctan2(lateral_y, x_from_basket))

        if shot_distance >= 40:
            return "Heave"
        if shot_distance < 5:
            return "At Rim"
        if raw_x >= 28 and abs(lateral_y) <= 6 and shot_distance >= 5:
            return "Paint (Non-Rim)"

        is_three_pointer = shot_distance >= 22.15

        if is_three_pointer:
            if shot_angle > 55:   return "Left Corner 3PT"
            if shot_angle > 25:   return "Left Wing 3PT"
            if shot_angle > -25:  return "Center 3PT"
            if shot_angle > -55:  return "Right Wing 3PT"
            return "Right Corner 3PT"
        else:
            if shot_angle > 60:   return "Left Baseline Mid"
            if shot_angle > 25:   return "Left Mid-Range"
            if shot_angle > -25:  return "Center Mid-Range"
            if shot_angle > -60:  return "Right Mid-Range"
            return "Right Baseline Mid"

    def _calculate_zone_stats(self, shot_group_df):
        """Compute makes, attempts, and FG% by zone for a set of shots.

        Free throws are excluded.  Returns a DataFrame with columns:
        zone, makes, attempts, fg_pct.
        """
        is_free_throw_mask = (
            shot_group_df["type_text"].str.contains("Free", case=False, na=False)
            | shot_group_df["text"].str.contains("free throw", case=False, na=False)
        )
        field_goals_df = shot_group_df[~is_free_throw_mask].copy()
        field_goals_df["zone"] = field_goals_df.apply(self._classify_zone, axis=1)

        zone_stats_df = (
            field_goals_df.groupby("zone")
            .agg(makes=("scoring_play", "sum"), attempts=("scoring_play", "count"))
            .reset_index()
        )
        zone_stats_df["fg_pct"] = (zone_stats_df["makes"] / zone_stats_df["attempts"]) * 100
        return zone_stats_df

    # ── Internal rendering engines ────────────────────────────────────────────

    def _scatter_plot(self, shots_df, chart_title, chart_subtitle, return_fig=False):
        """Scatter shot chart: individual dots colored by make (green) / miss (red)."""
        if len(shots_df) == 0:
            return None

        is_free_throw_mask = (
            shots_df["type_text"].str.contains("Free", case=False, na=False)
            | shots_df["text"].str.contains("free throw", case=False, na=False)
        )
        field_goals_df = shots_df[~is_free_throw_mask].copy()
        if field_goals_df.empty:
            return None

        field_goals_df = field_goals_df.dropna(subset=["coordinate_x", "coordinate_y"])
        field_goals_df["plot_x"] = field_goals_df["coordinate_x"].abs()
        field_goals_df["plot_y"] = field_goals_df["coordinate_y"]

        made_shots_df   = field_goals_df[field_goals_df["scoring_play"] == True]
        missed_shots_df = field_goals_df[field_goals_df["scoring_play"] == False]

        total_attempts   = len(field_goals_df)
        total_makes      = len(made_shots_df)
        field_goal_pct   = (total_makes / total_attempts * 100) if total_attempts > 0 else 0

        court = Court(court_type="ncaa", origin="center", units="ft")
        fig, ax = court.draw(orientation="h", court_color="#D2A679", line_color="white")
        fig.set_size_inches(12, 10)
        ax.set_xlim(0, 47)
        ax.set_ylim(-27, 34)

        ax.scatter(missed_shots_df["plot_x"], missed_shots_df["plot_y"],
                   c="#E53935", s=25, alpha=0.7, label=f"Miss ({len(missed_shots_df)})",
                   zorder=3, linewidths=0.3, edgecolors="white")
        ax.scatter(made_shots_df["plot_x"], made_shots_df["plot_y"],
                   c="#43A047", s=25, alpha=0.85, label=f"Make ({total_makes})",
                   zorder=4, linewidths=0.3, edgecolors="white")
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

        shot_summary_text = f" FG: {total_makes}/{total_attempts}  ({field_goal_pct:.1f}%) "
        ax.text(46, 30, shot_summary_text, fontsize=10, fontweight="bold",
                ha="right", va="top", family="monospace",
                bbox=dict(boxstyle="square,pad=0.5", facecolor="#F8F9FA", edgecolor="black", alpha=0.9))

        plt.title(f"{chart_title}\n{chart_subtitle}", fontsize=14, fontweight="bold", pad=15)

        if return_fig:
            return fig
        plt.close(fig)
        return None

    def _density_plot(self, shots_df, chart_title, chart_subtitle, return_fig=False):
        """KDE heatmap showing shot-frequency hot spots for season-level data."""
        if len(shots_df) == 0:
            return None

        is_free_throw_mask = (
            shots_df["type_text"].str.contains("Free", case=False, na=False)
            | shots_df["text"].str.contains("free throw", case=False, na=False)
        )
        field_goals_df = shots_df[~is_free_throw_mask].dropna(subset=["coordinate_x", "coordinate_y"]).copy()
        if len(field_goals_df) < 5:
            return None

        field_goals_df["plot_x"] = field_goals_df["coordinate_x"].abs()
        field_goals_df["plot_y"] = field_goals_df["coordinate_y"]

        x_coords = field_goals_df["plot_x"].values
        y_coords = field_goals_df["plot_y"].values

        court = Court(court_type="ncaa", origin="center", units="ft")
        fig, ax = court.draw(orientation="h", court_color="#1a1a2e", line_color="white")
        fig.set_size_inches(12, 10)
        ax.set_xlim(0, 47)
        ax.set_ylim(-27, 34)

        # Kernel density estimation on a half-court grid.
        grid_x_coords = np.linspace(0, 47, 200)
        grid_y_coords = np.linspace(-25, 25, 200)
        grid_xx, grid_yy = np.meshgrid(grid_x_coords, grid_y_coords)
        grid_positions   = np.vstack([grid_xx.ravel(), grid_yy.ravel()])
        kde_estimator    = gaussian_kde(np.vstack([x_coords, y_coords]), bw_method=0.15)
        density_values   = kde_estimator(grid_positions).reshape(grid_xx.shape)

        contour_fill = ax.contourf(grid_xx, grid_yy, density_values, levels=14,
                                   cmap="YlOrRd", alpha=0.75, zorder=2)
        color_bar = fig.colorbar(contour_fill, ax=ax, shrink=0.6, pad=0.02)
        color_bar.set_label("Shot Density", fontsize=10)
        color_bar.set_ticks([])

        total_makes    = int(field_goals_df["scoring_play"].sum())
        total_attempts = len(field_goals_df)
        fg_pct         = total_makes / total_attempts * 100 if total_attempts > 0 else 0
        ax.text(46, 30, f" {total_makes}/{total_attempts}  ({fg_pct:.1f}% FG) ",
                fontsize=10, fontweight="bold", ha="right", va="top", family="monospace",
                bbox=dict(boxstyle="square,pad=0.5", facecolor="#F8F9FA", edgecolor="white", alpha=0.9))

        plt.title(f"{chart_title}\n{chart_subtitle}", fontsize=14, fontweight="bold",
                  pad=15, color="white")
        fig.patch.set_facecolor("#1a1a2e")

        if return_fig:
            return fig
        plt.close(fig)
        return None

    def _plot(self, shots_df, chart_title, chart_subtitle, output_filename,
              specific_folder=None, show_summary=False, return_fig=False):
        """Core matplotlib zone-efficiency chart engine.

        Draws zone stat labels (makes/attempts/FG%) at predefined court coordinates.
        Labels are color-coded: green if FG% ≥ 45%, red if ≤ 35%, neutral otherwise.
        """
        if len(shots_df) == 0:
            print(f"No shot data found for {chart_title}. Skipping.")
            return

        zone_stats_df = self._calculate_zone_stats(shots_df)

        court = Court(court_type="ncaa", origin="center", units="ft")
        fig, ax = court.draw(orientation="h", court_color="#D2A679", line_color="white")
        fig.set_size_inches(12, 10)
        ax.set_xlim(0, 47)
        ax.set_ylim(-27, 34)

        plt.title(f"{chart_title} - Zone Efficiency\n{chart_subtitle}",
                  fontsize=14, fontweight="bold", pad=15)

        if show_summary:
            valid_zones_df  = zone_stats_df[~zone_stats_df["zone"].isin(["Heave", "Unknown"])]
            total_makes     = valid_zones_df["makes"].sum()
            total_attempts  = valid_zones_df["attempts"].sum()
            overall_fg_pct  = (total_makes / total_attempts) * 100 if total_attempts > 0 else 0

            three_point_rows_df = valid_zones_df[
                valid_zones_df["zone"].astype(str).str.contains("3PT", na=False)
            ]
            three_pt_makes    = three_point_rows_df["makes"].sum()
            three_pt_attempts = three_point_rows_df["attempts"].sum()
            three_pt_pct      = (three_pt_makes / three_pt_attempts) * 100 if three_pt_attempts > 0 else 0

            two_point_rows_df = valid_zones_df[
                ~valid_zones_df["zone"].astype(str).str.contains("3PT", na=False)
            ]
            two_pt_makes    = two_point_rows_df["makes"].sum()
            two_pt_attempts = two_point_rows_df["attempts"].sum()
            two_pt_pct      = (two_pt_makes / two_pt_attempts) * 100 if two_pt_attempts > 0 else 0

            summary_text = (
                f" SHOOTING SPLITS \n"
                f"-----------------\n"
                f"Overall: {overall_fg_pct:.1f}% ({int(total_makes)}/{int(total_attempts)})\n"
                f"2PT FG:  {two_pt_pct:.1f}% ({int(two_pt_makes)}/{int(two_pt_attempts)})\n"
                f"3PT FG:  {three_pt_pct:.1f}% ({int(three_pt_makes)}/{int(three_pt_attempts)})"
            )
            ax.text(46, 27, summary_text, fontsize=10, fontweight="bold",
                    ha="right", va="bottom", family="monospace",
                    bbox=dict(boxstyle="square,pad=0.6", facecolor="#F8F9FA",
                              edgecolor="black", alpha=0.9))

        # Screen coordinates for each zone's label.
        zone_label_positions = {
            "At Rim":            (42.25,  0),
            "Paint (Non-Rim)":   (34,     0),
            "Center Mid-Range":  (23,     0),
            "Left Mid-Range":    (28,    12),
            "Right Mid-Range":   (28,   -12),
            "Left Baseline Mid": (38,    16),
            "Right Baseline Mid":(38,   -16),
            "Center 3PT":        (13,     0),
            "Left Wing 3PT":     (15,    20),
            "Right Wing 3PT":    (15,   -20),
            "Left Corner 3PT":   (40,    24),
            "Right Corner 3PT":  (40,   -24),
        }

        for _, zone_row in zone_stats_df.iterrows():
            zone_name = zone_row["zone"]
            if zone_name not in zone_label_positions:
                continue
            label_x, label_y = zone_label_positions[zone_name]
            zone_makes    = zone_row["makes"]
            zone_attempts = zone_row["attempts"]
            zone_fg_pct   = zone_row["fg_pct"]

            label_bg_color = "#C8E6C9" if zone_fg_pct >= 45 else ("#FFCDD2" if zone_fg_pct <= 35 else "#F5F5F5")
            label_text     = f"{zone_name}\n{int(zone_makes)}/{int(zone_attempts)}\n{zone_fg_pct:.1f}%"

            ax.text(label_x, label_y, label_text, fontsize=9, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=label_bg_color,
                              edgecolor="black", alpha=0.9))

        if return_fig:
            return fig

        save_directory = specific_folder if specific_folder else self.output_folder
        output_filepath = f"{save_directory}/{output_filename}.png"
        plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ── Territory map ─────────────────────────────────────────────────────────

    def plot_team_zone_leaders(self, team_name, filename=None, specific_folder=None,
                                return_fig=False, show_summary=False):
        """Plot a 'Territory Map' showing the top-scoring player in each zone."""
        team_full_name, zone_leaders_df = self.get_team_zone_leaders(team_name)
        if zone_leaders_df is None or zone_leaders_df.empty:
            return

        court = Court(court_type="ncaa", origin="center", units="ft")
        fig, ax = court.draw(orientation="h", court_color="#D2A679", line_color="white")
        fig.set_size_inches(12, 10)
        ax.set_xlim(0, 47)
        ax.set_ylim(-27, 34)

        plt.title(f"{team_full_name} - Top Scorers by Zone (Territory Map)\n{self.season} Season",
                  fontsize=14, fontweight="bold", pad=15)

        zone_label_positions = {
            "At Rim":            (42.25,  0),
            "Paint (Non-Rim)":   (34,     0),
            "Center Mid-Range":  (20,     0),
            "Left Mid-Range":    (28,    12),
            "Right Mid-Range":   (28,   -12),
            "Left Baseline Mid": (38,    16),
            "Right Baseline Mid":(38,   -16),
            "Center 3PT":        (13,     0),
            "Left Wing 3PT":     (15,    20),
            "Right Wing 3PT":    (15,   -20),
            "Left Corner 3PT":   (40,    24),
            "Right Corner 3PT":  (40,   -24),
        }

        for _, leader_row in zone_leaders_df.iterrows():
            zone_name = leader_row["zone"]
            if zone_name not in zone_label_positions:
                continue
            label_x, label_y = zone_label_positions[zone_name]
            label_text = f"{zone_name}\n{leader_row['athlete_display_name']}\n{leader_row['pts']} pts"
            ax.text(label_x, label_y, label_text, fontsize=8, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA",
                              edgecolor="black", alpha=0.9))

        if return_fig:
            return fig

        save_directory = specific_folder if specific_folder else self.output_folder
        plt.savefig(f"{save_directory}/{filename}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ── Public methods ────────────────────────────────────────────────────────

    def team_game_chart(self, team_name, opponent_name, date, return_fig=False):
        """Zone chart for a single game."""
        team_id, team_full = self._get_team_id(team_name)
        game_id, is_home   = self._get_game_id_from_matchup(team_name, opponent_name, date)
        shots_df = self._get_shots(team_id, game_id)
        location_label = "vs" if is_home else "@"
        output_filename = f"{self.season}_{team_full.replace(' ', '_')}_{location_label}_{opponent_name.replace(' ', '_')}_{date}"
        return self._plot(shots_df, team_full,
                          f"{location_label} {opponent_name.title()} ({date})",
                          output_filename, return_fig=return_fig)

    def team_season_chart(self, team_name, return_fig=False):
        """Zone chart for the full season."""
        team_id, team_full = self._get_team_id(team_name)
        shots_df = self._get_shots(team_id)
        output_filename = f"{self.season}_{team_full.replace(' ', '_')}_Season"
        return self._plot(shots_df, team_full, f"{self.season} Full Season",
                          output_filename, show_summary=True, return_fig=return_fig)

    def player_game_chart(self, player_name, team_name, opponent_name, date, return_fig=False):
        """Zone chart for one player in a single game."""
        team_id, team_full = self._get_team_id(team_name)
        game_id, is_home   = self._get_game_id_from_matchup(team_name, opponent_name, date)
        shots_df           = self._get_shots(team_id, game_id)
        player_shots_df    = shots_df[
            shots_df["athlete_display_name"].str.contains(player_name, case=False, na=False)
        ]
        location_label  = "vs" if is_home else "@"
        output_filename = f"{self.season}_{player_name.replace(' ', '_')}_{location_label}_{opponent_name.replace(' ', '_')}_{date}"
        return self._plot(player_shots_df, player_name,
                          f"{team_full} {location_label} {opponent_name.title()} ({date})",
                          output_filename, return_fig=return_fig)

    def player_season_chart(self, player_name, team_name, return_fig=False):
        """Zone chart for one player's full season."""
        team_id, team_full = self._get_team_id(team_name)
        shots_df           = self._get_shots(team_id)
        player_shots_df    = shots_df[
            shots_df["athlete_display_name"].str.contains(player_name, case=False, na=False)
        ]
        output_filename = f"{self.season}_{player_name.replace(' ', '_')}_Season"
        return self._plot(player_shots_df, player_name,
                          f"{team_full} - {self.season} Full Season",
                          output_filename, show_summary=True, return_fig=return_fig)

    def team_season_roster_batch(self, team_name):
        """Save a zone chart PNG for every player on the team's roster."""
        team_id, team_full = self._get_team_id(team_name)
        all_team_shots_df  = self._get_shots(team_id)
        roster_player_names = all_team_shots_df["athlete_display_name"].dropna().unique()

        batch_output_folder = f"{self.output_folder}/{self.season}_{team_full.replace(' ', '_')}_Roster_Season"
        if not os.path.exists(batch_output_folder):
            os.makedirs(batch_output_folder)

        print(f"\nProcessing {len(roster_player_names)} players for {team_full} ({self.season})...")
        for player_display_name in sorted(roster_player_names):
            player_shots_df = all_team_shots_df[
                all_team_shots_df["athlete_display_name"] == player_display_name
            ]
            if len(player_shots_df) > 0:
                output_filename = f"{self.season}_{player_display_name.replace(' ', '_')}"
                self._plot(player_shots_df, player_display_name,
                           f"{team_full} - {self.season} Season Update",
                           output_filename, specific_folder=batch_output_folder,
                           show_summary=True)
                print(f"  Processed: {player_display_name}")
        print(f"\nBatch complete! All player charts saved to: {batch_output_folder}")

    def get_team_zone_leaders(self, team_name):
        """Return (team_full_name, DataFrame) of top scorer per zone.

        The DataFrame has columns: zone, athlete_display_name, pts.
        Used by the Plotly layer for territory maps.
        """
        team_id, team_full = self._get_team_id(team_name)
        all_shots_df       = self._get_shots(team_id)

        if all_shots_df.empty:
            print(f"No shot data found for {team_full}.")
            return None, None

        all_shots_df = all_shots_df.copy()
        all_shots_df["zone"] = all_shots_df.apply(self._classify_zone, axis=1)

        made_shots_df = all_shots_df[all_shots_df["scoring_play"] == True].copy()
        made_shots_df["pts"] = made_shots_df["zone"].apply(lambda z: 3 if "3PT" in str(z) else 2)

        zone_player_points_df = (
            made_shots_df.groupby(["zone", "athlete_display_name"])["pts"].sum().reset_index()
        )
        zone_leaders_df = (
            zone_player_points_df.sort_values("pts", ascending=False)
            .groupby("zone").head(1)
            .reset_index(drop=True)
        )
        return team_full, zone_leaders_df

    # ── Density + scatter public methods ─────────────────────────────────────

    def player_season_density(self, player_name, team_name, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        shots_df = self._get_shots(team_id)
        player_shots_df = shots_df[
            shots_df["athlete_display_name"].str.contains(player_name, case=False, na=False)
        ]
        return self._density_plot(player_shots_df, player_name,
                                   f"{team_full} — {self.season} Full Season", return_fig=return_fig)

    def team_season_density(self, team_name, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        shots_df = self._get_shots(team_id)
        return self._density_plot(shots_df, team_full,
                                   f"{self.season} Full Season", return_fig=return_fig)

    def player_season_scatter(self, player_name, team_name, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        shots_df = self._get_shots(team_id)
        player_shots_df = shots_df[
            shots_df["athlete_display_name"].str.contains(player_name, case=False, na=False)
        ]
        return self._scatter_plot(player_shots_df, player_name,
                                   f"{team_full} — {self.season} Full Season", return_fig=return_fig)

    def player_game_scatter(self, player_name, team_name, opponent_name, date, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        game_id, is_home   = self._get_game_id_from_matchup(team_name, opponent_name, date)
        shots_df           = self._get_shots(team_id, game_id)
        player_shots_df    = shots_df[
            shots_df["athlete_display_name"].str.contains(player_name, case=False, na=False)
        ]
        location_label = "vs" if is_home else "@"
        return self._scatter_plot(player_shots_df, player_name,
                                   f"{team_full} {location_label} {opponent_name} ({date})",
                                   return_fig=return_fig)

    def team_season_scatter(self, team_name, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        shots_df = self._get_shots(team_id)
        return self._scatter_plot(shots_df, team_full,
                                   f"{self.season} Full Season", return_fig=return_fig)

    def team_game_scatter(self, team_name, opponent_name, date, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        game_id, is_home   = self._get_game_id_from_matchup(team_name, opponent_name, date)
        shots_df           = self._get_shots(team_id, game_id)
        location_label = "vs" if is_home else "@"
        return self._scatter_plot(shots_df, team_full,
                                   f"{location_label} {opponent_name} ({date})",
                                   return_fig=return_fig)

    # ── Utility / diagnostics ─────────────────────────────────────────────────

    def calculate_zone_areas(self, resolution=0.1):
        """Estimate square footage of each zone via grid approximation.

        Useful for verifying that zone boundaries are geometrically balanced.
        """
        print("Calculating zone areas using grid approximation...")
        grid_x = np.arange(0, 47, resolution)
        grid_y = np.arange(-25, 25, resolution)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
        grid_df = pd.DataFrame({
            "coordinate_x": grid_xx.flatten(),
            "coordinate_y": grid_yy.flatten()
        })
        grid_df["zone"] = grid_df.apply(self._classify_zone, axis=1)
        point_area_sq_ft = resolution ** 2
        zone_area_df     = (grid_df["zone"].value_counts() * point_area_sq_ft).reset_index()
        zone_area_df.columns = ["Zone", "Estimated Area (sq ft)"]
        return zone_area_df.round(1)

    def plot_zone_boundaries(self, resolution=0.2, filename="zone_definitions_map"):
        """Visualize zone boundaries by classifying a dense grid of court coordinates."""
        print(f"Generating grid points at {resolution}ft resolution…")
        grid_x = np.arange(0, 47.1, resolution)
        grid_y = np.arange(-25.1, 25.1, resolution)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
        grid_df = pd.DataFrame({
            "coordinate_x": grid_xx.flatten(),
            "coordinate_y": grid_yy.flatten()
        })
        print(f"Classifying {len(grid_df):,} grid points…")
        grid_df["zone"] = grid_df.apply(self._classify_zone, axis=1)
        grid_df = grid_df[grid_df["zone"] != "Unknown"]

        zone_color_map = {
            "At Rim":            "#D32F2F",
            "Paint (Non-Rim)":   "#F57C00",
            "Center Mid-Range":  "#FBC02D",
            "Left Mid-Range":    "#7CB342",
            "Right Mid-Range":   "#7CB342",
            "Left Baseline Mid": "#388E3C",
            "Right Baseline Mid":"#388E3C",
            "Center 3PT":        "#0288D1",
            "Left Wing 3PT":     "#1976D2",
            "Right Wing 3PT":    "#1976D2",
            "Left Corner 3PT":   "#512DA8",
            "Right Corner 3PT":  "#512DA8",
            "Heave":             "#616161",
        }

        court = Court(court_type="ncaa", origin="center", units="ft")
        fig, ax = court.draw(orientation="h", court_color="#E0C8B0", line_color="black")
        fig.set_size_inches(14, 12)
        ax.set_xlim(-5, 52)
        ax.set_ylim(-30, 30)
        plt.title("NCAA Zone Definition Boundaries\n(Visualized via Grid Classification)",
                  fontsize=16, fontweight="bold", pad=20)

        already_in_legend = set()
        for zone_name, dot_color in zone_color_map.items():
            zone_points = grid_df[grid_df["zone"] == zone_name]
            if not zone_points.empty:
                base_name    = zone_name.replace("Left ", "").replace("Right ", "")
                legend_label = base_name if base_name not in already_in_legend else "_nolegend_"
                if legend_label != "_nolegend_":
                    already_in_legend.add(base_name)
                ax.scatter(zone_points["coordinate_x"], zone_points["coordinate_y"],
                           c=dot_color, s=3, alpha=0.4, marker="o",
                           label=legend_label, zorder=2)

        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0,
                  title="Zone Categories", fontsize=10, markerscale=3)
        output_path = f"{self.output_folder}/{filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Zone boundary map saved to: {output_path}")


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
#
# --- MBBZoneEfficiencyVisualizer.__init__() ---
# season                       int     Calendar year the season ends.
# output_folder                str     Directory for saved PNG files.
# shots_parquet_path           str     Path to shots_{season}.parquet.
# boxscore_parquet_path        str     Path to box_{season}.parquet.
# box_df                       DataFrame  athlete_id/name + team_id/name; player identity lookup.
# raw_shot_parquet             DataFrame  All shooting rows from the parquet (before filtering).
# game_index                   DataFrame  One row per game: game_id, home/away IDs, date.
# pbp_df                       DataFrame  Shooting plays (filtered from the parquet).
# player_map                   DataFrame  Deduplicated athlete_id → display_name.
# _date_col                    str     "game_date" or "date" depending on sportsdataverse version.
#
# --- _classify_zone() ---
# shot_row                     Series  One row of shot data with coordinate_x, coordinate_y.
# raw_x / lateral_y            float   Shot coordinates from ESPN (raw_x is always positive).
# x_from_basket                float   Distance from x baseline shifted to measure from hoop.
# shot_distance                float   Euclidean distance from the basket.
# shot_angle                   float   Angle in degrees (arctan2); positive = left side.
# is_three_pointer             bool    True if shot_distance >= 22.15 ft (NCAA 3PT line).
#
# --- _calculate_zone_stats() ---
# shot_group_df                DataFrame  Shots to analyze (team, player, or game subset).
# is_free_throw_mask           Series   Boolean; True for free-throw rows to exclude.
# field_goals_df               DataFrame  Non-FT shots with zone labels attached.
# zone_stats_df                DataFrame  [zone, makes, attempts, fg_pct].
#
# --- _scatter_plot() ---
# field_goals_df               DataFrame  Non-FT shots with plot_x / plot_y coords.
# made_shots_df / missed_shots_df  DataFrame  Filtered subsets for green/red dots.
# total_attempts / total_makes int     Shot totals for the summary text box.
# field_goal_pct               float   Overall FG% for the summary text.
#
# --- _density_plot() ---
# x_coords / y_coords          ndarray  Shot coordinates for KDE input.
# grid_xx / grid_yy            ndarray  Meshgrid of the half-court display area.
# kde_estimator                gaussian_kde  SciPy KDE object fitted to the shots.
# density_values               ndarray  KDE output shaped to the grid.
# contour_fill                 QuadContourSet  Matplotlib contourf result.
#
# --- _plot() ---
# zone_stats_df                DataFrame  Output of _calculate_zone_stats.
# zone_label_positions         dict     {zone_name: (screen_x, screen_y)}.
# label_bg_color               str     "#C8E6C9" (green), "#FFCDD2" (red), or "#F5F5F5" (neutral).
#
# --- get_team_zone_leaders() ---
# made_shots_df                DataFrame  Subset where scoring_play == True.
# pts                          int     2 for non-3PT zones, 3 for 3PT zones.
# zone_player_points_df        DataFrame  Total pts per (zone, player).
# zone_leaders_df              DataFrame  Top scorer per zone (one row per zone).
#
# --- calculate_zone_areas() ---
# grid_x / grid_y              ndarray  1D coordinate ranges at the given resolution.
# grid_xx / grid_yy / grid_df  ndarray / DataFrame  Full half-court grid for zone classification.
# point_area_sq_ft             float   Area of one grid point in sq ft (resolution^2).
