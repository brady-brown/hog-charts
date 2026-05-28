import sportsdataverse.mbb as mbb
import pandas as pd
import matplotlib.pyplot as plt
from mplbasketball import Court
import numpy as np
import os
from scipy.stats import gaussian_kde


class MBBZoneEfficiencyVisualizer:
    def __init__(self, season=2026, output_folder="charts/mbb_zone_efficiency", team_filter=None):
        print(f"Loading {season} MBB shot data{f' for {team_filter}' if team_filter else ''}...")
        self.season = season

        base = "/tmp" if not os.access(".", os.W_OK) else "."
        self.output_folder = os.path.join(base, f"{output_folder}_{self.season}")

        shots_file = os.path.join(os.path.dirname(__file__), "shots_2026.parquet")
        box_file   = os.path.join(os.path.dirname(__file__), "box_2026.parquet")

        if os.path.exists(shots_file) and os.path.exists(box_file):
            # ── Fast path: load from pre-built parquet files ─────────────────
            self.box_df = pd.read_parquet(box_file)
            raw = pd.read_parquet(shots_file)
            self._date_col = raw["_date_col"].iloc[0] if "_date_col" in raw.columns else "game_date"
            game_cols = ["game_id", "home_team_id", "away_team_id", self._date_col]
            self.game_index = (
                raw[[c for c in game_cols if c in raw.columns]]
                .drop_duplicates(subset=["game_id"])
                .copy()
            )
            if team_filter is not None:
                team_id = self._resolve_team_id(team_filter)
                self.pbp_df = raw[raw["team_id"] == team_id].copy() if team_id else raw.copy()
            else:
                self.pbp_df = raw.copy()
            del raw
        else:
            # ── Slow path: fetch from sportsdataverse (local dev only) ────────
            raw_box = mbb.load_mbb_player_boxscore(seasons=[self.season], return_as_pandas=True)
            box_cols = ["athlete_id", "athlete_display_name", "team_id", "team_display_name"]
            self.box_df = raw_box[[c for c in box_cols if c in raw_box.columns]].copy()
            del raw_box

            raw_pbp = mbb.load_mbb_pbp(seasons=[self.season], return_as_pandas=True)
            self._date_col = "game_date" if "game_date" in raw_pbp.columns else "date"
            game_cols = ["game_id", "home_team_id", "away_team_id", self._date_col]
            self.game_index = (
                raw_pbp[[c for c in game_cols if c in raw_pbp.columns]]
                .drop_duplicates(subset=["game_id"])
                .copy()
            )
            shot_cols = ["game_id", "team_id", "athlete_id_1", "coordinate_x",
                         "coordinate_y", "scoring_play", "type_text", "text"]
            available = [c for c in shot_cols if c in raw_pbp.columns]
            mask = raw_pbp["shooting_play"] == True
            if team_filter is not None:
                team_box = self.box_df[
                    self.box_df["team_display_name"].str.contains(team_filter, case=False, na=False)
                ]
                if not team_box.empty:
                    mask = mask & (raw_pbp["team_id"] == team_box.iloc[0]["team_id"])
            self.pbp_df = raw_pbp.loc[mask, available].copy()
            del raw_pbp

        self.player_map = (
            self.box_df[["athlete_id", "athlete_display_name"]]
            .drop_duplicates(subset=["athlete_id"])
            .copy()
        )
        self.player_map["athlete_id"] = self.player_map["athlete_id"].astype(float)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        print("Data loaded successfully!\n")

    def _resolve_team_id(self, team_name):
        row = self.box_df[self.box_df["team_display_name"].str.contains(team_name, case=False, na=False)]
        return row.iloc[0]["team_id"] if not row.empty else None

    def _get_team_id(self, team_name):
        """Helper to find team ID from string."""
        team_box = self.box_df[
            self.box_df["team_display_name"].str.contains(
                team_name, case=False, na=False
            )
        ]
        if len(team_box) == 0:
            raise ValueError(f"❌ Error: Team '{team_name}' not found.")
        return team_box.iloc[0]["team_id"], team_box.iloc[0]["team_display_name"]

    def _get_game_id_from_matchup(self, team_name, opponent_name, date):
        """Finds a game_id based on team matchups and a specific date string."""
        team_id, team_full = self._get_team_id(team_name)
        opp_id, opp_full = self._get_team_id(opponent_name)

        matchups = self.game_index[
            ((self.game_index["home_team_id"] == team_id) & (self.game_index["away_team_id"] == opp_id))
            | ((self.game_index["away_team_id"] == team_id) & (self.game_index["home_team_id"] == opp_id))
        ]

        if matchups.empty:
            raise ValueError(
                f"❌ Error: No games found between {team_full} and {opp_full} in this dataset."
            )

        specific_game = matchups[
            matchups[self._date_col].astype(str).str.contains(date, na=False)
        ]

        if specific_game.empty:
            available_dates = matchups[self._date_col].astype(str).str[:10].unique()
            raise ValueError(
                f"❌ Error: No game found on {date}. {team_full} and {opp_full} played on: {', '.join(available_dates)}"
            )

        game_row = specific_game.iloc[0]
        is_home = game_row["home_team_id"] == team_id
        return game_row["game_id"], is_home

    def _get_shots(self, team_id, game_id=None):
        """Helper to filter shot data. If game_id is None, returns full season."""
        shots = self.pbp_df[
            (self.pbp_df["team_id"] == team_id) & (self.pbp_df["shooting_play"] == True)
        ].copy()
        if game_id:
            shots = shots[shots["game_id"] == game_id]

        shots = shots.merge(
            self.player_map, left_on="athlete_id_1", right_on="athlete_id", how="left"
        )
        return shots

    @staticmethod
    def _classify_zone(row):
        """Categorizes an NCAA shot using distances and radial angles from the hoop."""
        x = abs(row["coordinate_x"])
        y = row["coordinate_y"]

        if pd.isna(x) or pd.isna(y):
            return "Unknown"

        x_shifted = 41.75 - x
        dist = np.sqrt(x_shifted**2 + y**2)
        angle = np.degrees(np.arctan2(y, x_shifted))

        if dist >= 40:
            return "Heave"

        if dist < 5:
            return "At Rim"

        # NCAA Paint definition: inside lane lines (12ft wide -> +/- 6), between baseline and FT line
        if x >= 28 and abs(y) <= 6 and dist >= 5:
            return "Paint (Non-Rim)"

        # NCAA 3-Point Zone (Current line is roughly 22.15 ft)
        is_3pt = dist >= 22.15

        if is_3pt:
            # Pushed corners deeper, widened center slightly for balance
            if angle > 55:
                return "Left Corner 3PT"
            elif angle > 25:
                return "Left Wing 3PT"
            elif angle > -25:
                return "Center 3PT"
            elif angle > -55:
                return "Right Wing 3PT"
            else:
                return "Right Corner 3PT"
        else:
            # Mid-Range Zones
            # Widened center to wrap around the paint, pushed baseline deeper
            if angle > 60:
                return "Left Baseline Mid"
            elif angle > 25:
                return "Left Mid-Range"
            elif angle > -25:
                return "Center Mid-Range"
            elif angle > -60:
                return "Right Mid-Range"
            else:
                return "Right Baseline Mid"

    def _calculate_zone_stats(self, player_shots):
        """Calculates makes, attempts, and FG% for each zone."""
        is_ft = (
            player_shots["type_text"].str.contains("Free", case=False, na=False)
        ) | (player_shots["text"].str.contains("free throw", case=False, na=False))
        fgs = player_shots[~is_ft].copy()

        fgs["zone"] = fgs.apply(self._classify_zone, axis=1)

        zone_stats = (
            fgs.groupby("zone")
            .agg(makes=("scoring_play", "sum"), attempts=("scoring_play", "count"))
            .reset_index()
        )

        zone_stats["fg_pct"] = (zone_stats["makes"] / zone_stats["attempts"]) * 100
        return zone_stats

    def _scatter_plot(self, shots_df, title, subtitle, return_fig=False):
        """Scatter shot chart — individual dots colored by make/miss."""
        if len(shots_df) == 0:
            return None

        is_ft = (
            shots_df["type_text"].str.contains("Free", case=False, na=False)
        ) | (shots_df["text"].str.contains("free throw", case=False, na=False))
        fgs = shots_df[~is_ft].copy()

        if fgs.empty:
            return None

        fgs = fgs.dropna(subset=["coordinate_x", "coordinate_y"])
        fgs["plot_x"] = fgs["coordinate_x"].abs()
        fgs["plot_y"] = fgs["coordinate_y"]

        makes = fgs[fgs["scoring_play"] == True]
        misses = fgs[fgs["scoring_play"] == False]

        total = len(fgs)
        n_makes = len(makes)
        fg_pct = (n_makes / total * 100) if total > 0 else 0

        court = Court(court_type="ncaa", origin="center", units="ft")
        fig, ax = court.draw(orientation="h", court_color="#D2A679", line_color="white")
        fig.set_size_inches(12, 10)
        ax.set_xlim(0, 47)
        ax.set_ylim(-27, 34)

        ax.scatter(misses["plot_x"], misses["plot_y"], c="#E53935", s=25, alpha=0.7,
                   label=f"Miss ({len(misses)})", zorder=3, linewidths=0.3, edgecolors="white")
        ax.scatter(makes["plot_x"], makes["plot_y"], c="#43A047", s=25, alpha=0.85,
                   label=f"Make ({n_makes})", zorder=4, linewidths=0.3, edgecolors="white")

        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

        summary = f" FG: {n_makes}/{total}  ({fg_pct:.1f}%) "
        ax.text(46, 30, summary, fontsize=10, fontweight="bold", ha="right", va="top",
                family="monospace",
                bbox=dict(boxstyle="square,pad=0.5", facecolor="#F8F9FA", edgecolor="black", alpha=0.9))

        plt.title(f"{title}\n{subtitle}", fontsize=14, fontweight="bold", pad=15)

        if return_fig:
            return fig
        plt.close(fig)
        return None

    def _density_plot(self, shots_df, title, subtitle, return_fig=False):
        """KDE shot density heatmap for season-level data."""
        if len(shots_df) == 0:
            return None

        is_ft = (
            shots_df["type_text"].str.contains("Free", case=False, na=False)
        ) | (shots_df["text"].str.contains("free throw", case=False, na=False))
        fgs = shots_df[~is_ft].dropna(subset=["coordinate_x", "coordinate_y"]).copy()

        if len(fgs) < 5:
            return None

        fgs["plot_x"] = fgs["coordinate_x"].abs()
        fgs["plot_y"] = fgs["coordinate_y"]

        x = fgs["plot_x"].values
        y = fgs["plot_y"].values

        court = Court(court_type="ncaa", origin="center", units="ft")
        fig, ax = court.draw(orientation="h", court_color="#1a1a2e", line_color="white")
        fig.set_size_inches(12, 10)
        ax.set_xlim(0, 47)
        ax.set_ylim(-27, 34)

        # KDE on a half-court grid
        xi = np.linspace(0, 47, 200)
        yi = np.linspace(-25, 25, 200)
        xx, yy = np.meshgrid(xi, yi)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        kde = gaussian_kde(np.vstack([x, y]), bw_method=0.15)
        zz = kde(positions).reshape(xx.shape)

        cf = ax.contourf(xx, yy, zz, levels=14, cmap="YlOrRd", alpha=0.75, zorder=2)
        cbar = fig.colorbar(cf, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Shot Density", fontsize=10)
        cbar.set_ticks([])

        n_makes = int(fgs["scoring_play"].sum())
        n_total = len(fgs)
        fg_pct = n_makes / n_total * 100 if n_total > 0 else 0
        ax.text(46, 30, f" {n_makes}/{n_total}  ({fg_pct:.1f}% FG) ",
                fontsize=10, fontweight="bold", ha="right", va="top", family="monospace",
                bbox=dict(boxstyle="square,pad=0.5", facecolor="#F8F9FA", edgecolor="white", alpha=0.9))

        plt.title(f"{title}\n{subtitle}", fontsize=14, fontweight="bold", pad=15, color="white")
        fig.patch.set_facecolor("#1a1a2e")

        if return_fig:
            return fig
        plt.close(fig)
        return None

    def _plot(
        self,
        shots_df,
        title,
        subtitle,
        filename,
        specific_folder=None,
        show_summary=False,
        return_fig=False,
    ):
        """Core plotting engine for NCAA Men's Basketball."""
        if len(shots_df) == 0:
            print(f"⚠️ No shot data found for {title}. Skipping.")
            return

        stats = self._calculate_zone_stats(shots_df)

        # Set to NCAA court
        court = Court(court_type="ncaa", origin="center", units="ft")
        fig, ax = court.draw(orientation="h", court_color="#D2A679", line_color="white")
        fig.set_size_inches(12, 10)

        ax.set_xlim(0, 47)
        ax.set_ylim(-27, 34)

        plt.title(
            f"{title} - Zone Efficiency\n{subtitle}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        if show_summary:
            valid_stats = stats[~stats["zone"].isin(["Heave", "Unknown"])]

            total_makes = valid_stats["makes"].sum()
            total_attempts = valid_stats["attempts"].sum()
            overall_pct = (
                (total_makes / total_attempts) * 100 if total_attempts > 0 else 0
            )

            # THE FIX: Added .astype(str) to avoid errors when a player has no valid field goals
            threes = valid_stats[
                valid_stats["zone"].astype(str).str.contains("3PT", na=False)
            ]
            fg3_m = threes["makes"].sum()
            fg3_a = threes["attempts"].sum()
            fg3_pct = (fg3_m / fg3_a) * 100 if fg3_a > 0 else 0

            # THE FIX: Added .astype(str) here too
            twos = valid_stats[
                ~valid_stats["zone"].astype(str).str.contains("3PT", na=False)
            ]
            fg2_m = twos["makes"].sum()
            fg2_a = twos["attempts"].sum()
            fg2_pct = (fg2_m / fg2_a) * 100 if fg2_a > 0 else 0

            summary_text = (
                f" SHOOTING SPLITS \n"
                f"-----------------\n"
                f"Overall: {overall_pct:.1f}% ({int(total_makes)}/{int(total_attempts)})\n"
                f"2PT FG:  {fg2_pct:.1f}% ({int(fg2_m)}/{int(fg2_a)})\n"
                f"3PT FG:  {fg3_pct:.1f}% ({int(fg3_m)}/{int(fg3_a)})"
            )

            ax.text(
                46,
                27,
                summary_text,
                fontsize=10,
                fontweight="bold",
                ha="right",
                va="bottom",
                family="monospace",
                bbox=dict(
                    boxstyle="square,pad=0.6",
                    facecolor="#F8F9FA",
                    edgecolor="black",
                    alpha=0.9,
                ),
            )

        zone_locations = {
            "At Rim": (42.25, 0),
            "Paint (Non-Rim)": (34, 0),
            "Center Mid-Range": (23, 0),
            "Left Mid-Range": (28, 12),
            "Right Mid-Range": (28, -12),
            "Left Baseline Mid": (38, 16),
            "Right Baseline Mid": (38, -16),
            "Center 3PT": (13, 0),
            "Left Wing 3PT": (15, 20),
            "Right Wing 3PT": (15, -20),
            "Left Corner 3PT": (40, 24),
            "Right Corner 3PT": (40, -24),
        }

        for _, row in stats.iterrows():
            zone = row["zone"]
            if zone in zone_locations:
                loc_x, loc_y = zone_locations[zone]
                makes, attempts, pct = row["makes"], row["attempts"], row["fg_pct"]

                if pct >= 45:
                    color = "#C8E6C9"
                elif pct <= 35:
                    color = "#FFCDD2"
                else:
                    color = "#F5F5F5"

                text_str = f"{zone}\n{int(makes)}/{int(attempts)}\n{pct:.1f}%"
                ax.text(
                    loc_x,
                    loc_y,
                    text_str,
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor=color,
                        edgecolor="black",
                        alpha=0.9,
                    ),
                )

        if return_fig:
            return fig  # Send the figure back to Streamlit

        # Original saving logic
        save_dir = specific_folder if specific_folder else self.output_folder
        filepath = f"{save_dir}/{filename}.png"

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_team_zone_leaders(
        self,
        team_name,
        filename=None,
        specific_folder=None,
        return_fig=False,
        show_summary=False,
    ):
        """Plots a 'Territory Map' showing the top scoring player for each zone."""
        team_full, leaders = self.get_team_zone_leaders(team_name)

        if leaders is None or leaders.empty:
            return

        # Set up the court
        court = Court(court_type="ncaa", origin="center", units="ft")
        fig, ax = court.draw(orientation="h", court_color="#D2A679", line_color="white")
        fig.set_size_inches(12, 10)

        ax.set_xlim(0, 47)
        ax.set_ylim(-27, 34)

        plt.title(
            f"{team_full} - Top Scorers by Zone (Territory Map)\n{self.season} Season",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        # Map zones to court coordinates
        zone_locations = {
            "At Rim": (42.25, 0),
            "Paint (Non-Rim)": (34, 0),
            "Center Mid-Range": (20, 0),
            "Left Mid-Range": (28, 12),
            "Right Mid-Range": (28, -12),
            "Left Baseline Mid": (38, 16),
            "Right Baseline Mid": (38, -16),
            "Center 3PT": (13, 0),
            "Left Wing 3PT": (15, 20),
            "Right Wing 3PT": (15, -20),
            "Left Corner 3PT": (40, 24),
            "Right Corner 3PT": (40, -24),
        }

        # Plot each leader
        for _, row in leaders.iterrows():
            zone = row["zone"]
            if zone in zone_locations:
                loc_x, loc_y = zone_locations[zone]
                player = row["athlete_display_name"]
                pts = row["pts"]

                # Format the text box
                text_str = f"{zone}\n{player}\n{pts} pts"
                ax.text(
                    loc_x,
                    loc_y,
                    text_str,
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor="#F8F9FA",
                        edgecolor="black",
                        alpha=0.9,
                    ),
                )

        # REPLACE the bottom saving logic with this:
        if return_fig:
            return fig  # Send the figure back to Streamlit

        # Original saving logic
        save_dir = specific_folder if specific_folder else self.output_folder
        filepath = f"{save_dir}/{filename}.png"

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ==========================================
    # PUBLIC METHODS
    # ==========================================

    def team_game_chart(self, team_name, opponent_name, date, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        game_id, is_home = self._get_game_id_from_matchup(
            team_name, opponent_name, date
        )

        shots = self._get_shots(team_id, game_id)

        loc_str = "vs" if is_home else "@"
        filename = f"{self.season}_{team_full.replace(' ', '_')}_{loc_str}_{opponent_name.replace(' ', '_')}_{date}"
        return self._plot(
            shots, team_full, f"{loc_str} {opponent_name.title()} ({date})", filename,
            return_fig=return_fig,
        )

    def team_season_chart(self, team_name, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        shots = self._get_shots(team_id)

        filename = f"{self.season}_{team_full.replace(' ', '_')}_Season"
        return self._plot(
            shots, team_full, f"{self.season} Full Season", filename,
            show_summary=True, return_fig=return_fig,
        )

    def player_game_chart(self, player_name, team_name, opponent_name, date, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        game_id, is_home = self._get_game_id_from_matchup(
            team_name, opponent_name, date
        )

        shots = self._get_shots(team_id, game_id)
        player_shots = shots[
            shots["athlete_display_name"].str.contains(
                player_name, case=False, na=False
            )
        ]

        loc_str = "vs" if is_home else "@"
        filename = f"{self.season}_{player_name.replace(' ', '_')}_{loc_str}_{opponent_name.replace(' ', '_')}_{date}"
        return self._plot(
            player_shots,
            player_name,
            f"{team_full} {loc_str} {opponent_name.title()} ({date})",
            filename,
            return_fig=return_fig,
        )

    def player_season_chart(self, player_name, team_name, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        shots = self._get_shots(team_id)
        player_shots = shots[
            shots["athlete_display_name"].str.contains(
                player_name, case=False, na=False
            )
        ]

        filename = f"{self.season}_{player_name.replace(' ', '_')}_Season"
        return self._plot(
            player_shots,
            player_name,
            f"{team_full} - {self.season} Full Season",
            filename,
            show_summary=True,
            return_fig=return_fig,
        )

    def team_season_roster_batch(self, team_name):
        """Loops through every active player and creates a season zone chart in a dedicated folder."""
        team_id, team_full = self._get_team_id(team_name)
        shots = self._get_shots(team_id)

        players = shots["athlete_display_name"].dropna().unique()

        folder = f"{self.output_folder}/{self.season}_{team_full.replace(' ', '_')}_Roster_Season"
        if not os.path.exists(folder):
            os.makedirs(folder)

        print(f"\nProcessing {len(players)} players for {team_full} ({self.season})...")
        for player in sorted(players):
            player_shots = shots[shots["athlete_display_name"] == player]
            if len(player_shots) > 0:
                filename = f"{self.season}_{player.replace(' ', '_')}"
                self._plot(
                    player_shots,
                    player,
                    f"{team_full} - {self.season} Season Update",
                    filename,
                    specific_folder=folder,
                    show_summary=True,
                )
                print(f"  ✓ Processed: {player}")

        print(f"\n✅ Batch complete! All player charts saved to: {folder}")

    def get_team_zone_leaders(self, team_name):
        """Identifies which player has scored the most points in each zone for a team."""
        team_id, team_full = self._get_team_id(team_name)
        shots = self._get_shots(team_id)

        if shots.empty:
            print(f"⚠️ No shot data found for {team_full}.")
            return None, None

        # 1. Classify zones for all shots
        shots["zone"] = shots.apply(self._classify_zone, axis=1)

        # 2. Filter for only successful field goals
        makes = shots[shots["scoring_play"] == True].copy()

        # 3. Assign point values (3 points for 3PT zones, 2 for everything else)
        makes["pts"] = makes["zone"].apply(lambda x: 3 if "3PT" in str(x) else 2)

        # 4. Aggregate total points by zone and player
        leaderboard = (
            makes.groupby(["zone", "athlete_display_name"])["pts"].sum().reset_index()
        )

        # 5. Sort by points (highest first), then group by zone and keep only the top player
        zone_leaders = (
            leaderboard.sort_values("pts", ascending=False)
            .groupby("zone")
            .head(1)
            .reset_index(drop=True)
        )

        return team_full, zone_leaders

    # ── Density chart public methods ──────────────────────────────────────────

    def player_season_density(self, player_name, team_name, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        shots = self._get_shots(team_id)
        player_shots = shots[shots["athlete_display_name"].str.contains(player_name, case=False, na=False)]
        return self._density_plot(player_shots, player_name, f"{team_full} — {self.season} Full Season", return_fig=return_fig)

    def team_season_density(self, team_name, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        shots = self._get_shots(team_id)
        return self._density_plot(shots, team_full, f"{self.season} Full Season", return_fig=return_fig)

    # ── Scatter chart public methods ──────────────────────────────────────────

    def player_season_scatter(self, player_name, team_name, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        shots = self._get_shots(team_id)
        player_shots = shots[shots["athlete_display_name"].str.contains(player_name, case=False, na=False)]
        return self._scatter_plot(player_shots, player_name, f"{team_full} — {self.season} Full Season", return_fig=return_fig)

    def player_game_scatter(self, player_name, team_name, opponent_name, date, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        game_id, is_home = self._get_game_id_from_matchup(team_name, opponent_name, date)
        shots = self._get_shots(team_id, game_id)
        player_shots = shots[shots["athlete_display_name"].str.contains(player_name, case=False, na=False)]
        loc_str = "vs" if is_home else "@"
        return self._scatter_plot(player_shots, player_name, f"{team_full} {loc_str} {opponent_name} ({date})", return_fig=return_fig)

    def team_season_scatter(self, team_name, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        shots = self._get_shots(team_id)
        return self._scatter_plot(shots, team_full, f"{self.season} Full Season", return_fig=return_fig)

    def team_game_scatter(self, team_name, opponent_name, date, return_fig=False):
        team_id, team_full = self._get_team_id(team_name)
        game_id, is_home = self._get_game_id_from_matchup(team_name, opponent_name, date)
        shots = self._get_shots(team_id, game_id)
        loc_str = "vs" if is_home else "@"
        return self._scatter_plot(shots, team_full, f"{loc_str} {opponent_name} ({date})", return_fig=return_fig)

    def calculate_zone_areas(self, resolution=0.1):
        """
        Estimates the square footage of each zone using a high-density grid.
        resolution: distance between points in feet (0.1 means 100 points per sq ft).
        """
        print("Calculating zone areas using grid approximation...")

        # Standard NCAA half-court dimensions
        # x goes from center court (0) to baseline (47)
        # y goes from left sideline (25) to right sideline (-25)
        x_coords = np.arange(0, 47, resolution)
        y_coords = np.arange(-25, 25, resolution)

        # Create a mesh grid of every coordinate
        xv, yv = np.meshgrid(x_coords, y_coords)
        grid_df = pd.DataFrame(
            {"coordinate_x": xv.flatten(), "coordinate_y": yv.flatten()}
        )

        # Apply your exact classification logic
        grid_df["zone"] = grid_df.apply(self._classify_zone, axis=1)

        # Each point represents a square area of (resolution * resolution)
        point_area = resolution**2

        # Aggregate the areas
        area_stats = grid_df["zone"].value_counts() * point_area
        area_df = area_stats.reset_index()
        area_df.columns = ["Zone", "Estimated Area (sq ft)"]

        return area_df.round(1)

    def plot_zone_boundaries(self, resolution=0.2, filename="zone_definitions_map"):
        """
        Creates a visual map of the zone definitions by classifying a dense grid
        of points across the half-court and coloring them.
        """
        print(
            f"Generating grid points at {resolution}ft resolution for visualization..."
        )

        # 1. Generate Grid Data covering the half court plus sidelines
        # Go slightly past boundaries (47.1 and 25.1) to ensure edges are caught
        x_coords = np.arange(0, 47.1, resolution)
        y_coords = np.arange(-25.1, 25.1, resolution)

        xv, yv = np.meshgrid(x_coords, y_coords)
        grid_df = pd.DataFrame(
            {"coordinate_x": xv.flatten(), "coordinate_y": yv.flatten()}
        )

        # 2. Classify points using your defined logic
        print(f"Classifying {len(grid_df):,} points...")
        grid_df["zone"] = grid_df.apply(self._classify_zone, axis=1)

        # Filter out "Unknown" which usually means off-court data errors
        grid_df = grid_df[grid_df["zone"] != "Unknown"]

        # 3. Define a distinct color palette for regions
        # Grouping similar zones with similar hues for visual clarity
        zone_colors = {
            # Interior (Warm/Red)
            "At Rim": "#D32F2F",  # Deep Red
            "Paint (Non-Rim)": "#F57C00",  # Dark Orange
            # Mid-Range (Yellow/Greens)
            "Center Mid-Range": "#FBC02D",  # Mustard Yellow
            "Left Mid-Range": "#7CB342",  # Light Green
            "Right Mid-Range": "#7CB342",
            "Left Baseline Mid": "#388E3C",  # Forest Green
            "Right Baseline Mid": "#388E3C",
            # 3-Pointers (Blues/Purples)
            "Center 3PT": "#0288D1",  # Light Blue
            "Left Wing 3PT": "#1976D2",  # Medium Blue
            "Right Wing 3PT": "#1976D2",
            "Left Corner 3PT": "#512DA8",  # Deep Purple
            "Right Corner 3PT": "#512DA8",
            # Other
            "Heave": "#616161",  # Dark Grey
        }

        # 4. Set up the court plot
        court = Court(court_type="ncaa", origin="center", units="ft")
        # Using standard wooden colors so the zone dots pop out
        fig, ax = court.draw(orientation="h", court_color="#E0C8B0", line_color="black")
        fig.set_size_inches(14, 12)  # Slightly larger to accommodate legend

        ax.set_xlim(-5, 52)  # Show a little area behind halfcourt and baseline
        ax.set_ylim(-30, 30)  # Show past sidelines

        plt.title(
            "NCAA Zone Definition Boundaries\n(Visualized via Grid Classification)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # 5. Plot each zone group
        # We iterate through the predefined color keys to ensure consistent legend order
        already_plotted = set()

        for zone_name, color in zone_colors.items():
            subset = grid_df[grid_df["zone"] == zone_name]

            if not subset.empty:
                # Handle left/right mirroring for the legend:
                # If we already plotted "Left Wing", don't add "Right Wing" to legend again
                base_name = zone_name.replace("Left ", "").replace("Right ", "")
                label = base_name if base_name not in already_plotted else "_nolegend_"
                if label != "_nolegend_":
                    already_plotted.add(base_name)

                # s=3 gives small dots, alpha=0.4 makes them semi-transparent so the lines show through
                ax.scatter(
                    subset["coordinate_x"],
                    subset["coordinate_y"],
                    c=color,
                    s=3,
                    alpha=0.4,
                    marker="o",
                    label=label,
                    zorder=2,
                )

        # Add legend outside the plot area
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            title="Zone Categories",
            fontsize=10,
            markerscale=3,
        )

        filepath = f"{self.output_folder}/{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Zone boundary map saved to: {filepath}")
