"""
build_conf_stats.py — Build player_stats_conf_{SEASON}.csv for conference-only stats.

WHY THIS FILE EXISTS
--------------------
build_player_stats.py already writes a conference CSV using game IDs from the
live schedule, but this standalone version is useful when you want to regenerate
the conference stats for a specific season without re-running the full pipeline.
It also serves as a readable, self-contained reference for how conference stats
are computed.

Outputs:
    player_stats_conf_{SEASON}.csv   Conference-only season totals + per-game averages.

Run locally:
    python3 build_conf_stats.py
    OVERRIDE_SEASON=2025 python3 build_conf_stats.py
"""
import os
from datetime import date as _date

import pandas as pd
import sportsdataverse.mbb as mbb

# ---------------------------------------------------------------------------
# Season detection
# ---------------------------------------------------------------------------
from hoglib.season import detect_season
SEASON = detect_season()

# ---------------------------------------------------------------------------
# Identify conference game IDs from the schedule
# ---------------------------------------------------------------------------
print("Loading schedule...")
from hoglib import feeds  # cached by build_ingest.py (step 0)
schedule_df = feeds.load_schedule(SEASON)

conference_game_id_set = set(
    schedule_df.loc[schedule_df["conference_competition"] == True, "game_id"]
)
print(f"  {len(conference_game_id_set)} conference games found")

# ---------------------------------------------------------------------------
# Load player box scores and filter to conference games
# ---------------------------------------------------------------------------
print("Loading player boxscores...")
full_player_boxscores_df = feeds.load_player_box(SEASON)

conference_boxscores_df = full_player_boxscores_df[
    full_player_boxscores_df["game_id"].isin(conference_game_id_set)
    & (full_player_boxscores_df["did_not_play"] == False)
    & (full_player_boxscores_df["active"] == True)
].copy()
print(f"  {len(conference_boxscores_df)} player-game rows in conference games")


# ---------------------------------------------------------------------------
# Parse minutes from "MM:SS" string format to decimal float
# ---------------------------------------------------------------------------
def parse_minutes_string(minutes_string):
    """Convert ESPN's 'MM:SS' minutes format to a decimal float.

    Examples: '32:45' → 32.75,  '8' → 8.0,  NaN → 0.0
    """
    try:
        if pd.isna(minutes_string):
            return 0.0
        parts = str(minutes_string).split(":")
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1]) / 60
        return float(parts[0])
    except Exception:
        return 0.0


conference_boxscores_df["minutes_decimal"] = conference_boxscores_df["minutes"].apply(
    parse_minutes_string
)

# ---------------------------------------------------------------------------
# Pull in conference label and position from the overall player_stats CSV
# (which already has conference labels from build_player_stats.py)
# ---------------------------------------------------------------------------
overall_player_stats_csv = f"player_stats_{SEASON}.csv"
player_identity_info_df = pd.read_csv(overall_player_stats_csv)[
    ["athlete_id", "conf.", "athlete_position_name"]
].drop_duplicates(subset=["athlete_id"])

# ---------------------------------------------------------------------------
# Aggregate: one row per player across all conference games
# ---------------------------------------------------------------------------
COUNTING_STAT_COLUMNS = [
    "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "rebounds",
    "assists", "steals", "blocks", "turnovers", "fouls", "points",
]

player_season_aggregates_df = (
    conference_boxscores_df.groupby(
        ["athlete_id", "athlete_display_name", "team_id", "team_display_name"]
    )
    .agg(
        games_played=("game_id", "nunique"),
        minutes=("minutes_decimal", "sum"),
        **{stat_col: (stat_col, "sum") for stat_col in COUNTING_STAT_COLUMNS},
    )
    .reset_index()
)

player_season_aggregates_df = player_season_aggregates_df.merge(
    player_identity_info_df, on="athlete_id", how="left"
)

# ---------------------------------------------------------------------------
# Shooting percentages
# ---------------------------------------------------------------------------
def safe_divide(numerator_col, denominator_col):
    """Divide two Series, replacing division-by-zero with NaN."""
    return numerator_col / denominator_col.replace(0, float("nan"))


player_season_aggregates_df["fg_pct"] = safe_divide(
    player_season_aggregates_df["field_goals_made"],
    player_season_aggregates_df["field_goals_attempted"]
)
player_season_aggregates_df["3pt_pct"] = safe_divide(
    player_season_aggregates_df["three_point_field_goals_made"],
    player_season_aggregates_df["three_point_field_goals_attempted"]
)
player_season_aggregates_df["ft_pct"] = safe_divide(
    player_season_aggregates_df["free_throws_made"],
    player_season_aggregates_df["free_throws_attempted"]
)
player_season_aggregates_df["efg_pct"] = safe_divide(
    player_season_aggregates_df["field_goals_made"]
    + 0.5 * player_season_aggregates_df["three_point_field_goals_made"],
    player_season_aggregates_df["field_goals_attempted"]
)

# ---------------------------------------------------------------------------
# Per-game averages
# ---------------------------------------------------------------------------
games_played_series = player_season_aggregates_df["games_played"]

per_game_average_column_pairs = [
    ("minutes",                                "minute_avg"),
    ("field_goals_made",                       "fgm_avg"),
    ("field_goals_attempted",                  "fga_avg"),
    ("three_point_field_goals_made",           "3ptm_avg"),
    ("three_point_field_goals_attempted",      "3pta_avg"),
    ("free_throws_made",                       "ftm_avg"),
    ("free_throws_attempted",                  "fta_avg"),
    ("offensive_rebounds",                     "oreb_avg"),
    ("defensive_rebounds",                     "dreb_avg"),
    ("rebounds",                               "reb_avg"),
    ("assists",                                "ast_avg"),
    ("steals",                                 "steal_avg"),
    ("blocks",                                 "blocks_avg"),
    ("turnovers",                              "to_avg"),
    ("points",                                 "points_avg"),
]

for total_column, average_column in per_game_average_column_pairs:
    player_season_aggregates_df[average_column] = (
        player_season_aggregates_df[total_column] / games_played_series
    ).round(2)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output_filename = f"player_stats_conf_{SEASON}.csv"
player_season_aggregates_df.to_csv(output_filename, index=False)
print(f"\nSaved {output_filename} — {len(player_season_aggregates_df)} players")


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
#
# SEASON                          int     Calendar year the season ends (e.g. 2026).
# _season_override                str     Value of OVERRIDE_SEASON env var; None if unset.
# schedule_df                     DataFrame  Full ESPN schedule for the season.
# conference_game_id_set          set     game_id values where conference_competition == True.
# full_player_boxscores_df        DataFrame  Every player-game box score row.
# conference_boxscores_df         DataFrame  Subset: conference games, active/played rows only.
# overall_player_stats_csv        str     Filename of the overall stats CSV (source of conf. labels).
# player_identity_info_df         DataFrame  athlete_id → conf. + position (from overall CSV).
# COUNTING_STAT_COLUMNS           list    Column names summed across games.
# player_season_aggregates_df     DataFrame  One row per player; season totals + averages.
# games_played_series             Series  games_played column, used as denominator for averages.
# per_game_average_column_pairs   list    [(total_col, avg_col), …] pairs for per-game avg computation.
# output_filename                 str     "player_stats_conf_{SEASON}.csv"
