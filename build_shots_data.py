"""
build_shots_data.py — Download and save shot-coordinate data for the current season.

WHY THIS FILE EXISTS
--------------------
Loading the full play-by-play (PBP) from sportsdataverse takes 60–90 seconds
and requires a network connection.  Running that at cloud-app startup time would
be unacceptable.  Instead, this script runs locally (or in GitHub Actions) and
saves a compact parquet file with only shooting plays.  The cloud app reads
the parquet instantly without any network calls.

Outputs:
    shots_{SEASON}.parquet   Shooting rows only, with game metadata joined in.
    box_{SEASON}.parquet     Player box scores (used to look up athlete names and teams).

Run locally:
    python build_shots_data.py
    OVERRIDE_SEASON=2025 python build_shots_data.py
"""

import os
from datetime import date as _date

import pandas as pd
import sportsdataverse.mbb as mbb

# ---------------------------------------------------------------------------
# Season detection
# ---------------------------------------------------------------------------
_season_override = os.environ.get("OVERRIDE_SEASON")
if _season_override:
    SEASON = int(_season_override)
else:
    _today = _date.today()
    SEASON = _today.year + 1 if _today.month >= 11 else _today.year

SHOTS_OUTPUT_FILE = f"shots_{SEASON}.parquet"

# Columns kept from the full PBP — everything else is dropped to keep the file small.
SHOT_COLUMNS_TO_KEEP = [
    "game_id", "team_id", "athlete_id_1",
    "coordinate_x", "coordinate_y",
    "scoring_play", "type_text", "text",
]
# Game-level metadata we join back onto each shot row.
GAME_METADATA_COLUMNS = ["game_id", "home_team_id", "away_team_id"]

# ---------------------------------------------------------------------------
# Load full play-by-play
# ---------------------------------------------------------------------------
print(f"Loading {SEASON} PBP data...")
raw_play_by_play = mbb.load_mbb_pbp(seasons=[SEASON], return_as_pandas=True)

# Determine the date column name (varies between sportsdataverse versions).
date_column_name = "game_date" if "game_date" in raw_play_by_play.columns else "date"
GAME_METADATA_COLUMNS.append(date_column_name)

# ---------------------------------------------------------------------------
# Filter to shooting plays only
# ---------------------------------------------------------------------------
print("Filtering to shooting plays...")
shooting_plays_df = raw_play_by_play.loc[
    raw_play_by_play["shooting_play"] == True,
    [c for c in SHOT_COLUMNS_TO_KEEP if c in raw_play_by_play.columns]
].copy()

# One row per game carrying the metadata we need (home/away team IDs and date).
game_metadata_df = (
    raw_play_by_play[[c for c in GAME_METADATA_COLUMNS if c in raw_play_by_play.columns]]
    .drop_duplicates(subset=["game_id"])
    .copy()
)
game_metadata_df["_is_game_row"] = True

# Free memory — the full PBP is large.
del raw_play_by_play

# Join game metadata (home/away IDs + date) onto every shot row.
shots_with_game_info_df = pd.merge(shooting_plays_df, game_metadata_df, on="game_id", how="left")
shots_with_game_info_df["_date_col"] = date_column_name    # store the column name for downstream readers

# ---------------------------------------------------------------------------
# Load player box scores (for athlete name + team lookup)
# ---------------------------------------------------------------------------
print(f"Loading boxscore data...")
raw_player_boxscores = mbb.load_mbb_player_boxscore(seasons=[SEASON], return_as_pandas=True)

player_boxscore_columns = ["athlete_id", "athlete_display_name", "team_id", "team_display_name"]
player_identity_df = raw_player_boxscores[
    [c for c in player_boxscore_columns if c in raw_player_boxscores.columns]
].copy()

del raw_player_boxscores

# ---------------------------------------------------------------------------
# Save parquet files
# ---------------------------------------------------------------------------
player_identity_df.to_parquet(f"box_{SEASON}.parquet", index=False)
shots_with_game_info_df.to_parquet(SHOTS_OUTPUT_FILE, index=False)

print(f"\nDone.")
print(f"  shots_{SEASON}.parquet  — {len(shots_with_game_info_df):,} shot rows")
print(f"  box_{SEASON}.parquet    — {len(player_identity_df):,} player rows")


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
#
# SEASON                       int     Calendar year the season ends (e.g. 2026).
# _season_override             str     Value of OVERRIDE_SEASON env var; None if unset.
# SHOTS_OUTPUT_FILE            str     Output filename, e.g. "shots_2026.parquet".
# SHOT_COLUMNS_TO_KEEP         list    PBP columns kept in the shot parquet (coordinates, etc.).
# GAME_METADATA_COLUMNS        list    Game-level columns joined onto each shot row.
# date_column_name             str     "game_date" or "date" depending on sportsdataverse version.
# raw_play_by_play             DataFrame  Full PBP — every play from every game. ~5M rows.
# shooting_plays_df            DataFrame  Subset: only rows where shooting_play == True.
# game_metadata_df             DataFrame  One row per game_id with home/away team IDs and date.
# shots_with_game_info_df      DataFrame  shooting_plays_df joined with game_metadata_df.
# raw_player_boxscores         DataFrame  Full player box score (every player, every game).
# player_identity_df           DataFrame  athlete_id/name + team_id/name; used for name lookup.
