"""
build_lineups.py — Generate 1/2/3/5-man lineup combo CSVs for overall and conference games.

WHY THIS FILE EXISTS
--------------------
The Hog Charts lineup page shows how every combination of 1, 2, 3, or 5
players on the same team performed together — net rating, offensive rating,
defensive rating, and per-100 counting stats.

This requires:
  1. Reconstructing exact on-court lineups from substitution events in the
     play-by-play (PBP) data.
  2. Slicing each game into "stints" — continuous segments where the lineup
     doesn't change.
  3. Accumulating raw counting stats (FGA, FTA, ORB, TOV, points scored,
     points allowed) for each stint's lineup.
  4. Rolling up stints into season totals per lineup.
  5. Computing all sub-combinations (e.g. from a 5-man lineup, all 5
     individual players, all 10 pairs, all 10 triples).

Outputs (8 CSV files):
    1_man_overall_stats_{SEASON}.csv      1_man_conference_stats_{SEASON}.csv
    2_man_overall_stats_{SEASON}.csv      2_man_conference_stats_{SEASON}.csv
    3_man_overall_stats_{SEASON}.csv      3_man_conference_stats_{SEASON}.csv
    5_man_overall_stats_{SEASON}.csv      5_man_conference_stats_{SEASON}.csv

Run locally:
    python3 build_lineups.py
    OVERRIDE_SEASON=2025 python3 build_lineups.py
"""

import os
from datetime import date as _date
from itertools import combinations

import numpy as np
import pandas as pd
import sportsdataverse.mbb as mbb
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Season detection
# ---------------------------------------------------------------------------
from hoglib.season import detect_season
SEASON = detect_season()

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
print("Loading data...")
from hoglib import feeds
play_by_play_df = feeds.load_pbp(SEASON)          # cached by build_ingest.py (step 0)
player_box_df   = feeds.load_player_box(SEASON)
schedule_df     = feeds.load_schedule(SEASON)

# ---------------------------------------------------------------------------
# Build team name and conference lookup maps
# ---------------------------------------------------------------------------
home_team_info_df = schedule_df[
    ["home_id", "home_location", "home_name", "home_conference_id"]
].rename(columns={
    "home_id": "team_id", "home_location": "team_location",
    "home_name": "team_name", "home_conference_id": "conference_id"
})
away_team_info_df = schedule_df[
    ["away_id", "away_location", "away_name", "away_conference_id"]
].rename(columns={
    "away_id": "team_id", "away_location": "team_location",
    "away_name": "team_name", "away_conference_id": "conference_id"
})
all_teams_info_df = (
    pd.concat([home_team_info_df, away_team_info_df])
    .dropna(subset=["team_id"])
    .drop_duplicates("team_id")
)
all_teams_info_df["full_display_name"] = (
    all_teams_info_df["team_location"].astype(str)
    + " "
    + all_teams_info_df["team_name"].astype(str)
)

team_id_to_display_name = dict(
    zip(all_teams_info_df["team_id"].astype(int), all_teams_info_df["full_display_name"])
)
team_id_to_conference_id = dict(
    zip(all_teams_info_df["team_id"].astype(int), all_teams_info_df["conference_id"])
)

# Only process teams with enough games to have meaningful lineup data.
MIN_GAMES_TO_QUALIFY = 15
schedule_teams_long_df = schedule_df.melt(
    id_vars=["game_id"], value_vars=["home_id", "away_id"], value_name="team_id"
)
games_per_team = schedule_teams_long_df.groupby("team_id")["game_id"].nunique()
qualifying_team_ids = games_per_team[games_per_team >= MIN_GAMES_TO_QUALIFY].index.astype(int).tolist()
print(f"  {len(qualifying_team_ids)} qualifying teams (>= {MIN_GAMES_TO_QUALIFY} games)")


# ---------------------------------------------------------------------------
# Clock utilities
# ---------------------------------------------------------------------------

def clock_to_seconds(clock_minutes_value, clock_seconds_value):
    """Convert separate clock_minutes / clock_seconds columns to total seconds remaining."""
    try:
        return int(clock_minutes_value) * 60 + int(clock_seconds_value)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Per-game lineup reconstruction
# ---------------------------------------------------------------------------

def calculate_game_lineup_stints(game_id, team_id, all_pbp_df, all_box_df):
    """Reconstruct lineup stints for one team in one game.

    A "stint" is a continuous segment of game time where the 5-man lineup
    doesn't change.  We walk the play-by-play row by row, tracking:
      - Which players are currently on the floor.
      - How many seconds elapsed since the last event.
      - Counting stats (points, rebounds, etc.) accumulated in that stint.

    Returns a DataFrame where each row is one (lineup_key, stint_accumulation).
    The "Lineup" column is a comma-separated string of player display names.
    """
    numeric_team_id = int(team_id)
    game_pbp_df     = all_pbp_df[all_pbp_df["game_id"] == game_id].copy()
    game_pbp_df     = game_pbp_df.sort_values("game_play_number")

    # Forward-fill the score columns so we can track the running margin.
    if "home_score" in game_pbp_df.columns:
        game_pbp_df["home_score"] = game_pbp_df["home_score"].ffill().fillna(0)
        game_pbp_df["away_score"] = game_pbp_df["away_score"].ffill().fillna(0)

    team_box_for_game = all_box_df[
        (all_box_df["game_id"] == game_id) & (all_box_df["team_id"] == numeric_team_id)
    ]
    player_id_to_name = dict(zip(team_box_for_game["athlete_id"], team_box_for_game["athlete_display_name"]))

    # Seed the starting lineup from the box score; fall back to first 5 if starters flag is missing.
    starting_player_ids = team_box_for_game[team_box_for_game["starter"] == True]["athlete_id"].tolist()
    if len(starting_player_ids) != 5:
        starting_player_ids = team_box_for_game["athlete_id"].head(5).tolist()

    current_on_court_ids = set(starting_player_ids)
    current_stint_stats  = {}   # {lineup_tuple: {stat: total}}
    previous_period      = 1
    previous_clock_secs  = 20 * 60  # regulation period = 1200 seconds

    for _, play_row in game_pbp_df.iterrows():
        current_period    = play_row["period_number"]
        current_clock_secs = clock_to_seconds(play_row["clock_minutes"], play_row["clock_seconds"])

        # Periods reset the clock; OT periods are 5 minutes each.
        if current_period != previous_period:
            previous_clock_secs = 20 * 60 if current_period <= 2 else 5 * 60
            previous_period = current_period

        seconds_in_stint = max(0, previous_clock_secs - current_clock_secs)

        # The lineup key is a sorted tuple of player IDs (immutable, hashable).
        lineup_tuple_key = tuple(sorted(list(current_on_court_ids)))
        if lineup_tuple_key not in current_stint_stats:
            current_stint_stats[lineup_tuple_key] = {
                "Seconds": 0, "PF": 0, "PA": 0,
                "FGM": 0, "FGA": 0, "FG3M": 0, "FG3A": 0, "FTM": 0, "FTA": 0,
                "ORB": 0, "DRB": 0, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "Foul": 0,
                "Opp_FGA": 0, "Opp_FTA": 0, "Opp_ORB": 0, "Opp_TOV": 0,
            }

        current_lineup_accumulator = current_stint_stats[lineup_tuple_key]
        current_lineup_accumulator["Seconds"] += seconds_in_stint

        # Classify the play and attribute stats.
        play_type_text = str(play_row["type_text"]).lower()
        play_desc_text = str(play_row["text"]).lower()
        play_team_id   = int(play_row["team_id"]) if pd.notnull(play_row["team_id"]) else -1
        is_this_team   = (play_team_id == numeric_team_id)

        if play_row["scoring_play"]:
            if is_this_team:
                current_lineup_accumulator["PF"] += play_row["score_value"]
            else:
                current_lineup_accumulator["PA"] += play_row["score_value"]

        if is_this_team:
            if play_row["shooting_play"] and "free throw" not in play_type_text:
                current_lineup_accumulator["FGA"] += 1
                if "three point" in play_desc_text:
                    current_lineup_accumulator["FG3A"] += 1
                if play_row["scoring_play"]:
                    current_lineup_accumulator["FGM"] += 1
                    if "three point" in play_desc_text:
                        current_lineup_accumulator["FG3M"] += 1
                    if "(" in play_desc_text and ")" in play_desc_text:
                        current_lineup_accumulator["AST"] += 1
            if "free throw" in play_type_text:
                current_lineup_accumulator["FTA"] += 1
                if play_row["scoring_play"]:
                    current_lineup_accumulator["FTM"] += 1
            if "offensive rebound" in play_type_text: current_lineup_accumulator["ORB"]  += 1
            if "defensive rebound" in play_type_text: current_lineup_accumulator["DRB"]  += 1
            if "steal"             in play_type_text: current_lineup_accumulator["STL"]  += 1
            if "block"             in play_type_text: current_lineup_accumulator["BLK"]  += 1
            if "turnover"          in play_type_text: current_lineup_accumulator["TOV"]  += 1
            if "foul"              in play_type_text: current_lineup_accumulator["Foul"] += 1
        else:
            if play_row["shooting_play"] and "free throw" not in play_type_text:
                current_lineup_accumulator["Opp_FGA"] += 1
            if "free throw"        in play_type_text: current_lineup_accumulator["Opp_FTA"] += 1
            if "offensive rebound" in play_type_text: current_lineup_accumulator["Opp_ORB"] += 1
            if "turnover"          in play_type_text: current_lineup_accumulator["Opp_TOV"] += 1

        # Apply substitutions after accruing the current play's time.
        if play_row["type_text"] == "Substitution" and play_team_id == numeric_team_id:
            substituting_player_id = play_row["athlete_id_1"]
            if "subbing in"  in play_desc_text:
                current_on_court_ids.add(substituting_player_id)
            elif "subbing out" in play_desc_text:
                if substituting_player_id in current_on_court_ids:
                    current_on_court_ids.remove(substituting_player_id)

        previous_clock_secs = current_clock_secs

    # Convert accumulator dict → list of rows for DataFrame creation.
    stint_record_list = []
    for lineup_tuple_key, stint_stats in current_stint_stats.items():
        if stint_stats["Seconds"] > 0:
            stint_stats["Lineup"] = ", ".join(
                [player_id_to_name.get(pid, str(pid)) for pid in lineup_tuple_key]
            )
            stint_record_list.append(stint_stats)

    return pd.DataFrame(stint_record_list)


# ---------------------------------------------------------------------------
# Combo stat computation
# ---------------------------------------------------------------------------

def compute_combo_stats(full_lineup_aggregates_df, combo_size, min_avg_possessions=20):
    """Roll up season lineup totals into per-combo stats.

    For each row in full_lineup_aggregates_df (a full 5-man lineup with season
    totals), we generate all C(5, n) sub-combinations and sum their raw stats.
    This produces e.g. all 5 individual players, all 10 pairs, all 10 triples.

    Returns a filtered DataFrame sorted by NetRtg.  Combos below
    min_avg_possessions are dropped to reduce noise from small samples.
    """
    # Columns present in every lineup row.
    raw_stat_columns = [
        "Seconds", "PF", "PA", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "ORB", "DRB", "AST", "STL", "BLK", "TOV", "Foul",
        "Opp_FGA", "Opp_FTA", "Opp_ORB", "Opp_TOV",
    ]
    raw_stat_cols_present = [c for c in raw_stat_columns if c in full_lineup_aggregates_df.columns]

    # Aggregate raw stats by combo key.
    combo_accumulator = {}
    for _, lineup_row in full_lineup_aggregates_df.iterrows():
        player_names_in_lineup = lineup_row["Lineup"].split(", ")
        for player_combo in combinations(sorted(player_names_in_lineup), combo_size):
            combo_key = tuple(player_combo)
            if combo_key not in combo_accumulator:
                combo_accumulator[combo_key] = {stat_col: 0 for stat_col in raw_stat_cols_present}
            for stat_col in raw_stat_cols_present:
                combo_accumulator[combo_key][stat_col] += lineup_row[stat_col]

    # Build DataFrame from the accumulator.
    combo_rows = [{"Combo": ", ".join(combo_key), **stat_totals}
                  for combo_key, stat_totals in combo_accumulator.items()]
    combo_stats_df = pd.DataFrame(combo_rows)
    if combo_stats_df.empty:
        return pd.DataFrame()

    # Possession estimates (standard formula: FGA - ORB + TOV + 0.475 * FTA).
    combo_stats_df["Poss_Off"] = (
        combo_stats_df["FGA"] - combo_stats_df["ORB"]
        + combo_stats_df["TOV"] + 0.475 * combo_stats_df["FTA"]
    )
    combo_stats_df["Poss_Def"] = (
        combo_stats_df["Opp_FGA"] - combo_stats_df["Opp_ORB"]
        + combo_stats_df["Opp_TOV"] + 0.475 * combo_stats_df["Opp_FTA"]
    )
    # Fallback: if no defensive possessions recorded, mirror offensive possessions.
    combo_stats_df.loc[combo_stats_df["Poss_Def"] <= 0, "Poss_Def"] = combo_stats_df["Poss_Off"]

    combo_stats_df["Avg_Poss"] = ((combo_stats_df["Poss_Off"] + combo_stats_df["Poss_Def"]) / 2).round(1)
    combo_stats_df["ORtg"]     = (combo_stats_df["PF"] / combo_stats_df["Poss_Off"].replace(0, np.nan)) * 100
    combo_stats_df["DRtg"]     = (combo_stats_df["PA"] / combo_stats_df["Poss_Def"].replace(0, np.nan)) * 100
    combo_stats_df["NetRtg"]   = (combo_stats_df["ORtg"].fillna(0) - combo_stats_df["DRtg"].fillna(0)).round(2)

    # Per-100-possession rates.
    per_100_stat_names = ["PF", "PA", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                          "ORB", "DRB", "AST", "STL", "BLK", "TOV", "Foul"]
    for stat_name in per_100_stat_names:
        if stat_name in combo_stats_df.columns:
            combo_stats_df[f"{stat_name}_100"] = (
                (combo_stats_df[stat_name] / combo_stats_df["Poss_Off"].replace(0, np.nan)) * 100
            ).round(2)

    combo_stats_df["Mins"]    = (combo_stats_df["Seconds"] / 60).round(1)
    combo_stats_df["REB_100"] = (combo_stats_df["ORB_100"] + combo_stats_df["DRB_100"]).round(2)

    # Filter by minimum possessions.
    filtered_df = combo_stats_df[combo_stats_df["Avg_Poss"] >= min_avg_possessions]

    output_columns = (
        ["Combo", "Mins", "Avg_Poss", "NetRtg", "ORtg", "DRtg"]
        + [f"{s}_100" for s in per_100_stat_names if f"{s}_100" in combo_stats_df.columns]
        + ["REB_100"]
    )
    return filtered_df[output_columns].sort_values("NetRtg", ascending=False)


# ---------------------------------------------------------------------------
# Process all teams
# ---------------------------------------------------------------------------

def process_all_teams(qualifying_teams, game_scope="overall"):
    """Build lineup combo CSVs for all qualifying teams.

    game_scope: "overall" processes all games; "conference" restricts to
    conference matchups only (fewer games → lower minimum possessions threshold).
    """
    # Per-combo-size raw stat DataFrames accumulated across all teams.
    combo_frames_by_size = {1: [], 2: [], 3: [], 5: []}

    for team_id in tqdm(qualifying_teams, desc=f"Processing {game_scope.upper()}"):
        # Get this team's games for the selected scope.
        team_game_rows = schedule_df[
            (schedule_df["home_id"].astype(int) == team_id)
            | (schedule_df["away_id"].astype(int) == team_id)
        ]
        if game_scope == "conference":
            team_game_rows = team_game_rows[team_game_rows["conference_competition"] == True]

        # Reconstruct lineups game-by-game.
        all_game_lineup_stints = []
        for game_id_value in team_game_rows["game_id"].unique():
            try:
                game_stint_df = calculate_game_lineup_stints(
                    game_id_value, team_id, play_by_play_df, player_box_df
                )
                if not game_stint_df.empty and "Lineup" in game_stint_df.columns:
                    all_game_lineup_stints.append(game_stint_df)
            except Exception:
                continue

        if not all_game_lineup_stints:
            continue

        # Sum across all games → season totals per lineup.
        season_lineup_totals_df = (
            pd.concat(all_game_lineup_stints)
            .groupby("Lineup")
            .sum(numeric_only=True)
            .reset_index()
        )

        # Minimum average possessions varies by scope and combo size.
        minimum_poss_threshold = 30 if game_scope == "conference" else 50

        for combo_size in [1, 2, 3, 5]:
            combo_df = compute_combo_stats(
                season_lineup_totals_df,
                combo_size=combo_size,
                min_avg_possessions=minimum_poss_threshold
            )
            if not combo_df.empty:
                combo_df["Team"] = team_id_to_display_name.get(team_id, f"Team {team_id}")
                combo_frames_by_size[combo_size].append(combo_df)

    # Write one CSV per combo size.
    for combo_size, frame_list in combo_frames_by_size.items():
        if frame_list:
            combined_df = (
                pd.concat(frame_list, ignore_index=True)
                .sort_values(["Team", "NetRtg"], ascending=[True, False])
            )
            output_csv = f"{combo_size}_man_{game_scope}_stats_{SEASON}.csv"
            combined_df.to_csv(output_csv, index=False)
            print(f"  Saved {output_csv} — {len(combined_df)} combos")


process_all_teams(qualifying_team_ids, "overall")
process_all_teams(qualifying_team_ids, "conference")

print("\nDone. 8 lineup CSVs saved.")


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
#
# SEASON                           int     Calendar year the season ends (e.g. 2026).
# play_by_play_df                  DataFrame  Full play-by-play for all games.
# player_box_df                    DataFrame  Player boxscore (starter flag, athlete names, team).
# schedule_df                      DataFrame  Full ESPN schedule.
# home_team_info_df                DataFrame  Home team columns from schedule.
# away_team_info_df                DataFrame  Away team columns from schedule.
# all_teams_info_df                DataFrame  Union of home + away teams; one row per team.
# team_id_to_display_name          dict     {team_id: "School Name"}.
# team_id_to_conference_id         dict     {team_id: conference_id}.
# MIN_GAMES_TO_QUALIFY             int     Minimum games for a team to be processed (15).
# schedule_teams_long_df           DataFrame  Melted schedule; one row per team per game.
# games_per_team                   Series   Distinct game count per team.
# qualifying_team_ids              list    team_ids with >= MIN_GAMES_TO_QUALIFY games.
#
# --- calculate_game_lineup_stints() ---
# game_id                          int/str  The game to process.
# team_id                          int     The team whose lineups we track.
# game_pbp_df                      DataFrame  PBP rows for this game only, sorted by play number.
# team_box_for_game                DataFrame  Box score rows for this team in this game.
# player_id_to_name                dict     {athlete_id: display_name} for this game.
# starting_player_ids              list    athlete_ids in the starting lineup.
# current_on_court_ids             set     athlete_ids currently on the court.
# current_stint_stats              dict    {lineup_tuple: {stat: total}}.
# lineup_tuple_key                 tuple   Sorted player IDs for the current lineup.
# current_lineup_accumulator       dict    Stat totals for the current lineup.
# seconds_in_stint                 int     Seconds elapsed since previous play.
# play_type_text / play_desc_text  str     Lowercase type and description for classification.
# play_team_id                     int     Team that made this play (-1 if unknown).
# is_this_team                     bool    True if this play belongs to the tracked team.
# substituting_player_id           int/str athlete_id of the player entering/leaving.
# stint_record_list                list    Row dicts ready for DataFrame construction.
#
# --- compute_combo_stats() ---
# full_lineup_aggregates_df        DataFrame  Season totals per 5-man lineup.
# combo_size                       int     1, 2, 3, or 5.
# min_avg_possessions              int     Minimum Avg_Poss for a combo to be included.
# combo_accumulator                dict    {combo_tuple: {stat: total}}.
# combo_rows                       list    List of dicts for DataFrame construction.
# combo_stats_df                   DataFrame  All combos with raw totals + computed rates.
# Poss_Off / Poss_Def              Series   Offensive/defensive possessions estimated per stint.
# Avg_Poss                         Series   (Poss_Off + Poss_Def) / 2 (possession quality filter).
# ORtg / DRtg / NetRtg             Series   Points per 100 possessions and net differential.
# per_100_stat_names               list    Stat names scaled to per-100 possessions.
#
# --- process_all_teams() ---
# game_scope                       str     "overall" or "conference".
# combo_frames_by_size             dict    {combo_size: [DataFrame, …]}.
# team_game_rows                   DataFrame  This team's game rows from the schedule.
# all_game_lineup_stints           list    List of per-game stint DataFrames.
# season_lineup_totals_df          DataFrame  Grouped sum across all games for this team.
# minimum_poss_threshold           int     50 (overall) or 30 (conference).
# combined_df                      DataFrame  All teams' combos for one size and scope.
