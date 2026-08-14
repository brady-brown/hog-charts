"""
Generates on/off splits and RAPM for both overall and conference games.

Run locally:
    python3 build_onoff_rapm.py

Outputs (season auto-detected from current date):
    mbb_onoff_{SEASON}_v2.csv          regular season
    mbb_onoff_{SEASON}_conf_v2.csv     conference games
    mbb_onoff_{SEASON}_all_v2.csv      regular + postseason (no RAPM)
    mbb_rapm_{SEASON-1}{SEASON%100:02d}.csv
    presence_full.parquet              regular season only
    player_lookup.csv

HIGH-LEVEL APPROACH
-------------------
1.  Load play-by-play (PBP), player box scores, team box scores, and schedule
    from ESPN via sportsdataverse.
2.  Split every game into "stints" — continuous segments of play where neither
    team makes a substitution.  Each stint has a fixed 5-man lineup for both
    home and away.
3.  Assign points scored and possessions to each stint for each team side.
4.  Build a "presence table": one row per (player, stint), marking whether the
    player was on or off the court.  Aggregating on/off gives each player's
    on-court and off-court net rating.
5.  Fit RAPM (Regularized Adjusted Plus-Minus) via Ridge regression over the
    design matrix, shrinking every player toward 0.

SCOPES
------
The pipeline runs three times.  "reg" (regular season) is the canonical pass —
it writes presence_full.parquet and the RAPM tables, and its on/off file is what
the site treats as the season on/off metric.  "conf" restricts to conference
games.  "all" covers regular + postseason and exists so the site's all-games
scope can compute stint-based advanced rates against what actually happened on
the floor instead of a minutes-share approximation; it skips the RAPM solve,
since RAPM stays a regular-season metric.
"""

import pandas as pd
import numpy as np
import warnings
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
import sportsdataverse.mbb as mbb

warnings.filterwarnings("ignore")

import os as _os
from datetime import date as _date

# ---------------------------------------------------------------------------
# Season detection
# ---------------------------------------------------------------------------
from hoglib.season import detect_season
SEASON = detect_season()

REGULAR_SEASON_TYPE = 2   # sportsdataverse code for regular season (1=pre, 3=post)
POSTSEASON_TYPE     = 3   # NCAA / NIT / etc.
# Season types the stint pipeline can ever see. Preseason is always dropped; the
# per-scope masks inside run_pipeline narrow this further.
PLAYED_SEASON_TYPES = [REGULAR_SEASON_TYPE, POSTSEASON_TYPE]

# ---------------------------------------------------------------------------
# Load raw data from ESPN (once, then filter inside run_pipeline)
# ---------------------------------------------------------------------------
print("Loading play-by-play...")
# Keep the full feed (all season types) for the assisted-FG share tables below,
# then narrow to regular + postseason: the "reg"/"conf" passes mask down to
# season_type 2, and the "all" pass needs the postseason rows too.
from hoglib import feeds  # feeds cached by build_ingest.py (step 0)
raw_play_by_play_full = feeds.load_pbp(SEASON)
raw_play_by_play = raw_play_by_play_full[
    raw_play_by_play_full["season_type"].isin(PLAYED_SEASON_TYPES)
].reset_index(drop=True)

print("Loading player box scores...")
raw_player_boxscores = feeds.load_player_box(SEASON)
raw_player_boxscores = raw_player_boxscores[
    raw_player_boxscores["season_type"].isin(PLAYED_SEASON_TYPES)
].reset_index(drop=True)

print("Loading team box scores...")
raw_team_boxscores = feeds.load_team_box(SEASON)
raw_team_boxscores = raw_team_boxscores[
    raw_team_boxscores["season_type"].isin(PLAYED_SEASON_TYPES)
].reset_index(drop=True)

print("Loading schedule...")
schedule_df = feeds.load_schedule(SEASON)
schedule_df = schedule_df[schedule_df["season_type"] == REGULAR_SEASON_TYPE]
conference_game_id_set = set(schedule_df.loc[schedule_df["conference_competition"] == True, "game_id"])

print(f"  PBP: {len(raw_play_by_play):,} plays | Players: {len(raw_player_boxscores):,} rows | Teams: {len(raw_team_boxscores):,} rows")


# ---------------------------------------------------------------------------
# Assisted-FG share tables (one per player-stats scope)
# ---------------------------------------------------------------------------
# Pure individual stat: of a player's OWN made field goals, how many a teammate
# set up. On a made-FG row athlete_id_1 is the scorer and athlete_id_2 is the
# assister (NaN when unassisted). This needs no stint/lineup coverage, so we
# compute it straight from the PBP for every scope — including postseason,
# which has no on/off pipeline. build_site reads astd_fgm / fgm_pbp and divides.
def _assisted_fg_share_table(pbp_frame):
    """Per-scorer assisted/total made-FG counts for a PBP subset."""
    is_free_throw = pbp_frame["type_text"].str.contains("FreeThrow", na=False)
    is_made_field_goal = (
        pbp_frame["shooting_play"].astype(bool)
        & ~is_free_throw
        & pbp_frame["scoring_play"].astype(bool)
    )
    scorer_id   = pd.to_numeric(pbp_frame["athlete_id_1"], errors="coerce")
    assister_id = pd.to_numeric(pbp_frame["athlete_id_2"], errors="coerce")
    made_field_goals = pd.DataFrame({
        "athlete_id": scorer_id[is_made_field_goal],
        "_is_assisted": assister_id[is_made_field_goal].notna().astype(int),
    }).dropna(subset=["athlete_id"])
    return (
        made_field_goals.groupby("athlete_id")
        .agg(astd_fgm=("_is_assisted", "sum"), fgm_pbp=("_is_assisted", "size"))
        .reset_index()
    )

print("Building assisted-FG share tables...")
_is_regular    = raw_play_by_play_full["season_type"] == REGULAR_SEASON_TYPE
_is_postseason = raw_play_by_play_full["season_type"] == POSTSEASON_TYPE   # NCAA + NIT + …
_is_conference = raw_play_by_play_full["game_id"].isin(conference_game_id_set)
assist_share_scopes = {
    "":         raw_play_by_play_full[_is_regular],
    "_all":     raw_play_by_play_full[_is_regular | _is_postseason],
    "_post":    raw_play_by_play_full[_is_postseason],
    "_conf":    raw_play_by_play_full[_is_regular & _is_conference],
    "_nonconf": raw_play_by_play_full[_is_regular & ~_is_conference],
}
for _scope_suffix, _scope_pbp in assist_share_scopes.items():
    _assisted_fg_share_table(_scope_pbp).to_csv(
        f"assist_share_{SEASON}{_scope_suffix}.csv", index=False)
    print(f"  assist_share_{SEASON}{_scope_suffix}.csv — {_scope_pbp['game_id'].nunique()} games")

# The full (all-season-type) PBP feed is only needed for the assisted-FG tables
# above. Free it before the memory-heavy on/off + RAPM passes below — retaining
# it for the whole run OOM-kills the 7 GB GitHub-hosted nightly runner.
del raw_play_by_play_full, assist_share_scopes, _scope_pbp
del _is_regular, _is_postseason, _is_conference
import gc as _gc
_gc.collect()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def _scope_mask(frame, game_filter):
    """Row mask selecting one scope's games out of a regular+postseason frame.

    reg  — regular season only (the canonical pass)
    conf — regular-season conference games
    all  — regular season + postseason
    """
    is_regular = frame["season_type"] == REGULAR_SEASON_TYPE
    if game_filter == "reg":
        return is_regular
    if game_filter == "conf":
        return is_regular & frame["game_id"].isin(conference_game_id_set)
    if game_filter == "all":
        return frame["season_type"].isin(PLAYED_SEASON_TYPES)
    raise ValueError(f"unknown game_filter: {game_filter!r}")


def run_pipeline(game_filter="reg", fit_rapm=True):
    """Run the on/off (and optionally RAPM) pipeline for one game scope.

    game_filter: "reg" | "conf" | "all"  (see _scope_mask).
    fit_rapm:    False skips the ridge solve and writes no RAPM table. RAPM is a
                 regular-season metric, so only the "reg" and "conf" passes fit it.

    Returns (presence_full, player_on_off_stats) for the scope.
    """
    filter_label = {"reg":  "regular season",
                    "conf": "conference only",
                    "all":  "regular + postseason"}[game_filter]
    print(f"\n{'='*60}")
    print(f"Running pipeline: {filter_label}")
    print(f"{'='*60}")

    # --- Filter raw data to the appropriate game set ---
    play_by_play     = raw_play_by_play[_scope_mask(raw_play_by_play, game_filter)].reset_index(drop=True)
    player_boxscores = raw_player_boxscores[_scope_mask(raw_player_boxscores, game_filter)].reset_index(drop=True)
    team_boxscores   = raw_team_boxscores[_scope_mask(raw_team_boxscores, game_filter)].reset_index(drop=True)

    print(f"  {play_by_play['game_id'].nunique()} games, {len(play_by_play):,} plays")

    # -----------------------------------------------------------------------
    # Prepare play-by-play
    # -----------------------------------------------------------------------
    base_columns = [
        "game_id", "sequence_number", "type_text", "text", "team_id",
        "athlete_id_1", "athlete_id_2", "scoring_play", "score_value",
        "shooting_play", "home_team_id", "away_team_id",
    ]
    has_points_attempted_column = "points_attempted" in play_by_play.columns
    if has_points_attempted_column:
        base_columns.append("points_attempted")
    play_by_play = play_by_play[[c for c in base_columns if c in play_by_play.columns]].copy()
    if not has_points_attempted_column:
        play_by_play["points_attempted"] = float("nan")

    play_by_play["sequence_number"]  = pd.to_numeric(play_by_play["sequence_number"],  errors="coerce")
    play_by_play["score_value"]      = pd.to_numeric(play_by_play["score_value"],      errors="coerce").fillna(0).astype(int)
    play_by_play["points_attempted"] = pd.to_numeric(play_by_play["points_attempted"], errors="coerce")
    play_by_play["scoring_play"]     = play_by_play["scoring_play"].astype(bool)
    play_by_play["shooting_play"]    = play_by_play["shooting_play"].astype(bool)
    for col in ["team_id", "athlete_id_1", "athlete_id_2", "home_team_id", "away_team_id"]:
        play_by_play[col] = pd.to_numeric(play_by_play[col], errors="coerce")

    # Boolean flags for every event type we care about.
    play_by_play["is_substitution"]     = play_by_play["type_text"] == "Substitution"
    play_by_play["is_subbing_in"]       = play_by_play["is_substitution"] & play_by_play["text"].str.contains("subbing in",  case=False, na=False)
    play_by_play["is_subbing_out"]      = play_by_play["is_substitution"] & play_by_play["text"].str.contains("subbing out", case=False, na=False)
    play_by_play["is_free_throw"]       = play_by_play["type_text"].str.contains("FreeThrow", na=False)
    play_by_play["is_field_goal_att"]   = play_by_play["shooting_play"] & ~play_by_play["is_free_throw"]
    play_by_play["is_turnover"]         = play_by_play["type_text"].str.contains("Turnover",  na=False)
    play_by_play["is_field_goal_made"]  = play_by_play["is_field_goal_att"] & play_by_play["scoring_play"]
    play_by_play["is_three_att"]        = play_by_play["is_field_goal_att"] & (play_by_play["points_attempted"] == 3)
    play_by_play["is_three_made"]       = play_by_play["is_three_att"] & play_by_play["scoring_play"]
    play_by_play["is_free_throw_made"]  = play_by_play["is_free_throw"] & play_by_play["scoring_play"]
    play_by_play["is_off_rebound"]      = play_by_play["type_text"].str.contains("Offensive Rebound", na=False)
    play_by_play["is_def_rebound"]      = play_by_play["type_text"].str.contains("Defensive Rebound", na=False)
    play_by_play = play_by_play.sort_values(["game_id", "sequence_number"]).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Game-level possession estimates from team box scores
    # (used later to scale stint-level possession weights)
    # -----------------------------------------------------------------------
    team_boxscores["box_possessions"] = (
        team_boxscores["field_goals_attempted"]
        + 0.44 * team_boxscores["free_throws_attempted"]
        + team_boxscores["total_turnovers"]
        - team_boxscores["offensive_rebounds"]
    )
    team_boxscores["team_id_numeric"] = pd.to_numeric(team_boxscores["team_id"], errors="coerce")
    game_team_possession_dict = (
        team_boxscores.set_index(["game_id", "team_id_numeric"])["box_possessions"].to_dict()
    )

    # -----------------------------------------------------------------------
    # Starters
    # -----------------------------------------------------------------------
    player_boxscores["athlete_id"]   = pd.to_numeric(player_boxscores["athlete_id"],  errors="coerce")
    player_boxscores["team_id_numeric"] = pd.to_numeric(player_boxscores["team_id"], errors="coerce")
    starters_by_game_team = (
        player_boxscores[(player_boxscores["starter"] == True) & player_boxscores["athlete_id"].notna()]
        .groupby(["game_id", "team_id_numeric"])["athlete_id"]
        .apply(frozenset).to_dict()
    )

    # -----------------------------------------------------------------------
    # Build stints
    # Each stint is a period between two substitutions where lineups are fixed.
    # -----------------------------------------------------------------------
    print("  Building stints...")
    from hoglib import stints
    play_by_play, stint_lineup_info = stints.build_stints(play_by_play, starters_by_game_team)
    print(f"  {len(stint_lineup_info):,} stints built")

    # -----------------------------------------------------------------------
    # Aggregate per-stint statistics
    # -----------------------------------------------------------------------

    # --- Points scored per (game, stint, scoring_team) ---
    scoring_plays_df = play_by_play[play_by_play["scoring_play"]].copy()
    points_by_stint_team = (
        scoring_plays_df.groupby(["game_id", "stint_id", "team_id"])["score_value"]
        .sum().reset_index()
        .rename(columns={"score_value": "pts_scored", "team_id": "scoring_team_id"})
    )

    # --- Possessions per (game, stint, team) ---
    # Each field goal attempt = 1 possession, each FT counts as 0.44 possessions,
    # each turnover = 1 possession.  This is the standard Possessions formula.
    possession_events = play_by_play[
        play_by_play["is_field_goal_att"] | play_by_play["is_free_throw"] | play_by_play["is_turnover"]
    ].copy()
    possession_events["possession_weight"] = np.where(
        possession_events["is_field_goal_att"], 1.0,
        np.where(possession_events["is_free_throw"], 0.44, 1.0)
    )
    stint_possession_weights_df = (
        possession_events.groupby(["game_id", "stint_id", "team_id"])["possession_weight"]
        .sum().reset_index()
        .rename(columns={"team_id": "possession_team_id"})
    )

    # Scale stint-level possession fractions up to match the team's box-score
    # total (PBP undercounts slightly).
    game_total_possession_weights = (
        stint_possession_weights_df.groupby(["game_id", "possession_team_id"])["possession_weight"]
        .sum().reset_index()
        .rename(columns={"possession_weight": "game_possession_weight_total",
                         "possession_team_id": "team_id_for_join"})
    )
    game_possession_estimates = pd.DataFrame(
        [(g, t, p) for (g, t), p in game_team_possession_dict.items()],
        columns=["game_id", "team_id_for_join", "box_score_possessions"]
    )
    game_total_possession_weights = game_total_possession_weights.merge(
        game_possession_estimates, on=["game_id", "team_id_for_join"], how="left"
    )
    stint_possession_weights_df = stint_possession_weights_df.merge(
        game_total_possession_weights.rename(columns={"team_id_for_join": "possession_team_id"}),
        on=["game_id", "possession_team_id"], how="left"
    )
    stint_possession_weights_df["scaled_possessions"] = np.where(
        stint_possession_weights_df["game_possession_weight_total"] > 0,
        stint_possession_weights_df["possession_weight"]
            / stint_possession_weights_df["game_possession_weight_total"]
            * stint_possession_weights_df["box_score_possessions"],
        0
    )

    # --- Shooting stats per (game, stint, team) ---
    # Maps each counting stat name to the boolean flag column in play_by_play.
    PLAY_TYPE_TO_FLAG_MAP = {
        "fga": "is_field_goal_att",
        "fgm": "is_field_goal_made",
        "tpa": "is_three_att",
        "tpm": "is_three_made",
        "fta": "is_free_throw",
        "ftm": "is_free_throw_made",
        "orb": "is_off_rebound",
        "drb": "is_def_rebound",
    }
    COUNTING_STAT_NAMES = list(PLAY_TYPE_TO_FLAG_MAP.keys())
    shooting_stats_by_stint_team = None
    for stat_name, flag_column in PLAY_TYPE_TO_FLAG_MAP.items():
        stat_counts = (
            play_by_play[play_by_play[flag_column]]
            .groupby(["game_id", "stint_id", "team_id"])
            .size().reset_index(name=stat_name)
        )
        shooting_stats_by_stint_team = (
            stat_counts if shooting_stats_by_stint_team is None
            else shooting_stats_by_stint_team.merge(
                stat_counts, on=["game_id", "stint_id", "team_id"], how="outer"
            )
        )
    shooting_stats_by_stint_team = shooting_stats_by_stint_team.fillna(0)
    for stat_name in COUNTING_STAT_NAMES:
        shooting_stats_by_stint_team[stat_name] = shooting_stats_by_stint_team[stat_name].astype(int)

    # -----------------------------------------------------------------------
    # Pivot home vs. away — each stint row gets home and away columns
    # -----------------------------------------------------------------------
    stint_stats = stint_lineup_info[
        ["game_id", "stint_id", "home_team_id", "away_team_id"]
    ].copy()

    def _aggregate_side(df, points_or_poss_col, output_col, group_cols):
        return (df.groupby(group_cols)[points_or_poss_col]
                .sum().reset_index()
                .rename(columns={points_or_poss_col: output_col}))

    # Points
    points_with_home_info = points_by_stint_team.merge(
        stint_lineup_info[["game_id", "stint_id", "home_team_id"]],
        on=["game_id", "stint_id"]
    )
    points_with_home_info["is_home_team"] = (
        points_with_home_info["scoring_team_id"] == points_with_home_info["home_team_id"]
    )
    home_points_by_stint = _aggregate_side(
        points_with_home_info[points_with_home_info["is_home_team"]],
        "pts_scored", "home_pts", ["game_id", "stint_id"]
    )
    away_points_by_stint = _aggregate_side(
        points_with_home_info[~points_with_home_info["is_home_team"]],
        "pts_scored", "away_pts", ["game_id", "stint_id"]
    )
    stint_stats = stint_stats.merge(home_points_by_stint, on=["game_id", "stint_id"], how="left")
    stint_stats = stint_stats.merge(away_points_by_stint, on=["game_id", "stint_id"], how="left")

    # Possessions
    poss_with_home_info = stint_possession_weights_df.merge(
        stint_lineup_info[["game_id", "stint_id", "home_team_id"]],
        on=["game_id", "stint_id"]
    )
    poss_with_home_info["is_home_team"] = (
        poss_with_home_info["possession_team_id"] == poss_with_home_info["home_team_id"]
    )
    home_possessions_by_stint = _aggregate_side(
        poss_with_home_info[poss_with_home_info["is_home_team"]],
        "scaled_possessions", "home_poss", ["game_id", "stint_id"]
    )
    away_possessions_by_stint = _aggregate_side(
        poss_with_home_info[~poss_with_home_info["is_home_team"]],
        "scaled_possessions", "away_poss", ["game_id", "stint_id"]
    )
    stint_stats = stint_stats.merge(home_possessions_by_stint, on=["game_id", "stint_id"], how="left")
    stint_stats = stint_stats.merge(away_possessions_by_stint, on=["game_id", "stint_id"], how="left")
    stint_stats[["home_pts", "away_pts", "home_poss", "away_poss"]] = (
        stint_stats[["home_pts", "away_pts", "home_poss", "away_poss"]].fillna(0)
    )

    # Shooting stats (side-split)
    shooting_with_home_info = shooting_stats_by_stint_team.merge(
        stint_lineup_info[["game_id", "stint_id", "home_team_id"]],
        on=["game_id", "stint_id"]
    )
    shooting_with_home_info["is_home_team"] = (
        shooting_with_home_info["team_id"] == shooting_with_home_info["home_team_id"]
    )
    home_shooting_by_stint = (shooting_with_home_info[shooting_with_home_info["is_home_team"]]
                               .groupby(["game_id", "stint_id"])[COUNTING_STAT_NAMES].sum().reset_index())
    away_shooting_by_stint = (shooting_with_home_info[~shooting_with_home_info["is_home_team"]]
                               .groupby(["game_id", "stint_id"])[COUNTING_STAT_NAMES].sum().reset_index())
    home_shooting_by_stint = home_shooting_by_stint.rename(columns={c: f"home_{c}" for c in COUNTING_STAT_NAMES})
    away_shooting_by_stint = away_shooting_by_stint.rename(columns={c: f"away_{c}" for c in COUNTING_STAT_NAMES})
    stint_stats = stint_stats.merge(home_shooting_by_stint, on=["game_id", "stint_id"], how="left")
    stint_stats = stint_stats.merge(away_shooting_by_stint, on=["game_id", "stint_id"], how="left")
    shooting_stat_columns = ([f"home_{c}" for c in COUNTING_STAT_NAMES]
                              + [f"away_{c}" for c in COUNTING_STAT_NAMES])
    stint_stats[shooting_stat_columns] = stint_stats[shooting_stat_columns].fillna(0).astype(int)

    # -----------------------------------------------------------------------
    # Presence table
    # One row per (player, stint) — on_court = True if player was in the lineup.
    # -----------------------------------------------------------------------
    print("  Building presence table...")
    # "active" = played at least one minute (not DNP).
    active_players_by_game_team = (
        player_boxscores[player_boxscores["did_not_play"] != True]
        .dropna(subset=["athlete_id", "minutes"])
        .groupby(["game_id", "team_id_numeric"])["athlete_id"]
        .apply(set).to_dict()
    )

    presence_rows = []
    for _, stint_row in stint_lineup_info.iterrows():
        game_id      = stint_row["game_id"]
        stint_id     = stint_row["stint_id"]
        home_team_id = stint_row["home_team_id"]
        away_team_id = stint_row["away_team_id"]
        home_lineup_frozenset = stint_row["home_lineup"]
        away_lineup_frozenset = stint_row["away_lineup"]

        for team_id, lineup_frozenset in [(home_team_id, home_lineup_frozenset),
                                          (away_team_id, away_lineup_frozenset)]:
            for player_id in active_players_by_game_team.get((game_id, team_id), set()):
                presence_rows.append({
                    "game_id":      game_id,
                    "stint_id":     stint_id,
                    "athlete_id":   player_id,
                    "team_id":      team_id,
                    "is_on_court":  player_id in lineup_frozenset,
                })

    presence_df = pd.DataFrame(presence_rows)
    presence_full = presence_df.merge(
        stint_stats[[
            "game_id", "stint_id", "home_team_id", "away_team_id",
            "home_pts", "away_pts", "home_poss", "away_poss"
        ] + shooting_stat_columns],
        on=["game_id", "stint_id"], how="left"
    )

    # Assign each row the correct "for" and "against" from team perspective.
    presence_full["is_home_team"]  = presence_full["team_id"] == presence_full["home_team_id"]
    presence_full["pts_for"]       = np.where(presence_full["is_home_team"], presence_full["home_pts"], presence_full["away_pts"])
    presence_full["pts_against"]   = np.where(presence_full["is_home_team"], presence_full["away_pts"], presence_full["home_pts"])
    presence_full["poss_off"]      = np.where(presence_full["is_home_team"], presence_full["home_poss"], presence_full["away_poss"])
    presence_full["poss_def"]      = np.where(presence_full["is_home_team"], presence_full["away_poss"], presence_full["home_poss"])
    for stat_name in COUNTING_STAT_NAMES:
        presence_full[f"{stat_name}_for"]     = np.where(presence_full["is_home_team"],
                                                          presence_full[f"home_{stat_name}"],
                                                          presence_full[f"away_{stat_name}"])
        presence_full[f"{stat_name}_against"] = np.where(presence_full["is_home_team"],
                                                          presence_full[f"away_{stat_name}"],
                                                          presence_full[f"home_{stat_name}"])

    # -----------------------------------------------------------------------
    # Aggregate on/off splits
    # -----------------------------------------------------------------------
    print("  Aggregating on/off stats...")
    shooting_for_and_against_cols = (
        [f"{c}_for" for c in COUNTING_STAT_NAMES]
        + [f"{c}_against" for c in COUNTING_STAT_NAMES]
    )
    on_off_aggregates = (
        presence_full
        .groupby(["athlete_id", "team_id", "is_on_court"])
        [["pts_for", "pts_against", "poss_off", "poss_def"] + shooting_for_and_against_cols]
        .sum().reset_index()
    )

    on_court_stats  = (on_off_aggregates[on_off_aggregates["is_on_court"]].drop(columns="is_on_court")
                       .add_suffix("_on")
                       .rename(columns={"athlete_id_on": "athlete_id", "team_id_on": "team_id"}))
    off_court_stats = (on_off_aggregates[~on_off_aggregates["is_on_court"]].drop(columns="is_on_court")
                       .add_suffix("_off")
                       .rename(columns={"athlete_id_off": "athlete_id", "team_id_off": "team_id"}))
    player_on_off_stats = (on_court_stats
                            .merge(off_court_stats, on=["athlete_id", "team_id"], how="outer")
                            .fillna(0))

    def _net_rating(points_scored, possessions):
        """Points per 100 possessions; NaN when fewer than 10 possessions."""
        return np.where(possessions > 10, points_scored / possessions * 100, np.nan)

    def _safe_divide(numerator, denominator):
        return np.where(denominator > 0, numerator / denominator, np.nan)

    player_on_off_stats["ortg_on"]  = _net_rating(player_on_off_stats["pts_for_on"],     player_on_off_stats["poss_off_on"])
    player_on_off_stats["drtg_on"]  = _net_rating(player_on_off_stats["pts_against_on"], player_on_off_stats["poss_def_on"])
    player_on_off_stats["nrtg_on"]  = player_on_off_stats["ortg_on"]  - player_on_off_stats["drtg_on"]
    player_on_off_stats["ortg_off"] = _net_rating(player_on_off_stats["pts_for_off"],     player_on_off_stats["poss_off_off"])
    player_on_off_stats["drtg_off"] = _net_rating(player_on_off_stats["pts_against_off"], player_on_off_stats["poss_def_off"])
    player_on_off_stats["nrtg_off"] = player_on_off_stats["ortg_off"] - player_on_off_stats["drtg_off"]
    player_on_off_stats["on_off"]   = player_on_off_stats["nrtg_on"]  - player_on_off_stats["nrtg_off"]

    # Shooting split ratings for both on-court and off-court.
    for court_suffix in ["on", "off"]:
        fga  = player_on_off_stats[f"fga_for_{court_suffix}"]
        fgm  = player_on_off_stats[f"fgm_for_{court_suffix}"]
        tpa  = player_on_off_stats[f"tpa_for_{court_suffix}"]
        tpm  = player_on_off_stats[f"tpm_for_{court_suffix}"]
        fta  = player_on_off_stats[f"fta_for_{court_suffix}"]
        ftm  = player_on_off_stats[f"ftm_for_{court_suffix}"]
        orb  = player_on_off_stats[f"orb_for_{court_suffix}"]
        drb  = player_on_off_stats[f"drb_for_{court_suffix}"]
        poss_off_side  = player_on_off_stats[f"poss_off_{court_suffix}"]
        poss_def_side  = player_on_off_stats[f"poss_def_{court_suffix}"]
        opp_fga = player_on_off_stats[f"fga_against_{court_suffix}"]
        opp_fgm = player_on_off_stats[f"fgm_against_{court_suffix}"]
        opp_tpa = player_on_off_stats[f"tpa_against_{court_suffix}"]
        opp_tpm = player_on_off_stats[f"tpm_against_{court_suffix}"]
        opp_fta = player_on_off_stats[f"fta_against_{court_suffix}"]
        opp_orb = player_on_off_stats[f"orb_against_{court_suffix}"]
        opp_drb = player_on_off_stats[f"drb_against_{court_suffix}"]

        player_on_off_stats[f"fg_pct_{court_suffix}"]       = _safe_divide(fgm, fga)
        player_on_off_stats[f"efg_pct_{court_suffix}"]      = _safe_divide(fgm + 0.5 * tpm, fga)
        player_on_off_stats[f"3p_pct_{court_suffix}"]       = _safe_divide(tpm, tpa)
        player_on_off_stats[f"3p_rate_{court_suffix}"]      = _safe_divide(tpa, fga)
        player_on_off_stats[f"ft_rate_{court_suffix}"]      = _safe_divide(fta, fga)
        player_on_off_stats[f"ft_pct_{court_suffix}"]       = _safe_divide(ftm, fta)
        player_on_off_stats[f"opp_fg_pct_{court_suffix}"]   = _safe_divide(opp_fgm, opp_fga)
        player_on_off_stats[f"opp_efg_pct_{court_suffix}"]  = _safe_divide(opp_fgm + 0.5 * opp_tpm, opp_fga)
        player_on_off_stats[f"opp_3p_pct_{court_suffix}"]   = _safe_divide(opp_tpm, opp_tpa)
        player_on_off_stats[f"opp_3p_rate_{court_suffix}"]  = _safe_divide(opp_tpa, opp_fga)
        player_on_off_stats[f"drb_pct_{court_suffix}"]      = _safe_divide(drb, drb + opp_orb)
        player_on_off_stats[f"orb_pct_{court_suffix}"]      = _safe_divide(orb, orb + opp_drb)
        player_on_off_stats[f"orb_per100_{court_suffix}"]      = _safe_divide(orb,     poss_off_side) * 100
        player_on_off_stats[f"drb_per100_{court_suffix}"]      = _safe_divide(drb,     poss_def_side) * 100
        player_on_off_stats[f"opp_orb_per100_{court_suffix}"]  = _safe_divide(opp_orb, poss_def_side) * 100

    # Attach player and team names.
    player_name_team_lookup = (
        player_boxscores[["athlete_id", "team_id_numeric",
                          "athlete_display_name", "team_display_name"]]
        .drop_duplicates()
        .rename(columns={"team_id_numeric": "team_id"})
    )
    player_on_off_stats = player_on_off_stats.merge(
        player_name_team_lookup, on=["athlete_id", "team_id"], how="left"
    )
    on_off_results = player_on_off_stats.copy().sort_values("on_off", ascending=False)

    # -----------------------------------------------------------------------
    # Save on/off CSV
    # -----------------------------------------------------------------------
    output_column_names = [
        "athlete_id", "athlete_display_name", "team_id", "team_display_name",
        "ortg_on", "drtg_on", "nrtg_on", "ortg_off", "drtg_off", "nrtg_off", "on_off",
        "ortg_on_pct", "drtg_on_pct", "nrtg_on_pct",
        "ortg_off_pct", "drtg_off_pct", "nrtg_off_pct", "on_off_pct",
        "fg_pct_on", "efg_pct_on", "3p_pct_on", "3p_rate_on", "ft_rate_on", "ft_pct_on",
        "opp_fg_pct_on", "opp_efg_pct_on", "opp_3p_pct_on", "opp_3p_rate_on",
        "drb_pct_on", "orb_pct_on", "drb_per100_on", "orb_per100_on", "opp_orb_per100_on",
        "poss_off_on", "poss_def_on", "poss_off_off", "poss_def_off",
        "pts_for_on", "pts_against_on", "pts_for_off", "pts_against_off",
        "fga_for_on", "fgm_for_on", "tpa_for_on", "tpm_for_on",
        "fga_against_on", "fgm_against_on", "tpa_against_on", "tpm_against_on",
        "orb_for_on", "drb_for_on", "orb_against_on", "drb_against_on",
    ]
    for percentile_col in ["ortg_on_pct", "drtg_on_pct", "nrtg_on_pct",
                           "ortg_off_pct", "drtg_off_pct", "nrtg_off_pct", "on_off_pct"]:
        if percentile_col not in on_off_results.columns:
            on_off_results[percentile_col] = np.nan

    # The regular-season pass keeps the unsuffixed filename: downstream scripts
    # treat mbb_onoff_{SEASON}_v2.csv as THE season on/off table.
    filename_suffix = "" if game_filter == "reg" else f"_{game_filter}"
    onoff_output_filename = f"mbb_onoff_{SEASON}{filename_suffix}_v2.csv"
    on_off_results[[c for c in output_column_names if c in on_off_results.columns]].round(4).to_csv(
        onoff_output_filename, index=False
    )
    print(f"  Saved {onoff_output_filename} — {len(on_off_results)} players")

    if not fit_rapm:
        print("  Skipping RAPM (regular-season metric)")
        return presence_full, player_on_off_stats

    # -----------------------------------------------------------------------
    # RAPM — Regularized Adjusted Plus-Minus
    # -----------------------------------------------------------------------
    print("  Fitting RAPM...")
    rapm_stint_data = stint_stats.merge(
        stint_lineup_info[["game_id", "stint_id", "home_lineup", "away_lineup"]],
        on=["game_id", "stint_id"]
    ).copy()

    # Build the universe of all player IDs who appeared in any lineup.
    all_players = sorted({
        player_id
        for _, row in rapm_stint_data[["home_lineup", "away_lineup"]].iterrows()
        for lineup in (row["home_lineup"], row["away_lineup"])
        for player_id in lineup
        if pd.notna(player_id)
    })
    player_to_matrix_index = {player_id: idx for idx, player_id in enumerate(all_players)}
    num_players = len(all_players)

    # Build sparse design matrix X of shape (num_stints * 2, num_players * 2).
    # For each (stint, side) observation:
    #   - Offensive players get +1 in their offensive column (cols 0..n-1)
    #   - Defensive players get -1 in their defensive column (cols n..2n-1)
    # The target y is the net rating (pts / poss * 100) for the offensive team.
    matrix_row_indices, matrix_col_indices, matrix_values = [], [], []
    net_ratings_per_stint, possession_weights_per_stint = [], []
    stint_observation_count = 0

    for row in rapm_stint_data.itertuples():
        for offensive_lineup, defensive_lineup, points_scored, possessions in [
            (row.home_lineup, row.away_lineup, row.home_pts, row.home_poss),
            (row.away_lineup, row.home_lineup, row.away_pts, row.away_poss),
        ]:
            if possessions <= 0:
                continue
            if len(offensive_lineup) != 5 or len(defensive_lineup) != 5:
                continue
            for player_id in offensive_lineup:
                if pd.isna(player_id):
                    continue
                matrix_row_indices.append(stint_observation_count)
                matrix_col_indices.append(player_to_matrix_index[player_id])
                matrix_values.append(1.0)
            for player_id in defensive_lineup:
                if pd.isna(player_id):
                    continue
                matrix_row_indices.append(stint_observation_count)
                matrix_col_indices.append(player_to_matrix_index[player_id] + num_players)
                matrix_values.append(-1.0)
            net_ratings_per_stint.append(points_scored / possessions * 100)
            possession_weights_per_stint.append(possessions)
            stint_observation_count += 1

    RIDGE_REGULARIZATION_ALPHA = 4000  # tuned to balance bias-variance in backtests

    # Assemble the Ridge regression inputs.
    stint_design_matrix     = csr_matrix(
        (matrix_values, (matrix_row_indices, matrix_col_indices)),
        shape=(stint_observation_count, 2 * num_players)
    )
    observed_net_ratings     = np.array(net_ratings_per_stint)
    stint_possession_weights = np.array(possession_weights_per_stint)

    league_avg_net_rating    = np.average(observed_net_ratings, weights=stint_possession_weights)
    sqrt_possession_weights  = np.sqrt(stint_possession_weights)
    centered_net_ratings     = observed_net_ratings - league_avg_net_rating
    # Weighted Ridge: multiply both X and y by sqrt(w) so Ridge minimizes
    # the possession-weighted residual sum of squares.
    weighted_design_matrix   = stint_design_matrix.multiply(sqrt_possession_weights[:, None]).tocsr()

    # RAPM: every player shrinks toward 0.
    ridge_model = Ridge(alpha=RIDGE_REGULARIZATION_ALPHA, fit_intercept=False)
    ridge_model.fit(weighted_design_matrix, centered_net_ratings * sqrt_possession_weights)
    off_rapm = ridge_model.coef_[:num_players]
    def_rapm = ridge_model.coef_[num_players:]

    # Assemble output DataFrame.
    raw_rapm_estimates = pd.DataFrame({
        "athlete_id":  all_players,
        "o_rapm":      off_rapm,
        "d_rapm":      def_rapm,
        "rapm":        off_rapm + def_rapm,
    })

    # Possession totals for the minimum-sample filter.
    player_on_court_possessions = (
        presence_full[presence_full["is_on_court"]]
        .groupby("athlete_id")["poss_off"].sum().reset_index()
        .rename(columns={"poss_off": "total_poss"})
    )
    player_name_most_recent = (
        player_boxscores[["athlete_id", "athlete_display_name",
                          "team_id", "team_display_name", "game_id"]]
        .sort_values("game_id").drop_duplicates("athlete_id", keep="last")
        .drop(columns="game_id")
        .rename(columns={"team_id": "team_id"})
    )
    qualified_rapm_players = (
        raw_rapm_estimates
        .merge(player_name_most_recent, on="athlete_id", how="left")
        .merge(player_on_court_possessions, on="athlete_id", how="left")
        .query("total_poss >= 500")
        .sort_values("rapm", ascending=False)
        .reset_index(drop=True)
    )
    for rapm_col in ["o_rapm", "d_rapm"]:
        qualified_rapm_players[rapm_col] = qualified_rapm_players[rapm_col].round(2)
    qualified_rapm_players["rapm"] = (qualified_rapm_players["o_rapm"]
                                      + qualified_rapm_players["d_rapm"])

    season_file_suffix  = f"{SEASON - 1}{str(SEASON)[2:]}"
    rapm_output_filename = f"mbb_rapm_{season_file_suffix}{filename_suffix}.csv"
    qualified_rapm_players.to_csv(rapm_output_filename, index=False)
    print(f"  Saved {rapm_output_filename} — {len(qualified_rapm_players)} qualified players")

    return presence_full, player_on_off_stats


# ---------------------------------------------------------------------------
# Run every scope and save the shared outputs
# ---------------------------------------------------------------------------
# "reg" is the canonical pass — its presence table and player lookup are what the
# rest of the pipeline consumes, and it owns the unsuffixed filenames.
presence_regular, player_on_off_regular = run_pipeline("reg")
run_pipeline("conf")
# Regular + postseason. Exists so build_site can compute stint-based advanced
# rates for the all-games scope instead of the minutes-share approximation. No
# RAPM: that stays a regular-season metric, and skipping the solve keeps the
# added nightly cost to the stint/presence build alone.
run_pipeline("all", fit_rapm=False)

# Presence parquet: used downstream by build_lineups.py for WOWY synergy, and by
# build_points_resp.py for the on-court points denominator. REGULAR SEASON ONLY —
# Resp% is documented as a regular-season metric and depends on this staying so.
SHOT_STAT_NAMES = ["fga", "fgm", "tpa", "tpm", "fta", "ftm", "orb", "drb"]
columns_to_save = (
    ["game_id", "stint_id", "athlete_id", "team_id", "is_on_court",
     "pts_for", "pts_against", "poss_off", "poss_def"]
    + [f"{c}_for"     for c in SHOT_STAT_NAMES]
    + [f"{c}_against" for c in SHOT_STAT_NAMES]
)
presence_regular[[c for c in columns_to_save if c in presence_regular.columns]].to_parquet(
    "presence_full.parquet", index=False
)

# Player lookup: a lightweight name/team map used by the web app.
player_on_off_regular[
    ["athlete_id", "athlete_display_name", "team_id", "team_display_name"]
].drop_duplicates("athlete_id").reset_index(drop=True).to_csv("player_lookup.csv", index=False)

_season_suffix = f"{SEASON-1}{str(SEASON)[2:]}"
print("\nDone. Files saved:")
print(f"  mbb_onoff_{SEASON}_v2.csv")
print(f"  mbb_onoff_{SEASON}_conf_v2.csv")
print(f"  mbb_onoff_{SEASON}_all_v2.csv")
print(f"  mbb_rapm_{_season_suffix}.csv")
print(f"  mbb_rapm_{_season_suffix}_conf.csv")
print("  presence_full.parquet")
print("  player_lookup.csv")


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
#
# SEASON                      int   Calendar year the season ENDS (e.g. 2026 for the 2025-26 season).
# REGULAR_SEASON_TYPE         int   sportsdataverse code for regular-season games (2).
# POSTSEASON_TYPE             int   sportsdataverse code for postseason games (3).
# PLAYED_SEASON_TYPES         list  Season types the stint pipeline can see ([2, 3]).
# _season_override            str   Value of OVERRIDE_SEASON env var; used to build historical seasons.
#
# raw_play_by_play            DataFrame   Every play from every regular-season + postseason game.
# raw_player_boxscores        DataFrame   Per-player per-game box score stats (regular + postseason).
# raw_team_boxscores          DataFrame   Per-team per-game box score stats (regular + postseason).
# schedule_df                 DataFrame   Game schedule; used to identify conference games.
# conference_game_id_set      set[int]    game_ids flagged as conference competition.
#
# --- Inside run_pipeline() ---
# play_by_play                DataFrame   PBP filtered to the current game_filter.
# player_boxscores            DataFrame   Player box scores filtered to current game_filter.
# team_boxscores              DataFrame   Team box scores filtered to current game_filter.
# game_team_possession_dict   dict        {(game_id, team_id): box_possessions} — ground-truth poss counts.
# starters_by_game_team       dict        {(game_id, team_id): frozenset(athlete_ids)} — opening lineups.
#
# play_stint_ids              ndarray     Stint number assigned to every play row.
# lineup_change_records       list[dict]  One dict per lineup state transition (game × stint).
# stint_lineup_info           DataFrame   Deduplicated lineup state per (game_id, stint_id).
# stint_stats                 DataFrame   Aggregated pts/poss/shooting for each stint (home & away).
#
# PLAY_TYPE_TO_FLAG_MAP       dict        Maps short stat name (e.g. "fga") to its boolean flag column.
# COUNTING_STAT_NAMES         list[str]   Short names of counting stats: fga, fgm, tpa, tpm, fta, ftm, orb, drb.
# shooting_stats_by_stint_team DataFrame  Counting stat totals per (game, stint, team).
# shooting_stat_columns       list[str]   Column names for home_*/away_* shooting stats in stint_stats.
#
# points_by_stint_team        DataFrame   Points scored per (game_id, stint_id, scoring_team).
# possession_events           DataFrame   PBP rows that represent an offensive possession ending.
# stint_possession_weights_df DataFrame   Possession totals per (game_id, stint_id, team).
# game_total_possession_weights DataFrame Denominator for scaling: total possession weight per (game, team).
# game_possession_estimates   DataFrame   Box-score-derived possession counts per (game, team).
#
# active_players_by_game_team dict        {(game_id, team_id): set(athlete_ids)} — players who actually played.
# presence_rows               list[dict]  Raw rows for the presence table (player × stint × on/off).
# presence_df                 DataFrame   Presence table before merging stint stats.
# presence_full               DataFrame   Presence table with pts/poss/shooting attached per stint.
#
# on_off_aggregates           DataFrame   Sum of on-court and off-court stats by (athlete, team, is_on_court).
# on_court_stats              DataFrame   Aggregates for is_on_court=True rows.
# off_court_stats             DataFrame   Aggregates for is_on_court=False rows.
# player_on_off_stats         DataFrame   One row per player with both on and off totals merged.
# player_name_team_lookup     DataFrame   Athlete names + team names looked up from box scores.
# on_off_results              DataFrame   player_on_off_stats sorted by on_off descending.
# output_column_names         list[str]   Columns to retain in the final on/off CSV.
# onoff_output_filename       str         e.g. "mbb_onoff_2026_v2.csv".
# filename_suffix             str         "" for regular season, "_conf" / "_all" otherwise.
#
# --- RAPM section ---
# rapm_stint_data             DataFrame   Stint stats merged with home/away lineups for RAPM input.
# all_players                 list[int]   Sorted list of all player IDs who appeared in any lineup.
# player_to_matrix_index      dict        {athlete_id: column_index} for the design matrix.
# num_players                 int         Number of unique players; matrix has 2*num_players columns.
# matrix_row_indices          list[int]   Row index for each nonzero entry in the design matrix.
# matrix_col_indices          list[int]   Column index for each nonzero entry.
# matrix_values               list[float] +1 (offensive player) or -1 (defensive player).
# net_ratings_per_stint       list[float] Observed net rating (pts/poss*100) for each stint observation.
# possession_weights_per_stint list[float] Possession count for each stint observation (regression weight).
# stint_observation_count     int         Total number of (stint, side) rows in the design matrix.
# RIDGE_REGULARIZATION_ALPHA  int         Ridge penalty λ; higher = more shrinkage toward 0.
# stint_design_matrix         csr_matrix  Sparse X, shape (observations, 2*num_players).
# observed_net_ratings        ndarray     Target y: net rating per observation.
# stint_possession_weights    ndarray     Regression weights w (possession counts).
# league_avg_net_rating       float       Possession-weighted mean net rating across all stints.
# sqrt_possession_weights     ndarray     √w; used to convert Ridge to a WLS problem.
# centered_net_ratings        ndarray     y - ȳ (mean-centered target for Ridge).
# weighted_design_matrix      csr_matrix  X * √w elementwise (for WLS via Ridge).
# ridge_model                 Ridge       sklearn Ridge, every player shrunk toward 0.
# off_rapm                    ndarray     Offensive RAPM coefficients.
# def_rapm                    ndarray     Defensive RAPM coefficients.
# raw_rapm_estimates          DataFrame   RAPM estimates for every player.
# player_on_court_possessions DataFrame   Total on-court possessions per player (for minimum filter).
# player_name_most_recent     DataFrame   Most recent name/team per athlete_id.
# qualified_rapm_players      DataFrame   Players with ≥500 on-court possessions; sorted by RAPM.
# season_file_suffix          str         e.g. "202526" — used in filenames.
# rapm_output_filename        str         e.g. "mbb_rapm_202526.csv".
#
# --- Top-level outputs ---
# presence_regular            DataFrame   Full presence table from the regular-season run.
# player_on_off_regular       DataFrame   On/off stats from the regular-season run.
# SHOT_STAT_NAMES             list[str]   Stat names retained in the parquet output.
# columns_to_save             list[str]   Columns written to presence_full.parquet.
