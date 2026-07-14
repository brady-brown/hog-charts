"""
build_player_stats.py — Aggregate per-game box scores to per-player season totals.

WHY THIS FILE EXISTS
--------------------
The Hog Charts player-stats page shows season averages (PPG, RPG, APG, etc.)
plus shooting percentages and efficiency ratings.  The raw data from
sportsdataverse is one row per player per game.  This script groups those rows
by player and computes season totals, per-game averages, and shooting splits.

Two outputs:
    player_stats_{SEASON}.csv        All regular-season games.
    player_stats_conf_{SEASON}.csv   Conference games only.

Run locally:
    python3 build_player_stats.py
    OVERRIDE_SEASON=2025 python3 build_player_stats.py
"""

import os
from datetime import date as _date

import numpy as np
import pandas as pd
import sportsdataverse.mbb as mbb

from conferences import team_id_to_label
from predictor import calculate_efficiency, calculate_adjusted_efficiency

# ---------------------------------------------------------------------------
# Season detection
# ---------------------------------------------------------------------------
from hoglib.season import detect_season
SEASON = detect_season()

# Drop players with fewer than this many minutes from the box score (eliminates DNPs).
MIN_MINUTES_THRESHOLD = 1.0

# ---------------------------------------------------------------------------
# Load schedule to identify conference games
# ---------------------------------------------------------------------------
print("Loading schedule…")
season_schedule_df = mbb.load_mbb_schedule(seasons=SEASON, return_as_pandas=True)

conference_game_ids = (
    season_schedule_df[season_schedule_df["conference_competition"] == True]["game_id"]
    .astype(int)
    .unique()
)
print(f"  {len(conference_game_ids):,} conference game IDs")

# Conference TOURNAMENT games are a quirk: ESPN tags them season_type == 2
# (regular season) even though they're played in March after league play. They
# carry conference_competition == True plus a notes headline naming the event
# ("SEC Tournament", "Big Ten Tournament", "ASUN Championship", "America East
# Playoffs", …). We detect them so they can be grouped with the postseason
# instead of the regular season. (Early-season multi-team events like the Maui
# Invitational are non-conference, so conference_competition is False and they
# are correctly excluded.)
_notes = season_schedule_df.get("notes_headline")
if _notes is not None:
    is_conf_tournament_game = (
        (season_schedule_df["conference_competition"] == True)
        & _notes.astype(str).str.contains("Tournament|Championship|Playoffs", case=False, na=False)
    )
    conference_tournament_game_ids = set(
        season_schedule_df.loc[is_conf_tournament_game, "game_id"].astype(int).unique()
    )
else:
    conference_tournament_game_ids = set()
print(f"  {len(conference_tournament_game_ids):,} conference-tournament game IDs (grouped into postseason)")

# ---------------------------------------------------------------------------
# Build conference label map: team_id → short site label (e.g. "SEC", "Big 12").
# Sourced from ESPN's season standings (see conferences.py) so it works for every
# season at build time, including in CI — no dependency on a local reference CSV.
# ---------------------------------------------------------------------------
team_conference_label_map = team_id_to_label(SEASON)
print(f"  {len([v for v in team_conference_label_map.values() if v]):,} teams labeled by conference")

# ---------------------------------------------------------------------------
# Load per-game player box scores
# ---------------------------------------------------------------------------
print("Loading per-game player box scores…")
offline_boxscore_csv = f"offline_player_{SEASON}.csv"
if os.path.exists(offline_boxscore_csv):
    raw_player_boxscores_df = pd.read_csv(offline_boxscore_csv, low_memory=False)
else:
    print(f"  {offline_boxscore_csv} not found — fetching from sportsdataverse…")
    raw_player_boxscores_df = mbb.load_mbb_player_boxscore(
        seasons=[SEASON], return_as_pandas=True
    )

# Keep regular-season (season_type == 2) AND postseason (season_type == 3) games.
# Preseason (1) and any other types are dropped.  Each scope below is a mask over
# this combined frame.
played_boxscores_df = raw_player_boxscores_df[
    raw_player_boxscores_df["season_type"].isin([2, 3])
].copy()
played_boxscores_df["game_id"]     = played_boxscores_df["game_id"].astype(int)
played_boxscores_df["team_id"]     = played_boxscores_df["team_id"].astype(int)
played_boxscores_df["season_type"] = played_boxscores_df["season_type"].astype(int)
played_boxscores_df["minutes"] = pd.to_numeric(
    played_boxscores_df["minutes"], errors="coerce"
).fillna(0)
played_boxscores_df = played_boxscores_df[
    played_boxscores_df["minutes"] >= MIN_MINUTES_THRESHOLD
]

# Per-row scope flags.
#   is_conf_tournament : season_type-2 game flagged as a conference tournament
#                        (reclassified into the postseason, out of regular/conf).
#   is_postseason      : NCAA/NIT/etc (season_type 3) OR a conference tournament.
#   is_regular_game    : season_type 2 that is NOT a conference tournament.
#   is_conference_game : regular-season LEAGUE games only (no conf tournament).
played_boxscores_df["is_conf_tournament"] = (
    played_boxscores_df["game_id"].isin(conference_tournament_game_ids)
)
played_boxscores_df["is_postseason"] = (
    (played_boxscores_df["season_type"] == 3) | played_boxscores_df["is_conf_tournament"]
)
played_boxscores_df["is_regular_game"] = (
    (played_boxscores_df["season_type"] == 2) & ~played_boxscores_df["is_conf_tournament"]
)
played_boxscores_df["is_conference_game"] = (
    played_boxscores_df["is_regular_game"]
    & played_boxscores_df["game_id"].isin(conference_game_ids)
)
print(
    f"  {len(played_boxscores_df):,} player-game rows  "
    f"({played_boxscores_df['is_regular_game'].sum():,} regular, "
    f"{played_boxscores_df['is_postseason'].sum():,} postseason "
    f"[{played_boxscores_df['is_conf_tournament'].sum():,} from conf tournaments], "
    f"{played_boxscores_df['is_conference_game'].sum():,} conference)"
)

# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------
# Columns that are summed (totals across all games).
COUNTING_STAT_COLUMNS = [
    "minutes",
    "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "rebounds",
    "assists", "steals", "blocks", "turnovers", "fouls", "points",
]


def aggregate_player_stats(player_game_rows_df):
    """Group per-game rows by player and compute season totals + averages.

    Returns one row per player with:
      - Season totals for every counting stat
      - games_played  (distinct game_ids)
      - Shooting percentages (FG%, 3PT%, FT%, eFG%)
      - Per-game averages for minutes and each box score category
    """
    player_groups = player_game_rows_df.groupby(
        ["athlete_id", "athlete_display_name",
         "team_id",    "team_display_name",
         "athlete_position_name"],
        dropna=False
    )

    season_totals_df = player_groups[COUNTING_STAT_COLUMNS].sum()
    games_played_series = player_groups["game_id"].nunique().rename("games_played")

    player_season_df = pd.concat([season_totals_df, games_played_series], axis=1).reset_index()

    # Conference label from the team_id → label map (ESPN standings).
    player_season_df["conf."] = player_season_df["team_id"].map(team_conference_label_map)

    # --- Shooting percentages ---
    def safe_divide(numerator, denominator):
        """Divide two Series; return None where denominator is zero."""
        return np.where(denominator > 0, numerator / denominator, None)

    player_season_df["fg_pct"] = safe_divide(
        player_season_df["field_goals_made"],
        player_season_df["field_goals_attempted"]
    )
    player_season_df["3pt_pct"] = safe_divide(
        player_season_df["three_point_field_goals_made"],
        player_season_df["three_point_field_goals_attempted"]
    )
    player_season_df["ft_pct"] = safe_divide(
        player_season_df["free_throws_made"],
        player_season_df["free_throws_attempted"]
    )
    # eFG% = (FGM + 0.5 × 3PM) / FGA  (weights threes at 1.5× twos)
    player_season_df["efg_pct"] = safe_divide(
        player_season_df["field_goals_made"]
        + 0.5 * player_season_df["three_point_field_goals_made"],
        player_season_df["field_goals_attempted"]
    )

    # --- Per-game averages ---
    per_game_average_pairs = [
        ("minutes",                             "minute_avg"),
        ("field_goals_made",                    "fgm_avg"),
        ("field_goals_attempted",               "fga_avg"),
        ("three_point_field_goals_made",        "3ptm_avg"),
        ("three_point_field_goals_attempted",   "3pta_avg"),
        ("offensive_rebounds",                  "oreb_avg"),
        ("defensive_rebounds",                  "dreb_avg"),
        ("rebounds",                            "reb_avg"),
        ("assists",                             "ast_avg"),
        ("steals",                              "steal_avg"),
        ("blocks",                              "blocks_avg"),
        ("turnovers",                           "to_avg"),
        ("points",                              "points_avg"),
    ]
    games_played_col = player_season_df["games_played"]
    for total_column, average_column in per_game_average_pairs:
        player_season_df[average_column] = safe_divide(player_season_df[total_column], games_played_col)

    # --- Round ---
    shooting_pct_columns = ["fg_pct", "3pt_pct", "ft_pct", "efg_pct"]
    per_game_avg_columns = [col for col in player_season_df.columns if col.endswith("_avg")]
    player_season_df[shooting_pct_columns] = player_season_df[shooting_pct_columns].round(6)
    player_season_df[per_game_avg_columns] = player_season_df[per_game_avg_columns].round(2)

    return player_season_df


# ---------------------------------------------------------------------------
# Build and write — one CSV per scope.
#   reg  → player_stats_{SEASON}.csv       (regular season; consumed by other
#          scripts, so its name/meaning is kept stable)
#   all  → player_stats_all_{SEASON}.csv   (regular + postseason)
#   post → player_stats_post_{SEASON}.csv  (postseason only)
#   conf → player_stats_conf_{SEASON}.csv  (regular-season conference games)
# ---------------------------------------------------------------------------
# Non-conference: regular-season games that are NOT conference league games (and
# not conference tournaments, which is_regular_game already excludes). This is the
# complement of the conference scope within the regular season — early-season
# tournaments, buy games, rivalry non-league matchups, etc.
played_boxscores_df["is_nonconf_game"] = (
    played_boxscores_df["is_regular_game"]
    & ~played_boxscores_df["game_id"].isin(conference_game_ids)
)

PLAYER_STAT_SCOPES = [
    ("reg",     f"player_stats_{SEASON}.csv",         played_boxscores_df["is_regular_game"]),
    ("all",     f"player_stats_all_{SEASON}.csv",     played_boxscores_df["season_type"].isin([2, 3])),
    ("post",    f"player_stats_post_{SEASON}.csv",    played_boxscores_df["is_postseason"]),
    ("conf",    f"player_stats_conf_{SEASON}.csv",    played_boxscores_df["is_conference_game"]),
    ("nonconf", f"player_stats_nonconf_{SEASON}.csv", played_boxscores_df["is_nonconf_game"]),
]

for scope_name, output_filename, scope_mask in PLAYER_STAT_SCOPES:
    scope_boxscores_df = played_boxscores_df[scope_mask]
    if scope_boxscores_df.empty:
        print(f"  {output_filename}  — no games, skipped ({scope_name})")
        continue
    scope_player_stats_df = aggregate_player_stats(scope_boxscores_df)
    scope_player_stats_df.to_csv(output_filename, index=False)
    print(f"  {output_filename}  — {len(scope_player_stats_df):,} players ({scope_name})")

# ---------------------------------------------------------------------------
# Team context (opponent-faced totals) — powers the rate-stat percentages
# (ORB%, DRB%, TRB%, STL%, BLK%) in build_site.py, which need the rebounds,
# possessions and shot attempts the OPPONENT generated against each team.
# ---------------------------------------------------------------------------
TEAM_BOX_NUMERIC_COLUMNS = [
    "field_goals_attempted", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "total_rebounds",
    "three_point_field_goals_attempted", "turnovers",
]


def build_team_context(team_box_df):
    """Sum the totals each team's OPPONENTS produced against them.

    Group by ``opponent_team_id`` so the result is keyed by the team being
    scouted: every row whose opponent is team X is one of X's opponents'
    games, so the aggregate is what X's defense faced all season.

    Returns a DataFrame keyed by ``team_id`` with opponent offensive
    rebounds, defensive rebounds, total rebounds, FGA, 3PA and possessions
    (FGA + 0.44·FTA − ORB + TOV).
    """
    rows = team_box_df.copy()
    for col in TEAM_BOX_NUMERIC_COLUMNS:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0)
    rows["off_poss"] = (
        rows["field_goals_attempted"] + 0.44 * rows["free_throws_attempted"]
        - rows["offensive_rebounds"] + rows["turnovers"]
    )
    context_df = (
        rows.groupby("opponent_team_id")
        .agg(
            opp_orb=("offensive_rebounds", "sum"),
            opp_drb=("defensive_rebounds", "sum"),
            opp_trb=("total_rebounds", "sum"),
            opp_fga=("field_goals_attempted", "sum"),
            opp_3pa=("three_point_field_goals_attempted", "sum"),
            opp_poss=("off_poss", "sum"),
        )
        .reset_index()
        .rename(columns={"opponent_team_id": "team_id"})
    )
    return context_df


# Box columns summed for the full team-stats table (powers the Team Stats site
# page). Superset of TEAM_BOX_NUMERIC_COLUMNS — also needs makes/points/assists
# /steals/blocks for the box, shooting and four-factor stats on both ends.
TEAM_STATS_NUMERIC_COLUMNS = [
    "team_score",
    "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "total_rebounds",
    "assists", "steals", "blocks", "turnovers",
]

# How each summed box column maps onto the short CSV key, for both the team's
# own totals and the opponent totals (what opponents did against the team).
_TEAM_STAT_AGG = {
    "pts": "team_score",
    "fgm": "field_goals_made",         "fga": "field_goals_attempted",
    "tpm": "three_point_field_goals_made", "tpa": "three_point_field_goals_attempted",
    "ftm": "free_throws_made",         "fta": "free_throws_attempted",
    "orb": "offensive_rebounds",       "drb": "defensive_rebounds", "trb": "total_rebounds",
    "ast": "assists", "stl": "steals", "blk": "blocks", "tov": "turnovers",
}


def build_team_stats(team_box_df):
    """Build a per-team table of season box totals + opponent box totals.

    Own totals are grouped by ``team_id`` (what the team did); opponent totals
    are grouped by ``opponent_team_id`` (what opponents did against the team,
    i.e. the team's defense). Both carry possessions, so build_site.py can
    derive box per-game, shooting, and offensive/defensive four-factor rates.
    Keyed by ``team_id``.
    """
    rows = team_box_df.copy()
    for col in TEAM_STATS_NUMERIC_COLUMNS:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0)
    rows["off_poss"] = (
        rows["field_goals_attempted"] + 0.44 * rows["free_throws_attempted"]
        - rows["offensive_rebounds"] + rows["turnovers"]
    )

    own_agg = {k: (src, "sum") for k, src in _TEAM_STAT_AGG.items()}
    own_agg["games"] = ("game_id", "nunique")
    own_agg["poss"]  = ("off_poss", "sum")
    own_df = rows.groupby("team_id").agg(**own_agg).reset_index()

    opp_agg = {f"opp_{k}": (src, "sum") for k, src in _TEAM_STAT_AGG.items()}
    opp_agg["opp_poss"] = ("off_poss", "sum")
    opp_df = (
        rows.groupby("opponent_team_id").agg(**opp_agg)
        .reset_index().rename(columns={"opponent_team_id": "team_id"})
    )

    return own_df.merge(opp_df, on="team_id", how="left")


# Columns the efficiency engine (predictor.calculate_efficiency) consumes.
_EFF_INPUT_COLUMNS = [
    "team_score", "opponent_team_score",
    "field_goals_attempted", "free_throws_attempted",
    "offensive_rebounds", "turnovers",
]


def build_scope_net_ratings(scope_team_box_df):
    """Opponent-adjusted Off/Def/Net efficiency + SOS for ONE scope's games.

    Re-runs the predictor's season solve (calculate_adjusted_efficiency) but on
    just this scope's game set, so the Team Stats ratings reflect only those
    games. No prior-season blend — a scope is its own sample. "D-I" self-limits
    to teams appearing in the scope, and SOS is the mean adjusted net rating of
    opponents faced within the scope.

    Returns a DataFrame: team_id, off_eff, def_eff, net_eff, sos.
    """
    games = scope_team_box_df.copy()
    games["season"] = SEASON   # single-season solve; guarantee the grouping key
    for col in _EFF_INPUT_COLUMNS:
        games[col] = pd.to_numeric(games[col], errors="coerce").fillna(0)

    adjusted = calculate_adjusted_efficiency(games, calculate_efficiency(games))
    net_eff_by_team = adjusted.set_index("team_id")["net_eff"]

    vs_rated = games[games["opponent_team_id"].isin(net_eff_by_team.index)].copy()
    vs_rated["opp_net_eff"] = vs_rated["opponent_team_id"].map(net_eff_by_team)
    sos_per_team = vs_rated.groupby("team_id")["opp_net_eff"].mean().rename("sos")

    out = adjusted[["team_id", "off_eff", "def_eff", "net_eff"]].merge(
        sos_per_team, on="team_id", how="left"
    )
    return out.round({"off_eff": 1, "def_eff": 1, "net_eff": 1, "sos": 1})


print("Loading team box scores for opponent context…")
raw_team_boxscores_df = mbb.load_mbb_team_boxscore(seasons=[SEASON], return_as_pandas=True)
team_boxscores_df = raw_team_boxscores_df[
    raw_team_boxscores_df["season_type"].isin([2, 3])
].copy()
team_boxscores_df["game_id"]          = team_boxscores_df["game_id"].astype(int)
team_boxscores_df["team_id"]          = team_boxscores_df["team_id"].astype(int)
team_boxscores_df["season_type"]      = team_boxscores_df["season_type"].astype(int)
team_boxscores_df["opponent_team_id"] = pd.to_numeric(
    team_boxscores_df["opponent_team_id"], errors="coerce"
)
team_boxscores_df = team_boxscores_df.dropna(subset=["opponent_team_id"])
team_boxscores_df["opponent_team_id"] = team_boxscores_df["opponent_team_id"].astype(int)

# One context file per scope — must mirror the player-stat scope masks above
# (including the conference-tournament reclassification) so the advanced rate
# stats use the matching opponent totals.
tc_is_conf_tourney = team_boxscores_df["game_id"].isin(conference_tournament_game_ids)
tc_is_regular      = (team_boxscores_df["season_type"] == 2) & ~tc_is_conf_tourney
tc_is_conference   = tc_is_regular & team_boxscores_df["game_id"].isin(conference_game_ids)
tc_is_nonconf      = tc_is_regular & ~team_boxscores_df["game_id"].isin(conference_game_ids)
TEAM_CONTEXT_SCOPES = [
    ("reg",     f"team_context_{SEASON}.csv",         tc_is_regular),
    ("all",     f"team_context_all_{SEASON}.csv",     team_boxscores_df["season_type"].isin([2, 3])),
    ("post",    f"team_context_post_{SEASON}.csv",    (team_boxscores_df["season_type"] == 3) | tc_is_conf_tourney),
    ("conf",    f"team_context_conf_{SEASON}.csv",    tc_is_conference),
    ("nonconf", f"team_context_nonconf_{SEASON}.csv", tc_is_nonconf),
]

for scope_name, output_filename, scope_mask in TEAM_CONTEXT_SCOPES:
    scope_team_box_df = team_boxscores_df[scope_mask]
    if scope_team_box_df.empty:
        print(f"  {output_filename}  — no games, skipped ({scope_name})")
        continue
    scope_context_df = build_team_context(scope_team_box_df)
    scope_context_df.to_csv(output_filename, index=False)
    print(f"  {output_filename}  — {len(scope_context_df):,} teams ({scope_name})")

# Full team-stats tables (own + opponent box totals) — one per scope. Same
# scope masks as the context files so the Team Stats site page lines up with
# the player-stats scope toggle (All / Conf / Reg / Post).
TEAM_STATS_SCOPES = [
    ("reg",     f"team_stats_{SEASON}.csv",         tc_is_regular),
    ("all",     f"team_stats_all_{SEASON}.csv",     team_boxscores_df["season_type"].isin([2, 3])),
    ("post",    f"team_stats_post_{SEASON}.csv",    (team_boxscores_df["season_type"] == 3) | tc_is_conf_tourney),
    ("conf",    f"team_stats_conf_{SEASON}.csv",    tc_is_conference),
    ("nonconf", f"team_stats_nonconf_{SEASON}.csv", tc_is_nonconf),
]
for scope_name, output_filename, scope_mask in TEAM_STATS_SCOPES:
    scope_team_box_df = team_boxscores_df[scope_mask]
    if scope_team_box_df.empty:
        print(f"  {output_filename}  — no games, skipped ({scope_name})")
        continue
    scope_team_stats_df = build_team_stats(scope_team_box_df)
    scope_team_stats_df.to_csv(output_filename, index=False)
    print(f"  {output_filename}  — {len(scope_team_stats_df):,} teams ({scope_name})")

    # Scope-specific opponent-adjusted net ratings (re-solved on this scope only).
    net_filename = f"net_ratings_{scope_name}_{SEASON}.csv"
    scope_net_df = build_scope_net_ratings(scope_team_box_df)
    scope_net_df.to_csv(net_filename, index=False)
    print(f"  {net_filename}  — {len(scope_net_df):,} teams ({scope_name})")

print("\nDone.")


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
#
# SEASON                           int     Calendar year the season ends (e.g. 2026).
# _season_override                 str     Value of OVERRIDE_SEASON env var; None if unset.
# MIN_MINUTES_THRESHOLD            float   Rows with < 1 minute are treated as DNP and dropped.
#
# season_schedule_df               DataFrame  Full ESPN schedule for the current season.
# conference_game_ids              ndarray  game_id values where conference_competition == True.
# home_teams_in_schedule           DataFrame  Home team id/conf/name columns from schedule.
# away_teams_in_schedule           DataFrame  Away team id/conf/name columns from schedule.
# all_unique_teams_df              DataFrame  Union of home/away teams; one row per team.
# existing_player_stats_df         DataFrame  Previous player_stats.csv for conference labels.
# conference_label_map             dict     {team_display_name: conference_abbreviation}.
#
# offline_boxscore_csv             str     Filename of the cached per-game boxscore CSV.
# raw_player_boxscores_df          DataFrame  All seasons/types from sportsdataverse.
# regular_season_boxscores_df      DataFrame  Filtered to season_type==2 and minutes > threshold.
# is_conference_game               Series   Boolean flag: True if game_id is in conference_game_ids.
#
# COUNTING_STAT_COLUMNS            list    Columns summed across games per player.
#
# --- aggregate_player_stats() ---
# player_game_rows_df              DataFrame  Input: one row per player per game.
# player_groups                    GroupBy  Grouped by athlete/team identity columns.
# season_totals_df                 DataFrame  Sum of every counting stat per player.
# games_played_series              Series   Distinct game_id count per player.
# player_season_df                 DataFrame  Merged totals + games_played.
# per_game_average_pairs           list    [(total_col, avg_col), …] for per-game computation.
# games_played_col                 Series   games_played column used as denominator.
# shooting_pct_columns             list    FG%/3PT%/FT%/eFG% columns rounded to 6 decimals.
# per_game_avg_columns             list    All columns ending in "_avg" rounded to 2 decimals.
#
# overall_player_stats_df          DataFrame  Aggregate over all regular-season games.
# conference_only_boxscores_df     DataFrame  Subset of regular_season_boxscores_df for conf games.
# conference_player_stats_df       DataFrame  Aggregate over conference games only.
#
# --- build_team_context() ---
# TEAM_BOX_NUMERIC_COLUMNS         list    Team-box columns coerced to numeric before summing.
# team_box_df                      DataFrame  Regular-season team-game rows (one per team per game).
# rows                             DataFrame  Copy with off_poss (FGA + 0.44·FTA − ORB + TOV) added.
# context_df                       DataFrame  Opponent-faced totals keyed by team_id.
# raw_team_boxscores_df            DataFrame  All team-game rows from sportsdataverse.
# team_boxscores_df                DataFrame  Filtered to regular season with numeric opponent_team_id.
# overall_team_context_df          DataFrame  Opponent context across all games → team_context CSV.
# conference_team_context_df       DataFrame  Opponent context, conference games only.
