"""
build_player_stats.py — Generate player_stats.csv and player_stats_conf.csv.

Aggregates per-game box scores from offline_player.csv, joining to the ESPN
schedule to identify conference games.

Run locally:
    python3 build_player_stats.py

Outputs:
    player_stats.csv        full-season per-player aggregates
    player_stats_conf.csv   conference-games-only aggregates
"""

import os
from datetime import date as _date

import numpy as np
import pandas as pd
import sportsdataverse.mbb as mbb

_override = os.environ.get("OVERRIDE_SEASON")
if _override:
    SEASON = int(_override)
else:
    _today = _date.today()
    SEASON = _today.year + 1 if _today.month >= 11 else _today.year
MIN_MINS = 1.0   # drop DNP rows

# ── Load schedule to get conference_competition flag ──────────────────────────
print("Loading schedule…")
schedule = mbb.load_mbb_schedule(seasons=SEASON, return_as_pandas=True)

conf_games = (schedule[schedule["conference_competition"] == True]["game_id"]
              .astype(int)
              .unique())
print(f"  {len(conf_games):,} conference game IDs")

# Conference name map: team_id → conference name string
home = (schedule[["home_id","home_conference_id","home_location","home_name"]]
        .rename(columns={"home_id":"team_id","home_conference_id":"conf_id",
                         "home_location":"loc","home_name":"nm"}))
away = (schedule[["away_id","away_conference_id","away_location","away_name"]]
        .rename(columns={"away_id":"team_id","away_conference_id":"conf_id",
                         "away_location":"loc","away_name":"nm"}))
all_teams = pd.concat([home, away]).dropna(subset=["team_id"]).drop_duplicates("team_id")

# Build human-readable conference labels from the existing player_stats.csv if
# available, otherwise fall back to ESPN conf IDs.
try:
    existing = pd.read_csv("player_stats.csv", usecols=["team_display_name","conf."])
    conf_label_map = (existing.dropna()
                              .drop_duplicates("team_display_name")
                              .set_index("team_display_name")["conf."]
                              .to_dict())
except Exception:
    conf_label_map = {}

# ── Load per-game player box scores ──────────────────────────────────────────
print("Loading per-game player box scores…")
if os.path.exists("offline_player.csv"):
    box = pd.read_csv("offline_player.csv", low_memory=False)
else:
    print("  offline_player.csv not found — fetching from sportsdataverse…")
    box = mbb.load_mbb_player_boxscore(seasons=[SEASON], return_as_pandas=True)

# Filter to regular season (season_type == 2) only
box = box[box["season_type"] == 2].copy()
box["game_id"]  = box["game_id"].astype(int)
box["team_id"]  = box["team_id"].astype(int)
box["minutes"]  = pd.to_numeric(box["minutes"],  errors="coerce").fillna(0)
box = box[box["minutes"] >= MIN_MINS]

# Mark conference games
box["is_conf"] = box["game_id"].isin(conf_games)
print(f"  {len(box):,} player-game rows  "
      f"({box['is_conf'].sum():,} from conference games)")

# ── Aggregate helper ──────────────────────────────────────────────────────────
COUNTING = [
    "minutes",
    "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "rebounds",
    "assists", "steals", "blocks", "turnovers", "fouls", "points",
]

def aggregate(df):
    """Aggregate per-game rows to per-player season totals + averages."""
    grp = df.groupby(
        ["athlete_id", "athlete_display_name",
         "team_id",    "team_display_name",
         "athlete_position_name"],
        dropna=False
    )

    totals = grp[COUNTING].sum()
    gp     = grp["game_id"].nunique().rename("games_played")

    out = pd.concat([totals, gp], axis=1).reset_index()

    # Conference label
    out["conf."] = out["team_display_name"].map(conf_label_map)

    # Shooting pct
    def safe_div(num, den):
        return np.where(den > 0, num / den, None)

    out["fg_pct"]   = safe_div(out["field_goals_made"],              out["field_goals_attempted"])
    out["3pt_pct"]  = safe_div(out["three_point_field_goals_made"],  out["three_point_field_goals_attempted"])
    out["ft_pct"]   = safe_div(out["free_throws_made"],              out["free_throws_attempted"])
    out["efg_pct"]  = safe_div(out["field_goals_made"] + 0.5 * out["three_point_field_goals_made"],
                                out["field_goals_attempted"])

    # Per-game averages
    for raw, avg in [
        ("minutes",                         "minute_avg"),
        ("field_goals_made",                "fgm_avg"),
        ("field_goals_attempted",           "fga_avg"),
        ("three_point_field_goals_made",    "3ptm_avg"),
        ("three_point_field_goals_attempted","3pta_avg"),
        ("offensive_rebounds",              "oreb_avg"),
        ("defensive_rebounds",              "dreb_avg"),
        ("rebounds",                        "reb_avg"),
        ("assists",                         "ast_avg"),
        ("steals",                          "steal_avg"),
        ("blocks",                          "blocks_avg"),
        ("turnovers",                       "to_avg"),
        ("points",                          "points_avg"),
    ]:
        out[avg] = safe_div(out[raw], out["games_played"])

    # Round
    pct_cols = ["fg_pct","3pt_pct","ft_pct","efg_pct"]
    avg_cols = [c for c in out.columns if c.endswith("_avg")]
    out[pct_cols] = out[pct_cols].round(6)
    out[avg_cols] = out[avg_cols].round(2)

    return out


# ── Build and write ───────────────────────────────────────────────────────────
print("Aggregating overall stats…")
overall = aggregate(box)
overall.to_csv(f"player_stats_{SEASON}.csv", index=False)
print(f"  player_stats_{SEASON}.csv  — {len(overall):,} players")

print("Aggregating conference-only stats…")
conf_df = aggregate(box[box["is_conf"]])
conf_df.to_csv(f"player_stats_conf_{SEASON}.csv", index=False)
print(f"  player_stats_conf_{SEASON}.csv  — {len(conf_df):,} players")

print("\nDone.")
