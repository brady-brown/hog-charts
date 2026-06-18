"""
Run locally to regenerate player_stats_conf.csv:
    python3 build_conf_stats.py
"""
import pandas as pd
import sportsdataverse.mbb as mbb

SEASON = 2026

print("Loading schedule...")
sched = mbb.load_mbb_schedule(seasons=[SEASON], return_as_pandas=True)
conf_game_ids = set(sched.loc[sched["conference_competition"] == True, "game_id"])
print(f"  {len(conf_game_ids)} conference games found")

print("Loading player boxscores...")
box = mbb.load_mbb_player_boxscore(seasons=[SEASON], return_as_pandas=True)

# Filter to conference games only, exclude did-not-play rows
conf_box = box[
    box["game_id"].isin(conf_game_ids) &
    (box["did_not_play"] == False) &
    (box["active"] == True)
].copy()
print(f"  {len(conf_box)} player-game rows in conference games")

# Parse minutes (stored as "MM:SS" string)
def parse_minutes(m):
    try:
        if pd.isna(m): return 0.0
        parts = str(m).split(":")
        return int(parts[0]) + int(parts[1]) / 60 if len(parts) == 2 else float(parts[0])
    except:
        return 0.0

conf_box["minutes_f"] = conf_box["minutes"].apply(parse_minutes)

# Pull in conference from original player_stats.csv
player_info = pd.read_csv("player_stats.csv")[
    ["athlete_id", "conf.", "athlete_position_name"]
].drop_duplicates(subset=["athlete_id"])

stat_cols = [
    "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "rebounds",
    "assists", "steals", "blocks", "turnovers", "fouls", "points",
]

agg = (
    conf_box.groupby(["athlete_id", "athlete_display_name", "team_id", "team_display_name"])
    .agg(
        games_played=("game_id", "nunique"),
        minutes=("minutes_f", "sum"),
        **{c: (c, "sum") for c in stat_cols},
    )
    .reset_index()
)

agg = agg.merge(player_info, on="athlete_id", how="left")

# Shooting percentages
agg["fg_pct"]  = agg["field_goals_made"] / agg["field_goals_attempted"].replace(0, float("nan"))
agg["3pt_pct"] = agg["three_point_field_goals_made"] / agg["three_point_field_goals_attempted"].replace(0, float("nan"))
agg["ft_pct"]  = agg["free_throws_made"] / agg["free_throws_attempted"].replace(0, float("nan"))
agg["efg_pct"] = (agg["field_goals_made"] + 0.5 * agg["three_point_field_goals_made"]) / agg["field_goals_attempted"].replace(0, float("nan"))

# Per-game averages
gp = agg["games_played"]
agg["minute_avg"]  = (agg["minutes"] / gp).round(2)
agg["fgm_avg"]     = (agg["field_goals_made"] / gp).round(2)
agg["fga_avg"]     = (agg["field_goals_attempted"] / gp).round(2)
agg["3ptm_avg"]    = (agg["three_point_field_goals_made"] / gp).round(2)
agg["3pta_avg"]    = (agg["three_point_field_goals_attempted"] / gp).round(2)
agg["ftm_avg"]     = (agg["free_throws_made"] / gp).round(2)
agg["fta_avg"]     = (agg["free_throws_attempted"] / gp).round(2)
agg["oreb_avg"]    = (agg["offensive_rebounds"] / gp).round(2)
agg["dreb_avg"]    = (agg["defensive_rebounds"] / gp).round(2)
agg["reb_avg"]     = (agg["rebounds"] / gp).round(2)
agg["ast_avg"]     = (agg["assists"] / gp).round(2)
agg["steal_avg"]   = (agg["steals"] / gp).round(2)
agg["blocks_avg"]  = (agg["blocks"] / gp).round(2)
agg["to_avg"]      = (agg["turnovers"] / gp).round(2)
agg["points_avg"]  = (agg["points"] / gp).round(2)

agg = agg.rename(columns={"conf.": "conf.", "athlete_position_name": "athlete_position_name"})

agg.to_csv("player_stats_conf.csv", index=False)
print(f"\nSaved player_stats_conf.csv — {len(agg)} players")
