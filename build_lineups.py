"""
Generates all lineup combo CSVs (1/2/3/5-man, overall + conference).

Run locally:
    python3 build_lineups.py

Outputs (8 files):
    1_man_overall_stats.csv      1_man_conference_stats.csv
    2_man_overall_stats.csv      2_man_conference_stats.csv
    3_man_overall_stats.csv      3_man_conference_stats.csv
    5_man_overall_stats.csv      5_man_conference_stats.csv
"""

import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from datetime import date as _date
import sportsdataverse.mbb as mbb

_today = _date.today()
SEASON = _today.year + 1 if _today.month >= 11 else _today.year

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
pbp      = mbb.load_mbb_pbp(seasons=SEASON,          return_as_pandas=True)
box      = mbb.load_mbb_player_boxscore(seasons=SEASON, return_as_pandas=True)
schedule = mbb.load_mbb_schedule(seasons=SEASON,      return_as_pandas=True)

# Team name and conference maps
home = schedule[["home_id","home_location","home_name","home_conference_id"]].rename(
    columns={"home_id":"team_id","home_location":"loc","home_name":"nm","home_conference_id":"conf"})
away = schedule[["away_id","away_location","away_name","away_conference_id"]].rename(
    columns={"away_id":"team_id","away_location":"loc","away_name":"nm","away_conference_id":"conf"})
all_teams = pd.concat([home, away]).dropna(subset=["team_id"]).drop_duplicates("team_id")
all_teams["full_name"] = all_teams["loc"].astype(str) + " " + all_teams["nm"].astype(str)
team_map = dict(zip(all_teams["team_id"].astype(int), all_teams["full_name"]))
conf_map = dict(zip(all_teams["team_id"].astype(int), all_teams["conf"]))

# Eligible teams (played >= 15 games)
games_melted  = schedule.melt(id_vars=["game_id"], value_vars=["home_id","away_id"], value_name="team_id")
team_counts   = games_melted.groupby("team_id")["game_id"].nunique()
eligible      = team_counts[team_counts >= 15].index.astype(int).tolist()
print(f"  {len(eligible)} eligible teams")


def get_seconds(row):
    try: return int(row["clock_minutes"]) * 60 + int(row["clock_seconds"])
    except: return 0


def calculate_game_lineups(game_id, team_id, pbp_df, box_df):
    t_id     = int(team_id)
    game_pbp = pbp_df[pbp_df["game_id"] == game_id].copy().sort_values("game_play_number")

    if "home_score" in game_pbp.columns:
        game_pbp["home_score"] = game_pbp["home_score"].ffill().fillna(0)
        game_pbp["away_score"] = game_pbp["away_score"].ffill().fillna(0)

    team_box   = box_df[(box_df["game_id"] == game_id) & (box_df["team_id"] == t_id)]
    id_to_name = dict(zip(team_box["athlete_id"], team_box["athlete_display_name"]))
    starters   = team_box[team_box["starter"] == True]["athlete_id"].tolist()
    if len(starters) != 5:
        starters = team_box["athlete_id"].head(5).tolist()

    current_lineup = set(starters)
    lineup_stats   = {}
    prev_period    = 1
    prev_secs      = 20 * 60
    current_margin = 0

    for _, row in game_pbp.iterrows():
        curr_period = row["period_number"]
        curr_secs   = get_seconds(row)
        if curr_period != prev_period:
            prev_secs  = 20 * 60 if curr_period <= 2 else 5 * 60
            prev_period = curr_period
        duration = max(0, prev_secs - curr_secs)

        key = tuple(sorted(list(current_lineup)))
        if key not in lineup_stats:
            lineup_stats[key] = {
                "Seconds": 0, "PF": 0, "PA": 0,
                "FGM": 0, "FGA": 0, "FG3M": 0, "FG3A": 0, "FTM": 0, "FTA": 0,
                "ORB": 0, "DRB": 0, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "Foul": 0,
                "Opp_FGA": 0, "Opp_FTA": 0, "Opp_ORB": 0, "Opp_TOV": 0,
            }
        ls = lineup_stats[key]
        ls["Seconds"] += duration

        type_txt  = str(row["type_text"]).lower()
        text_txt  = str(row["text"]).lower()
        row_tid   = int(row["team_id"]) if pd.notnull(row["team_id"]) else -1
        is_team   = (row_tid == t_id)

        if row["scoring_play"]:
            if is_team: ls["PF"] += row["score_value"]
            else:       ls["PA"] += row["score_value"]

        if is_team:
            if row["shooting_play"] and "free throw" not in type_txt:
                ls["FGA"] += 1
                if "three point" in text_txt: ls["FG3A"] += 1
                if row["scoring_play"]:
                    ls["FGM"] += 1
                    if "three point" in text_txt: ls["FG3M"] += 1
                    if "(" in text_txt and ")" in text_txt: ls["AST"] += 1
            if "free throw" in type_txt:
                ls["FTA"] += 1
                if row["scoring_play"]: ls["FTM"] += 1
            if "offensive rebound" in type_txt: ls["ORB"] += 1
            if "defensive rebound" in type_txt: ls["DRB"] += 1
            if "steal"    in type_txt: ls["STL"]  += 1
            if "block"    in type_txt: ls["BLK"]  += 1
            if "turnover" in type_txt: ls["TOV"]  += 1
            if "foul"     in type_txt: ls["Foul"] += 1
        else:
            if row["shooting_play"] and "free throw" not in type_txt: ls["Opp_FGA"] += 1
            if "free throw"        in type_txt: ls["Opp_FTA"] += 1
            if "offensive rebound" in type_txt: ls["Opp_ORB"] += 1
            if "turnover"          in type_txt: ls["Opp_TOV"] += 1

        if row["type_text"] == "Substitution" and row_tid == t_id:
            pid = row["athlete_id_1"]
            if "subbing in"  in text_txt: current_lineup.add(pid)
            elif "subbing out" in text_txt:
                if pid in current_lineup: current_lineup.remove(pid)

        prev_secs      = curr_secs
        current_margin = abs(row.get("home_score", 0) - row.get("away_score", 0))

    results = []
    for lineup_tuple, s in lineup_stats.items():
        if s["Seconds"] > 0:
            s["Lineup"] = ", ".join([id_to_name.get(p, str(p)) for p in lineup_tuple])
            results.append(s)
    return pd.DataFrame(results)


def calculate_possession_combos(lineup_df, n, min_poss=20):
    core = ["Seconds","PF","PA","FGM","FGA","FG3M","FG3A","FTM","FTA",
            "ORB","DRB","AST","STL","BLK","TOV","Foul",
            "Opp_FGA","Opp_FTA","Opp_ORB","Opp_TOV"]
    stat_cols  = [c for c in core if c in lineup_df.columns]
    stats_map  = {}

    for _, row in lineup_df.iterrows():
        players = row["Lineup"].split(", ")
        for combo in combinations(sorted(players), n):
            key = tuple(combo)
            if key not in stats_map:
                stats_map[key] = {c: 0 for c in stat_cols}
            for c in stat_cols:
                stats_map[key][c] += row[c]

    rows = [{"Combo": ", ".join(k), **v} for k, v in stats_map.items()]
    df   = pd.DataFrame(rows)
    if df.empty: return pd.DataFrame()

    df["Poss_Off"] = df["FGA"] - df["ORB"] + df["TOV"] + 0.475 * df["FTA"]
    df["Poss_Def"] = df["Opp_FGA"] - df["Opp_ORB"] + df["Opp_TOV"] + 0.475 * df["Opp_FTA"]
    df.loc[df["Poss_Def"] <= 0, "Poss_Def"] = df["Poss_Off"]
    df["Avg_Poss"] = ((df["Poss_Off"] + df["Poss_Def"]) / 2).round(1)
    df["ORtg"]     = (df["PF"] / df["Poss_Off"].replace(0, np.nan)) * 100
    df["DRtg"]     = (df["PA"] / df["Poss_Def"].replace(0, np.nan)) * 100
    df["NetRtg"]   = (df["ORtg"].fillna(0) - df["DRtg"].fillna(0)).round(2)

    per100 = ["PF","PA","FGM","FGA","FG3M","FG3A","FTM","FTA",
              "ORB","DRB","AST","STL","BLK","TOV","Foul"]
    for s in per100:
        if s in df.columns:
            df[f"{s}_100"] = ((df[s] / df["Poss_Off"].replace(0, np.nan)) * 100).round(2)

    df["Mins"]    = (df["Seconds"] / 60).round(1)
    df["REB_100"] = (df["ORB_100"] + df["DRB_100"]).round(2)

    result = df[df["Avg_Poss"] >= min_poss].sort_values("NetRtg", ascending=False)
    cols   = (["Combo","Mins","Avg_Poss","NetRtg","ORtg","DRtg"]
              + [f"{s}_100" for s in per100 if f"{s}_100" in df.columns]
              + ["REB_100"])
    return result[cols]


def process_all_teams(teams, mode="overall"):
    results = {1: [], 2: [], 3: [], 5: []}

    for t_id in tqdm(teams, desc=f"Processing {mode.upper()}"):
        t_games = schedule[(schedule["home_id"].astype(int) == t_id) |
                           (schedule["away_id"].astype(int) == t_id)]
        if mode == "conference":
            t_games = t_games[t_games["conference_competition"] == True]

        stints = []
        for g_id in t_games["game_id"].unique():
            try:
                gdf = calculate_game_lineups(g_id, t_id, pbp, box)
                if not gdf.empty and "Lineup" in gdf.columns:
                    stints.append(gdf)
            except: continue

        if not stints: continue
        lineup_agg = pd.concat(stints).groupby("Lineup").sum(numeric_only=True).reset_index()

        for n in [1, 2, 3, 5]:
            min_p = 30 if mode == "conference" else 50
            combo = calculate_possession_combos(lineup_agg, n=n, min_poss=min_p)
            if not combo.empty:
                combo["Team"] = team_map.get(t_id, f"Team {t_id}")
                results[n].append(combo)

    for n, frames in results.items():
        if frames:
            out = (pd.concat(frames, ignore_index=True)
                   .sort_values(["Team","NetRtg"], ascending=[True,False]))
            out.to_csv(f"{n}_man_{mode}_stats.csv", index=False)
            print(f"  Saved {n}_man_{mode}_stats.csv — {len(out)} combos")


process_all_teams(eligible, "overall")
process_all_teams(eligible, "conference")

print("\nDone. 8 lineup CSVs saved.")
