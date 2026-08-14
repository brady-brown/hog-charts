"""
build_lineups.py — Generate 1/2/3/5-man lineup combo CSVs for overall and conference games.

WHY THIS FILE EXISTS
--------------------
The Hog Charts lineup page shows how every combination of 1, 2, 3, or 5
players on the same team performed together — net rating, offensive rating,
defensive rating, and per-100 counting stats.

Lineups are reconstructed by the SAME stint engine build_onoff_rapm uses
(hoglib.stints). Per-play weights (FGA 1, FT 0.44, TOV 1) only DISTRIBUTE
possessions across stints; each game/team's stint possessions are then rescaled
to the box-score total, which is the project-wide possession estimate
FGA + 0.44·FTA − ORB + TOV. So the Lineups page and the on/off table agree: a
1-man "lineup" matches that player's ortg_on/drtg_on in mbb_onoff_*.csv.

Steps:
  1. hoglib.stints.build_stints reconstructs on-court lineups from substitution
     events (stint = a run of plays with fixed lineups on both teams).
  2. Aggregate per (game, stint, team): points, box counting stats, scaled
     possessions, and clock seconds.
  3. Roll each team's stints into season totals per 5-man lineup.
  4. Compute all sub-combinations (from a 5-man lineup: 5 singles, 10 pairs,
     10 triples) and per-100 rates off the scaled possessions.

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
from itertools import combinations

import numpy as np
import pandas as pd
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
team_box_df     = feeds.load_team_box(SEASON)     # for box-score possession scaling

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


# ---------------------------------------------------------------------------
# Unified stint engine
# ---------------------------------------------------------------------------
# Lineups are reconstructed by the SAME engine build_onoff_rapm uses
# (hoglib.stints), so the Lineups page and the on/off table agree for the same
# players. Per-play weights (FGA 1, FT 0.44, TOV 1) only distribute possessions
# across stints; the per-game total is rescaled to the box-score possession
# estimate FGA + 0.44·FTA − ORB + TOV. A 1-man "lineup" therefore matches that
# player's ortg_on / drtg_on in mbb_onoff_*.csv.
from hoglib import stints

# Per-team stat columns carried through the roll-up (displayed as per-100 rates).
COUNTING_COLS = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                 "ORB", "DRB", "AST", "STL", "BLK", "TOV", "Foul"]
SUM_COLS = ["Seconds", "PF", "PA", "Poss_Off", "Poss_Def"] + COUNTING_COLS

print("Preparing play-by-play + stints…")
pbp = play_by_play_df.copy()
pbp["sequence_number"]  = pd.to_numeric(pbp["sequence_number"], errors="coerce")
pbp["score_value"]      = pd.to_numeric(pbp["score_value"], errors="coerce").fillna(0).astype(int)
pbp["points_attempted"] = pd.to_numeric(pbp["points_attempted"], errors="coerce")
pbp["scoring_play"]     = pbp["scoring_play"].astype(bool)
pbp["shooting_play"]    = pbp["shooting_play"].astype(bool)
for _c in ["team_id", "athlete_id_1", "home_team_id", "away_team_id", "period_number"]:
    pbp[_c] = pd.to_numeric(pbp[_c], errors="coerce")

_tt  = pbp["type_text"].astype(str)
_txt = pbp["text"].astype(str)
pbp["is_substitution"] = pbp["type_text"] == "Substitution"
pbp["is_subbing_in"]   = pbp["is_substitution"] & _txt.str.contains("subbing in",  case=False, na=False)
pbp["is_subbing_out"]  = pbp["is_substitution"] & _txt.str.contains("subbing out", case=False, na=False)
pbp["is_ft"]   = _tt.str.contains("FreeThrow", na=False)
pbp["is_fga"]  = pbp["shooting_play"] & ~pbp["is_ft"]
pbp["is_tov"]  = _tt.str.contains("Turnover", na=False)
pbp["is_fgm"]  = pbp["is_fga"] & pbp["scoring_play"]
pbp["is_3a"]   = pbp["is_fga"] & (pbp["points_attempted"] == 3)
pbp["is_3m"]   = pbp["is_3a"] & pbp["scoring_play"]
pbp["is_ftm"]  = pbp["is_ft"] & pbp["scoring_play"]
pbp["is_orb"]  = _tt.str.contains("Offensive Rebound", na=False)
pbp["is_drb"]  = _tt.str.contains("Defensive Rebound", na=False)
pbp["is_stl"]  = _tt.str.contains("Steal", na=False)
pbp["is_blk"]  = _tt.str.contains("Block", na=False)
pbp["is_foul"] = _tt.str.contains("Foul", na=False)
# Assist heuristic: an assisted made FG names the passer in parentheses.
pbp["is_ast"]  = pbp["is_fgm"] & _txt.str.contains(r"\(", na=False) & _txt.str.contains(r"\)", na=False)
pbp = pbp.sort_values(["game_id", "sequence_number"]).reset_index(drop=True)

# Box-score possession total per (game, team) — the scale target.
tb = team_box_df.copy()
tb["team_id"] = pd.to_numeric(tb["team_id"], errors="coerce")
tb["box_poss"] = (tb["field_goals_attempted"] + 0.44 * tb["free_throws_attempted"]
                  + tb["total_turnovers"] - tb["offensive_rebounds"])
box_poss_map = tb.set_index(["game_id", "team_id"])["box_poss"].to_dict()

# Starters (same source as build_onoff) + id→name for lineup labels.
pb = player_box_df.copy()
pb["athlete_id"]      = pd.to_numeric(pb["athlete_id"], errors="coerce")
pb["team_id_numeric"] = pd.to_numeric(pb["team_id"], errors="coerce")
starters_by_game_team = (
    pb[(pb["starter"] == True) & pb["athlete_id"].notna()]
    .groupby(["game_id", "team_id_numeric"])["athlete_id"].apply(frozenset).to_dict()
)
player_id_to_name = dict(zip(pb["athlete_id"], pb["athlete_display_name"]))

pbp, stint_lineup_info = stints.build_stints(pbp, starters_by_game_team)

# --- per (game, stint, team): points, counting stats, raw possession weight ---
print("Aggregating stint stats…")
grouped = pbp.groupby(["game_id", "stint_id", "team_id"])
stint_team = grouped.agg(
    PF=("score_value", lambda s: s[pbp.loc[s.index, "scoring_play"]].sum()),
    FGM=("is_fgm", "sum"),   FGA=("is_fga", "sum"),
    FG3M=("is_3m", "sum"),   FG3A=("is_3a", "sum"),
    FTM=("is_ftm", "sum"),   FTA=("is_ft", "sum"),
    ORB=("is_orb", "sum"),   DRB=("is_drb", "sum"),
    AST=("is_ast", "sum"),   STL=("is_stl", "sum"), BLK=("is_blk", "sum"),
    TOV=("is_tov", "sum"),   Foul=("is_foul", "sum"),
).reset_index()
stint_team["team_id"] = pd.to_numeric(stint_team["team_id"], errors="coerce")

# Raw possession weight per stint, scaled to the box total per (game, team).
pbp["_pw"] = np.where(pbp["is_fga"], 1.0, np.where(pbp["is_ft"], 0.44, np.where(pbp["is_tov"], 1.0, 0.0)))
poss = pbp.groupby(["game_id", "stint_id", "team_id"])["_pw"].sum().reset_index()
poss["team_id"] = pd.to_numeric(poss["team_id"], errors="coerce")
game_totals = poss.groupby(["game_id", "team_id"])["_pw"].sum().reset_index().rename(columns={"_pw": "_gtot"})
poss = poss.merge(game_totals, on=["game_id", "team_id"], how="left")
poss["_box"] = [box_poss_map.get((g, t), np.nan) for g, t in zip(poss["game_id"], poss["team_id"])]
poss["Poss_Off"] = np.where(poss["_gtot"] > 0, poss["_pw"] / poss["_gtot"] * poss["_box"], 0.0)
stint_team = stint_team.merge(poss[["game_id", "stint_id", "team_id", "Poss_Off"]],
                              on=["game_id", "stint_id", "team_id"], how="left")

# Seconds elapsed per (game, stint) from the clock (attributed to both lineups).
pbp["_clock"] = (pbp["clock_minutes"].fillna(0).astype(float) * 60
                 + pbp["clock_seconds"].fillna(0).astype(float))

def _game_stint_seconds(game_df):
    prev_clock, prev_period, elapsed = None, None, []
    for period, clock in zip(game_df["period_number"].values, game_df["_clock"].values):
        if period != prev_period:
            prev_clock = 1200.0 if (period is None or period <= 2) else 300.0
            prev_period = period
        elapsed.append(max(0.0, prev_clock - clock))
        prev_clock = clock
    return pd.Series(elapsed, index=game_df.index)

pbp["_secs"] = pbp.groupby("game_id", group_keys=False).apply(_game_stint_seconds)
stint_seconds = (pbp.groupby(["game_id", "stint_id"])["_secs"].sum()
                 .reset_index().rename(columns={"_secs": "Seconds"}))

# --- expand each stint into one row per team (own stats + opponent PF/Poss) ---
expanded_rows = []
for rec in stint_lineup_info.itertuples(index=False):
    expanded_rows.append((rec.game_id, rec.stint_id, int(rec.home_team_id),
                          int(rec.away_team_id), rec.home_lineup))
    expanded_rows.append((rec.game_id, rec.stint_id, int(rec.away_team_id),
                          int(rec.home_team_id), rec.away_lineup))
lineup_stints = pd.DataFrame(expanded_rows,
                             columns=["game_id", "stint_id", "team_id", "opp_id", "lineup"])
lineup_stints = lineup_stints.merge(stint_team, on=["game_id", "stint_id", "team_id"], how="left")
_opp = stint_team[["game_id", "stint_id", "team_id", "PF", "Poss_Off"]].rename(
    columns={"team_id": "opp_id", "PF": "PA", "Poss_Off": "Poss_Def"})
lineup_stints = lineup_stints.merge(_opp, on=["game_id", "stint_id", "opp_id"], how="left")
lineup_stints = lineup_stints.merge(stint_seconds, on=["game_id", "stint_id"], how="left")
for _c in SUM_COLS:
    lineup_stints[_c] = pd.to_numeric(lineup_stints[_c], errors="coerce").fillna(0.0)
lineup_stints["lineup_key"] = lineup_stints["lineup"].apply(lambda s: tuple(sorted(s)))

# Conference games (for the conference scope).
conference_game_ids = set(
    schedule_df.loc[schedule_df["conference_competition"] == True, "game_id"].astype(int).unique()
)


# ---------------------------------------------------------------------------
# Combo roll-up
# ---------------------------------------------------------------------------

def compute_combo_stats(season_lineup_totals, combo_size, min_avg_possessions):
    """Roll season 5-man lineup totals into all C(5, n) sub-combos.

    season_lineup_totals: rows of {lineup_key (tuple of names), SUM_COLS…}.
    Possessions are the already-scaled on/off possessions, summed across stints
    — NOT recomputed — so ratings match the on/off table.
    """
    combo_accumulator = {}
    for lineup_row in season_lineup_totals.itertuples(index=False):
        names = list(lineup_row.names)
        for combo in combinations(sorted(names), combo_size):
            acc = combo_accumulator.setdefault(combo, {c: 0.0 for c in SUM_COLS})
            for c in SUM_COLS:
                acc[c] += getattr(lineup_row, c)

    if not combo_accumulator:
        return pd.DataFrame()
    df = pd.DataFrame([{"Combo": ", ".join(k), **v} for k, v in combo_accumulator.items()])

    # Defensive possessions fall back to offensive if none were recorded.
    df.loc[df["Poss_Def"] <= 0, "Poss_Def"] = df["Poss_Off"]
    df["Avg_Poss"] = ((df["Poss_Off"] + df["Poss_Def"]) / 2).round(1)
    df["ORtg"] = (df["PF"] / df["Poss_Off"].replace(0, np.nan)) * 100
    df["DRtg"] = (df["PA"] / df["Poss_Def"].replace(0, np.nan)) * 100
    df["NetRtg"] = (df["ORtg"].fillna(0) - df["DRtg"].fillna(0)).round(2)
    df["ORtg"] = df["ORtg"].round(1)
    df["DRtg"] = df["DRtg"].round(1)

    per_100 = ["PF", "PA"] + COUNTING_COLS
    for stat in per_100:
        df[f"{stat}_100"] = ((df[stat] / df["Poss_Off"].replace(0, np.nan)) * 100).round(2)
    df["Mins"] = (df["Seconds"] / 60).round(1)
    df["REB_100"] = (df["ORB_100"] + df["DRB_100"]).round(2)

    df = df[df["Avg_Poss"] >= min_avg_possessions]
    output_columns = (["Combo", "Mins", "Avg_Poss", "NetRtg", "ORtg", "DRtg"]
                      + [f"{s}_100" for s in per_100] + ["REB_100"])
    return df[output_columns].sort_values("NetRtg", ascending=False)


# ---------------------------------------------------------------------------
# Process all teams
# ---------------------------------------------------------------------------

def process_all_teams(qualifying_teams, game_scope="overall"):
    """Write 1/2/3/5-man combo CSVs for all qualifying teams and a game scope."""
    combo_frames_by_size = {1: [], 2: [], 3: [], 5: []}
    scope_stints = (lineup_stints[lineup_stints["game_id"].isin(conference_game_ids)]
                    if game_scope == "conference" else lineup_stints)
    minimum_poss_threshold = 30 if game_scope == "conference" else 50

    for team_id in tqdm(qualifying_teams, desc=f"Processing {game_scope.upper()}"):
        team_rows = scope_stints[scope_stints["team_id"] == team_id]
        if team_rows.empty:
            continue

        # Season totals per full lineup for this team.
        season_totals = team_rows.groupby("lineup_key")[SUM_COLS].sum().reset_index()
        season_totals["names"] = season_totals["lineup_key"].apply(
            lambda k: [player_id_to_name.get(pid, str(pid)) for pid in k])
        # Only 5-man lineups roll up cleanly into sub-combos.
        season_totals = season_totals[season_totals["names"].apply(len) == 5]
        if season_totals.empty:
            continue

        for combo_size in [1, 2, 3, 5]:
            combo_df = compute_combo_stats(season_totals, combo_size, minimum_poss_threshold)
            if not combo_df.empty:
                combo_df["Team"] = team_id_to_display_name.get(team_id, f"Team {team_id}")
                combo_frames_by_size[combo_size].append(combo_df)

    for combo_size, frame_list in combo_frames_by_size.items():
        if frame_list:
            combined_df = (pd.concat(frame_list, ignore_index=True)
                           .sort_values(["Team", "NetRtg"], ascending=[True, False]))
            output_csv = f"{combo_size}_man_{game_scope}_stats_{SEASON}.csv"
            combined_df.to_csv(output_csv, index=False)
            print(f"  Saved {output_csv} — {len(combined_df)} combos")


process_all_teams(qualifying_team_ids, "overall")
process_all_teams(qualifying_team_ids, "conference")

print("\nDone. 8 lineup CSVs saved.")
