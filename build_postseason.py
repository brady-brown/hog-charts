"""
build_postseason.py — NCAA tournament bracket + tournament-only leaders.

WHY THIS FILE EXISTS
--------------------
The Hog Charts "Postseason" page shows the NCAA tournament bracket (regions,
rounds, scores, the champion) plus player and team leaders computed *only* from
tournament games.  All of this is derivable from data sportsdataverse already
exposes — it is just dropped by the other builders:

    schedule.season_type == 3  AND  tournament_id == 22  →  NCAA tournament games
    schedule.notes_headline    →  region + round  (e.g. "... - East Region - Sweet 16")
    schedule home/away scores  →  matchup results + champion
    player boxscore filtered to tournament game_ids  →  tournament-only leaders

Like the other builders this runs nightly in CI (network access is fine at build
time); the browser only ever reads the static JSON it produces.

Output (written into site/data/{SEASON}/):
    postseason.json   { bracket, champion, player_leaders, team_summary, meta }

Run locally:
    python build_postseason.py
    OVERRIDE_SEASON=2025 python build_postseason.py
"""

import json
import os
import re
from datetime import date as _date

import numpy as np
import pandas as pd
import sportsdataverse.mbb as mbb

# ---------------------------------------------------------------------------
# Season detection (identical convention to the other builders)
# ---------------------------------------------------------------------------
from hoglib.season import detect_season
SEASON = detect_season()

PROJECT_ROOT    = os.path.dirname(os.path.abspath(__file__))
SEASON_DATA_DIR = os.path.join(PROJECT_ROOT, "site", "data", str(SEASON))
os.makedirs(SEASON_DATA_DIR, exist_ok=True)

# ESPN's tournament_id for the NCAA Division-I men's championship.
NCAA_TOURNAMENT_ID = 22

# Round labels ordered earliest → latest.  Index = bracket depth, used both for
# sorting games and for deciding how "far" a team advanced.
ROUND_ORDER = [
    "First Four",
    "1st Round",
    "2nd Round",
    "Sweet 16",
    "Elite 8",
    "Final Four",
    "National Championship",
]
ROUND_RANK = {round_name: idx for idx, round_name in enumerate(ROUND_ORDER)}

# Minimum minutes for a player-game row to count (drops DNPs); matches build_player_stats.
MIN_MINUTES_THRESHOLD = 1.0

# Counting columns summed across a player's tournament games.
COUNTING_STAT_COLUMNS = [
    "minutes",
    "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "rebounds",
    "assists", "steals", "blocks", "turnovers", "fouls", "points",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# NaN-safe JSON sanitizing now lives in hoglib.io (shared, must not drift).
from hoglib.io import sanitize_for_json


def parse_region_and_round(notes_headline):
    """Extract (region, round) from an ESPN tournament notes headline.

    Examples:
        "NCAA Men's Basketball Championship - East Region - Sweet 16"
            → ("East", "Sweet 16")
        "NCAA Men's Basketball Championship - Final Four"
            → (None, "Final Four")
        "NCAA Men's Basketball Championship - National Championship"
            → (None, "National Championship")

    Returns (region_or_None, round_or_None).
    """
    if not isinstance(notes_headline, str):
        return None, None

    # Older seasons use ALL-CAPS headlines ("... - EAST REGION - ELITE 8"), so
    # match case-insensitively and normalize the captured region to title case.
    region = None
    region_match = re.search(r"-\s*([A-Za-z]+)\s+Region", notes_headline, re.IGNORECASE)
    if region_match:
        region = region_match.group(1).title()

    found_round = None
    for round_name in ROUND_ORDER:
        if round_name.lower() in notes_headline.lower():
            found_round = round_name
            break
    return region, found_round


# ---------------------------------------------------------------------------
# 1. Load schedule and isolate the NCAA tournament
# ---------------------------------------------------------------------------
print(f"Loading {SEASON} schedule…")
from hoglib import feeds  # cached by build_ingest.py (step 0)
schedule_df = feeds.load_schedule(SEASON)

tournament_games_df = schedule_df[
    (schedule_df["season_type"] == 3)
    & (schedule_df["tournament_id"] == NCAA_TOURNAMENT_ID)
].copy()

if tournament_games_df.empty:
    print(f"  [warn] no NCAA tournament games found for {SEASON} — writing empty file.")

# ESPN lists each game once (home perspective), but to be safe drop duplicate
# game_ids and keep only completed games with both scores.
tournament_games_df = tournament_games_df.drop_duplicates(subset=["game_id"])
tournament_games_df = tournament_games_df[
    tournament_games_df["status_type_completed"].fillna(False).astype(bool)
    if "status_type_completed" in tournament_games_df.columns
    else tournament_games_df["home_score"].notna()
].copy()

tournament_game_ids = set(tournament_games_df["game_id"].astype(int).tolist())
print(f"  {len(tournament_game_ids)} completed NCAA tournament games")


# ---------------------------------------------------------------------------
# 2. Build the bracket: one record per game, grouped by region + round
# ---------------------------------------------------------------------------
def build_game_record(schedule_row):
    """Convert one schedule row into a compact bracket game record."""
    home_score = schedule_row.get("home_score")
    away_score = schedule_row.get("away_score")
    home_won = bool(schedule_row.get("home_winner")) if pd.notna(schedule_row.get("home_winner")) else None
    region, round_name = parse_region_and_round(schedule_row.get("notes_headline"))

    home_name = schedule_row.get("home_display_name") or schedule_row.get("home_name")
    away_name = schedule_row.get("away_display_name") or schedule_row.get("away_name")
    winner_name = None
    if home_won is True:
        winner_name = home_name
    elif home_won is False:
        winner_name = away_name

    return {
        "game_id": int(schedule_row["game_id"]),
        "region":  region,
        "round":   round_name,
        "round_rank": ROUND_RANK.get(round_name, -1),
        "date":    str(schedule_row.get("date"))[:10],
        "home":    home_name,
        "away":    away_name,
        "home_id": int(schedule_row["home_id"]) if pd.notna(schedule_row.get("home_id")) else None,
        "away_id": int(schedule_row["away_id"]) if pd.notna(schedule_row.get("away_id")) else None,
        "home_score": int(home_score) if pd.notna(home_score) else None,
        "away_score": int(away_score) if pd.notna(away_score) else None,
        "winner":  winner_name,
    }


game_records = [build_game_record(row) for _, row in tournament_games_df.iterrows()]


def link_round_to_feeders(later_round_games, earlier_round_games):
    """Order `later_round_games` so each one sits above its two feeder games.

    A round-N game's two teams are the winners of two adjacent round-(N-1) games.
    We sort the later round by the position of its earliest feeder so the bracket
    chains correctly top-to-bottom even though the data carries no seed numbers.
    """
    winner_position = {}
    for position, feeder in enumerate(earlier_round_games):
        if feeder["winner"]:
            winner_position[feeder["winner"]] = position

    def earliest_feeder_position(game):
        positions = [winner_position[name]
                     for name in (game["home"], game["away"])
                     if name in winner_position]
        return min(positions) if positions else 1e9

    return sorted(later_round_games, key=earliest_feeder_position)


def order_region(region_games):
    """Return {round_name: [ordered game records]} for one region."""
    games_by_round = {}
    for game in region_games:
        games_by_round.setdefault(game["round"], []).append(game)

    ordered = {}
    previous_round_games = None
    for round_name in ROUND_ORDER:
        if round_name not in games_by_round:
            continue
        this_round = games_by_round[round_name]
        if previous_round_games is None:
            # First (deepest) round in the region: stable order by game_id.
            this_round = sorted(this_round, key=lambda g: g["game_id"])
        else:
            this_round = link_round_to_feeders(this_round, previous_round_games)
        ordered[round_name] = this_round
        previous_round_games = this_round
    return ordered


regions = ["East", "West", "South", "Midwest"]
bracket_by_region = {}
for region_name in regions:
    region_games = [g for g in game_records if g["region"] == region_name]
    if region_games:
        bracket_by_region[region_name] = order_region(region_games)

# Final Four + National Championship have no region.
final_four_games = sorted(
    [g for g in game_records if g["round"] == "Final Four"],
    key=lambda g: g["game_id"],
)
championship_games = [g for g in game_records if g["round"] == "National Championship"]
first_four_games = [g for g in game_records if g["round"] == "First Four"]

champion = championship_games[0]["winner"] if championship_games else None
runner_up = None
if championship_games:
    final_game = championship_games[0]
    runner_up = (final_game["away"] if final_game["winner"] == final_game["home"]
                 else final_game["home"])

print(f"  bracket: {len(bracket_by_region)} regions, "
      f"{len(final_four_games)} Final Four, "
      f"{'champion ' + champion if champion else 'no champion yet'}")


# ---------------------------------------------------------------------------
# 3. Team summary: how far each team advanced (from the bracket itself)
# ---------------------------------------------------------------------------
team_stats = {}   # team_name → {wins, losses, pf, pa, best_round_rank, region}
for game in game_records:
    if game["home_score"] is None or game["away_score"] is None:
        continue
    for side, opp_side in (("home", "away"), ("away", "home")):
        team_name = game[side]
        if not team_name:
            continue
        record = team_stats.setdefault(team_name, {
            "team": team_name, "wins": 0, "losses": 0,
            "pf": 0, "pa": 0, "games": 0,
            "best_round_rank": -1, "region": game["region"],
        })
        record["games"] += 1
        record["pf"] += game[f"{side}_score"]
        record["pa"] += game[f"{opp_side}_score"]
        if game["winner"] == team_name:
            record["wins"] += 1
        else:
            record["losses"] += 1
        # The deepest round a team appears in is how far it advanced.
        record["best_round_rank"] = max(record["best_round_rank"], game["round_rank"])
        if game["region"]:
            record["region"] = game["region"]

# Translate the deepest round reached into a human result label.
def result_label(record):
    if champion and record["team"] == champion:
        return "Champion"
    if runner_up and record["team"] == runner_up:
        return "Runner-Up"
    deepest = record["best_round_rank"]
    # A team that *lost* in round R was eliminated there; a team that won its
    # deepest round advanced beyond it.  We label by the round they exited.
    eliminated_rank = deepest if record["losses"] else deepest
    label_by_rank = {
        ROUND_RANK["First Four"]:           "First Four",
        ROUND_RANK["1st Round"]:            "Round of 64",
        ROUND_RANK["2nd Round"]:            "Round of 32",
        ROUND_RANK["Sweet 16"]:             "Sweet 16",
        ROUND_RANK["Elite 8"]:              "Elite 8",
        ROUND_RANK["Final Four"]:           "Final Four",
        ROUND_RANK["National Championship"]:"Final",
    }
    return label_by_rank.get(eliminated_rank, "")

team_summary = []
for record in team_stats.values():
    record["pt_diff"] = record["pf"] - record["pa"]
    record["result"] = result_label(record)
    team_summary.append(record)
team_summary.sort(key=lambda r: (-r["best_round_rank"], -r["pt_diff"]))


# ---------------------------------------------------------------------------
# 4. Player leaders — aggregate boxscores filtered to tournament games only
# ---------------------------------------------------------------------------
player_leaders = []
all_tournament = []   # top-5 standout players ("All-Tournament team")
highlights = {}       # featured single-stat standouts
if tournament_game_ids:
    print("Loading player boxscores…")
    offline_boxscore_csv = os.path.join(PROJECT_ROOT, f"offline_player_{SEASON}.csv")
    if os.path.exists(offline_boxscore_csv):
        raw_boxscores_df = pd.read_csv(offline_boxscore_csv, low_memory=False)
    else:
        raw_boxscores_df = feeds.load_player_box(SEASON)

    raw_boxscores_df["game_id"] = raw_boxscores_df["game_id"].astype(int)
    tournament_box_df = raw_boxscores_df[
        raw_boxscores_df["game_id"].isin(tournament_game_ids)
    ].copy()
    tournament_box_df["minutes"] = pd.to_numeric(
        tournament_box_df["minutes"], errors="coerce"
    ).fillna(0)
    tournament_box_df = tournament_box_df[tournament_box_df["minutes"] >= MIN_MINUTES_THRESHOLD]
    print(f"  {len(tournament_box_df):,} player-game rows in the tournament")

    player_groups = tournament_box_df.groupby(
        ["athlete_id", "athlete_display_name", "team_id", "team_display_name",
         "athlete_position_name"],
        dropna=False,
    )
    totals_df = player_groups[COUNTING_STAT_COLUMNS].sum()
    games_played = player_groups["game_id"].nunique().rename("gp")
    leaders_df = pd.concat([totals_df, games_played], axis=1).reset_index()

    def safe_divide(numerator, denominator):
        return np.where(denominator > 0, numerator / denominator, np.nan)

    fga = leaders_df["field_goals_attempted"]
    fta = leaders_df["free_throws_attempted"]
    fgm = leaders_df["field_goals_made"]
    tpm = leaders_df["three_point_field_goals_made"]

    leaders_df["ppg"] = safe_divide(leaders_df["points"], leaders_df["gp"])
    leaders_df["rpg"] = safe_divide(leaders_df["rebounds"], leaders_df["gp"])
    leaders_df["apg"] = safe_divide(leaders_df["assists"], leaders_df["gp"])
    leaders_df["spg"] = safe_divide(leaders_df["steals"], leaders_df["gp"])
    leaders_df["bpg"] = safe_divide(leaders_df["blocks"], leaders_df["gp"])
    leaders_df["mpg"] = safe_divide(leaders_df["minutes"], leaders_df["gp"])
    leaders_df["fg"]  = safe_divide(fgm, fga)
    leaders_df["fg3"] = safe_divide(tpm, leaders_df["three_point_field_goals_attempted"])
    leaders_df["ft"]  = safe_divide(leaders_df["free_throws_made"], fta)
    leaders_df["efg"] = safe_divide(fgm + 0.5 * tpm, fga)
    # True shooting: pts / (2 * (FGA + 0.44 * FTA))
    leaders_df["ts"]  = safe_divide(leaders_df["points"], 2 * (fga + 0.44 * fta))

    # Keep rotation contributors only (played at least 2 tournament games).
    leaders_df = leaders_df[leaders_df["gp"] >= 1].copy()

    rename_map = {
        "athlete_id": "id", "athlete_display_name": "n",
        "team_id": "tid", "team_display_name": "t",
        "athlete_position_name": "pos",
        "points": "pts", "rebounds": "reb", "assists": "ast",
        "steals": "stl", "blocks": "blk", "turnovers": "tov",
    }
    keep_cols = list(rename_map) + [
        "gp", "ppg", "rpg", "apg", "spg", "bpg", "mpg",
        "fg", "fg3", "ft", "efg", "ts",
    ]
    leaders_out = leaders_df[[c for c in keep_cols if c in leaders_df.columns]].rename(columns=rename_map)

    round_cols = ["ppg", "rpg", "apg", "spg", "bpg", "mpg"]
    pct_cols   = ["fg", "fg3", "ft", "efg", "ts"]
    for col in round_cols:
        leaders_out[col] = pd.to_numeric(leaders_out[col], errors="coerce").round(1)
    for col in pct_cols:
        leaders_out[col] = pd.to_numeric(leaders_out[col], errors="coerce").round(3)

    leaders_out = leaders_out.sort_values("pts", ascending=False)
    player_leaders = leaders_out.where(leaders_out.notna(), other=None).to_dict("records")
    print(f"  {len(player_leaders)} players with tournament stats")

    # --- All-Tournament team + highlight standouts ---
    # Blended tournament score from season-long totals, so it rewards both
    # production and advancing deep (more games = more chances to accumulate).
    blended_score = (
        leaders_df["points"] + 1.2 * leaders_df["rebounds"] + 1.5 * leaders_df["assists"]
        + 2 * leaders_df["steals"] + 2 * leaders_df["blocks"] - leaders_df["turnovers"]
    )
    score_by_athlete_id = dict(zip(leaders_df["athlete_id"], blended_score))
    for player_record in player_leaders:
        athlete_id = player_record.get("id")
        player_record["tscore"] = (
            round(float(score_by_athlete_id.get(athlete_id, 0)), 1)
            if athlete_id is not None else None
        )

    # Top 5 by blended score (require 2+ games so a single hot night can't crash it).
    all_tournament = sorted(
        [r for r in player_leaders if (r.get("gp") or 0) >= 2],
        key=lambda r: r.get("tscore") or 0, reverse=True,
    )[:5]

    # Best single-game scoring performance, with the opponent from the bracket.
    opponent_by_game_team = {}
    for game in game_records:
        if game["home"] and game["away"]:
            opponent_by_game_team[(game["game_id"], game["home"])] = game["away"]
            opponent_by_game_team[(game["game_id"], game["away"])] = game["home"]
    best_single_game = None
    if not tournament_box_df.empty:
        best_row = tournament_box_df.loc[tournament_box_df["points"].idxmax()]
        best_single_game = {
            "n":   best_row["athlete_display_name"],
            "t":   best_row["team_display_name"],
            "tid": int(best_row["team_id"]) if pd.notna(best_row["team_id"]) else None,
            "id":  int(best_row["athlete_id"]) if pd.notna(best_row["athlete_id"]) else None,
            "pts": int(best_row["points"]),
            "opp": opponent_by_game_team.get((int(best_row["game_id"]), best_row["team_display_name"])),
        }

    def _top_by(metric):
        ranked = sorted(player_leaders, key=lambda r: r.get(metric) or 0, reverse=True)
        return ranked[0] if ranked else None

    highlights = {
        "top_scorer":    _top_by("pts"),
        "top_rebounder": _top_by("reb"),
        "top_playmaker": _top_by("ast"),
        "best_game":     best_single_game,
    }


# ---------------------------------------------------------------------------
# 5. Write postseason.json
# ---------------------------------------------------------------------------
# Use the most recent tournament game date as the build stamp so the committed
# JSON stays byte-identical when nothing has changed (matches build_site.py).
if not tournament_games_df.empty:
    stamp_dates = pd.to_datetime(tournament_games_df["date"], errors="coerce")
    data_stamp = stamp_dates.max().date().isoformat() if stamp_dates.notna().any() else f"{SEASON}-04-30"
else:
    data_stamp = f"{SEASON}-04-30"

output = {
    "bracket": {
        "regions":      bracket_by_region,
        "first_four":   first_four_games,
        "final_four":   final_four_games,
        "championship": championship_games,
    },
    "champion":       champion,
    "runner_up":      runner_up,
    "all_tournament": all_tournament,
    "highlights":     highlights,
    "player_leaders": player_leaders,
    "team_summary":   team_summary,
    "meta": {"built_at": data_stamp, "season": SEASON, "n_games": len(tournament_game_ids)},
}

output_path = os.path.join(SEASON_DATA_DIR, "postseason.json")
with open(output_path, "w") as json_file:
    json_file.write(json.dumps(sanitize_for_json(output), separators=(",", ":")))
print(f"\nWrote {output_path}  ({os.path.getsize(output_path) / 1024:.0f} KB)")
print("Done.")


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
#
# SEASON                   int    Calendar year the season ends (e.g. 2026).
# NCAA_TOURNAMENT_ID       int    ESPN tournament_id for the NCAA D-I championship (22).
# ROUND_ORDER              list   Round names earliest→latest (index = bracket depth).
# ROUND_RANK               dict   {round_name: depth index}.
# COUNTING_STAT_COLUMNS    list   Boxscore columns summed per player.
#
# schedule_df              DataFrame  Full ESPN schedule for the season.
# tournament_games_df      DataFrame  Completed games where tournament_id == 22.
# tournament_game_ids      set    int game_ids of all NCAA tournament games.
# game_records             list   Compact dict per game (region/round/scores/winner).
# bracket_by_region        dict   {region: {round_name: [ordered game records]}}.
# final_four_games         list   Final Four game records (no region).
# championship_games       list   National Championship game record(s).
# first_four_games         list   First Four play-in game records.
# champion / runner_up     str    Team names from the title game.
# team_stats / team_summary       Per-team tournament W/L, point diff, round reached.
# tournament_box_df        DataFrame  Player boxscores filtered to tournament games.
# leaders_df / leaders_out DataFrame  Aggregated per-player tournament stats.
# player_leaders           list   Per-player tournament stat records for the page.
# data_stamp               str    Most-recent tournament game date (build stamp).
# output                   dict   Everything serialized into postseason.json.
