"""Per-team game files for the Games views on the player and team cards.

Writes site/data/{season}/games/teams/{team_id}.json: every game that team
played that season, each with its result and every player's line. That one file
serves every view:
  - a player's game log is his lines from his team's file;
  - a team's game log is its games and totals;
  - a game's box score is the team's entry plus the same game in the opponent's file.

The nightly rebuilds the live season; past seasons are written once from their
archived box scores and frozen.

One file per team rather than per player or per game: a team's file only changes
on nights it plays, which keeps the nightly commit small, and ~700 files a season
stays well under Cloudflare's 20,000-file cap (one file per game would be ~12,000).

Compact on purpose (six seasons of box scores would otherwise be ~250 MB): names
and positions are listed once per file in "roster", and each player line or total
is a plain array in "cols" order. site/js/games.js decodes it.
"""
import json
import os
import shutil

import pandas as pd

from hoglib.io import sanitize_for_json

# player_box column → short key, in the order of every line/total array.
STAT_COLUMNS = {
    "minutes": "min", "points": "pts", "rebounds": "reb", "offensive_rebounds": "orb",
    "assists": "ast", "steals": "stl", "blocks": "blk", "turnovers": "tov", "fouls": "pf",
    "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "tpm", "three_point_field_goals_attempted": "tpa",
    "free_throws_made": "ftm", "free_throws_attempted": "fta",
}
COLS = list(STAT_COLUMNS.values())


def _number(value):
    """Whole floats as ints (27.0 → 27) so the JSON stays short."""
    if pd.isna(value):
        return None
    return int(value) if float(value).is_integer() else round(float(value), 1)


def write_game_logs(player_box, season_dir):
    """Rebuild season_dir/games/teams/ from one season's player box scores."""
    box = player_box[player_box["season_type"].isin([2, 3]) & player_box["athlete_id"].notna()].copy()
    box["played"] = ~box["did_not_play"].fillna(False).astype(bool)
    box["date"] = box["game_date"].astype(str).str[:10]
    for col in STAT_COLUMNS:
        box[col] = pd.to_numeric(box[col], errors="coerce")

    out_dir = os.path.join(season_dir, "games", "teams")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)        # fully regenerated; drops teams no longer in the feed
    os.makedirs(out_dir)

    games_written = 0
    for team_id, team_box in box.groupby("team_id"):
        roster = {}
        games = []
        for game_id, game in team_box.sort_values(["date", "game_id"]).groupby("game_id", sort=False):
            first = game.iloc[0]
            lines, dnp = [], []
            for _, p in game.sort_values(["starter", "minutes"], ascending=False).iterrows():
                athlete_id = int(p["athlete_id"])
                roster.setdefault(str(athlete_id), [p["athlete_display_name"], p["athlete_position_abbreviation"]])
                if p["played"]:
                    lines.append([athlete_id, int(bool(p["starter"]))]
                                 + [_number(p[col]) for col in STAT_COLUMNS])
                else:
                    dnp.append(athlete_id)
            played = game[game["played"]]
            games.append({
                "g": int(game_id), "d": first["date"], "ha": first["home_away"],
                "o": first["opponent_team_location"], "oid": first["opponent_team_id"],
                "ts": first["team_score"], "os": first["opponent_team_score"], "w": int(bool(first["team_winner"])),
                "p": lines, "dnp": dnp,
                "tot": [None] + [_number(played[col].sum()) for col in list(STAT_COLUMNS)[1:]],
            })
        with open(os.path.join(out_dir, f"{int(team_id)}.json"), "w") as team_file:
            team_file.write(json.dumps(
                sanitize_for_json({"t": team_box["team_display_name"].iloc[0], "cols": COLS,
                                   "roster": roster, "games": games}),
                separators=(",", ":"), allow_nan=False))
        games_written += len(games)
    print(f"  games/teams/{{team_id}}.json  {box['team_id'].nunique():,} teams, "
          f"{games_written // 2:,} games")
