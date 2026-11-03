"""Per-team game files for the player card's Games tab (live season only).

Writes site/data/{season}/games/teams/{team_id}.json: every game that team
played this season, each with its result and every player's line. That one file
serves both views on the card:
  - a player's game log is his lines from his team's file;
  - a game's box score is his team's entry plus the same game in the opponent's file.

One file per team rather than per player or per game: a team's file only changes
on nights it plays, which keeps the nightly commit small, and ~700 files stays well
under Cloudflare's 20,000-file cap (one file per game would be ~12,000 a season).
"""
import json
import os
import shutil

import pandas as pd

from hoglib.io import sanitize_for_json

# player_box column → short key for each player's line.
STAT_KEYS = {
    "minutes": "min", "points": "pts", "rebounds": "reb", "offensive_rebounds": "orb",
    "assists": "ast", "steals": "stl", "blocks": "blk", "turnovers": "tov", "fouls": "pf",
    "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "tpm", "three_point_field_goals_attempted": "tpa",
    "free_throws_made": "ftm", "free_throws_attempted": "fta",
}


def _stat_line(row):
    return {short: row[col] for col, short in STAT_KEYS.items()}


def write_game_logs(player_box, season_dir):
    """Rebuild season_dir/games/teams/ from one season's player box scores."""
    box = player_box[player_box["season_type"].isin([2, 3])].copy()
    box["played"] = ~box["did_not_play"].fillna(False).astype(bool)
    box["date"] = box["game_date"].astype(str).str[:10]
    for col in STAT_KEYS:
        box[col] = pd.to_numeric(box[col], errors="coerce")

    out_dir = os.path.join(season_dir, "games", "teams")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)        # fully regenerated; drops teams no longer in the feed
    os.makedirs(out_dir)

    games_written = 0
    for team_id, team_box in box.groupby("team_id"):
        games = []
        for game_id, game in team_box.sort_values(["date", "game_id"]).groupby("game_id", sort=False):
            first = game.iloc[0]
            played = game[game["played"]]
            games.append({
                "g": int(game_id), "d": first["date"], "ha": first["home_away"],
                "o": first["opponent_team_location"], "oid": first["opponent_team_id"],
                "ts": first["team_score"], "os": first["opponent_team_score"], "w": bool(first["team_winner"]),
                "players": [{"id": int(p["athlete_id"]), "n": p["athlete_display_name"],
                             "pos": p["athlete_position_abbreviation"], "st": bool(p["starter"]),
                             "dnp": not p["played"], **_stat_line(p)}
                            for _, p in game.sort_values(["starter", "minutes"], ascending=False).iterrows()],
                "totals": {short: played[col].sum() for col, short in STAT_KEYS.items() if col != "minutes"},
            })
        games.sort(key=lambda g: (g["d"], g["g"]))
        with open(os.path.join(out_dir, f"{int(team_id)}.json"), "w") as team_file:
            team_file.write(json.dumps(
                sanitize_for_json({"t": team_box["team_display_name"].iloc[0], "games": games}),
                separators=(",", ":"), allow_nan=False))
        games_written += len(games)
    print(f"  games/teams/{{team_id}}.json  {box['team_id'].nunique():,} teams, "
          f"{games_written // 2:,} games")
