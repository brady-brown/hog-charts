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
positions and jersey numbers are listed once per file in "roster" ([name, pos, jersey]),
"c" is the team's [primary, alternate] color, and each player line or total
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


def _team_color(series):
    """A team's most common hex color in the feed ("a32136"), or None."""
    values = series.dropna().astype(str).str.strip().str.lstrip("#").str.lower()
    values = values[values.str.fullmatch(r"[0-9a-f]{6}")]
    return values.mode().iloc[0] if not values.empty else None


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
                jersey = p.get("athlete_jersey")
                roster.setdefault(str(athlete_id), [p["athlete_display_name"], p["athlete_position_abbreviation"],
                                                    None if pd.isna(jersey) else str(jersey)])
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
                                   # [primary, alternate] hex, no "#" (the Game Chart jersey)
                                   "c": [_team_color(team_box[col]) if col in team_box else None
                                         for col in ("team_color", "team_alternate_color")],
                                   "roster": roster, "games": games}),
                separators=(",", ":"), allow_nan=False))
        games_written += len(games)
    print(f"  games/teams/{{team_id}}.json  {box['team_id'].nunique():,} teams, "
          f"{games_written // 2:,} games")
    write_game_index(season_dir)


def write_game_index(season_dir):
    """season_dir/games/index.json — one row per game, for the Games scoreboard.

    Built from the team files just written (so it also works on a frozen past
    season without its box scores). Each game appears once, away team first:
      games: [[game_id, date, away_id, away_pts, home_id, home_pts, away_top, home_top]]
      teams: {team_id: [name, short]}   short = school only ("Duke"), from the
                                         opponents' "o" fields; falls back to name
    *_top is that team's leading scorer as [name, pts], or null when the team has
    no file (non-D-I opponents). Opponents without a file are named from the
    other team's "o" field.
    """
    teams_dir = os.path.join(season_dir, "games", "teams")
    names, short, by_game = {}, {}, {}
    for filename in sorted(os.listdir(teams_dir)):
        with open(os.path.join(teams_dir, filename)) as team_file:
            data = json.load(team_file)
        team_id = int(filename.split(".")[0])
        names[team_id] = data["t"]
        cols = data["cols"]
        pts_at = cols.index("pts") + 2            # player line = [id, starter, *cols]
        for g in data["games"]:
            if g["oid"] is not None and g["o"]:
                short.setdefault(int(g["oid"]), g["o"])
            top = max(g["p"], key=lambda line: line[pts_at] or 0, default=None)
            side = {"tid": team_id, "pts": g["ts"],
                    "top": [data["roster"][str(top[0])][0], top[pts_at]] if top and top[pts_at] else None}
            entry = by_game.setdefault(g["g"], {"d": g["d"]})
            entry[g["ha"]] = side
            # Placeholder for the other side, replaced if the opponent has a file.
            other = "home" if g["ha"] == "away" else "away"
            if g["oid"] is not None:
                entry.setdefault(other, {"tid": int(g["oid"]), "pts": g["os"], "top": None, "name": g["o"]})
    rows = []
    for game_id, e in by_game.items():
        away, home = e.get("away"), e.get("home")
        if not away or not home:
            continue
        for side in (away, home):
            if side["tid"] not in names and side.get("name"):
                names[side["tid"]] = side["name"]
        rows.append([game_id, e["d"], away["tid"], away["pts"], home["tid"], home["pts"], away["top"], home["top"]])
    rows.sort(key=lambda r: (r[1], r[0]))
    with open(os.path.join(season_dir, "games", "index.json"), "w") as index_file:
        index_file.write(json.dumps(sanitize_for_json({"teams": {str(k): [v, short.get(k, v)] for k, v in names.items()},
                                                      "games": rows}),
                                    separators=(",", ":"), allow_nan=False))
    print(f"  games/index.json  {len(rows):,} games")


def _text(value):
    return None if value is None or pd.isna(value) or str(value) == "" else str(value)


def write_upcoming(schedule, season_dir, today=None, max_days=21):
    """season_dir/games/upcoming.json — games not yet played, for the Games page.

    Covers every D-I team's next game: all scheduled games from today (Eastern)
    through the latest date any D-I team plays next, capped at max_days out. Games
    already in games/index.json (finals) are left out.
      games: [[game_id, date, start_iso, away_id, home_id, neutral, tbd,
               broadcast, venue, line_provider, line_details, home_spread, over_under]]
      teams: {team_id: [name, short, abbreviation]}
    The line columns are ESPN's scoreboard line (null until a book posts one).
    """
    import datetime as dt
    from zoneinfo import ZoneInfo

    today = today or dt.datetime.now(ZoneInfo("America/New_York")).date()
    games_dir = os.path.join(season_dir, "games")
    os.makedirs(games_dir, exist_ok=True)
    finals = set()
    index_path = os.path.join(games_dir, "index.json")
    if os.path.exists(index_path):
        with open(index_path) as index_file:
            finals = {row[0] for row in json.load(index_file)["games"]}

    # Only D-I teams (ESPN gives them a conference id; small colleges have none)
    # count toward "everyone's next game", or one NAIA team's December date
    # would stretch the window.
    division_one = set()
    for side in ("away", "home"):
        has_conf = schedule[f"{side}_conference_id"].notna()
        division_one |= set(schedule.loc[has_conf, f"{side}_id"].dropna().astype("int64"))

    s = schedule.copy()
    s["day"] = pd.to_datetime(s["game_date"].astype("string"), errors="coerce").dt.date
    s = s[(s["status_type_state"].astype("string") == "pre") & s["home_id"].notna() & s["away_id"].notna()
          & s["day"].notna() & (s["day"] >= today)
          & (s["day"] <= today + dt.timedelta(days=max_days))]
    s = s[~s["game_id"].astype("int64").isin(finals)]

    rows, teams = [], {}
    if len(s):
        long = pd.concat([s[["away_id", "day"]].rename(columns={"away_id": "tid"}),
                          s[["home_id", "day"]].rename(columns={"home_id": "tid"})])
        long = long[long["tid"].isin(division_one)]
        last_day = long.groupby("tid")["day"].min().max()       # the latest "next game"
        s = s[s["day"] <= last_day].sort_values(["game_date_time", "game_id"])
        for _, g in s.iterrows():
            start = _text(g["game_date_time"])
            venue = ", ".join(x for x in (_text(g["venue_full_name"]), _text(g["venue_address_city"]),
                                          _text(g["venue_address_state"])) if x)
            for side in ("away", "home"):
                teams.setdefault(str(int(g[f"{side}_id"])),
                                 [_text(g[f"{side}_display_name"]), _text(g[f"{side}_location"]),
                                  _text(g[f"{side}_abbreviation"])])
            rows.append([
                int(g["game_id"]), g["day"].isoformat(), start.replace(" ", "T") if start else None,
                int(g["away_id"]), int(g["home_id"]), int(bool(g["neutral_site"])) if pd.notna(g["neutral_site"]) else 0,
                int((_text(g["status_type_detail"]) or "").upper() == "TBD"),
                _text(g.get("broadcast")), venue or None,
                _text(g.get("odds_provider")), _text(g.get("odds_details")),
                _number(g.get("odds_home_spread")), _number(g.get("odds_over_under")),
            ])
    with open(os.path.join(games_dir, "upcoming.json"), "w") as out:
        out.write(json.dumps(sanitize_for_json({"built": today.isoformat(), "teams": teams, "games": rows}),
                             separators=(",", ":"), allow_nan=False))
    print(f"  games/upcoming.json  {len(rows):,} games")
