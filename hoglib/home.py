"""The home page's daily cards: site/data/{season}/home.json.

    statlines  the best individual lines from the last completed day, scored
               PTS + REB + AST + STL + BLK (points break ties). Top 8 kept.
    upset      that day's biggest upset against the closing line: how far the
               underdog beat the spread, preferring underdogs that won outright
               (a 20-point favorite winning by 50 is far from the spread but
               isn't an upset). Falls back to the biggest underdog cover.
    best_game  today's game between the two best teams by average Net rating
               (Team Stats, team-stats-all.json).

"Last completed day" = the most recent date with finals before today (Eastern),
so a midday build never shows a half-finished day. Reads files build_site.py has
already written (games/index.json, games/teams/*.json, games/upcoming.json,
games/lines.json, team-stats-all.json) plus the ingest's closing lines.
"""
import datetime as dt
import json
import os
from zoneinfo import ZoneInfo

import pandas as pd

from hoglib.io import sanitize_for_json

EASTERN = ZoneInfo("America/New_York")
STAT_KEYS = ("pts", "reb", "ast", "stl", "blk")


def _load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as handle:
        return json.load(handle)


def _statlines(teams_dir, day, top=8):
    lines = []
    for filename in os.listdir(teams_dir):
        data = _load(os.path.join(teams_dir, filename))
        cols = data["cols"]
        at = {c: i + 2 for i, c in enumerate(cols)}          # player line = [id, starter, *cols]
        team_id = int(filename.split(".")[0])
        for g in data["games"]:
            if g["d"] != day:
                continue
            for line in g["p"]:
                stat = {c: line[at[c]] for c in cols}
                if any(stat.get(k) is None for k in STAT_KEYS):
                    continue
                lines.append({
                    "pid": line[0], "name": data["roster"][str(line[0])][0],
                    "pos": data["roster"][str(line[0])][1], "tid": team_id, "team": data["t"],
                    "g": g["g"], "oid": g["oid"], "opp": g["o"], "ha": g["ha"],
                    "ts": g["ts"], "os": g["os"], "w": g["w"],
                    "score": sum(stat[k] for k in STAT_KEYS), **stat,
                })
    lines.sort(key=lambda r: (r["score"], r["pts"]), reverse=True)
    return lines[:top]


def _closing_lines(games_dir, lines_frame):
    """game_id → home closing spread: the ingest's lines feed, else the line log."""
    close = {}
    log = _load(os.path.join(games_dir, "lines.json"))
    if log:
        for row in log["games"]:
            r = dict(zip(log["cols"], row))
            if r.get("close_spread") is not None:
                close[r["g"]] = (r["close_spread"], r.get("provider"))
    if lines_frame is not None and len(lines_frame):
        for r in lines_frame.to_dict("records"):
            if pd.notna(r.get("close_home_spread")):
                close[int(r["game_id"])] = (float(r["close_home_spread"]), r.get("provider"))
    return close


def _upset(finals, close):
    best = None
    for g, _d, away, away_pts, home, home_pts, *_ in finals:
        if g not in close or close[g][0] == 0:
            continue
        spread, provider = close[g]
        margin = home_pts - away_pts                     # home margin
        home_dog = spread > 0                            # home line > 0 → home is the underdog
        dog, fav = (home, away) if home_dog else (away, home)
        dog_margin = margin if home_dog else -margin
        beat_by = dog_margin - (-abs(spread))            # how far the underdog beat the number
        if beat_by <= 0:
            continue
        cand = {"g": g, "dog": dog, "fav": fav, "spread": abs(spread), "provider": provider,
                "dog_margin": dog_margin, "beat_by": beat_by, "outright": dog_margin > 0}
        if best is None or (cand["outright"], cand["beat_by"]) > (best["outright"], best["beat_by"]):
            best = cand
    return best


def _best_game(upcoming, today, net):
    best = None
    for g, day, start, away, home, neutral, tbd, tv, venue, provider, details, *_ in upcoming.get("games", []):
        if day != today or away not in net or home not in net:
            continue
        avg = (net[away] + net[home]) / 2
        if best is None or avg > best["avg_net"]:
            best = {"g": g, "away": away, "home": home, "start": None if tbd else start,
                    "neutral": neutral, "tv": tv, "venue": venue, "line": details, "provider": provider,
                    "away_net": net[away], "home_net": net[home], "avg_net": round(avg, 1)}
    return best


def _teams(ids, index, upcoming, stats):
    """Display info for every team a card mentions: names, rank, record today."""
    names = {int(k): v for k, v in index.get("teams", {}).items()}
    for k, v in upcoming.get("teams", {}).items():
        names.setdefault(int(k), v)
        if len(v) > 2 and len(names[int(k)]) < 3:
            names[int(k)] = v
    rank = {t["team_id"]: t.get("rank") for t in stats["team_stats"]}
    wl = {}
    for _g, _d, away, away_pts, home, home_pts, *_ in index["games"]:
        for me, mine, theirs in ((away, away_pts, home_pts), (home, home_pts, away_pts)):
            r = wl.setdefault(me, [0, 0]); r[0 if mine > theirs else 1] += 1
    out = {}
    for tid in ids:
        n = names.get(tid, [str(tid), str(tid)])
        out[str(tid)] = {"name": n[0], "short": n[1] if len(n) > 1 else n[0],
                         "abbr": n[2] if len(n) > 2 else None, "rank": rank.get(tid),
                         "record": "-".join(map(str, wl.get(tid, [0, 0])))}
    return out


def write_home(season_dir, lines_frame=None, now=None):
    now = now or dt.datetime.now(EASTERN)
    today = now.date().isoformat()
    games_dir = os.path.join(season_dir, "games")
    index = _load(os.path.join(games_dir, "index.json"), {"games": []})
    days = sorted({row[1] for row in index["games"] if row[1] < today})
    day = days[-1] if days else None

    statlines = _statlines(os.path.join(games_dir, "teams"), day) if day else []
    finals = [row for row in index["games"] if row[1] == day]
    upset = _upset(finals, _closing_lines(games_dir, lines_frame)) if day else None
    if upset:
        row = next(r for r in finals if r[0] == upset["g"])
        upset.update({"away": row[2], "away_pts": row[3], "home": row[4], "home_pts": row[5],
                      "away_top": row[6], "home_top": row[7]})       # [name, pts] or None

    stats = _load(os.path.join(season_dir, "team-stats-all.json"), {"team_stats": []})
    net = {t["team_id"]: t["net_eff"] for t in stats["team_stats"] if t.get("net_eff") is not None}
    upcoming = _load(os.path.join(games_dir, "upcoming.json"), {"games": []})
    best_game = _best_game(upcoming, today, net)

    ids = {s["tid"] for s in statlines} | {s["oid"] for s in statlines if s.get("oid") is not None}
    for card in (upset, best_game):
        if card:
            ids |= {card["away"], card["home"]}
    out = {"built": now.isoformat(timespec="minutes"), "day": day, "today": today,
           "statlines": statlines, "upset": upset, "best_game": best_game,
           "teams": _teams(ids, index, upcoming, stats)}
    with open(os.path.join(season_dir, "home.json"), "w") as handle:
        handle.write(json.dumps(sanitize_for_json(out), separators=(",", ":"), allow_nan=False))
    top = statlines[0] if statlines else None
    print(f"  home.json  day={day}  statline={top and (top['name'], top['score'])}  "
          f"upset={upset and upset['g']}  best_game={best_game and best_game['g']}")
