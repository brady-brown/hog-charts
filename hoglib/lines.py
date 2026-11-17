"""Model vs. market log: our pre-game numbers next to the book's line.

Writes site/data/{season}/games/lines.json, one row per game we predicted:
  - the model's spread, home win probability and projected total, frozen at the
    last nightly run before tip (each run overwrites them only while the game
    is still in the future, so they never see the result);
  - the book's opening and closing spread/total (raw_{season}/lines.parquet,
    from ESPN's summary pickcenter; arrives once the game is final);
  - the final score (games/index.json).

The file is cumulative: each nightly reads the committed copy and adds to it,
so it is the only record of what the model said before each game. Spreads are
home-relative everywhere (negative = home favored), the same convention as the
book's.

line_report.py reads it to compare the model with the line.
"""
import datetime as dt
import json
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from hoglib.io import sanitize_for_json

COLS = ["g", "d", "away", "home", "neutral",
        "model_spread", "model_prob", "model_total", "model_at",
        "provider", "open_spread", "close_spread", "open_total", "close_total",
        "away_pts", "home_pts"]

EASTERN = ZoneInfo("America/New_York")


def _num(value, digits=2):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(value) else round(value, digits)


def predict(model, away, home, neutral, league_def):
    """The model on one game — must match site/js/predict.js (t1 = away, t2 = home).

    Returns (home_spread, home_win_prob, projected_total); home_spread < 0 = home favored.
    """
    league_adv = model["league_home_adv"]
    home_adv = home.get("home_adv")
    home_adv = league_adv if home_adv is None or np.isnan(home_adv) else home_adv
    home_court = 0.0 if neutral else -home_adv

    exp_pace = (away["pace"] + home["pace"]) / 2
    tempo_adj = (away["net_eff"] - home["net_eff"]) * exp_pace / model["league_avg_pace"]
    form_diff = away["form"] - home["form"]
    c0, c1, c2 = model["coef"]
    away_margin = model["intercept"] + c0 * tempo_adj + c1 * home_court + c2 * form_diff
    away_prob = float(np.interp(away_margin, model["iso_x"], model["iso_y"]))

    pp_away = away["off_eff"] + (home["def_eff"] - league_def)
    pp_home = home["off_eff"] + (away["def_eff"] - league_def)
    total = (pp_away + pp_home) / 100 * exp_pace
    # Betting convention: the home line is minus the home margin, i.e. the away
    # margin. Ark +2.5 at IU means IU is expected to win by 2.5 → home line −2.5.
    return away_margin, 1 - away_prob, total


def _load(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as handle:
        return json.load(handle)


def write_line_log(season_dir, lines_frame, now=None):
    now = now or dt.datetime.now(EASTERN)
    games_dir = os.path.join(season_dir, "games")
    log_path = os.path.join(games_dir, "lines.json")

    existing = _load(log_path, {"cols": COLS, "games": []})
    log = {row[0]: dict(zip(existing["cols"], row)) for row in existing["games"]}

    # 1. Freeze the model's numbers for every game that hasn't tipped yet.
    predictor = _load(os.path.join(season_dir, "predictor.json"), None)
    upcoming = _load(os.path.join(games_dir, "upcoming.json"), {"games": []})
    captured = 0
    if predictor:
        model = predictor["model"]
        teams = {t["team_id"]: t for t in predictor["teams"]}
        league_def = float(np.mean([t["def_eff"] for t in predictor["teams"]]))
        needed = ("net_eff", "off_eff", "def_eff", "pace", "form")
        for g, day, start, away_id, home_id, neutral, *_ in upcoming["games"]:
            if start and dt.datetime.fromisoformat(start) <= now:
                continue                         # already tipped: keep the frozen numbers
            away, home = teams.get(away_id), teams.get(home_id)
            if not away or not home or any(t.get(k) is None for t in (away, home) for k in needed):
                continue
            spread, prob, total = predict(model, away, home, bool(neutral), league_def)
            row = log.setdefault(g, {c: None for c in COLS})
            row.update({"g": g, "d": day, "away": away_id, "home": home_id, "neutral": int(bool(neutral)),
                        "model_spread": _num(spread), "model_prob": _num(prob, 4),
                        "model_total": _num(total, 1), "model_at": now.isoformat(timespec="minutes")})
            captured += 1

    # 2. The book's open/close, once the game is final.
    if lines_frame is not None and len(lines_frame):
        for line in lines_frame.to_dict("records"):
            row = log.get(int(line["game_id"]))
            if row is None:
                continue
            for src, dst in (("open_home_spread", "open_spread"), ("close_home_spread", "close_spread"),
                             ("open_total", "open_total"), ("close_total", "close_total")):
                row[dst] = _num(line.get(src), 1)
            row["provider"] = None if pd.isna(line.get("provider")) else line.get("provider")

    # 3. Final scores.
    index = _load(os.path.join(games_dir, "index.json"), {"games": []})
    for g, _d, away_id, away_pts, home_id, home_pts, *_ in index["games"]:
        row = log.get(g)
        if row is not None and row["away"] == away_id and row["home"] == home_id:
            row["away_pts"], row["home_pts"] = away_pts, home_pts

    rows = sorted(log.values(), key=lambda r: (r["d"] or "", r["g"]))
    os.makedirs(games_dir, exist_ok=True)
    with open(log_path, "w") as out:
        out.write(json.dumps(sanitize_for_json({"cols": COLS, "games": [[r.get(c) for c in COLS] for r in rows]}),
                             separators=(",", ":"), allow_nan=False))
    graded = sum(1 for r in rows if r["close_spread"] is not None and r["home_pts"] is not None)
    print(f"  games/lines.json  {len(rows):,} games logged ({captured:,} captured this run, {graded:,} graded)")
