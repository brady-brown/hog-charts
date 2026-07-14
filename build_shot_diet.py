"""
build_shot_diet.py — per-player "shot diet" tables (rim / mid-range / three).

A player's shot diet is WHERE he shoots from, as a share of his field-goal
attempts, collapsed into three intuitive buckets:

    rim   — Restricted Area (inside ~4 ft of the basket): layups, dunks, putbacks
    mid   — every other two-point attempt (short floaters out to long twos)
    three — three-point attempts

The three rates are computed from the SAME play-by-play shot coordinates, so for
a player with tracked shots they sum to 100% (heaves and coordinate-less shots
are dropped from the denominator).  This is deliberately coarser than the 14-zone
shot chart — it answers "does he live at the rim or settle for jumpers?" at a
glance, and drops straight into the player-stats table.

Coordinate → bucket boundaries MUST stay identical to classify_shot_zones() in
build_site.py (and classifyZone() in site/js/zone-chart.js) so the diet lines up
with the shot charts: RA < 4 ft, three at/beyond the 22.146 ft arc (or the corner
strip), everything else is mid.

SHOT DATA IS CURRENT-SEASON ONLY.  build_shots_data.py only writes
shots_{SEASON}.parquet for the current season (ESPN's historical shot
coordinates are unreliable — see the shot-chart pages), so past seasons get no
diet file and the columns come back blank in build_site.py, exactly like the
shot charts themselves.

Scopes mirror the player-stats scope masks (game-id sets from the schedule):
    shot_diet_{SEASON}.csv          — regular season   (season_type 2, no conf tourney)
    shot_diet_{SEASON}_all.csv      — regular + postseason
    shot_diet_{SEASON}_post.csv     — postseason        (season_type 3 + conf tourneys)
    shot_diet_{SEASON}_conf.csv     — conference        (regular league games)
    shot_diet_{SEASON}_nonconf.csv  — non-conference    (regular games, not league)
    each: athlete_id, team_id, fga, rim_rate, mid_rate, three_rate

Run:  python build_shot_diet.py            (current season)
      OVERRIDE_SEASON=2025 python build_shot_diet.py
"""

import os
from datetime import date as _date

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

from hoglib.season import detect_season
SEASON = detect_season()

# Below this many tracked FGA in a scope, the diet is too small a sample to trust,
# so the rates are left blank (matches the rotation-player focus of the table).
MIN_TRACKED_FGA = 20


def _game_scopes():
    """Game-id sets per scope, mirroring build_player_stats.py's scope masks.

    reg     = season_type 2 games that are NOT conference tournaments
    conf    = reg games flagged conference_competition (league games)
    nonconf = reg games that are NOT conference league games
    post    = season_type 3 (NCAA/NIT/…) plus the conference tournaments
    all     = reg ∪ post

    Conference tournaments are ESPN-tagged season_type 2 but belong to the
    postseason, so they are pulled out of reg/conf/nonconf and folded into post.
    """
    import sportsdataverse.mbb as mbb
    sched = mbb.load_mbb_schedule(seasons=SEASON, return_as_pandas=True)
    conf_comp_ids = set(
        sched.loc[sched["conference_competition"] == True, "game_id"].astype(int).unique())

    notes = sched.get("notes_headline")
    if notes is not None:
        is_conf_tourney = (
            (sched["conference_competition"] == True)
            & notes.astype(str).str.contains("Tournament|Championship|Playoffs",
                                             case=False, na=False))
        conf_tourney_ids = set(sched.loc[is_conf_tourney, "game_id"].astype(int).unique())
    else:
        conf_tourney_ids = set()

    type2_ids = set(sched.loc[sched["season_type"] == 2, "game_id"].astype(int).unique())
    type3_ids = set(sched.loc[sched["season_type"] == 3, "game_id"].astype(int).unique())

    reg_games     = type2_ids - conf_tourney_ids
    conf_games    = (conf_comp_ids - conf_tourney_ids) & reg_games
    nonconf_games = reg_games - conf_games
    post_games    = type3_ids | conf_tourney_ids
    all_games     = reg_games | post_games
    return {"": reg_games, "_all": all_games, "_post": post_games,
            "_conf": conf_games, "_nonconf": nonconf_games}


def _classified_shots():
    """Load the shot parquet and label each FGA rim / mid / three.

    Mirrors classify_shot_zones() in build_site.py but collapses the 14 zones to
    three buckets. Returns rows with game_id, team_id, athlete_id and one of the
    bucket columns set; heaves and coordinate-less shots are dropped.
    """
    shots_path = os.path.join(PROJECT_ROOT, f"shots_{SEASON}.parquet")
    if not os.path.exists(shots_path):
        print(f"  [warn] {shots_path} not found — no shot-diet tables written")
        return None

    shots = pd.read_parquet(
        shots_path, columns=["game_id", "team_id", "athlete_id_1",
                             "coordinate_x", "coordinate_y"])
    shots = shots.rename(columns={"athlete_id_1": "athlete_id"})
    for col in ("game_id", "team_id", "athlete_id"):
        shots[col] = pd.to_numeric(shots[col], errors="coerce").astype("Int64")
    shots = shots.dropna(subset=["game_id", "team_id", "athlete_id",
                                 "coordinate_x", "coordinate_y"]).copy()

    RA, THREE, CORNER_X = 4.0, 22.146, 21.65
    Y_MEET = np.sqrt(THREE**2 - CORNER_X**2)

    lateral  = -shots["coordinate_y"]
    toward   = 41.75 - shots["coordinate_x"].abs()
    distance = np.sqrt(lateral**2 + toward**2)

    is_heave = distance >= 40
    is_rim   = ~is_heave & (distance < RA)
    is_three = ~is_heave & ~is_rim & (
        (distance >= THREE) | ((lateral.abs() >= CORNER_X) & (toward <= Y_MEET)))
    is_mid   = ~is_heave & ~is_rim & ~is_three

    shots = shots[~is_heave].copy()
    shots["is_rim"]   = is_rim[~is_heave].astype(int)
    shots["is_mid"]   = is_mid[~is_heave].astype(int)
    shots["is_three"] = is_three[~is_heave].astype(int)
    return shots


def _scope_table(shots, games):
    """One scope's per-(team, player) shot-diet table for a game-id set."""
    cols = ["athlete_id", "team_id", "fga", "rim_rate", "mid_rate", "three_rate"]
    if not games:
        return pd.DataFrame(columns=cols)

    s = shots[shots["game_id"].isin(games)]
    grouped = (s.groupby(["team_id", "athlete_id"])[["is_rim", "is_mid", "is_three"]]
                 .sum().reset_index())
    grouped["fga"] = grouped[["is_rim", "is_mid", "is_three"]].sum(axis=1)
    grouped = grouped[grouped["fga"] > 0].copy()

    enough = grouped["fga"] >= MIN_TRACKED_FGA
    for bucket, rate in (("is_rim", "rim_rate"), ("is_mid", "mid_rate"),
                         ("is_three", "three_rate")):
        grouped[rate] = np.where(
            enough, (100 * grouped[bucket] / grouped["fga"]).round(1), np.nan)

    grouped["fga"] = grouped["fga"].astype(int)
    return (grouped[cols].sort_values("fga", ascending=False)
            .reset_index(drop=True))


def main():
    print(f"Building shot-diet tables for {SEASON} …")
    shots = _classified_shots()
    if shots is None:
        return
    scopes = _game_scopes()
    print(f"  {len(shots):,} tracked FGA; scope game counts: "
          + ", ".join(f"{k or 'reg'}={len(v)}" for k, v in scopes.items()))
    for suffix, games in scopes.items():
        tbl = _scope_table(shots, games)
        path = os.path.join(PROJECT_ROOT, f"shot_diet_{SEASON}{suffix}.csv")
        tbl.to_csv(path, index=False)
        print(f"  {os.path.basename(path)} — {len(tbl)} players")


if __name__ == "__main__":
    main()
