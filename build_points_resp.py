"""
build_points_resp.py — per-player "points responsible" tables for the site.

Points Responsible = of the points a team scores WHILE A PLAYER IS ON THE FLOOR,
the share he personally SCORED (every made FG + FT) or ASSISTED (the FG value of
baskets he set up).  A player can't assist his own make, so scored and assisted
never double-count for one player; they DO overlap across players, so a team's
column sums past 100%.  Blind spot: a pass that only draws a shooting foul earns
no assist (no made-FG row to attach it to).

Numerator comes from the full play-by-play (scoring_play rows: athlete_id_1 =
scorer, athlete_id_2 = assister, score_value = points).  Denominator = the team's
points scored while the player was on the floor (pts_for summed over his on-court
stints in presence_full.parquet).  Numerator and denominator are matched by
season type per scope so they cover the same games.

This is a REGULAR-SEASON metric: presence_full.parquet (the stint/on-court data
that supplies the denominator) is regular-season only, so there is no postseason
on-court-points denominator to compute an all-games or postseason version from.
Two scopes match the player-stats scope masks exactly (game sets from the
schedule): regular season and conference (regular league games only).

Output (project root, consumed by build_site.py and build_scout.py):
    points_resp_{SEASON}.csv          — regular season   (season_type 2, no conf tourney)
    points_resp_conf_{SEASON}.csv     — conference        (regular league games)
    points_resp_nonconf_{SEASON}.csv  — non-conference    (regular games, not league)
    each: athlete_id, team_id, pts_scored, pts_ast, on_court_pts,
          scored_pct, assist_pct, resp_pct

Run:  python build_points_resp.py            (current season)
      OVERRIDE_SEASON=2025 python build_points_resp.py
"""

import os
from datetime import date as _date

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

from hoglib.season import detect_season
SEASON = detect_season()

def _game_sets():
    """Regular-season and conference game-id sets, mirroring build_player_stats.py.

    reg  = season_type 2 games that are NOT conference tournaments
    conf = reg games that are conference (league) games (conference_competition)
    Conference tournaments are ESPN-tagged season_type 2 but belong to postseason,
    so they're excluded from both — matching the player-stats scope masks.
    """
    import sportsdataverse.mbb as mbb
    sched = mbb.load_mbb_schedule(seasons=SEASON, return_as_pandas=True)
    conf_comp = sched[sched["conference_competition"] == True]
    conf_comp_ids = set(conf_comp["game_id"].astype(int).unique())

    notes = sched.get("notes_headline")
    if notes is not None:
        is_conf_tourney = (
            (sched["conference_competition"] == True)
            & notes.astype(str).str.contains("Tournament|Championship|Playoffs",
                                             case=False, na=False))
        conf_tourney_ids = set(sched.loc[is_conf_tourney, "game_id"].astype(int).unique())
    else:
        conf_tourney_ids = set()

    reg_type2 = set(sched.loc[sched["season_type"] == 2, "game_id"].astype(int).unique())
    reg_games = reg_type2 - conf_tourney_ids
    conf_games = (conf_comp_ids - conf_tourney_ids) & reg_games
    nonconf_games = reg_games - conf_games
    return reg_games, conf_games, nonconf_games


def _scoring_frame():
    """Full-season scoring plays: team_id, scorer, assister, points, type, game."""
    import sportsdataverse.mbb as mbb
    pbp = mbb.load_mbb_pbp(seasons=[SEASON], return_as_pandas=True)
    keep = ["game_id", "team_id", "athlete_id_1", "athlete_id_2",
            "scoring_play", "score_value", "season_type"]
    sc = pbp[pbp["scoring_play"] == True][keep].copy()
    for col in ("team_id", "athlete_id_1", "athlete_id_2", "game_id", "season_type"):
        sc[col] = pd.to_numeric(sc[col], errors="coerce").astype("Int64")
    sc["score_value"] = pd.to_numeric(sc["score_value"], errors="coerce").fillna(0)
    return sc


def _presence():
    """On-court stint rows with team points (pts_for repeated per on-court player)."""
    pres = pd.read_parquet(
        os.path.join(PROJECT_ROOT, "presence_full.parquet"),
        columns=["game_id", "team_id", "athlete_id", "is_on_court", "pts_for"])
    pres = pres[pres["is_on_court"] == True].copy()
    for col in ("game_id", "team_id", "athlete_id"):
        pres[col] = pd.to_numeric(pres[col], errors="coerce").astype("Int64")
    return pres


def _scope_table(sc, pres, games):
    """One scope's per-(team, player) points-responsible table for a game-id set."""
    if not games:
        return pd.DataFrame(columns=["athlete_id", "team_id", "pts_scored",
                                     "pts_ast", "on_court_pts", "scored_pct",
                                     "assist_pct", "resp_pct"])

    s = sc[sc["game_id"].isin(games)]
    scored = (s.dropna(subset=["athlete_id_1"])
                .groupby(["team_id", "athlete_id_1"])["score_value"].sum()
                .rename("pts_scored").reset_index()
                .rename(columns={"athlete_id_1": "athlete_id"}))
    assisted = (s.dropna(subset=["athlete_id_2"])
                  .groupby(["team_id", "athlete_id_2"])["score_value"].sum()
                  .rename("pts_ast").reset_index()
                  .rename(columns={"athlete_id_2": "athlete_id"}))

    p = pres[pres["game_id"].isin(games)]
    on = (p.groupby(["team_id", "athlete_id"])["pts_for"].sum()
            .rename("on_court_pts").reset_index())

    out = (on.merge(scored, on=["team_id", "athlete_id"], how="left")
             .merge(assisted, on=["team_id", "athlete_id"], how="left"))
    out[["pts_scored", "pts_ast"]] = out[["pts_scored", "pts_ast"]].fillna(0)
    out = out[out["on_court_pts"] > 0].copy()
    resp = out["pts_scored"] + out["pts_ast"]
    out["scored_pct"] = (out["pts_scored"] / out["on_court_pts"]).round(4)
    out["assist_pct"] = (out["pts_ast"] / out["on_court_pts"]).round(4)
    out["resp_pct"]   = (resp / out["on_court_pts"]).round(4)
    for col in ("pts_scored", "pts_ast", "on_court_pts"):
        out[col] = out[col].round().astype(int)
    return out.sort_values("resp_pct", ascending=False).reset_index(drop=True)


def main():
    print(f"Building points-responsible tables for {SEASON} …")
    reg_games, conf_games, nonconf_games = _game_sets()
    print(f"  {len(reg_games):,} regular, {len(conf_games):,} conference, "
          f"{len(nonconf_games):,} non-conference game IDs")
    sc = _scoring_frame()
    pres = _presence()
    for suffix, games in (("", reg_games), ("_conf", conf_games), ("_nonconf", nonconf_games)):
        tbl = _scope_table(sc, pres, games)
        path = os.path.join(PROJECT_ROOT, f"points_resp{suffix}_{SEASON}.csv")
        tbl.to_csv(path, index=False)
        print(f"  {os.path.basename(path)} — {len(tbl)} players")


if __name__ == "__main__":
    main()
