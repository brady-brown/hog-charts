"""
build_scout.py — Coaches Suite scouting data for the site.

Turns the scout_lib engine (coaches_suite/scout_lib.py) into per-team JSON that
the browser renders on the Coaches Suite scout page.  Current-season only, like
the shot charts — it reads shots_{SEASON}.parquet and the on/off + PBP files at
the project root, all of which the nightly CI already produces.

Output (site/data/{SEASON}/scout/):
    scout-index.json         list of {tid, name, slug} for the team dropdown
    {slug}.json              one full scouting report per D-I team

Run:  python build_scout.py            (current season)
      OVERRIDE_SEASON=2025 python build_scout.py
"""

import os
import re
import sys
import json
from datetime import date as _date

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "coaches_suite"))
import scout_lib as SL  # noqa: E402


# ── season detection (matches build_site.py) ─────────────────────────────────
from hoglib.season import detect_season
SEASON = detect_season()

OUT_DIR = os.path.join(PROJECT_ROOT, "site", "data", str(SEASON), "scout")
os.makedirs(OUT_DIR, exist_ok=True)


# ── JSON helpers (copied from build_site.py conventions) ─────────────────────
# NaN-safe JSON sanitizing + slugify now live in hoglib.io (shared, must not drift).
from hoglib.io import sanitize_for_json, slugify


def write_json(data_object, output_filename, output_directory=OUT_DIR):
    path = os.path.join(output_directory, output_filename)
    with open(path, "w") as fh:
        json.dump(sanitize_for_json(data_object), fh,
                  separators=(",", ":"), allow_nan=False)


def _records(df):
    """DataFrame -> list of dict rows (empty list if None/empty)."""
    if df is None or getattr(df, "empty", True):
        return []
    return df.to_dict("records")


def load_jersey_map(season):
    """athlete_id (int) -> jersey number (str), from the player boxscore.

    ~99% coverage and season-accurate, unlike the ESPN roster endpoint that
    build_site.py's bios use (that returns the CURRENT roster, so it drops
    players who left after the season — leaving jn=None for them in player-stats).
    """
    try:
        import sportsdataverse.mbb as mbb
        pbox = mbb.load_mbb_player_boxscore(seasons=[season]).to_pandas()
        j = pbox.dropna(subset=["athlete_id", "athlete_jersey"]).copy()
        j["athlete_id"] = j["athlete_id"].astype(float).astype("int64")
        j["athlete_jersey"] = j["athlete_jersey"].astype(str)
        # most-frequent jersey per player (guards the odd mid-season change)
        return (j.groupby("athlete_id")["athlete_jersey"]
                  .agg(lambda s: s.mode().iloc[0]).to_dict())
    except Exception as exc:
        print(f"  [warn] jersey map unavailable: {exc}")
        return {}


def _freshness(season):
    p = os.path.join(PROJECT_ROOT, f"shots_{season}.parquet")
    try:
        d = pd.to_datetime(pd.read_parquet(p, columns=["game_date"])["game_date"],
                           errors="coerce")
        if d.notna().any():
            return d.max().date().isoformat()
    except Exception:
        pass
    return f"{season}-04-30"


# ── one team's full scouting report ──────────────────────────────────────────
def build_team_report(data, tid, name, baselines, jmap):
    off_shots = SL.team_shots(data, tid)
    def_shots = SL.opp_shots(data, tid)

    # 3PT board keys jerseys off player-stats, which drops deep-bench players;
    # backfill those from the fuller boxscore jersey map.
    threat = _records(SL.three_point_threat(data, tid))
    for row in threat:
        if pd.isna(row.get("jn")) and not pd.isna(row.get("athlete_id")):
            row["jn"] = jmap.get(int(row["athlete_id"]))

    # per-scorer directional + macro quality (coords stay in shots/{slug}.json)
    scorers = SL.top_scorers(data, tid, name)
    scorer_rows = []
    for _, s in scorers.iterrows():
        aid = s["id"]
        ps = SL.player_shots(data, aid)
        scorer_rows.append({
            **{k: s.get(k) for k in ("id", "n", "jn", "pos", "ppg", "usg")},
            "n_fga": int(len(ps)),
            "directional": _records(SL.directional_split(ps)),
            "quality": _records(SL.macro_quality(ps, baselines=baselines)),
        })

    form = SL.form_summary(off_shots, data, root=PROJECT_ROOT)

    return {
        "team": name,
        "tid": int(tid),
        "season": SEASON,
        "generated": _freshness(SEASON),
        "three_point_threat": threat,
        "offense": {
            "macro_freq": _records(SL.macro_frequency(off_shots, data)),
            "macro_qual": _records(SL.macro_quality(off_shots, baselines=baselines)),
            "zone_freq": _records(SL.zone_frequency(off_shots, data)["zone"]),
            "zone_qual": _records(SL.zone_quality(off_shots, baselines=baselines)),
            "n_fga": int(len(off_shots)),
        },
        "defense": {
            "macro_freq": _records(SL.macro_frequency(def_shots, data)),
            "macro_qual": _records(SL.macro_quality(def_shots, baselines=baselines)),
            "zone_qual": _records(SL.zone_quality(def_shots, baselines=baselines)),
            "n_fga": int(len(def_shots)),
        },
        "attack_board": _records(SL.attack_board(data, tid, name)),
        "usage_board": _records(SL.usage_board(data, tid, name)),
        "points_responsible": _records(SL.points_responsible_board(data, tid, name)),
        "foul_board": _records(SL.foul_board(data, tid, name)),
        "rebound_board": _records(SL.rebound_board(data, tid, name)),
        "top_scorers": scorer_rows,
        "form": {"table": _records(form["table"]),
                 "n_recent_games": form["n_recent_games"]},
    }


def main():
    print(f"Loading scout data for {SEASON} ...")
    data = SL.load_data(SEASON, root=PROJECT_ROOT)

    # Precompute the league 3PT pools ONCE — three_point_threat recomputes them
    # on every call otherwise (a full groupby over all D-I shots per team).
    _pools = SL.league_three_pools(data["shots"])
    SL.league_three_pools = lambda shots, min_fga=50: _pools

    # Overlay reliable jerseys from the player boxscore onto player-stats (the
    # roster-endpoint bios miss departed players). Every board reads jn from here.
    jmap = load_jersey_map(SEASON)
    ps_all = data["pstats"]
    if jmap and "id" in ps_all.columns:
        from_box = ps_all["id"].astype("Int64").map(jmap)
        base = ps_all["jn"] if "jn" in ps_all.columns else None
        ps_all["jn"] = from_box.where(from_box.notna(), base)
        print(f"  jerseys filled from boxscore: "
              f"{int(from_box.notna().sum())}/{len(ps_all)} players")

    baselines = SL._load_baselines(
        os.path.join(PROJECT_ROOT, "site", "data", "zone-baselines.json"))

    # D-I teams = those present in player-stats (367); tid from the same table.
    ps = data["pstats"]
    teams = (ps.dropna(subset=["t", "tid"])
               .drop_duplicates("tid")[["tid", "t"]]
               .sort_values("t"))

    index = []
    for _, row in teams.iterrows():
        tid, name = int(row["tid"]), row["t"]
        slug = slugify(name)
        try:
            report = build_team_report(data, tid, name, baselines, jmap)
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")
            continue
        write_json(report, f"{slug}.json")
        index.append({"tid": tid, "name": name, "slug": slug})

    write_json(index, "scout-index.json")
    print(f"Wrote {len(index)} team reports -> {OUT_DIR}")


if __name__ == "__main__":
    main()
