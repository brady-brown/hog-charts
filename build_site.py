"""
build_site.py — Export artifacts to static JSON files for Hog Charts.

Writes separate site/data/*.json files consumed by each page via fetch().
Designed to run nightly via GitHub Actions.

Usage:
    python build_site.py
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as _date

import numpy as np
import pandas as pd
import requests

_override = os.environ.get("OVERRIDE_SEASON")
if _override:
    SEASON = int(_override)
else:
    _today  = _date.today()
    SEASON  = _today.year + 1 if _today.month >= 11 else _today.year

BASE      = os.path.dirname(os.path.abspath(__file__))
ART       = os.path.join(BASE, "artifacts", str(SEASON))
OUT       = os.path.join(BASE, "site", "data", str(SEASON))
SHOTS_DIR = os.path.join(BASE, "site", "data", "shots")  # current season only, no subdir

os.makedirs(OUT, exist_ok=True)
os.makedirs(SHOTS_DIR, exist_ok=True)

ZONES = [
    "At Rim", "Paint (Non-Rim)",
    "Left Baseline Mid", "Right Baseline Mid",
    "Left Mid-Range", "Center Mid-Range", "Right Mid-Range",
    "Left Corner 3PT", "Left Wing 3PT", "Center 3PT",
    "Right Wing 3PT", "Right Corner 3PT",
]
ZONE_IDX    = {z: i for i, z in enumerate(ZONES)}
THREE_ZONES = {"Left Corner 3PT", "Left Wing 3PT", "Center 3PT",
               "Right Wing 3PT", "Right Corner 3PT"}


# ── helpers ───────────────────────────────────────────────────────────────────
def _clean(obj):
    if isinstance(obj, float):
        return None if (obj != obj or obj == float("inf") or obj == float("-inf")) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj

ROOT_DATA = os.path.join(BASE, "site", "data")

def write_json(data, filename, out_dir=None):
    path = os.path.join(out_dir or OUT, filename)
    with open(path, "w") as f:
        f.write(json.dumps(_clean(data), separators=(",", ":")))
    kb = os.path.getsize(path) / 1024
    print(f"  {filename:<35s} {kb:7.0f} KB")

def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def classify_zones(df):
    valid = df["coordinate_x"].notna() & df["coordinate_y"].notna()
    zone  = pd.Series("Unknown", index=df.index, dtype="object")
    x = df.loc[valid, "coordinate_x"].abs()
    y = df.loc[valid, "coordinate_y"]
    xs    = 41.75 - x
    dist  = np.sqrt(xs**2 + y**2)
    angle = np.degrees(np.arctan2(y, xs))
    heave = dist >= 40
    rim   = ~heave & (dist < 5)
    paint = ~heave & ~rim & (x >= 28) & (y.abs() <= 6)
    is3   = ~heave & ~rim & ~paint & (dist >= 22.15)
    mid   = ~heave & ~rim & ~paint & ~is3
    zone[valid & heave] = "Heave"
    zone[valid & rim]   = "At Rim"
    zone[valid & paint] = "Paint (Non-Rim)"
    zone[valid & is3 & (angle >  55)]                 = "Left Corner 3PT"
    zone[valid & is3 & (angle >  25) & (angle <=  55)]= "Left Wing 3PT"
    zone[valid & is3 & (angle > -25) & (angle <=  25)]= "Center 3PT"
    zone[valid & is3 & (angle > -55) & (angle <= -25)]= "Right Wing 3PT"
    zone[valid & is3 & (angle <= -55)]                = "Right Corner 3PT"
    zone[valid & mid & (angle >  60)]                 = "Left Baseline Mid"
    zone[valid & mid & (angle >  25) & (angle <=  60)]= "Left Mid-Range"
    zone[valid & mid & (angle > -25) & (angle <=  25)]= "Center Mid-Range"
    zone[valid & mid & (angle > -60) & (angle <= -25)]= "Right Mid-Range"
    zone[valid & mid & (angle <= -60)]                = "Right Baseline Mid"
    return zone

def zone_records(group):
    stats = (group.groupby("zone")
                  .agg(makes=("scoring_play", "sum"), attempts=("scoring_play", "count"))
                  .reset_index())
    return [[ZONE_IDX[r["zone"]], int(r["makes"]), int(r["attempts"])]
            for _, r in stats.iterrows() if r["zone"] in ZONE_IDX and int(r["attempts"]) > 0]


# ── load core artifacts ───────────────────────────────────────────────────────
print("\nLoading artifacts…")
with open(os.path.join(ART, "metadata.json")) as f:
    meta = json.load(f)
with open(os.path.join(ART, "model.json")) as f:
    model = json.load(f)

teams_df = pd.read_parquet(os.path.join(ART, "teams.parquet"))
nr_raw   = pd.read_parquet(os.path.join(ART, "net_ratings.parquet"))


# ── conference lookup ─────────────────────────────────────────────────────────
ps_path = os.path.join(BASE, f"player_stats_{SEASON}.csv")
if os.path.exists(ps_path):
    _ps_conf = pd.read_csv(ps_path, usecols=["team_display_name", "conf."])
    conf_map = (_ps_conf.dropna()
                        .drop_duplicates("team_display_name")
                        .set_index("team_display_name")["conf."]
                        .to_dict())
else:
    conf_map = {}
    print(f"  [warn] player_stats_{SEASON}.csv not found — conf will be empty")


# ══════════════════════════════════════════════════════════════════════════════
# 1. predictor.json
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding predictor.json…")
teams_df["conf"] = teams_df["team"].map(conf_map).fillna("")
rank_cols = [c for c in ["team", "rank", "off_rank", "def_rank"] if c in nr_raw.columns]
teams_df  = teams_df.merge(nr_raw[rank_cols], on="team", how="left")
for col in ["net_eff", "off_eff", "def_eff", "form"]:
    if col in teams_df.columns: teams_df[col] = teams_df[col].round(2)
for col in ["pace", "home_adv"]:
    if col in teams_df.columns: teams_df[col] = teams_df[col].round(1)
want = ["team", "team_id", "conf", "net_eff", "off_eff", "def_eff",
        "pace", "home_adv", "form", "rank", "off_rank", "def_rank"]
teams_records = (teams_df[[c for c in want if c in teams_df.columns]]
                 .sort_values("team")
                 .where(lambda df: ~df.isin([float("nan")]), other=None)
                 .to_dict("records"))
write_json({"model": model, "teams": teams_records, "meta": meta}, "predictor.json")


# ══════════════════════════════════════════════════════════════════════════════
# 2. net-ratings.json
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding net-ratings.json…")
nr = nr_raw.copy()
nr["conf"] = nr["team"].map(conf_map).fillna("")
if "wins" in nr.columns and "losses" in nr.columns:
    nr["record"] = (nr["wins"].astype(int).astype(str)
                    + "–" + nr["losses"].astype(int).astype(str))
want_nr = ["rank", "team", "team_id", "conf", "record", "games",
           "net_eff", "off_eff", "def_eff", "off_rank", "def_rank",
           "sos", "pace", "home_court", "form"]
nr_records = nr[[c for c in want_nr if c in nr.columns]].to_dict("records")
write_json({"net_ratings": nr_records, "meta": meta}, "net-ratings.json")


# ══════════════════════════════════════════════════════════════════════════════
# 3a. Fetch player bios from ESPN roster endpoints (height/weight/class/hometown)
# ══════════════════════════════════════════════════════════════════════════════
_ROSTER_URL = ("https://site.api.espn.com/apis/site/v2/sports/basketball"
               "/mens-college-basketball/teams/{tid}/roster")

def _fetch_team_bios(team_id):
    try:
        r = requests.get(_ROSTER_URL.format(tid=int(team_id)), timeout=10)
        if r.status_code != 200:
            return {}
        athletes = r.json().get("athletes", [])
        out = {}
        for a in athletes:
            aid = str(a.get("id", ""))
            if not aid:
                continue
            bp = a.get("birthPlace") or {}
            out[aid] = {
                "ht":  a.get("displayHeight"),
                "wt":  a.get("displayWeight"),
                "exp": (a.get("experience") or {}).get("displayValue"),
                "hw":  bp.get("displayText"),
                "jn":  a.get("jersey"),
            }
        return out
    except Exception:
        return {}

def build_player_bios(ps_csv_path):
    """Return dict {str(athlete_id): bio} by fetching ESPN rosters for all teams."""
    if not os.path.exists(ps_csv_path):
        return {}
    team_ids = pd.read_csv(ps_csv_path, usecols=["team_id"])["team_id"].dropna().unique()
    print(f"  Fetching ESPN rosters for {len(team_ids)} teams…")
    bios = {}
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(_fetch_team_bios, tid): tid for tid in team_ids}
        done = 0
        for fut in as_completed(futs):
            bios.update(fut.result())
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{len(team_ids)} teams fetched…")
    print(f"  → {len(bios)} player bios collected")
    return bios


# ══════════════════════════════════════════════════════════════════════════════
# 3. player-stats.json  +  player-stats-conf.json
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding player-stats JSON files…")

COL_MAP = {
    "athlete_id":            "id",
    "athlete_display_name":  "n",
    "team_id":               "tid",
    "team_display_name":     "t",
    "athlete_position_name": "pos",
    "conf.":                 "conf",
    "games_played":          "gp",
    "mpg":                   "mpg",
    "points_avg":            "ppg",
    "reb_avg":               "rpg",
    "ast_avg":               "apg",
    "steal_avg":             "spg",
    "blocks_avg":            "bpg",
    "to_avg":                "tpg",
    "fg_pct":                "fg",
    "3pt_pct":               "fg3",
    "ft_pct":                "ft",
    "efg_pct":               "efg",
    "ts_pct":                "ts",
    "3par":                  "par3",
    "ftr":                   "ftr",
    "usg":                   "usg",
    "on_off":                "on_off",
}

def build_player_json(csv_path, onoff_csv, bios=None, min_gp=8, min_mpg=8, onoff_min_poss=200, label=""):
    if not os.path.exists(csv_path):
        print(f"  [warn] {csv_path} not found — skipped")
        return None
    ps   = pd.read_csv(csv_path)
    fga  = ps["field_goals_attempted"]
    fg3a = ps["three_point_field_goals_attempted"]
    fta  = ps["free_throws_attempted"]
    tov  = ps["turnovers"]
    pts  = ps["points"]
    gp   = ps["games_played"]
    mins = ps["minutes"]
    ps["ts_pct"] = np.where(fga + 0.44*fta > 0, pts/(2*(fga + 0.44*fta)), None)
    ps["3par"]   = np.where(fga > 0, fg3a/fga, None)
    ps["ftr"]    = np.where(fga > 0, fta/fga,  None)
    ps["mpg"]    = np.where(gp  > 0, mins/gp,  None)
    # Usage%
    team_tots = (ps.groupby("team_display_name")
                   .agg(tm_fga=("field_goals_attempted", "sum"),
                        tm_fta=("free_throws_attempted",  "sum"),
                        tm_tov=("turnovers",              "sum"),
                        tm_min=("minutes",                "sum"))
                   .reset_index())
    ps = ps.merge(team_tots, on="team_display_name", how="left")
    team_poss   = ps["tm_fga"] + 0.44*ps["tm_fta"] + ps["tm_tov"]
    player_poss = fga + 0.44*fta + tov
    ps["usg"] = np.where((mins > 0) & (team_poss > 0),
                         100 * player_poss * (ps["tm_min"]/5) / (mins * team_poss), None)
    # On/off
    if onoff_csv and os.path.exists(onoff_csv):
        oo = pd.read_csv(onoff_csv, usecols=["athlete_id","on_off","poss_off_on"])
        oo = oo[oo["poss_off_on"] >= onoff_min_poss][["athlete_id","on_off"]]
        ps = ps.merge(oo, on="athlete_id", how="left")
    else:
        ps["on_off"] = None
    # Filter
    ps = ps[(ps["games_played"] >= min_gp) & (ps["mpg"].fillna(0) >= min_mpg)].copy()
    # Round
    for c in ["ts_pct","3par","ftr","efg_pct","fg_pct","3pt_pct","ft_pct"]:
        if c in ps.columns: ps[c] = ps[c].round(3)
    for c in ["mpg","points_avg","reb_avg","ast_avg","steal_avg","blocks_avg","to_avg","on_off","usg"]:
        if c in ps.columns: ps[c] = ps[c].round(1)
    keep   = [c for c in COL_MAP if c in ps.columns]
    ps_out = ps[keep].rename(columns=COL_MAP)
    records = ps_out.where(ps_out.notna(), other=None).to_dict("records")
    # Merge bio fields (height, weight, class year, hometown, jersey)
    if bios:
        for rec in records:
            bio = bios.get(str(int(rec["id"])) if rec.get("id") is not None else "", {})
            rec["ht"]  = bio.get("ht")
            rec["wt"]  = bio.get("wt")
            rec["exp"] = bio.get("exp")
            rec["hw"]  = bio.get("hw")
            rec["jn"]  = bio.get("jn")
    print(f"  → {len(records)} players {label}")
    return records

player_bios = build_player_bios(ps_path)

overall_records = build_player_json(
    ps_path,
    os.path.join(BASE, f"mbb_onoff_{SEASON}_v2.csv") if os.path.exists(os.path.join(BASE, f"mbb_onoff_{SEASON}_v2.csv")) else None,
    bios=player_bios, min_gp=8, min_mpg=8, onoff_min_poss=200, label="(overall)"
)
if overall_records is not None:
    write_json({"players": overall_records, "meta": meta}, "player-stats.json")

ps_conf_path = os.path.join(BASE, f"player_stats_conf_{SEASON}.csv")
conf_records = build_player_json(
    ps_conf_path,
    os.path.join(BASE, f"mbb_onoff_{SEASON}_conf_v2.csv") if os.path.exists(os.path.join(BASE, f"mbb_onoff_{SEASON}_conf_v2.csv")) else None,
    bios=player_bios, min_gp=4, min_mpg=8, onoff_min_poss=100, label="(conference)"
)
if conf_records is not None:
    write_json({"players": conf_records, "meta": meta}, "player-stats-conf.json")


# ══════════════════════════════════════════════════════════════════════════════
# 4. lineup-stats.json
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding lineup-stats.json + lineup-conf-stats.json…")
LINEUP_KEEP = ["Combo", "Mins", "Avg_Poss", "NetRtg", "ORtg", "DRtg",
               "AST_100", "TOV_100", "REB_100", "STL_100", "BLK_100"]
LINEUP_MIN  = {"1": 100, "2": 75, "3": 75, "5": 25}
LINEUP_MIN_CONF = {"1": 50, "2": 40, "3": 40, "5": 15}
ROUND_COLS  = ["Mins", "Avg_Poss", "NetRtg", "ORtg", "DRtg",
               "AST_100", "TOV_100", "REB_100", "STL_100", "BLK_100"]

def load_lineup_file(size, variant, min_poss):
    suffix = "overall" if variant == "all" else "conference"
    fname  = os.path.join(BASE, f"{size}_man_{suffix}_stats_{SEASON}.csv")
    if not os.path.exists(fname):
        return {}, set()
    df   = pd.read_csv(fname)
    df   = df[df["Avg_Poss"] >= min_poss].copy()
    keep = [c for c in LINEUP_KEEP if c in df.columns]
    rc   = [c for c in ROUND_COLS  if c in df.columns]
    df[rc] = df[rc].round(1)
    by_team = {}
    teams   = set()
    for team, grp in df.groupby("Team"):
        teams.add(team)
        by_team[str(team)] = grp[keep].sort_values("NetRtg", ascending=False).to_dict("records")
    return by_team, teams

all_teams = set()
overall_by_size = {}
conf_by_size    = {}
for size in ["1", "2", "3", "5"]:
    by_team, teams = load_lineup_file(size, "all",  LINEUP_MIN[size])
    overall_by_size[size] = by_team
    all_teams |= teams
    total = sum(len(v) for v in by_team.values())
    print(f"  {size}-man overall:     {total:5d} combos, {len(by_team)} teams")

    by_team_c, _ = load_lineup_file(size, "conf", LINEUP_MIN_CONF[size])
    conf_by_size[size] = by_team_c
    total_c = sum(len(v) for v in by_team_c.values())
    print(f"  {size}-man conference:  {total_c:5d} combos, {len(by_team_c)} teams")

# Write per-team files so the browser only downloads one team at a time (~20KB vs 10MB)
LINEUPS_DIR = os.path.join(OUT, "lineups")
os.makedirs(LINEUPS_DIR, exist_ok=True)
team_slugs = {}
for team in sorted(all_teams):
    slug = slugify(team)
    team_slugs[team] = slug
    team_data = {
        "team": team,
        "overall": {s: overall_by_size[s].get(team, []) for s in ["1","2","3","5"]},
        "conf":    {s: conf_by_size[s].get(team, [])    for s in ["1","2","3","5"]},
    }
    path = os.path.join(LINEUPS_DIR, f"{slug}.json")
    with open(path, "w") as f:
        f.write(json.dumps(_clean(team_data), separators=(",", ":")))

print(f"  lineup files:          {len(team_slugs):5d} teams → lineups/{{slug}}.json")

# Small index file: just team names + slug map so the page can build the dropdown
write_json({"teams": sorted(all_teams), "slugs": team_slugs, "meta": meta},
           "lineup-index.json")


# ══════════════════════════════════════════════════════════════════════════════
# 5. shots-meta.json  +  shots/{slug}.json
# ══════════════════════════════════════════════════════════════════════════════
shots_path = os.path.join(BASE, f"shots_{SEASON}.parquet")
box_path   = os.path.join(BASE, f"box_{SEASON}.parquet")

if os.path.exists(shots_path) and os.path.exists(box_path):
    print("\nBuilding shot data…")
    pbp = pd.read_parquet(shots_path)
    box = pd.read_parquet(box_path)

    player_meta     = (box[["athlete_id", "athlete_display_name", "team_id", "team_display_name"]]
                       .drop_duplicates("athlete_id").copy())
    team_id_to_name = (box[["team_id", "team_display_name"]]
                       .drop_duplicates()
                       .set_index("team_id")["team_display_name"]
                       .to_dict())

    pbp = pbp.merge(player_meta[["athlete_id", "athlete_display_name"]],
                    left_on="athlete_id_1", right_on="athlete_id", how="left")

    is_ft = (pbp["type_text"].str.contains("Free", case=False, na=False)
             | pbp["text"].str.contains("free throw", case=False, na=False))
    fgs   = pbp[~is_ft].copy()
    fgs   = fgs.dropna(subset=["coordinate_x", "coordinate_y"])
    fgs   = fgs[(fgs["coordinate_x"].abs() <= 50) & (fgs["coordinate_y"].abs() <= 30)]

    print("  Classifying zones…")
    fgs["zone"] = classify_zones(fgs)
    fgs = fgs[~fgs["zone"].isin(["Heave", "Unknown"])].copy()

    # ── team zones ───────────────────────────────────────────────────────────
    print("  Team zone stats…")
    team_zones = {}
    for team_id, grp in fgs.groupby("team_id"):
        name = team_id_to_name.get(team_id)
        if not name: continue
        z = zone_records(grp)
        if z:
            team_zones[name] = {"id": int(team_id), "slug": slugify(name), "z": z}

    # ── player zones (min 30 FGA) ─────────────────────────────────────────────
    print("  Player zone stats…")
    player_zones = {}
    for (team_id, player_name), grp in fgs.dropna(subset=["athlete_display_name"]).groupby(
            ["team_id", "athlete_display_name"]):
        if len(grp) < 30: continue
        team_name = team_id_to_name.get(team_id, "")
        z = zone_records(grp)
        if z:
            player_zones[f"{player_name}|{team_name}"] = {"n": player_name, "t": team_name, "z": z}

    # ── territory ─────────────────────────────────────────────────────────────
    print("  Territory maps…")
    fgs["pts"] = (fgs["scoring_play"]
                  .map(lambda s: 1 if s else 0)
                  * fgs["zone"].map(lambda z: 3 if z in THREE_ZONES else 2))
    territory = {}
    for team_id, grp in fgs.groupby("team_id"):
        name = team_id_to_name.get(team_id)
        if not name: continue
        leaders = []
        for zone_name, zgrp in grp.groupby("zone"):
            if zone_name not in ZONE_IDX: continue
            named = zgrp.dropna(subset=["athlete_display_name"])
            if named.empty: continue
            by_p  = named.groupby("athlete_display_name")["pts"].sum()
            top   = by_p.idxmax()
            if int(by_p.max()) > 0:
                leaders.append({"z": ZONE_IDX[zone_name], "n": top, "pts": int(by_p.max())})
        if leaders:
            territory[name] = leaders

    # ── per-team raw shots → individual files ─────────────────────────────────
    print("  Raw shot coordinates → shots/{slug}.json…")
    sched_path = os.path.join(BASE, "game_schedule.parquet")
    if os.path.exists(sched_path):
        sched = pd.read_parquet(sched_path)
        name_to_tid = {v: k for k, v in team_id_to_name.items()}
        sched2 = sched.copy()
        sched2["team_id"] = sched2["team"].map(name_to_tid)
        sched2 = sched2.dropna(subset=["team_id"])
        sched2["team_id"] = sched2["team_id"].astype(int)
        sched2 = sched2.sort_values(["team_id", "date"]).reset_index(drop=True)
        sched2["local_gi"] = sched2.groupby("team_id").cumcount()

        gi_lookup = sched2.set_index(["team_id", "game_id"])["local_gi"]
        fgs_idx   = pd.MultiIndex.from_arrays(
            [fgs["team_id"].astype(int), fgs["game_id"].astype(int)])
        fgs = fgs.copy()
        fgs["gi"]   = gi_lookup.reindex(fgs_idx).fillna(-1).astype(int).values
        fgs["px_i"] = (-fgs["coordinate_y"] * 10).round().astype(int)
        fgs["py_i"] = ((47 - fgs["coordinate_x"].abs()) * 10).round().astype(int)
        fgs["sc"]   = fgs["scoring_play"].astype(int)

        pi_parts = []
        for _, grp in fgs.groupby("team_id"):
            players = sorted(grp["athlete_display_name"].dropna().unique())
            p_map   = pd.Series(range(len(players)), index=players)
            pi_parts.append(grp["athlete_display_name"].map(p_map).fillna(-1).astype(int))
        fgs["pi"] = pd.concat(pi_parts).reindex(fgs.index).fillna(-1).astype(int)

        n_written = 0
        for team_id_val, grp in fgs.groupby("team_id"):
            tid   = int(team_id_val)
            tname = team_id_to_name.get(tid)
            if not tname: continue
            slug     = slugify(tname)
            t_sched  = sched2[sched2["team_id"] == tid].sort_values("date")
            games    = [{"id": int(r.game_id), "opp": r.opponent,
                         "date": str(r.date)[:10], "label": r.label}
                        for r in t_sched.itertuples()]
            if tname in team_zones:
                team_zones[tname]["gp"] = int(grp["gi"].nunique())
            players  = sorted(grp["athlete_display_name"].dropna().unique().tolist())
            arr      = grp[["gi", "px_i", "py_i", "sc", "pi"]].values
            shot_path = os.path.join(SHOTS_DIR, f"{slug}.json")
            with open(shot_path, "w") as f:
                f.write(json.dumps(_clean({"games": games, "players": players,
                                           "shots": arr.ravel().tolist()}),
                                   separators=(",", ":")))
            n_written += 1
        print(f"  → {n_written} team shot files")
    else:
        print("  [warn] game_schedule.parquet not found — per-game shots skipped")

    write_json({
        "zones":        ZONES,
        "team_zones":   team_zones,
        "player_zones": player_zones,
        "territory":    territory,
        "meta":         meta,
    }, "shots-meta.json", out_dir=ROOT_DATA)

else:
    print(f"\n  [warn] shots_{SEASON}.parquet not found — shot data skipped")

# ── seasons.json — list of built seasons for the UI dropdown ─────────────────
_built = sorted(
    [int(d) for d in os.listdir(ROOT_DATA)
     if d.isdigit() and os.path.isdir(os.path.join(ROOT_DATA, d))],
    reverse=True
)
write_json(_built, "seasons.json", out_dir=ROOT_DATA)
print(f"\nAvailable seasons: {_built}")


# ── summary ───────────────────────────────────────────────────────────────────
print("\n  Done. Serve site/ with: python -m http.server 8000 --directory site")
