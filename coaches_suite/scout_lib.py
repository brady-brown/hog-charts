"""
scout_lib.py — Coaches Suite scouting engine (shared by scout.ipynb).

Pure functions that turn the raw Hog Charts data files into the four scouting
views for a single opponent:

    1. Three-Point Threat Board  — who is a real threat from deep, in coach language
    2. Shot Diet                 — where the opponent likes to shoot (vs. D-I average)
    3. Shot Quality              — where the opponent actually shoots best (vs. NCAA baseline)
    4. Who To Attack             — the defensive weak links (drapm + on/off)

Everything is parametric on (team_name, season).  Nothing here prints or plots;
the notebook owns presentation.  Zone math is copied verbatim from
build_site.classify_shot_zones so the zones match the rest of the site.
"""

import json
import os

import numpy as np
import pandas as pd

# ── zone system (shared with build_site / build_shot_diet via hoglib.zones) ──
from hoglib.zones import ZONE_NAMES, classify_shot_zones
# Shot-diet families (coarser than the 14 wedges) and their member zones.
FAMILY_OF = {z: ("Rim" if z == "Restricted Area"
                 else "Close" if z.startswith("Close Mid")
                 else "Three" if z.startswith("3PT")
                 else "Mid") for z in ZONE_NAMES}
FAMILY_ORDER = ["Rim", "Close", "Mid", "Three"]
# Coarse 3-band grouping (paint = restricted area + close mid).
MACRO_OF = {z: ("Three" if z.startswith("3PT")
               else "Mid-Range" if z.startswith("Mid")
               else "Paint / Restricted") for z in ZONE_NAMES}
MACRO_ORDER = ["Paint / Restricted", "Mid-Range", "Three"]

# Field-goal shot types (everything else — free throws — is dropped).
FG_TYPES = {"JumpShot", "LayUpShot", "DunkShot", "TipShot", "Shot"}

LEAGUE_3P_PCT = 0.345    # D-I average, matches avg of the three-zone baselines


# ── data loading ────────────────────────────────────────────────────────────
def _find_rows(obj):
    """player-stats/impact JSON may be a dict wrapping the list — dig out the list."""
    if isinstance(obj, list):
        return obj
    for v in obj.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    raise ValueError("could not locate row list in JSON")


def load_data(season, root="."):
    """Load and pre-process every file the engine needs for a season.

    Returns a dict of frames.  `shots` is ALL D-I field-goal attempts with a
    zone / make / is_three / is_rim column already attached (league context), so
    both the team view and the league baselines come from one pass.
    """
    shots = pd.read_parquet(os.path.join(root, f"shots_{season}.parquet"))
    box = pd.read_parquet(os.path.join(root, f"box_{season}.parquet"))

    shots = shots[shots["type_text"].isin(FG_TYPES)].copy()
    shots["zone"] = classify_shot_zones(shots)
    shots = shots[~shots["zone"].isin(["Unknown", "Heave"])].copy()
    shots["make"] = shots["scoring_play"].astype(bool)
    shots["is_three"] = shots["zone"].str.startswith("3PT")
    shots["is_rim"] = shots["zone"] == "Restricted Area"
    shots["family"] = shots["zone"].map(FAMILY_OF)

    # athlete_id -> name / team (dedupe: keep the team a player shot most for)
    names = (box.dropna(subset=["athlete_id"])
                .drop_duplicates(subset=["athlete_id"])
                .set_index("athlete_id"))

    site = os.path.join(root, "site", "data", str(season))
    # Use the ALL-games player stats (regular + postseason) so every board's
    # PPG / USG% / rebound% etc. matches the Player Stats page, which defaults to
    # the "All games" scope. (player-stats.json is the regular-season-only scope.)
    _ps_all = os.path.join(site, "player-stats-all.json")
    _ps_path = _ps_all if os.path.exists(_ps_all) else os.path.join(site, "player-stats.json")
    pstats = pd.DataFrame(_find_rows(json.load(open(_ps_path))))
    impact = pd.DataFrame(_find_rows(json.load(open(os.path.join(site, "player-impact.json")))))
    # defensive on/off (drtg_on / drtg_off) — available early season, unlike RAPM
    onoff = pd.read_csv(os.path.join(root, f"mbb_onoff_{season}_v2.csv"),
                        usecols=["athlete_id", "team_id", "drtg_on", "drtg_off",
                                 "poss_off_on", "poss_def_on"])
    # personal fouls per player for the foul-trouble sheet. build_shots_data.py
    # emits fouls_{season}.csv from the standard PBP feed (so a CI scout build has
    # real foul data); fall back to a local offline_pbp.csv for ad-hoc runs, then
    # to an empty series. (See ARCHITECTURE_REVIEW.md G1.)
    fouls = pd.Series(dtype="int64", name="fouls")
    _fouls_csv = os.path.join(root, f"fouls_{season}.csv")
    try:
        if os.path.exists(_fouls_csv):
            fouls = pd.read_csv(_fouls_csv).set_index("athlete_id_1")["fouls"]
        else:
            pbp = pd.read_csv(os.path.join(root, "offline_pbp.csv"),
                              usecols=["type_text", "athlete_id_1", "season"])
            pf = pbp[(pbp["type_text"] == "PersonalFoul") & (pbp["season"] == season)]
            fouls = pf.groupby("athlete_id_1").size().rename("fouls")
    except Exception:
        fouls = pd.Series(dtype="int64", name="fouls")

    # points-responsible table (regular-season scored+assisted share of on-court
    # team points), precomputed by build_points_resp.py from the full PBP feed +
    # presence_full so the site and this engine report identical numbers.
    try:
        points_resp = pd.read_csv(os.path.join(root, f"points_resp_{season}.csv"))
        points_resp["athlete_id"] = points_resp["athlete_id"].astype("Int64")
        points_resp["team_id"] = points_resp["team_id"].astype("Int64")
    except Exception:
        points_resp = pd.DataFrame(columns=["athlete_id", "team_id", "pts_scored",
                                            "pts_ast", "on_court_pts", "scored_pct",
                                            "assist_pct", "resp_pct"])

    return {"shots": shots, "box": box, "names": names,
            "pstats": pstats, "impact": impact, "onoff": onoff,
            "fouls": fouls, "points_resp": points_resp}


def team_shots(data, tid):
    """This team's own field-goal attempts."""
    return data["shots"][data["shots"]["team_id"] == tid]


def opp_shots(data, tid):
    """Opponents' field-goal attempts IN games this team played (the defense they faced)."""
    shots = data["shots"]
    in_game = (shots["home_team_id"] == tid) | (shots["away_team_id"] == tid)
    return shots[in_game & (shots["team_id"] != tid)]


def resolve_team(data, team_name):
    """Return (team_id, canonical_display_name) for a fuzzy team name."""
    box = data["box"]
    hit = box[box["team_display_name"].astype(str).str.contains(team_name, case=False, na=False)]
    if hit.empty:
        raise ValueError(f"no team matching {team_name!r}")
    # prefer an exact (case-insensitive) match, else the most-shot team
    exact = hit[hit["team_display_name"].str.lower() == team_name.lower()]
    pick = exact if not exact.empty else hit
    tid = pick["team_id"].value_counts().idxmax()
    name = box.loc[box["team_id"] == tid, "team_display_name"].iloc[0]
    return int(tid), name


# ── league percentile helpers ────────────────────────────────────────────────
def _pct_rank(value, pool):
    """Fraction of pool strictly below value (0..1); nan-safe."""
    pool = np.asarray(pool, dtype=float)
    pool = pool[~np.isnan(pool)]
    if len(pool) == 0 or value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return float((pool < value).mean())


def league_three_pools(shots, min_fga=50):
    """League per-player 3PA/g and 3P% pools (rotation-ish) for gravity percentiles."""
    g = shots.groupby("athlete_id_1")
    tot = g.agg(fga=("make", "size"),
                games=("game_id", "nunique"),
                tpa=("is_three", "sum")).reset_index()
    tpm = shots[shots["is_three"]].groupby("athlete_id_1")["make"].sum().rename("tpm")
    tot = tot.merge(tpm, left_on="athlete_id_1", right_index=True, how="left").fillna({"tpm": 0})
    rot = tot[tot["fga"] >= min_fga].copy()
    rot["tpa_pg"] = rot["tpa"] / rot["games"].clip(lower=1)
    rot["tp_pct"] = np.where(rot["tpa"] > 0, rot["tpm"] / rot["tpa"], np.nan)
    return rot["tpa_pg"].to_numpy(), rot["tp_pct"].dropna().to_numpy()


# ── 1. Three-point threat board ──────────────────────────────────────────────
def three_point_threat(data, tid, min_tpa=15):
    """Per-player deep-shooting profile with a 0-100 gravity score and a tier label.

    Gravity = blend of league volume percentile (55%) and accuracy percentile (45%).
    Tiers translate volume+accuracy into a defensive instruction.
    """
    shots = data["shots"]
    vol_pool, acc_pool = league_three_pools(shots)

    team = shots[shots["team_id"] == tid]
    g = team.groupby("athlete_id_1")
    rows = []
    for aid, grp in g:
        games = grp["game_id"].nunique()
        tpa = int(grp["is_three"].sum())
        tpm = int(grp.loc[grp["is_three"], "make"].sum())
        if tpa == 0 and grp.shape[0] < 20:
            continue
        tpa_pg = tpa / max(games, 1)
        tp_pct = tpm / tpa if tpa else 0.0
        vol_p = _pct_rank(tpa_pg, vol_pool)
        acc_p = _pct_rank(tp_pct, acc_pool) if tpa >= 10 else np.nan
        # gravity: if too few attempts to trust accuracy, lean on volume only
        if np.isnan(acc_p):
            gravity = 100 * (0.7 * (vol_p if not np.isnan(vol_p) else 0))
        else:
            gravity = 100 * (0.55 * vol_p + 0.45 * acc_p)
        rows.append(dict(athlete_id=aid, games=games, tpa=tpa, tpm=tpm,
                         tpa_pg=round(tpa_pg, 2), tp_pct=round(tp_pct, 3),
                         gravity=round(gravity, 1),
                         tier=_threat_tier(tpa, tpa_pg, tp_pct, min_tpa)))
    out = pd.DataFrame(rows)
    out["name"] = out["athlete_id"].map(data["names"]["athlete_display_name"])
    out["jn"] = out["athlete_id"].map(_jersey_map(data))
    return out.sort_values("gravity", ascending=False).reset_index(drop=True)


def _jersey_map(data):
    """athlete_id -> jersey number (string) from player-stats; missing -> None."""
    ps = data["pstats"]
    if "id" not in ps.columns or "jn" not in ps.columns:
        return pd.Series(dtype="object")
    return ps.dropna(subset=["id"]).drop_duplicates("id").set_index("id")["jn"]


def _threat_tier(tpa, tpa_pg, pct, min_tpa):
    """Coach-language tier from volume + accuracy."""
    if tpa < 8 or tpa_pg < 1.0:
        return "IGNORE"
    small = tpa < min_tpa
    if tpa_pg >= 4.0 and pct >= 0.37:
        t = "DON'T LEAVE"
    elif pct >= 0.37 and tpa_pg >= 1.5:
        t = "CLOSE OUT"
    elif tpa_pg >= 5.5:                       # high volume even if average %
        t = "CLOSE OUT"
    elif 0.33 <= pct < 0.37 and tpa_pg >= 2.0:
        t = "RESPECT"
    elif pct < 0.30:
        t = "SAG OFF"
    else:
        t = "RESPECT"
    return t + (" *" if small else "")        # * = small sample (<min_tpa 3PA)


# ── 2/3. Zone frequency & quality (works on ANY shot subset) ─────────────────
def zone_frequency(shot_sub, data):
    """Share of FGA by zone (and family) for a shot subset, vs. the D-I average share.

    Pass team_shots(...) for offense (where they like to shoot) or opp_shots(...)
    for defense (where they force opponents to shoot).
    """
    lg_zone = data["shots"]["zone"].value_counts(normalize=True)
    lg_fam = data["shots"]["family"].value_counts(normalize=True)
    sub_zone = shot_sub["zone"].value_counts(normalize=True)
    sub_fam = shot_sub["family"].value_counts(normalize=True)

    zone = pd.DataFrame({
        "zone": ZONE_NAMES,
        "share": [sub_zone.get(z, 0.0) for z in ZONE_NAMES],
        "lg_share": [lg_zone.get(z, 0.0) for z in ZONE_NAMES],
        "att": [int((shot_sub["zone"] == z).sum()) for z in ZONE_NAMES],
    })
    zone["delta"] = zone["share"] - zone["lg_share"]
    fam = pd.DataFrame({
        "family": FAMILY_ORDER,
        "share": [sub_fam.get(f, 0.0) for f in FAMILY_ORDER],
        "lg_share": [lg_fam.get(f, 0.0) for f in FAMILY_ORDER],
    })
    fam["delta"] = fam["share"] - fam["lg_share"]
    return {"zone": zone, "family": fam, "n_fga": int(len(shot_sub))}


def zone_quality(shot_sub, baselines=None, root=".", min_att=10):
    """Points-per-shot (and FG%) by zone vs. the NCAA baseline, for a shot subset.

    PPS is the primary quality metric: FG% x point value, so a 3PT zone is
    rewarded 1.5x a 2PT zone for the same accuracy edge (`pps_vs_base` = the FG%
    delta scaled by the zone's point value).  Offense: how well THEY shoot from
    each spot.  Defense (opp_shots): how well opponents shoot against them.
    """
    if baselines is None:
        baselines = _load_baselines(os.path.join(root, "site", "data", "zone-baselines.json"))
    rows = []
    for z in ZONE_NAMES:
        zt = shot_sub[shot_sub["zone"] == z]
        att = int(len(zt))
        made = int(zt["make"].sum())
        fg = made / att if att else np.nan
        base = baselines.get(z, np.nan)
        pts = 3 if z.startswith("3PT") else 2
        pps = fg * pts if not np.isnan(fg) else np.nan
        pps_base = base * pts if not np.isnan(base) else np.nan
        pps_vs = pps - pps_base
        rows.append(dict(zone=z, att=att, made=made,
                         fg_pct=None if np.isnan(fg) else round(fg, 3),
                         baseline=None if np.isnan(base) else round(base, 3),
                         vs_base=None if (np.isnan(fg) or np.isnan(base)) else round(fg - base, 3),
                         pps=None if np.isnan(pps) else round(pps, 3),
                         pps_base=None if np.isnan(pps_base) else round(pps_base, 3),
                         pps_vs_base=None if np.isnan(pps_vs) else round(pps_vs, 3),
                         reliable=att >= min_att))
    return pd.DataFrame(rows)


def macro_frequency(shot_sub, data):
    """Share of FGA by the 3 coarse bands for a shot subset, vs. the D-I average.

    Offense: where THEY like to shoot.  Defense (opp_shots): where they force
    opponents to shoot.
    """
    lg = data["shots"]["zone"].map(MACRO_OF).value_counts(normalize=True)
    sub_macro = shot_sub["zone"].map(MACRO_OF)
    sub = sub_macro.value_counts(normalize=True)
    df = pd.DataFrame({
        "zone": MACRO_ORDER,
        "share": [sub.get(m, 0.0) for m in MACRO_ORDER],
        "lg_share": [lg.get(m, 0.0) for m in MACRO_ORDER],
        "att": [int((sub_macro == m).sum()) for m in MACRO_ORDER],
    })
    df["delta"] = df["share"] - df["lg_share"]
    return df


def macro_quality(shot_sub, baselines=None, root=".", min_att=25):
    """Points-per-shot (and FG%) by the 3 coarse bands (paint/restricted,
    mid-range, three) vs. an attempt-weighted NCAA baseline for that band.

    The band's expected FG% weights each member zone's NCAA baseline by how often
    THIS shot set was taken from that zone; PPS scales that by the band's point
    value (2, or 3 for the three band), so `pps_vs_base` = "how many more points
    per shot than a league-average team, given the same spots inside the band".
    """
    if baselines is None:
        baselines = _load_baselines(os.path.join(root, "site", "data", "zone-baselines.json"))
    sub = shot_sub.copy()
    sub["macro"] = sub["zone"].map(MACRO_OF)
    rows = []
    for macro in MACRO_ORDER:
        zm = sub[sub["macro"] == macro]
        att = int(len(zm))
        made = int(zm["make"].sum())
        fg = made / att if att else np.nan
        # attempt-weighted baseline across the member zones actually shot
        counts = zm["zone"].value_counts()
        wsum = sum(counts.get(z, 0) * baselines.get(z, np.nan) for z in counts.index)
        exp = wsum / att if att else np.nan
        pts = 3 if macro == "Three" else 2
        pps = fg * pts if not np.isnan(fg) else np.nan
        pps_base = exp * pts if not np.isnan(exp) else np.nan
        pps_vs = pps - pps_base
        rows.append(dict(zone=macro, att=att, made=made,
                         fg_pct=None if np.isnan(fg) else round(fg, 3),
                         baseline=None if np.isnan(exp) else round(exp, 3),
                         vs_base=None if (np.isnan(fg) or np.isnan(exp)) else round(fg - exp, 3),
                         pps=None if np.isnan(pps) else round(pps, 3),
                         pps_base=None if np.isnan(pps_base) else round(pps_base, 3),
                         pps_vs_base=None if np.isnan(pps_vs) else round(pps_vs, 3),
                         reliable=att >= min_att))
    return pd.DataFrame(rows)


def _load_baselines(path):
    if path is None:
        path = os.path.join("site", "data", "zone-baselines.json")
    try:
        b = json.load(open(path))
        return dict(zip(b["zones"], b["baselines"]))
    except Exception:
        return {}


# ── 4. Who to attack (defensive on/off — available early season, unlike RAPM) ─
def attack_board(data, tid, name, min_mpg=0.0, min_def_poss=0):
    """Rank the opponent's rotation by defensive on/off.

    def_onoff = drtg_on - drtg_off  (points allowed per 100 poss with the player
    ON minus OFF).  Positive = the team's defense gets WORSE with them on the
    floor -> attack them.  drtg_on = raw points allowed per 100 while on court.
    Both come from mbb_onoff_{season}_v2.csv, which is built from box/PBP and is
    ready at the start of the season (RAPM needs a stabilising sample first).

    Players below `min_def_poss` defensive possessions are dropped (noise);
    height / usage are shown as context only.
    """
    ps = data["pstats"]
    oo = data["onoff"]
    team_ps = ps[ps["t"] == name].copy()
    oo_by_id = oo.set_index("athlete_id")

    rows = []
    for _, p in team_ps.iterrows():
        if (p.get("mpg") or 0) < min_mpg:
            continue
        if p["id"] not in oo_by_id.index:
            continue
        r = oo_by_id.loc[p["id"]]
        if isinstance(r, pd.DataFrame):     # dupe athlete_id — keep most-poss row
            r = r.sort_values("poss_def_on").iloc[-1]
        if (r.get("poss_def_on") or 0) < min_def_poss:
            continue
        drtg_on, drtg_off = float(r["drtg_on"]), float(r["drtg_off"])
        rows.append(dict(name=p["n"], jn=p.get("jn"), pos=p.get("pos"), ht=p.get("ht"),
                         mpg=round(float(p.get("mpg") or 0), 1),
                         usg=p.get("usg"),
                         drtg_on=round(drtg_on, 1), drtg_off=round(drtg_off, 1),
                         def_onoff=round(drtg_on - drtg_off, 1),
                         def_poss=int(r["poss_def_on"])))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # rank by def_onoff first (team defends worse with them on), drtg_on breaks ties
    return out.sort_values(["def_onoff", "drtg_on"], ascending=False).reset_index(drop=True)


# ── 5. Usage / ball-handling (how heliocentric is the offense) ────────────────
def usage_board(data, tid, name, min_mpg=0.0):
    """Rotation ranked by usage %, with assist / turnover / assisted-FG context.

    usg   = share of team possessions a player uses while on court (heliocentrism)
    astp  = assist % (teammate FGs he sets up) -> on-ball creator vs. finisher
    tovp  = turnover %
    astdp = assisted-FG % — share of the player's OWN made FGs that were assisted
            (low = self-creator, guard the drive; high = spot-up, deny the feed)
    """
    ps = data["pstats"]
    cols = ["n", "jn", "pos", "mpg", "ppg", "usg", "astp", "tovp", "astdp"]
    team = ps[ps["t"] == name].copy()
    team = team[[c for c in cols if c in team.columns]]
    team = team[team["mpg"].fillna(0) >= min_mpg]
    return team.sort_values("usg", ascending=False).reset_index(drop=True)


# ── 5b. Points responsible (scored + assisted share of ON-COURT team points) ──
def points_responsible_board(data, tid, name):
    """Share of the team's on-court points each player is directly responsible for.

    For every stint a player is on the floor, what share of the points his team
    scores does he SCORE (every made FG + FT) or ASSIST (the FG value of baskets
    he set up)?  denominator = the team's points scored WHILE HE WAS ON THE FLOOR.

    A player can't assist his own make, so scored and assisted never double-count
    for one player.  Blind spot: a pass that only draws a shooting foul earns no
    assist.  Regular season only (see build_points_resp.py), which also computes
    the identical table the Player Stats page merges, so the two always agree.
    """
    pr = data.get("points_resp")
    if pr is None or pr.empty:
        return pd.DataFrame()

    tm = pr[pr["team_id"] == tid].copy()
    if tm.empty:
        return pd.DataFrame()

    # names / jersey / position from player-stats (jerseys already backfilled in
    # build_scout from the boxscore, so bench players resolve too)
    ps = data["pstats"]
    meta = (ps.dropna(subset=["id"]).drop_duplicates("id")
              .set_index("id")[[c for c in ("n", "jn", "pos") if c in ps.columns]])
    fallback = data["names"]["athlete_display_name"]

    rows = []
    for _, r in tm.iterrows():
        aid = r["athlete_id"]
        m = meta.loc[aid] if aid in meta.index else None
        nm = (m["n"] if m is not None and pd.notna(m.get("n")) else fallback.get(aid))
        rows.append(dict(
            name=nm,
            jn=(m["jn"] if m is not None else None),
            pos=(m["pos"] if m is not None else None),
            pts_scored=int(r["pts_scored"]), pts_ast=int(r["pts_ast"]),
            pts_resp=int(r["pts_scored"] + r["pts_ast"]),
            on_court_pts=int(r["on_court_pts"]),
            scored_pct=float(r["scored_pct"]), assist_pct=float(r["assist_pct"]),
            resp_pct=float(r["resp_pct"])))
    out = pd.DataFrame(rows)
    return out.sort_values("resp_pct", ascending=False).reset_index(drop=True)


# ── 6b. Foul-trouble sheet (fouls per possession, from full PBP) ──────────────
def foul_board(data, tid, name, min_def_poss=0, min_mpg=0.0):
    """Rotation ranked by personal fouls committed per 100 defensive possessions.

    High rate = attack him to put him in foul trouble.  Fouls come from the
    PersonalFoul play-by-play events; possessions from the on/off table.
    """
    ps = data["pstats"]
    fouls = data.get("fouls", pd.Series(dtype="int64"))
    oo = data["onoff"].set_index("athlete_id")
    team_ps = ps[ps["t"] == name]

    rows = []
    for _, p in team_ps.iterrows():
        if (p.get("mpg") or 0) < min_mpg:
            continue
        aid = p["id"]
        if aid not in oo.index:
            continue
        r = oo.loc[aid]
        if isinstance(r, pd.DataFrame):          # dupe athlete_id — keep most-poss row
            r = r.sort_values("poss_def_on").iloc[-1]
        dposs = float(r.get("poss_def_on") or 0)
        if dposs < min_def_poss:
            continue
        nf = int(fouls.get(aid, 0))
        rows.append(dict(name=p["n"], jn=p.get("jn"), pos=p.get("pos"),
                         mpg=round(float(p.get("mpg") or 0), 1), fouls=nf,
                         def_poss=int(dposs),
                         fouls_per100=round(100 * nf / dposs, 2) if dposs else np.nan))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("fouls_per100", ascending=False).reset_index(drop=True)


# ── 6c. Box-out board (who crashes the glass) ─────────────────────────────────
def rebound_board(data, tid, name, min_mpg=0.0):
    """Rotation ranked by offensive-rebound % — the players you have to box out.

    ORB% / DRB% / TRB% are share-of-available-rebounds rates (already in
    player-stats); reb_per100 is raw volume — total rebounds per 100 on-court
    possessions (rpg*gp over poss_off_on + poss_def_on).  Ranked by ORB% because
    the box-out job is on THEIR offensive glass.
    """
    ps = data["pstats"]
    oo = data["onoff"].set_index("athlete_id")
    team = ps[ps["t"] == name]

    rows = []
    for _, p in team.iterrows():
        if (p.get("mpg") or 0) < min_mpg:
            continue
        aid = p["id"]
        gp = float(p.get("gp") or 0)
        tot_reb = float(p.get("rpg") or 0) * gp
        reb_per100 = np.nan
        if aid in oo.index:
            r = oo.loc[aid]
            if isinstance(r, pd.DataFrame):
                r = r.sort_values("poss_def_on").iloc[-1]
            poss = float(r.get("poss_off_on") or 0) + float(r.get("poss_def_on") or 0)
            if poss > 0:
                reb_per100 = round(100 * tot_reb / poss, 1)
        rows.append(dict(name=p["n"], jn=p.get("jn"), pos=p.get("pos"), ht=p.get("ht"),
                         gp=int(gp), mpg=p.get("mpg"),
                         orbp=p.get("orbp"), drbp=p.get("drbp"), trbp=p.get("trbp"),
                         rpg=p.get("rpg"), reb=round(tot_reb), reb_per100=reb_per100))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("orbp", ascending=False).reset_index(drop=True)


# ── 7. Individual scoring — top scorers & their shots ─────────────────────────
def top_scorers(data, tid, name, n=6, min_mpg=0.0):
    """The team's top-`n` scorers by PPG (rotation only) — for per-player charts."""
    ps = data["pstats"]
    team = ps[ps["t"] == name].copy()
    team = team[team["mpg"].fillna(0) >= min_mpg]
    team = team.sort_values("ppg", ascending=False).head(n)
    keep = [c for c in ["id", "n", "jn", "pos", "ppg", "usg"] if c in team.columns]
    return team[keep].reset_index(drop=True)


def player_shots(data, aid):
    """One player's field-goal attempts (shooter = athlete_id_1)."""
    return data["shots"][data["shots"]["athlete_id_1"] == aid]


# ── 8. Directional tendency (force him to his weak side) ──────────────────────
def _side_of(zone):
    """Court side for a zone name: Left / Center / Right (rim/center -> Center)."""
    if zone.endswith("Right Center") or zone.endswith("Left Center"):
        return "Center"
    if zone.endswith("Right"):
        return "Right"
    if zone.endswith("Left"):
        return "Left"
    return "Center"          # Restricted Area, Mid/3PT - Center


def directional_split(shot_sub):
    """Share, FG% and PPS by court side (Left/Center/Right) for a shot subset.

    A location-based proxy for handedness/drive direction: if a scorer's PPS is
    far higher on one side, force him to the other ('force him weak').
    """
    sub = shot_sub.copy()
    sub["side"] = sub["zone"].map(_side_of)
    tot = len(sub)
    rows = []
    for side in ["Left", "Center", "Right"]:
        zs = sub[sub["side"] == side]
        att = int(len(zs))
        made = int(zs["make"].sum())
        fg = made / att if att else np.nan
        pts = np.where(zs["is_three"], 3, 2)
        pps = float((zs["make"].to_numpy() * pts).mean()) if att else np.nan
        rows.append(dict(side=side, att=att,
                         share=att / tot if tot else 0.0,
                         fg_pct=None if np.isnan(fg) else round(fg, 3),
                         pps=None if att == 0 else round(pps, 3)))
    return pd.DataFrame(rows)


# ── 9. Recent form (last N games) ─────────────────────────────────────────────
def recent_shots(shot_sub, n_games=5):
    """Restrict a shot subset to its most recent `n_games` (by game_date)."""
    if shot_sub.empty:
        return shot_sub
    games = (shot_sub[["game_id", "game_date"]].drop_duplicates()
             .sort_values("game_date").tail(n_games)["game_id"])
    return shot_sub[shot_sub["game_id"].isin(games)]


def form_summary(shot_sub, data, n_games=5, root="."):
    """Season vs. last-`n_games` shot mix & PPS by the 3 macro bands.

    Shows whether the offense's shot diet / shot-making has drifted recently —
    a scout of the November version is stale by March.
    """
    rec = recent_shots(shot_sub, n_games)
    full_f, rec_f = macro_frequency(shot_sub, data), macro_frequency(rec, data)
    full_q, rec_q = macro_quality(shot_sub, root=root), macro_quality(rec, root=root)
    rows = []
    for band in MACRO_ORDER:
        fs = float(full_f.loc[full_f["zone"] == band, "share"].iloc[0])
        rs = float(rec_f.loc[rec_f["zone"] == band, "share"].iloc[0])
        fp = full_q.loc[full_q["zone"] == band, "pps"].iloc[0]
        rp = rec_q.loc[rec_q["zone"] == band, "pps"].iloc[0]
        rows.append(dict(band=band, share=fs, share_recent=rs, share_delta=rs - fs,
                         pps=fp, pps_recent=rp,
                         pps_delta=None if (fp is None or rp is None) else round(rp - fp, 3)))
    n_rec = int(rec[["game_id"]].drop_duplicates().shape[0])
    return {"table": pd.DataFrame(rows), "n_recent_games": n_rec}
