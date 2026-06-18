"""
Generates on/off splits and RAPM for both overall and conference games.

Run locally:
    python3 build_onoff_rapm.py

Outputs (season auto-detected from current date):
    mbb_onoff_{SEASON}_v2.csv
    mbb_onoff_{SEASON}_conf_v2.csv
    mbb_rapm_{SEASON-1}{SEASON%100:02d}.csv
    presence_full.parquet
    player_lookup.csv
"""

import pandas as pd
import numpy as np
import warnings
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
import sportsdataverse.mbb as mbb

warnings.filterwarnings("ignore")

import os as _os
from datetime import date as _date
_override   = _os.environ.get("OVERRIDE_SEASON")
if _override:
    SEASON = int(_override)
else:
    _today = _date.today()
    SEASON = _today.year + 1 if _today.month >= 11 else _today.year
SEASON_TYPE = 2

# ── Load data (once) ──────────────────────────────────────────────────────────
print("Loading PBP...")
pbp_raw = mbb.load_mbb_pbp(seasons=SEASON, return_as_pandas=True)
pbp_raw = pbp_raw[pbp_raw["season_type"] == SEASON_TYPE].reset_index(drop=True)

print("Loading player boxscores...")
player_df = mbb.load_mbb_player_boxscore(seasons=SEASON, return_as_pandas=True)
player_df = player_df[player_df["season_type"] == SEASON_TYPE].reset_index(drop=True)

print("Loading team boxscores...")
team_df = mbb.load_mbb_team_boxscore(seasons=SEASON, return_as_pandas=True)
team_df = team_df[team_df["season_type"] == SEASON_TYPE].reset_index(drop=True)

print("Loading schedule...")
sched = mbb.load_mbb_schedule(seasons=SEASON, return_as_pandas=True)
sched = sched[sched["season_type"] == SEASON_TYPE]
conf_game_ids = set(sched.loc[sched["conference_competition"] == True, "game_id"])

print(f"  PBP: {len(pbp_raw):,} plays | Players: {len(player_df):,} rows | Teams: {len(team_df):,} rows")


def run_pipeline(game_filter="all"):
    """Run the full on/off + RAPM pipeline for a given game filter."""
    label = {"all": "all games", "conf": "conference only"}[game_filter]
    print(f"\n{'='*60}")
    print(f"Running pipeline: {label}")
    print(f"{'='*60}")

    if game_filter == "conf":
        pbp   = pbp_raw[pbp_raw["game_id"].isin(conf_game_ids)].reset_index(drop=True)
        plyr  = player_df[player_df["game_id"].isin(conf_game_ids)].reset_index(drop=True)
        team  = team_df[team_df["game_id"].isin(conf_game_ids)].reset_index(drop=True)
    else:
        pbp, plyr, team = pbp_raw.copy(), player_df.copy(), team_df.copy()

    print(f"  {pbp['game_id'].nunique()} games, {len(pbp):,} plays")

    # ── Prepare PBP ──────────────────────────────────────────────────────────
    pbp = pbp[[
        "game_id", "sequence_number", "type_text", "text", "team_id",
        "athlete_id_1", "athlete_id_2", "scoring_play", "score_value",
        "shooting_play", "points_attempted",
        "home_team_id", "away_team_id",
    ]].copy()

    pbp["sequence_number"]  = pd.to_numeric(pbp["sequence_number"],  errors="coerce")
    pbp["score_value"]      = pd.to_numeric(pbp["score_value"],      errors="coerce").fillna(0).astype(int)
    pbp["points_attempted"] = pd.to_numeric(pbp["points_attempted"], errors="coerce")
    pbp["scoring_play"]     = pbp["scoring_play"].astype(bool)
    pbp["shooting_play"]    = pbp["shooting_play"].astype(bool)
    for col in ["team_id", "athlete_id_1", "athlete_id_2", "home_team_id", "away_team_id"]:
        pbp[col] = pd.to_numeric(pbp[col], errors="coerce")

    pbp["is_sub"]     = pbp["type_text"] == "Substitution"
    pbp["is_sub_in"]  = pbp["is_sub"] & pbp["text"].str.contains("subbing in",  case=False, na=False)
    pbp["is_sub_out"] = pbp["is_sub"] & pbp["text"].str.contains("subbing out", case=False, na=False)
    pbp["is_ft"]      = pbp["type_text"].str.contains("FreeThrow", na=False)
    pbp["is_fga"]     = pbp["shooting_play"] & ~pbp["is_ft"]
    pbp["is_to"]      = pbp["type_text"].str.contains("Turnover",  na=False)
    pbp["is_fgm"]     = pbp["is_fga"] & pbp["scoring_play"]
    pbp["is_3pa"]     = pbp["is_fga"] & (pbp["points_attempted"] == 3)
    pbp["is_3pm"]     = pbp["is_3pa"] & pbp["scoring_play"]
    pbp["is_ftm"]     = pbp["is_ft"]  & pbp["scoring_play"]
    pbp["is_orb"]     = pbp["type_text"].str.contains("Offensive Rebound", na=False)
    pbp["is_drb"]     = pbp["type_text"].str.contains("Defensive Rebound", na=False)
    pbp = pbp.sort_values(["game_id", "sequence_number"]).reset_index(drop=True)

    # ── Game-level possessions ────────────────────────────────────────────────
    team["poss_box"]   = (team["field_goals_attempted"] + 0.44 * team["free_throws_attempted"]
                          + team["total_turnovers"] - team["offensive_rebounds"])
    team["team_id_n"]  = pd.to_numeric(team["team_id"], errors="coerce")
    game_poss          = team.set_index(["game_id", "team_id_n"])["poss_box"].to_dict()

    # ── Starters ─────────────────────────────────────────────────────────────
    plyr["athlete_id"]  = pd.to_numeric(plyr["athlete_id"],  errors="coerce")
    plyr["team_id_num"] = pd.to_numeric(plyr["team_id"],     errors="coerce")
    starters = (
        plyr[(plyr["starter"] == True) & plyr["athlete_id"].notna()]
        .groupby(["game_id", "team_id_num"])["athlete_id"]
        .apply(frozenset).to_dict()
    )

    # ── Build stints ─────────────────────────────────────────────────────────
    print("  Building stints...")
    games      = pbp["game_id"].unique()
    stint_ids  = np.zeros(len(pbp), dtype=np.int32)
    stint_rows = []

    for game_id in games:
        mask      = pbp["game_id"] == game_id
        game_idx  = pbp.index[mask].to_numpy()
        game_rows = pbp.loc[game_idx]

        home_id = int(game_rows["home_team_id"].iloc[0])
        away_id = int(game_rows["away_team_id"].iloc[0])
        home_lu = set(starters.get((game_id, home_id), set()))
        away_lu = set(starters.get((game_id, away_id), set()))

        cur_stint   = 0
        gstint_ids  = np.zeros(len(game_idx), dtype=np.int32)
        is_in_arr   = game_rows["is_sub_in"].to_numpy()
        is_out_arr  = game_rows["is_sub_out"].to_numpy()
        team_arr    = game_rows["team_id"].to_numpy()
        ath1_arr    = game_rows["athlete_id_1"].to_numpy()

        for j in range(len(game_idx)):
            if is_in_arr[j] or is_out_arr[j]:
                stint_rows.append({"game_id": game_id, "stint_id": cur_stint,
                                    "home_team_id": home_id, "away_team_id": away_id,
                                    "home_lineup": frozenset(home_lu),
                                    "away_lineup": frozenset(away_lu)})
                cur_stint += 1
                pid = ath1_arr[j]
                if pd.notna(pid):
                    pid = int(pid)
                    t   = team_arr[j]
                    if is_in_arr[j]:
                        if t == home_id: home_lu.add(pid)
                        elif t == away_id: away_lu.add(pid)
                    else:
                        if t == home_id: home_lu.discard(pid)
                        elif t == away_id: away_lu.discard(pid)
            gstint_ids[j] = cur_stint

        stint_rows.append({"game_id": game_id, "stint_id": cur_stint,
                            "home_team_id": home_id, "away_team_id": away_id,
                            "home_lineup": frozenset(home_lu),
                            "away_lineup": frozenset(away_lu)})
        stint_ids[game_idx] = gstint_ids

    pbp = pbp.copy()
    pbp["stint_id"] = stint_ids
    stint_info = (pd.DataFrame(stint_rows)
                  .drop_duplicates(subset=["game_id", "stint_id"], keep="last")
                  .reset_index(drop=True))
    print(f"  {len(stint_info):,} stints built")

    # ── Stint stats ───────────────────────────────────────────────────────────
    scoring  = pbp[pbp["scoring_play"]].copy()
    pts_df   = (scoring.groupby(["game_id", "stint_id", "team_id"])["score_value"]
                .sum().reset_index()
                .rename(columns={"score_value": "pts_scored", "team_id": "scoring_team_id"}))

    poss_ev  = pbp[pbp["is_fga"] | pbp["is_ft"] | pbp["is_to"]].copy()
    poss_ev["poss_weight"] = np.where(poss_ev["is_fga"], 1.0,
                             np.where(poss_ev["is_ft"],  0.44, 1.0))
    poss_w   = (poss_ev.groupby(["game_id", "stint_id", "team_id"])["poss_weight"]
                .sum().reset_index().rename(columns={"team_id": "poss_team_id"}))

    gpw = (poss_w.groupby(["game_id", "poss_team_id"])["poss_weight"]
           .sum().reset_index().rename(columns={"poss_weight": "game_poss_weight",
                                                "poss_team_id": "team_id_gw"}))
    gpdf = pd.DataFrame([(g, t, p) for (g, t), p in game_poss.items()],
                        columns=["game_id", "team_id_gw", "poss_box"])
    gpw  = gpw.merge(gpdf, on=["game_id", "team_id_gw"], how="left")
    poss_w = poss_w.merge(gpw.rename(columns={"team_id_gw": "poss_team_id"}),
                          on=["game_id", "poss_team_id"], how="left")
    poss_w["poss_scaled"] = np.where(
        poss_w["game_poss_weight"] > 0,
        poss_w["poss_weight"] / poss_w["game_poss_weight"] * poss_w["poss_box"], 0)

    STAT_FLAGS = {"fga": "is_fga", "fgm": "is_fgm", "tpa": "is_3pa", "tpm": "is_3pm",
                  "fta": "is_ft",  "ftm": "is_ftm",  "orb": "is_orb", "drb": "is_drb"}
    STAT_COLS  = list(STAT_FLAGS.keys())
    shoot_df   = None
    for stat, flag in STAT_FLAGS.items():
        sub = (pbp[pbp[flag]].groupby(["game_id", "stint_id", "team_id"])
               .size().reset_index(name=stat))
        shoot_df = sub if shoot_df is None else shoot_df.merge(
            sub, on=["game_id", "stint_id", "team_id"], how="outer")
    shoot_df = shoot_df.fillna(0)
    for c in STAT_COLS:
        shoot_df[c] = shoot_df[c].astype(int)

    # ── Pivot to home/away ────────────────────────────────────────────────────
    stint_full = stint_info[["game_id", "stint_id", "home_team_id", "away_team_id"]].copy()

    def side_agg(df, key_col, val_col, new_col, group_cols):
        return df.groupby(group_cols)[val_col].sum().reset_index().rename(columns={val_col: new_col})

    hp  = pts_df.merge(stint_info[["game_id", "stint_id", "home_team_id"]], on=["game_id", "stint_id"])
    hp["is_home"] = hp["scoring_team_id"] == hp["home_team_id"]
    hpts = side_agg(hp[hp["is_home"]],  None, "pts_scored", "home_pts", ["game_id", "stint_id"])
    apts = side_agg(hp[~hp["is_home"]], None, "pts_scored", "away_pts", ["game_id", "stint_id"])
    stint_full = stint_full.merge(hpts, on=["game_id", "stint_id"], how="left")
    stint_full = stint_full.merge(apts, on=["game_id", "stint_id"], how="left")

    pw  = poss_w.merge(stint_info[["game_id", "stint_id", "home_team_id"]], on=["game_id", "stint_id"])
    pw["is_home"] = pw["poss_team_id"] == pw["home_team_id"]
    hpos = side_agg(pw[pw["is_home"]],  None, "poss_scaled", "home_poss", ["game_id", "stint_id"])
    apos = side_agg(pw[~pw["is_home"]], None, "poss_scaled", "away_poss", ["game_id", "stint_id"])
    stint_full = stint_full.merge(hpos, on=["game_id", "stint_id"], how="left")
    stint_full = stint_full.merge(apos, on=["game_id", "stint_id"], how="left")
    stint_full[["home_pts", "away_pts", "home_poss", "away_poss"]] = \
        stint_full[["home_pts", "away_pts", "home_poss", "away_poss"]].fillna(0)

    sh = shoot_df.merge(stint_info[["game_id", "stint_id", "home_team_id"]], on=["game_id", "stint_id"])
    sh["is_home"] = sh["team_id"] == sh["home_team_id"]
    hsh = sh[sh["is_home"]].groupby(["game_id", "stint_id"])[STAT_COLS].sum().reset_index()
    ash = sh[~sh["is_home"]].groupby(["game_id", "stint_id"])[STAT_COLS].sum().reset_index()
    hsh = hsh.rename(columns={c: f"home_{c}" for c in STAT_COLS})
    ash = ash.rename(columns={c: f"away_{c}" for c in STAT_COLS})
    stint_full = stint_full.merge(hsh, on=["game_id", "stint_id"], how="left")
    stint_full = stint_full.merge(ash, on=["game_id", "stint_id"], how="left")
    new_cols = [f"home_{c}" for c in STAT_COLS] + [f"away_{c}" for c in STAT_COLS]
    stint_full[new_cols] = stint_full[new_cols].fillna(0).astype(int)

    # ── Presence table ────────────────────────────────────────────────────────
    print("  Building presence table...")
    active = (plyr[plyr["did_not_play"] != True].dropna(subset=["athlete_id", "minutes"])
              .groupby(["game_id", "team_id_num"])["athlete_id"].apply(set).to_dict())

    rows = []
    for _, s in stint_info.iterrows():
        gid  = s["game_id"]; sid  = s["stint_id"]
        htid = s["home_team_id"]; atid = s["away_team_id"]
        hl   = s["home_lineup"]; al   = s["away_lineup"]
        for team_id, lineup in [(htid, hl), (atid, al)]:
            for pid in active.get((gid, team_id), set()):
                rows.append({"game_id": gid, "stint_id": sid, "athlete_id": pid,
                              "team_id": team_id, "is_on_court": pid in lineup})

    presence_df = pd.DataFrame(rows)
    presence_full = presence_df.merge(
        stint_full[["game_id", "stint_id", "home_team_id", "away_team_id",
                    "home_pts", "away_pts", "home_poss", "away_poss"] + new_cols],
        on=["game_id", "stint_id"], how="left")

    presence_full["is_home"]     = presence_full["team_id"] == presence_full["home_team_id"]
    presence_full["pts_for"]     = np.where(presence_full["is_home"], presence_full["home_pts"], presence_full["away_pts"])
    presence_full["pts_against"] = np.where(presence_full["is_home"], presence_full["away_pts"], presence_full["home_pts"])
    presence_full["poss_off"]    = np.where(presence_full["is_home"], presence_full["home_poss"], presence_full["away_poss"])
    presence_full["poss_def"]    = np.where(presence_full["is_home"], presence_full["away_poss"], presence_full["home_poss"])
    for c in STAT_COLS:
        presence_full[f"{c}_for"]     = np.where(presence_full["is_home"], presence_full[f"home_{c}"], presence_full[f"away_{c}"])
        presence_full[f"{c}_against"] = np.where(presence_full["is_home"], presence_full[f"away_{c}"], presence_full[f"home_{c}"])

    # ── Aggregate on/off ──────────────────────────────────────────────────────
    print("  Aggregating on/off stats...")
    shoot_cols = [f"{c}_for" for c in STAT_COLS] + [f"{c}_against" for c in STAT_COLS]
    agg = (presence_full
           .groupby(["athlete_id", "team_id", "is_on_court"])
           [["pts_for", "pts_against", "poss_off", "poss_def"] + shoot_cols]
           .sum().reset_index())

    on_df  = (agg[agg["is_on_court"]].drop(columns="is_on_court")
              .add_suffix("_on").rename(columns={"athlete_id_on": "athlete_id", "team_id_on": "team_id"}))
    off_df = (agg[~agg["is_on_court"]].drop(columns="is_on_court")
              .add_suffix("_off").rename(columns={"athlete_id_off": "athlete_id", "team_id_off": "team_id"}))
    player_agg = on_df.merge(off_df, on=["athlete_id", "team_id"], how="outer").fillna(0)

    def rtg(pts, poss):
        return np.where(poss > 10, pts / poss * 100, np.nan)

    def safe_div(n, d):
        return np.where(d > 0, n / d, np.nan)

    player_agg["ortg_on"]  = rtg(player_agg["pts_for_on"],     player_agg["poss_off_on"])
    player_agg["drtg_on"]  = rtg(player_agg["pts_against_on"], player_agg["poss_def_on"])
    player_agg["nrtg_on"]  = player_agg["ortg_on"]  - player_agg["drtg_on"]
    player_agg["ortg_off"] = rtg(player_agg["pts_for_off"],     player_agg["poss_off_off"])
    player_agg["drtg_off"] = rtg(player_agg["pts_against_off"], player_agg["poss_def_off"])
    player_agg["nrtg_off"] = player_agg["ortg_off"] - player_agg["drtg_off"]
    player_agg["on_off"]   = player_agg["nrtg_on"]  - player_agg["nrtg_off"]

    for sfx in ["on", "off"]:
        fga = player_agg[f"fga_for_{sfx}"]; fgm = player_agg[f"fgm_for_{sfx}"]
        tpa = player_agg[f"tpa_for_{sfx}"]; tpm = player_agg[f"tpm_for_{sfx}"]
        fta = player_agg[f"fta_for_{sfx}"]; ftm = player_agg[f"ftm_for_{sfx}"]
        orb = player_agg[f"orb_for_{sfx}"]; drb = player_agg[f"drb_for_{sfx}"]
        poss   = player_agg[f"poss_off_{sfx}"]
        poss_d = player_agg[f"poss_def_{sfx}"]
        opp_fga = player_agg[f"fga_against_{sfx}"]; opp_fgm = player_agg[f"fgm_against_{sfx}"]
        opp_tpa = player_agg[f"tpa_against_{sfx}"]; opp_tpm = player_agg[f"tpm_against_{sfx}"]
        opp_fta = player_agg[f"fta_against_{sfx}"]
        opp_orb = player_agg[f"orb_against_{sfx}"]; opp_drb = player_agg[f"drb_against_{sfx}"]

        player_agg[f"fg_pct_{sfx}"]      = safe_div(fgm, fga)
        player_agg[f"efg_pct_{sfx}"]     = safe_div(fgm + 0.5 * tpm, fga)
        player_agg[f"3p_pct_{sfx}"]      = safe_div(tpm, tpa)
        player_agg[f"3p_rate_{sfx}"]     = safe_div(tpa, fga)
        player_agg[f"ft_rate_{sfx}"]     = safe_div(fta, fga)
        player_agg[f"ft_pct_{sfx}"]      = safe_div(ftm, fta)
        player_agg[f"opp_fg_pct_{sfx}"]  = safe_div(opp_fgm, opp_fga)
        player_agg[f"opp_efg_pct_{sfx}"] = safe_div(opp_fgm + 0.5 * opp_tpm, opp_fga)
        player_agg[f"opp_3p_pct_{sfx}"]  = safe_div(opp_tpm, opp_tpa)
        player_agg[f"opp_3p_rate_{sfx}"] = safe_div(opp_tpa, opp_fga)
        player_agg[f"drb_pct_{sfx}"]     = safe_div(drb, drb + opp_orb)
        player_agg[f"orb_pct_{sfx}"]     = safe_div(orb, orb + opp_drb)
        player_agg[f"orb_per100_{sfx}"]      = safe_div(orb,     poss)   * 100
        player_agg[f"drb_per100_{sfx}"]      = safe_div(drb,     poss_d) * 100
        player_agg[f"opp_orb_per100_{sfx}"]  = safe_div(opp_orb, poss_d) * 100

    names = (plyr[["athlete_id", "team_id_num", "athlete_display_name", "team_display_name"]]
             .drop_duplicates().rename(columns={"team_id_num": "team_id"}))
    player_agg = player_agg.merge(names, on=["athlete_id", "team_id"], how="left")
    result = player_agg.copy().sort_values("on_off", ascending=False)

    # ── Save on/off ───────────────────────────────────────────────────────────
    out_cols = [
        "athlete_id", "athlete_display_name", "team_id", "team_display_name",
        "ortg_on", "drtg_on", "nrtg_on", "ortg_off", "drtg_off", "nrtg_off", "on_off",
        "ortg_on_pct", "drtg_on_pct", "nrtg_on_pct",
        "ortg_off_pct", "drtg_off_pct", "nrtg_off_pct", "on_off_pct",
        "fg_pct_on", "efg_pct_on", "3p_pct_on", "3p_rate_on", "ft_rate_on", "ft_pct_on",
        "opp_fg_pct_on", "opp_efg_pct_on", "opp_3p_pct_on", "opp_3p_rate_on",
        "drb_pct_on", "orb_pct_on", "drb_per100_on", "orb_per100_on", "opp_orb_per100_on",
        "poss_off_on", "poss_def_on", "poss_off_off", "poss_def_off",
        "pts_for_on", "pts_against_on", "pts_for_off", "pts_against_off",
        "fga_for_on", "fgm_for_on", "tpa_for_on", "tpm_for_on",
        "fga_against_on", "fgm_against_on", "tpa_against_on", "tpm_against_on",
        "orb_for_on", "drb_for_on", "orb_against_on", "drb_against_on",
    ]
    # percentile cols may not exist if run on small conf subset — add them if absent
    for col in ["ortg_on_pct", "drtg_on_pct", "nrtg_on_pct",
                "ortg_off_pct", "drtg_off_pct", "nrtg_off_pct", "on_off_pct"]:
        if col not in result.columns:
            result[col] = np.nan
    sfx   = "" if game_filter == "all" else f"_{game_filter}"
    fname = f"mbb_onoff_{SEASON}{sfx}_v2.csv"
    result[[c for c in out_cols if c in result.columns]].round(4).to_csv(fname, index=False)
    print(f"  Saved {fname} — {len(result)} players")

    # ── RAPM ─────────────────────────────────────────────────────────────────
    print("  Fitting RAPM...")
    rapm_stints = stint_full.merge(
        stint_info[["game_id", "stint_id", "home_lineup", "away_lineup"]],
        on=["game_id", "stint_id"]).copy()

    all_players = sorted({
        pid for _, row in rapm_stints[["home_lineup", "away_lineup"]].iterrows()
        for lineup in (row["home_lineup"], row["away_lineup"])
        for pid in lineup if pd.notna(pid)})
    p2i = {pid: i for i, pid in enumerate(all_players)}
    n_p = len(all_players)

    obs_r, obs_c, obs_v, y_v, w_v = [], [], [], [], []
    idx = 0
    for row in rapm_stints.itertuples():
        for is_home, lineup_off, lineup_def, pts, poss in [
            (True,  row.home_lineup, row.away_lineup, row.home_pts, row.home_poss),
            (False, row.away_lineup, row.home_lineup, row.away_pts, row.away_poss),
        ]:
            if poss <= 0: continue
            if len(lineup_off) != 5 or len(lineup_def) != 5: continue
            for pid in lineup_off:
                if pd.isna(pid): continue
                obs_r.append(idx); obs_c.append(p2i[pid]);       obs_v.append(1.0)
            for pid in lineup_def:
                if pd.isna(pid): continue
                obs_r.append(idx); obs_c.append(p2i[pid] + n_p); obs_v.append(-1.0)
            y_v.append(pts / poss * 100)
            w_v.append(poss)
            idx += 1

    X  = csr_matrix((obs_v, (obs_r, obs_c)), shape=(idx, 2 * n_p))
    y  = np.array(y_v); w = np.array(w_v)
    ym = np.average(y, weights=w)
    sw = np.sqrt(w)
    ridge = Ridge(alpha=4000, fit_intercept=False)
    ridge.fit(X.multiply(sw[:, None]).tocsr(), (y - ym) * sw)

    rapm_raw = pd.DataFrame({"athlete_id": all_players,
                              "o_rapm": ridge.coef_[:n_p],
                              "d_rapm": ridge.coef_[n_p:],
                              "rapm":   ridge.coef_[:n_p] + ridge.coef_[n_p:]})
    poss_pl = (presence_full[presence_full["is_on_court"]]
               .groupby("athlete_id")["poss_off"].sum().reset_index()
               .rename(columns={"poss_off": "total_poss"}))
    pnames  = (plyr[["athlete_id", "athlete_display_name", "team_id", "team_display_name", "game_id"]]
               .sort_values("game_id").drop_duplicates("athlete_id", keep="last")
               .drop(columns="game_id"))
    rapm_df = (rapm_raw.merge(pnames, on="athlete_id", how="left")
               .merge(poss_pl, on="athlete_id", how="left")
               .query("total_poss >= 500")
               .sort_values("rapm", ascending=False).reset_index(drop=True))
    rapm_df["o_rapm"] = rapm_df["o_rapm"].round(2)
    rapm_df["d_rapm"] = rapm_df["d_rapm"].round(2)
    rapm_df["rapm"]   = rapm_df["o_rapm"] + rapm_df["d_rapm"]

    season_str = f"{SEASON-1}{str(SEASON)[2:]}"
    rfname = f"mbb_rapm_{season_str}{sfx}.csv"
    rapm_df.to_csv(rfname, index=False)
    print(f"  Saved {rfname} — {len(rapm_df)} qualified players")

    return presence_full, player_agg


# ── Run both passes ───────────────────────────────────────────────────────────
presence_all, player_agg_all = run_pipeline("all")
run_pipeline("conf")

# ── Save shared outputs (from the all-games run) ──────────────────────────────
SHOT_COLS = ["fga", "fgm", "tpa", "tpm", "fta", "ftm", "orb", "drb"]
save_cols = (["game_id", "stint_id", "athlete_id", "team_id", "is_on_court",
               "pts_for", "pts_against", "poss_off", "poss_def"]
             + [f"{c}_for" for c in SHOT_COLS]
             + [f"{c}_against" for c in SHOT_COLS])
presence_all[[c for c in save_cols if c in presence_all.columns]].to_parquet(
    "presence_full.parquet", index=False)

player_agg_all[["athlete_id", "athlete_display_name", "team_id", "team_display_name"]]\
    .drop_duplicates("athlete_id").reset_index(drop=True)\
    .to_csv("player_lookup.csv", index=False)

_ss = f"{SEASON-1}{str(SEASON)[2:]}"
print("\nDone. Files saved:")
print(f"  mbb_onoff_{SEASON}_v2.csv")
print(f"  mbb_onoff_{SEASON}_conf_v2.csv")
print(f"  mbb_rapm_{_ss}.csv")
print(f"  mbb_rapm_{_ss}_conf.csv")
print("  presence_full.parquet")
print("  player_lookup.csv")
