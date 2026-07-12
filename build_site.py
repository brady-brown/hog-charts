"""
build_site.py — Export artifacts to static JSON files for Hog Charts.

WHY THIS FILE EXISTS
--------------------
The Hog Charts site is a collection of static HTML/JS pages that load all data
via JavaScript fetch() calls.  This script reads the parquet/CSV artifacts
produced by build_artifacts.py, build_player_stats.py, build_lineups.py, and
build_onoff_rapm.py, and writes compact JSON files the browser can consume
without a server-side database.

Outputs (written into site/data/{SEASON}/):
    predictor.json          Teams + model coefficients for the game predictor page.
    net-ratings.json        Efficiency leaderboard for every D-I team.
    player-stats.json       Per-player overall season stats.
    player-stats-conf.json  Per-player conference-only stats.
    player-impact.json      RAPM + on/off impact hub (overall).
    player-impact-conf.json RAPM + on/off impact hub (conference only).
    lineup-index.json       Small index of teams + URL slugs for the dropdown.
    lineups/{slug}.json     One per team; 1/2/3/5-man lineup combo data.
    shots-meta.json         Zone stats, territory maps, player zones (shots page).
    site/data/shots/{slug}.json  Raw shot coordinates per team for interactive charts.
    site/data/seasons.json  List of all built season years (for the season dropdown).

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

# ---------------------------------------------------------------------------
# Season detection
# ---------------------------------------------------------------------------
_season_override = os.environ.get("OVERRIDE_SEASON")
if _season_override:
    SEASON = int(_season_override)
else:
    _today = _date.today()
    SEASON = _today.year + 1 if _today.month >= 11 else _today.year

PROJECT_ROOT    = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR   = os.path.join(PROJECT_ROOT, "artifacts", str(SEASON))
SEASON_DATA_DIR = os.path.join(PROJECT_ROOT, "site", "data", str(SEASON))   # per-season JSON output
SHOTS_DATA_DIR  = os.path.join(PROJECT_ROOT, "site", "data", "shots")       # current-season shot files
ROOT_DATA_DIR   = os.path.join(PROJECT_ROOT, "site", "data")                # parent of per-season dirs

os.makedirs(SEASON_DATA_DIR, exist_ok=True)
os.makedirs(SHOTS_DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shot zone definitions
# ---------------------------------------------------------------------------
# 14-wedge "2K-style" scheme: a restricted-area disk, then concentric rings
# (close mid / mid / three) sliced into angular wedges.  Index order MUST match
# the ZoneChart module and classifyZone() in site/js/zone-chart.js.
ZONE_NAMES = [
    "Restricted Area",                                              # 0
    "Close Mid - Right", "Close Mid - Center", "Close Mid - Left",  # 1-3
    "Mid - Right", "Mid - Right Center", "Mid - Center",            # 4-6
    "Mid - Left Center", "Mid - Left",                             # 7-8
    "3PT - Right", "3PT - Right Center", "3PT - Center",            # 9-11
    "3PT - Left Center", "3PT - Left",                             # 12-13
]
ZONE_NAME_TO_INDEX = {zone_name: idx for idx, zone_name in enumerate(ZONE_NAMES)}
THREE_POINT_ZONES  = {"3PT - Right", "3PT - Right Center", "3PT - Center",
                      "3PT - Left Center", "3PT - Left"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_for_json(python_obj):
    """Recursively replace NaN/Inf floats with None so JSON.parse never breaks.

    Python's json.dumps writes bare `NaN` and `Infinity` which are not valid
    JSON — the browser's JSON.parse will throw a SyntaxError and the page
    will stall on 'Loading…'.
    """
    if isinstance(python_obj, float):
        return None if (python_obj != python_obj
                        or python_obj == float("inf")
                        or python_obj == float("-inf")) else python_obj
    # pandas nullable scalars (pd.NA from Int64 columns, NaT) aren't floats and
    # aren't JSON-serializable — collapse them to null.
    if python_obj is pd.NA or python_obj is pd.NaT:
        return None
    if isinstance(python_obj, dict):
        return {key: _sanitize_for_json(val) for key, val in python_obj.items()}
    if isinstance(python_obj, list):
        return [_sanitize_for_json(item) for item in python_obj]
    return python_obj


def write_json(data_object, output_filename, output_directory=None):
    """Write a Python object as compact JSON; log filename and size."""
    output_path = os.path.join(output_directory or SEASON_DATA_DIR, output_filename)
    with open(output_path, "w") as json_file:
        json_file.write(json.dumps(_sanitize_for_json(data_object), separators=(",", ":")))
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"  {output_filename:<35s} {file_size_kb:7.0f} KB")


def slugify(display_name):
    """Convert a team/player name to a URL-safe slug (e.g. 'Arkansas Razorbacks' → 'arkansas-razorbacks')."""
    return re.sub(r"[^a-z0-9]+", "-", display_name.lower()).strip("-")


def get_data_freshness_date(season):
    """Return the ISO date of the most recent game in this season's data.

    Using the data date (not wall-clock time) means the JSON files stay
    byte-identical when no new games have arrived, so a nightly run with
    nothing new produces no diff and triggers no deploy.
    """
    for parquet_path, date_column in [
        (os.path.join(PROJECT_ROOT, f"shots_{season}.parquet"), "game_date"),
        (os.path.join(PROJECT_ROOT, "game_schedule.parquet"),   "date"),
    ]:
        if os.path.exists(parquet_path):
            try:
                date_series = pd.to_datetime(
                    pd.read_parquet(parquet_path, columns=[date_column])[date_column],
                    errors="coerce"
                )
                if date_series.notna().any():
                    return date_series.max().date().isoformat()
            except Exception:
                pass
    return f"{season}-04-30"    # stable fallback for historical seasons without shot data


def classify_shot_zones(shot_df):
    """Assign each shot to a zone based on its court coordinates.

    Coordinate convention (ESPN/sportsdataverse):
        coordinate_x: distance from center-court baseline (0 = baseline, 47 = halfcourt)
        coordinate_y: distance from court centerline (positive = left side, negative = right)

    14-wedge "2K-style" scheme.  Boundaries MUST match the SVG polygons and
    classifyZone() in site/js/zone-chart.js:
        rings    RA<4ft · close 4-11ft · mid 11ft-arc · three beyond arc
        3PT line college arc r=22.146ft, corner straights at |lat|=21.65ft
        angle    0deg = right baseline, 90deg = straight out, 180deg = left
    Returns a Series of zone name strings.
    """
    RA, CLOSE, THREE, CORNER_X = 4.0, 11.0, 22.146, 21.65
    Y_MEET = np.sqrt(THREE**2 - CORNER_X**2)          # ~4.658 ft, corner/arc junction

    valid_coords_mask = shot_df["coordinate_x"].notna() & shot_df["coordinate_y"].notna()
    zone_series = pd.Series("Unknown", index=shot_df.index, dtype="object")

    x_abs   = shot_df.loc[valid_coords_mask, "coordinate_x"].abs()
    y_coord = shot_df.loc[valid_coords_mask, "coordinate_y"]

    lateral  = -y_coord            # right side positive (matches screen x / svg +x)
    toward   = 41.75 - x_abs       # distance from basket toward halfcourt
    distance = np.sqrt(lateral**2 + toward**2)
    angle    = np.degrees(np.arctan2(toward, lateral))
    angle    = angle.where(angle >= -90, angle + 360)   # unwrap left-baseline corner

    is_heave = distance >= 40
    is_rim   = ~is_heave & (distance < RA)
    is_three = ~is_heave & ~is_rim & (
        (distance >= THREE) | ((lateral.abs() >= CORNER_X) & (toward <= Y_MEET))
    )
    is_close = ~is_heave & ~is_rim & ~is_three & (distance < CLOSE)
    is_mid   = ~is_heave & ~is_rim & ~is_three & ~is_close

    m = valid_coords_mask
    zone_series[m & is_heave] = "Heave"
    zone_series[m & is_rim]   = "Restricted Area"

    zone_series[m & is_close & (angle <  60)]                  = "Close Mid - Right"
    zone_series[m & is_close & (angle >= 60) & (angle < 120)]  = "Close Mid - Center"
    zone_series[m & is_close & (angle >= 120)]                 = "Close Mid - Left"

    zone_series[m & is_mid & (angle <  36)]                    = "Mid - Right"
    zone_series[m & is_mid & (angle >= 36)  & (angle < 72)]    = "Mid - Right Center"
    zone_series[m & is_mid & (angle >= 72)  & (angle < 108)]   = "Mid - Center"
    zone_series[m & is_mid & (angle >= 108) & (angle < 144)]   = "Mid - Left Center"
    zone_series[m & is_mid & (angle >= 144)]                   = "Mid - Left"

    zone_series[m & is_three & (angle <  36)]                  = "3PT - Right"
    zone_series[m & is_three & (angle >= 36)  & (angle < 72)]  = "3PT - Right Center"
    zone_series[m & is_three & (angle >= 72)  & (angle < 108)] = "3PT - Center"
    zone_series[m & is_three & (angle >= 108) & (angle < 144)] = "3PT - Left Center"
    zone_series[m & is_three & (angle >= 144)]                 = "3PT - Left"
    return zone_series


def compute_zone_baselines(field_goals_df):
    """NCAA league-average FG% for each zone index (fraction 0..1, or None).

    Feeds the relative-to-average colour scale in the browser so the placeholder
    family averages are replaced with the real season-wide numbers.
    """
    zone_stats = (
        field_goals_df.groupby("zone")
        .agg(makes=("scoring_play", "sum"), attempts=("scoring_play", "count"))
    )
    baselines = []
    for zone_name in ZONE_NAMES:
        if zone_name in zone_stats.index and int(zone_stats.loc[zone_name, "attempts"]) > 0:
            baselines.append(round(
                float(zone_stats.loc[zone_name, "makes"]) / float(zone_stats.loc[zone_name, "attempts"]), 4))
        else:
            baselines.append(None)
    return baselines


def build_zone_records(shot_group_df):
    """Aggregate makes and attempts by zone for one team or player.

    Returns a list of [zone_index, makes, attempts] triples — compact format
    for the browser since zone names are stored in the ZONE_NAMES array.
    """
    zone_stats = (
        shot_group_df.groupby("zone")
        .agg(makes=("scoring_play", "sum"), attempts=("scoring_play", "count"))
        .reset_index()
    )
    return [
        [ZONE_NAME_TO_INDEX[row["zone"]], int(row["makes"]), int(row["attempts"])]
        for _, row in zone_stats.iterrows()
        if row["zone"] in ZONE_NAME_TO_INDEX and int(row["attempts"]) > 0
    ]


# ---------------------------------------------------------------------------
# Load core artifacts
# ---------------------------------------------------------------------------
print("\nLoading artifacts…")
with open(os.path.join(ARTIFACTS_DIR, "metadata.json")) as metadata_file:
    build_metadata = json.load(metadata_file)

# Override the wall-clock build timestamp with the data freshness date.
# This keeps committed JSON byte-identical when no new games have arrived.
data_freshness_date = get_data_freshness_date(SEASON)
build_metadata["built_at"] = data_freshness_date

model_json_path = os.path.join(ARTIFACTS_DIR, "model.json")
model_json_data = json.load(open(model_json_path)) if os.path.exists(model_json_path) else None

teams_ratings_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "teams.parquet"))
net_ratings_raw_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "net_ratings.parquet"))


# ---------------------------------------------------------------------------
# Conference label lookup: team_display_name → conference abbreviation
# ---------------------------------------------------------------------------
current_player_stats_csv_path = os.path.join(PROJECT_ROOT, f"player_stats_{SEASON}.csv")
if os.path.exists(current_player_stats_csv_path):
    _player_stats_conf_col = pd.read_csv(current_player_stats_csv_path,
                                         usecols=["team_display_name", "conf."])
    conference_label_map = (
        _player_stats_conf_col.dropna()
        .drop_duplicates("team_display_name")
        .set_index("team_display_name")["conf."]
        .to_dict()
    )
else:
    conference_label_map = {}
    print(f"  [warn] player_stats_{SEASON}.csv not found — conf will be empty")


# ===========================================================================
# 1. predictor.json
# ===========================================================================
if model_json_data is not None:
    print("\nBuilding predictor.json…")
    model_json_data["built_at"] = data_freshness_date
    teams_ratings_df["conf"] = teams_ratings_df["team"].map(conference_label_map).fillna("")

    # Attach net-ratings ranks for the team search dropdowns.
    rank_columns = [c for c in ["team", "rank", "off_rank", "def_rank"] if c in net_ratings_raw_df.columns]
    teams_ratings_df = teams_ratings_df.merge(net_ratings_raw_df[rank_columns], on="team", how="left")

    for eff_col in ["net_eff", "off_eff", "def_eff", "form"]:
        if eff_col in teams_ratings_df.columns:
            teams_ratings_df[eff_col] = teams_ratings_df[eff_col].round(2)
    for display_col in ["pace", "home_adv"]:
        if display_col in teams_ratings_df.columns:
            teams_ratings_df[display_col] = teams_ratings_df[display_col].round(1)

    wanted_team_columns = [
        "team", "team_id", "conf", "net_eff", "off_eff", "def_eff",
        "pace", "home_adv", "form", "rank", "off_rank", "def_rank"
    ]
    teams_json_records = (
        teams_ratings_df[[c for c in wanted_team_columns if c in teams_ratings_df.columns]]
        .sort_values("team")
        .where(lambda df: ~df.isin([float("nan")]), other=None)
        .to_dict("records")
    )
    write_json({"model": model_json_data, "teams": teams_json_records, "meta": build_metadata},
               "predictor.json")
else:
    print("\nSkipping predictor.json (ratings-only build)")


# ===========================================================================
# 2. net-ratings.json
# ===========================================================================
print("\nBuilding net-ratings.json…")
net_ratings_df = net_ratings_raw_df.copy()
net_ratings_df["conf"] = net_ratings_df["team"].map(conference_label_map).fillna("")
if "wins" in net_ratings_df.columns and "losses" in net_ratings_df.columns:
    net_ratings_df["record"] = (
        net_ratings_df["wins"].astype(int).astype(str)
        + "–"
        + net_ratings_df["losses"].astype(int).astype(str)
    )
wanted_net_rating_columns = [
    "rank", "team", "team_id", "conf", "record", "games",
    "net_eff", "off_eff", "def_eff", "off_rank", "def_rank",
    "sos", "pace", "home_court", "form"
]
net_ratings_records = (
    net_ratings_df[[c for c in wanted_net_rating_columns if c in net_ratings_df.columns]]
    .to_dict("records")
)
write_json({"net_ratings": net_ratings_records, "meta": build_metadata}, "net-ratings.json")


# ===========================================================================
# 2b. team-stats-*.json  (powers the Team Stats page — the team analog of
#     player-stats: box + shooting + four-factor rates, plus the net ratings)
# ===========================================================================
print("\nBuilding team-stats JSONs…")

# Identity columns carried over from the net-ratings artifact. The actual
# Net/Off/Def/SOS ratings are NOT taken from here — each scope re-solves its own
# opponent-adjusted ratings (net_ratings_<scope>_<season>.csv from
# build_player_stats.py) so the toggle reflects just that scope's games.
# home_court stays season-overall (a venue property) and is blanked on the
# neutral-floor postseason scope inside build_team_stats_json().
TEAM_IDENTITY_COLS = ["team", "team_id", "conf", "record", "home_court"]
# Final ordered columns written to each team-stats JSON.
TEAM_STATS_OUTPUT_COLS = [
    "team", "team_id", "conf", "record", "rank",
    "net_eff", "off_eff", "def_eff", "off_rank", "def_rank", "sos", "home_court",
    "games", "pace",
    "ppg", "rpg", "apg", "spg", "bpg", "tpg",
    "fg", "fg3", "ft", "efg", "ts", "par3", "ftr",
    "opp_efg", "opp_fg3", "opp_par3", "opp_ftr",
    "tovp", "opp_tovp", "orbp", "drbp", "astp", "ast_to",
    "ppp", "opp_ppp",
]
team_identity_df = net_ratings_df[
    [c for c in TEAM_IDENTITY_COLS if c in net_ratings_df.columns]
].copy()


def _sdiv(numerator, denominator):
    """Element-wise divide → NaN where the denominator is 0 (sanitized later)."""
    return np.where(denominator > 0, numerator / denominator, np.nan)


def build_team_stats_json(csv_path, identity_df, net_ratings_csv_path, scope):
    """Compute the per-team stat table for one scope, merged with ratings.

    Driven by identity_df (the rated D-I teams) via an inner merge, so only
    leaderboard teams appear and they inherit team name / conf / record.
    Net/Off/Def/SOS come from this scope's own re-solved ratings file; ranks
    are recomputed within the scope.
    """
    box = pd.read_csv(csv_path)
    # Inner-merge: only rated D-I teams that actually have box data for this
    # scope (a team with no postseason games shouldn't appear in that scope).
    df = identity_df.merge(box, on="team_id", how="inner")

    # Scope-specific opponent-adjusted ratings (net/off/def/sos).
    if os.path.exists(net_ratings_csv_path):
        scope_net = pd.read_csv(net_ratings_csv_path)
        df = df.merge(scope_net[["team_id", "off_eff", "def_eff", "net_eff", "sos"]],
                      on="team_id", how="left")
    else:
        print(f"    (no {os.path.basename(net_ratings_csv_path)} — ratings left blank)")
        for col in ("off_eff", "def_eff", "net_eff", "sos"):
            df[col] = np.nan

    # home_court is a venue property — meaningless on neutral postseason floors.
    if scope == "post":
        df["home_court"] = np.nan

    g = df["games"]
    # Box per game
    for short, col in [("ppg", "pts"), ("rpg", "trb"), ("apg", "ast"),
                       ("spg", "stl"), ("bpg", "blk"), ("tpg", "tov")]:
        df[short] = np.round(_sdiv(df[col], g), 1)

    # Shooting — stored as 0-1 fractions (frontend ×100), like player-stats
    df["fg"]   = np.round(_sdiv(df["fgm"], df["fga"]), 3)
    df["fg3"]  = np.round(_sdiv(df["tpm"], df["tpa"]), 3)
    df["ft"]   = np.round(_sdiv(df["ftm"], df["fta"]), 3)
    df["efg"]  = np.round(_sdiv(df["fgm"] + 0.5 * df["tpm"], df["fga"]), 3)
    df["ts"]   = np.round(_sdiv(df["pts"], 2 * (df["fga"] + 0.44 * df["fta"])), 3)
    df["par3"] = np.round(_sdiv(df["tpa"], df["fga"]), 3)
    df["ftr"]  = np.round(_sdiv(df["fta"], df["fga"]), 3)
    # Defensive shooting allowed
    df["opp_efg"]  = np.round(_sdiv(df["opp_fgm"] + 0.5 * df["opp_tpm"], df["opp_fga"]), 3)
    df["opp_fg3"]  = np.round(_sdiv(df["opp_tpm"], df["opp_tpa"]), 3)
    df["opp_par3"] = np.round(_sdiv(df["opp_tpa"], df["opp_fga"]), 3)
    df["opp_ftr"]  = np.round(_sdiv(df["opp_fta"], df["opp_fga"]), 3)

    # Four-factor / rate stats — stored as percentages already (no ×100)
    df["tovp"]     = np.round(100 * _sdiv(df["tov"], df["poss"]), 1)
    df["opp_tovp"] = np.round(100 * _sdiv(df["opp_tov"], df["opp_poss"]), 1)
    df["orbp"]     = np.round(100 * _sdiv(df["orb"], df["orb"] + df["opp_drb"]), 1)
    df["drbp"]     = np.round(100 * _sdiv(df["drb"], df["drb"] + df["opp_orb"]), 1)
    df["astp"]     = np.round(100 * _sdiv(df["ast"], df["fgm"]), 1)
    df["ast_to"]   = np.round(_sdiv(df["ast"], df["tov"]), 2)

    # Efficiency from the box (points per 100 possessions) + scope-accurate pace
    df["ppp"]     = np.round(100 * _sdiv(df["pts"], df["poss"]), 1)
    df["opp_ppp"] = np.round(100 * _sdiv(df["opp_pts"], df["opp_poss"]), 1)
    df["pace"]    = np.round(_sdiv(df["poss"] + df["opp_poss"], 2 * g), 1)

    df["games"] = g.astype("Int64")

    # Ranks are recomputed within the scope from its own re-solved ratings.
    df["rank"]     = df["net_eff"].rank(ascending=False, method="min").astype("Int64")
    df["off_rank"] = df["off_eff"].rank(ascending=False, method="min").astype("Int64")
    df["def_rank"] = df["def_eff"].rank(ascending=True,  method="min").astype("Int64")

    return df[[c for c in TEAM_STATS_OUTPUT_COLS if c in df.columns]].to_dict("records")


TEAM_STATS_SCOPE_SPECS = [
    ("all",  f"team_stats_all_{SEASON}.csv",  f"net_ratings_all_{SEASON}.csv",  "team-stats-all.json"),
    ("reg",  f"team_stats_{SEASON}.csv",      f"net_ratings_reg_{SEASON}.csv",  "team-stats.json"),
    ("post", f"team_stats_post_{SEASON}.csv", f"net_ratings_post_{SEASON}.csv", "team-stats-post.json"),
    ("conf", f"team_stats_conf_{SEASON}.csv", f"net_ratings_conf_{SEASON}.csv", "team-stats-conf.json"),
]
for scope, csv_name, net_name, json_name in TEAM_STATS_SCOPE_SPECS:
    csv_full = os.path.join(PROJECT_ROOT, csv_name)
    if not os.path.exists(csv_full):
        print(f"  {json_name:<35s} skipped (no {csv_name})")
        continue
    net_full = os.path.join(PROJECT_ROOT, net_name)
    records = build_team_stats_json(csv_full, team_identity_df, net_full, scope)
    write_json({"team_stats": records, "meta": build_metadata}, json_name)


# ===========================================================================
# 3a. Fetch player bios from ESPN roster endpoint
# ===========================================================================
ESPN_ROSTER_URL_TEMPLATE = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/teams/{team_id}/roster"
)


def _fetch_single_team_bios(team_id):
    """Fetch height, weight, class year, hometown, jersey number for all rostered players.

    Returns {str(athlete_id): {ht, wt, exp, hw, jn}}.
    Empty dict on any network or parsing error.
    """
    try:
        response = requests.get(ESPN_ROSTER_URL_TEMPLATE.format(team_id=int(team_id)), timeout=10)
        if response.status_code != 200:
            return {}
        athlete_list = response.json().get("athletes", [])
        bio_by_athlete_id = {}
        for athlete in athlete_list:
            athlete_id_str = str(athlete.get("id", ""))
            if not athlete_id_str:
                continue
            birth_place = athlete.get("birthPlace") or {}
            bio_by_athlete_id[athlete_id_str] = {
                "ht":  athlete.get("displayHeight"),
                "wt":  athlete.get("displayWeight"),
                "exp": (athlete.get("experience") or {}).get("displayValue"),
                "hw":  birth_place.get("displayText"),
                "jn":  athlete.get("jersey"),
            }
        return bio_by_athlete_id
    except Exception:
        return {}


def fetch_all_player_bios(player_stats_csv_path):
    """Concurrently fetch ESPN rosters for every team in the player-stats CSV.

    Returns {str(athlete_id): bio_dict} across all teams.
    """
    if not os.path.exists(player_stats_csv_path):
        return {}
    all_team_ids = pd.read_csv(player_stats_csv_path, usecols=["team_id"])["team_id"].dropna().unique()
    print(f"  Fetching ESPN rosters for {len(all_team_ids)} teams…")
    bios_by_athlete_id = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_team = {executor.submit(_fetch_single_team_bios, tid): tid for tid in all_team_ids}
        teams_done = 0
        for completed_future in as_completed(future_to_team):
            bios_by_athlete_id.update(completed_future.result())
            teams_done += 1
            if teams_done % 100 == 0:
                print(f"    {teams_done}/{len(all_team_ids)} teams fetched…")
    print(f"  → {len(bios_by_athlete_id)} player bios collected")
    return bios_by_athlete_id


# ===========================================================================
# 3. player-stats.json  +  player-stats-conf.json
# ===========================================================================
print("\nBuilding player-stats JSON files…")

# Short key mapping: saves ~40% payload size vs full column names.
PLAYER_STATS_COLUMN_RENAME_MAP = {
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
    "fg3m_pg":               "fg3m",
    "fg3a_pg":               "fg3a",
    "fg2m_pg":               "fg2m",
    "fg2a_pg":               "fg2a",
    "fgm_pg":                "fgmpg",
    "fga_pg":                "fgapg",
    "ftm_pg":                "ftmpg",
    "fta_pg":                "ftapg",
    "efg_pct":               "efg",
    "ts_pct":                "ts",
    "3par":                  "par3",
    "ftr":                   "ftr",
    "usg":                   "usg",
    "ast_pct":               "astp",
    "tov_pct":               "tovp",
    "orb_pct":               "orbp",
    "drb_pct":               "drbp",
    "trb_pct":               "trbp",
    "stl_pct":               "stlp",
    "blk_pct":               "blkp",
    "ast_share":             "astdp",
    "adv_src":               "advsrc",
    "bpm":                   "bpm",
    "on_off":                "on_off",
    "resp_pct":              "resp",
}


def build_player_stats_json(csv_path, onoff_csv_path, bios_dict=None,
                             min_games_played=8, min_mpg=8,
                             min_onoff_possessions=200, label="",
                             team_context_csv_path=None, stint_csv_path=None,
                             assist_share_csv_path=None, min_team_games=0,
                             points_resp_csv_path=None):
    """Read a player-stats CSV, compute derived stats, and return JSON records.

    Derived stats added here (not in build_player_stats.py):
        ts_pct  True Shooting % = pts / (2 * (FGA + 0.44 * FTA))
        3par    Three-point attempt rate = 3PA / FGA
        ftr     Free-throw rate = FTA / FGA
        mpg     Minutes per game
        usg     Usage rate (% of team possessions used while on court)
        ast_pct Assist % = share of teammate field goals a player assisted on court
        tov_pct Turnover % = turnovers per 100 individual possessions used
        orb_pct Offensive-rebound % of available offensive boards while on court
        drb_pct Defensive-rebound % of available defensive boards while on court
        trb_pct Total-rebound % of all available boards while on court
        stl_pct Steal % of opponent possessions ended by a steal while on court
        blk_pct Block % of opponent two-point attempts blocked while on court
        bpm     Box Plus/Minus (a Hollinger Game Score proxy, centered on 0 = average)
        on_off  Net rating on-court minus net rating off-court (from RAPM pipeline)

    The rebound/steal/block rates need opponent totals (team_context_csv_path,
    keyed by team_id) since they compare against what opponents generated.
    """
    if not os.path.exists(csv_path):
        print(f"  [warn] {csv_path} not found — skipped")
        return None

    player_stats_df = pd.read_csv(csv_path)

    # Raw counting stat columns.
    field_goals_attempted       = player_stats_df["field_goals_attempted"]
    three_point_field_goals_att = player_stats_df["three_point_field_goals_attempted"]
    free_throws_attempted       = player_stats_df["free_throws_attempted"]
    turnovers                   = player_stats_df["turnovers"]
    total_points                = player_stats_df["points"]
    games_played                = player_stats_df["games_played"]
    total_minutes               = player_stats_df["minutes"]

    # --- Derived shooting metrics ---
    player_stats_df["ts_pct"] = np.where(
        field_goals_attempted + 0.44 * free_throws_attempted > 0,
        total_points / (2 * (field_goals_attempted + 0.44 * free_throws_attempted)),
        None
    )
    player_stats_df["3par"] = np.where(
        field_goals_attempted > 0, three_point_field_goals_att / field_goals_attempted, None
    )
    player_stats_df["ftr"] = np.where(
        field_goals_attempted > 0, free_throws_attempted / field_goals_attempted, None
    )
    player_stats_df["mpg"] = np.where(games_played > 0, total_minutes / games_played, None)

    # --- Per-game shooting volume (for stackable volume filters) ---
    # 3PM/3PA/FGM/FGA per game already exist as *_avg in the player CSV; 2-point
    # volume = field-goal volume minus 3-point volume. FT volume is per-game from
    # the season totals. These let users isolate, e.g., high-volume shooters.
    def _per_game(total_col):
        return np.where(games_played > 0, player_stats_df[total_col] / games_played, None)

    three_made_pg = player_stats_df.get("3ptm_avg")
    three_att_pg  = player_stats_df.get("3pta_avg")
    fg_made_pg    = player_stats_df.get("fgm_avg")
    fg_att_pg     = player_stats_df.get("fga_avg")
    if three_made_pg is not None and fg_made_pg is not None:
        player_stats_df["fg3m_pg"] = three_made_pg
        player_stats_df["fg3a_pg"] = three_att_pg
        player_stats_df["fg2m_pg"] = fg_made_pg - three_made_pg
        player_stats_df["fg2a_pg"] = fg_att_pg - three_att_pg
        player_stats_df["fgm_pg"]  = fg_made_pg
        player_stats_df["fga_pg"]  = fg_att_pg
    player_stats_df["ftm_pg"] = _per_game("free_throws_made")
    player_stats_df["fta_pg"] = _per_game("free_throws_attempted")

    # --- Usage rate ---
    # Usage% ≈ 100 * player_possessions_used * (team_minutes / 5) / (player_minutes * team_possessions)
    team_totals_df = (
        player_stats_df.groupby("team_display_name")
        .agg(
            team_total_fga=("field_goals_attempted", "sum"),
            team_total_fta=("free_throws_attempted",  "sum"),
            team_total_tov=("turnovers",              "sum"),
            team_total_min=("minutes",                "sum"),
            team_total_fgm=("field_goals_made",       "sum"),
            team_total_orb=("offensive_rebounds",     "sum"),
            team_total_drb=("defensive_rebounds",     "sum"),
            team_total_trb=("rebounds",               "sum"),
        ).reset_index()
    )
    player_stats_df = player_stats_df.merge(team_totals_df, on="team_display_name", how="left")
    team_possessions  = player_stats_df["team_total_fga"] + 0.44 * player_stats_df["team_total_fta"] + player_stats_df["team_total_tov"]
    player_possessions = field_goals_attempted + 0.44 * free_throws_attempted + turnovers
    player_stats_df["usg"] = np.where(
        (total_minutes > 0) & (team_possessions > 0),
        100 * player_possessions * (player_stats_df["team_total_min"] / 5) / (total_minutes * team_possessions),
        None
    )

    # --- Advanced rate percentages (Basketball-Reference style) ---
    # The (team_minutes / 5) factor below is the team's total game-minutes, i.e.
    # one full-time player's worth of minutes; dividing player minutes by it
    # scales each rate to the share of team play the player was on the floor for.
    assists          = player_stats_df["assists"]
    steals           = player_stats_df["steals"]
    blocks           = player_stats_df["blocks"]
    off_rebounds     = player_stats_df["offensive_rebounds"]
    def_rebounds     = player_stats_df["defensive_rebounds"]
    tot_rebounds     = player_stats_df["rebounds"]
    player_fgm       = player_stats_df["field_goals_made"]
    team_minutes_full = player_stats_df["team_total_min"] / 5   # team game-minutes

    # Turnover % — fully individual: turnovers per 100 possessions used.
    player_stats_df["tov_pct"] = np.where(
        player_possessions > 0, 100 * turnovers / player_possessions, None
    )

    # Assist % — share of teammates' made FGs the player assisted while on court.
    teammate_fgm_on_court = (
        (total_minutes / team_minutes_full) * player_stats_df["team_total_fgm"]
        - player_fgm
    )
    player_stats_df["ast_pct"] = np.where(
        (total_minutes > 0) & (teammate_fgm_on_court > 0),
        100 * assists / teammate_fgm_on_court, None
    )

    # Rebound / steal / block rates need opponent totals (team context).
    if team_context_csv_path and os.path.exists(team_context_csv_path):
        team_context_df = pd.read_csv(team_context_csv_path)
        player_stats_df = player_stats_df.merge(team_context_df, on="team_id", how="left")

        opp_orb  = player_stats_df["opp_orb"]
        opp_drb  = player_stats_df["opp_drb"]
        opp_trb  = player_stats_df["opp_trb"]
        opp_fga  = player_stats_df["opp_fga"]
        opp_3pa  = player_stats_df["opp_3pa"]
        opp_poss = player_stats_df["opp_poss"]

        def rate_pct(numerator_count, available_total):
            """100 · (count · team_minutes / player_minutes) / available_total."""
            return np.where(
                (total_minutes > 0) & (available_total > 0),
                100 * numerator_count * team_minutes_full / (total_minutes * available_total),
                None,
            )

        player_stats_df["orb_pct"] = rate_pct(off_rebounds, player_stats_df["team_total_orb"] + opp_drb)
        player_stats_df["drb_pct"] = rate_pct(def_rebounds, player_stats_df["team_total_drb"] + opp_orb)
        player_stats_df["trb_pct"] = rate_pct(tot_rebounds, player_stats_df["team_total_trb"] + opp_trb)
        player_stats_df["stl_pct"] = rate_pct(steals, opp_poss)
        player_stats_df["blk_pct"] = rate_pct(blocks, opp_fga - opp_3pa)
    else:
        for missing_col in ["orb_pct", "drb_pct", "trb_pct", "stl_pct", "blk_pct"]:
            player_stats_df[missing_col] = None
        if team_context_csv_path:
            print(f"  [warn] {team_context_csv_path} not found — rebound/steal/block% blank")

    # --- Stint-based advanced rates (preferred where lineup data supports them) ---
    # The on/off CSV carries each player's ON-COURT team and opponent box totals,
    # accumulated over the exact stints they played. That lets AST%/STL%/BLK% and
    # the rebound rates be computed against what actually happened on the floor
    # instead of season team totals. Used only where (a) the stint file matches
    # this scope's games (regular & conference — NOT all/post, whose box totals
    # include games the regular-season stint data doesn't) and (b) the player has
    # enough on-court possessions. Everyone else keeps the season approximation.
    STINT_COLS = ["poss_off_on", "poss_def_on", "fgm_for_on", "orb_for_on", "drb_for_on",
                  "fga_against_on", "tpa_against_on", "orb_against_on", "drb_against_on"]
    player_stats_df["adv_src"] = "season"
    if stint_csv_path and os.path.exists(stint_csv_path) \
            and all(c in pd.read_csv(stint_csv_path, nrows=0).columns for c in STINT_COLS):
        stint_df = pd.read_csv(stint_csv_path, usecols=["athlete_id", *STINT_COLS])
        # A player who changed teams mid-season has two on/off rows; keep the
        # higher-possession one so the merge stays one row per athlete (otherwise
        # it would duplicate player rows and break downstream length assumptions).
        stint_df["_tot_poss"] = stint_df["poss_off_on"].fillna(0) + stint_df["poss_def_on"].fillna(0)
        stint_df = (stint_df.sort_values("_tot_poss", ascending=False)
                    .drop_duplicates("athlete_id").drop(columns="_tot_poss"))
        player_stats_df = player_stats_df.merge(stint_df, on="athlete_id", how="left")
        on_court_poss = player_stats_df["poss_off_on"].fillna(0) + player_stats_df["poss_def_on"].fillna(0)
        has_stint = on_court_poss >= min_onoff_possessions

        teammate_fgm_floor = player_stats_df["fgm_for_on"] - player_stats_df["field_goals_made"]
        opp_two_pt_att     = player_stats_df["fga_against_on"] - player_stats_df["tpa_against_on"]
        available_orb      = player_stats_df["orb_for_on"] + player_stats_df["drb_against_on"]
        available_drb      = player_stats_df["drb_for_on"] + player_stats_df["orb_against_on"]

        def apply_stint(season_col, numerator, denominator):
            """Override the season estimate with the on-court rate where valid."""
            stint_val = np.where(has_stint & (denominator > 0),
                                 100 * numerator / denominator.where(denominator > 0, 1), np.nan)
            player_stats_df[season_col] = np.where(
                np.isfinite(stint_val), stint_val, player_stats_df[season_col])

        apply_stint("ast_pct", player_stats_df["assists"],             teammate_fgm_floor)
        apply_stint("stl_pct", player_stats_df["steals"],             player_stats_df["poss_def_on"])
        apply_stint("blk_pct", player_stats_df["blocks"],             opp_two_pt_att)
        apply_stint("orb_pct", player_stats_df["offensive_rebounds"], available_orb)
        apply_stint("drb_pct", player_stats_df["defensive_rebounds"], available_drb)
        apply_stint("trb_pct", player_stats_df["rebounds"],           available_orb + available_drb)
        player_stats_df.loc[has_stint, "adv_src"] = "stint"
        print(f"  stint-based advanced rates for {int(has_stint.sum())} players {label}")

    # --- Assisted-FG share (% of a player's own made FGs that were assisted) ---
    # Counted straight from play-by-play in build_onoff_rapm.py (astd_fgm /
    # fgm_pbp), one table per scope including postseason. Purely individual — no
    # stint/lineup coverage or possession gate needed.
    player_stats_df["ast_share"] = None
    if assist_share_csv_path and os.path.exists(assist_share_csv_path):
        share_df = pd.read_csv(assist_share_csv_path, usecols=["athlete_id", "astd_fgm", "fgm_pbp"])
        # A mid-season transfer can have two rows; keep the one with more made FGs.
        share_df = (share_df.sort_values("fgm_pbp", ascending=False)
                    .drop_duplicates("athlete_id"))
        player_stats_df = player_stats_df.merge(share_df, on="athlete_id", how="left")
        made_fg_pbp = player_stats_df["fgm_pbp"].fillna(0)
        player_stats_df["ast_share"] = np.where(
            made_fg_pbp > 0, 100 * player_stats_df["astd_fgm"] / made_fg_pbp.where(made_fg_pbp > 0, 1), None)

    # --- Box Plus-Minus (transparent proxy, not regressed BPM) ---
    # Scaled Game Score per 100 team possessions, centered on the minutes-weighted
    # league average so 0.0 ≈ an average rotation player.
    field_goals_made     = player_stats_df["field_goals_made"]
    free_throws_made     = player_stats_df["free_throws_made"]
    offensive_rebounds   = player_stats_df["offensive_rebounds"]
    defensive_rebounds   = player_stats_df["defensive_rebounds"]
    assists              = player_stats_df["assists"]
    steals               = player_stats_df["steals"]
    blocks               = player_stats_df["blocks"]
    personal_fouls       = player_stats_df["fouls"]

    hollinger_gamescore = (
        total_points + 0.4 * field_goals_made - 0.7 * field_goals_attempted
        - 0.4 * (free_throws_attempted - free_throws_made)
        + 0.7 * offensive_rebounds + 0.3 * defensive_rebounds
        + steals + 0.7 * assists + 0.7 * blocks - 0.4 * personal_fouls - turnovers
    )
    # Approximate on-court team possessions for this player's minutes.
    player_on_court_team_possessions = np.where(
        player_stats_df["team_total_min"] > 0,
        team_possessions * total_minutes / (player_stats_df["team_total_min"] / 5),
        np.nan
    )
    gamescore_per_100_team_poss = np.where(
        player_on_court_team_possessions > 0,
        100 * hollinger_gamescore / player_on_court_team_possessions,
        np.nan
    )
    # Weighted league average for centering.
    is_valid_bpm = np.isfinite(gamescore_per_100_team_poss) & (total_minutes.values > 0)
    league_avg_gamescore_per100 = (
        float(np.average(gamescore_per_100_team_poss[is_valid_bpm],
                         weights=total_minutes.values[is_valid_bpm]))
        if is_valid_bpm.any() else 0.0
    )
    player_stats_df["bpm"] = np.where(
        np.isfinite(gamescore_per_100_team_poss),
        gamescore_per_100_team_poss - league_avg_gamescore_per100,
        None
    )

    # --- On/off (from RAPM pipeline) ---
    if onoff_csv_path and os.path.exists(onoff_csv_path):
        on_off_df = pd.read_csv(onoff_csv_path, usecols=["athlete_id", "on_off", "poss_off_on"])
        on_off_df = on_off_df[on_off_df["poss_off_on"] >= min_onoff_possessions][["athlete_id", "on_off"]]
        player_stats_df = player_stats_df.merge(on_off_df, on="athlete_id", how="left")
    else:
        player_stats_df["on_off"] = None

    # --- Points responsible (regular-season on-court scored+assisted share) ---
    # Precomputed by build_points_resp.py; a season-level rate, so it is merged
    # onto the regular and all-games scopes (they share the same player pool).
    if points_resp_csv_path and os.path.exists(points_resp_csv_path):
        resp_df = pd.read_csv(points_resp_csv_path,
                              usecols=["athlete_id", "team_id", "resp_pct"])
        player_stats_df = player_stats_df.merge(
            resp_df, on=["athlete_id", "team_id"], how="left")
    else:
        player_stats_df["resp_pct"] = None

    # --- Filter ---
    # `min_team_games` keeps every player whose TEAM appeared in >= N games
    # (a team's game count ≈ the most games any one of its players played),
    # which drops non-D-I opponents that only surface a handful of times while
    # still showing deep-bench players on real teams. Per-player min_gp/min_mpg
    # stay available but are set to 0 for the scopes that use the team gate.
    team_games = player_stats_df.groupby("team_id")["games_played"].transform("max")
    player_stats_df = player_stats_df[
        (player_stats_df["games_played"] >= min_games_played)
        & (player_stats_df["mpg"].fillna(0) >= min_mpg)
        & (team_games.fillna(0) >= min_team_games)
    ].copy()

    # --- Round ---
    shooting_pct_cols = ["ts_pct", "3par", "ftr", "efg_pct", "fg_pct", "3pt_pct", "ft_pct", "resp_pct"]
    rate_avg_cols     = ["mpg", "points_avg", "reb_avg", "ast_avg", "steal_avg",
                         "blocks_avg", "to_avg", "on_off", "usg", "bpm",
                         "ast_pct", "tov_pct", "orb_pct", "drb_pct", "trb_pct",
                         "stl_pct", "blk_pct", "ast_share",
                         "fg3m_pg", "fg3a_pg", "fg2m_pg", "fg2a_pg",
                         "fgm_pg", "fga_pg", "ftm_pg", "fta_pg"]
    for col in shooting_pct_cols:
        if col in player_stats_df.columns:
            player_stats_df[col] = pd.to_numeric(player_stats_df[col], errors="coerce").round(3)
    for col in rate_avg_cols:
        if col in player_stats_df.columns:
            player_stats_df[col] = pd.to_numeric(player_stats_df[col], errors="coerce").round(1)

    # --- Select and rename ---
    columns_to_keep = [c for c in PLAYER_STATS_COLUMN_RENAME_MAP if c in player_stats_df.columns]
    player_stats_output_df = player_stats_df[columns_to_keep].rename(columns=PLAYER_STATS_COLUMN_RENAME_MAP)
    json_records = player_stats_output_df.where(player_stats_output_df.notna(), other=None).to_dict("records")

    # --- Merge bio fields ---
    if bios_dict:
        for player_record in json_records:
            player_athlete_id_str = (str(int(player_record["id"]))
                                     if player_record.get("id") is not None else "")
            bio = bios_dict.get(player_athlete_id_str, {})
            player_record["ht"]  = bio.get("ht")
            player_record["wt"]  = bio.get("wt")
            player_record["exp"] = bio.get("exp")
            player_record["hw"]  = bio.get("hw")
            player_record["jn"]  = bio.get("jn")

    print(f"  → {len(json_records)} players {label}")
    return json_records


def load_boxscore_jerseys(season):
    """{str(athlete_id): jersey} from the player boxscore — season-accurate.

    The roster-endpoint bios (fetch_all_player_bios) return the CURRENT roster,
    so any player who has since left comes back with jn=None. The player
    boxscore carries the jersey worn in each game this season (~99% coverage),
    so it fills those gaps and corrects mid-fetch roster churn.
    """
    try:
        import sportsdataverse.mbb as mbb
        pbox = mbb.load_mbb_player_boxscore(seasons=[season]).to_pandas()
        j = pbox.dropna(subset=["athlete_id", "athlete_jersey"]).copy()
        j["athlete_id"] = j["athlete_id"].astype(float).astype("int64").astype(str)
        j["athlete_jersey"] = j["athlete_jersey"].astype(str)
        return (j.groupby("athlete_id")["athlete_jersey"]
                  .agg(lambda s: s.mode().iloc[0]).to_dict())
    except Exception as exc:
        print(f"  [warn] boxscore jerseys unavailable: {exc}")
        return {}


all_player_bios = fetch_all_player_bios(current_player_stats_csv_path)

# Overlay season-accurate jerseys from the boxscore (authoritative for the
# season) over the roster-endpoint values, which miss departed players.
_boxscore_jerseys = load_boxscore_jerseys(SEASON)
for _aid, _jersey in _boxscore_jerseys.items():
    all_player_bios.setdefault(_aid, {})["jn"] = _jersey
print(f"  → jerseys overlaid from boxscore for {len(_boxscore_jerseys)} players")

# One JSON per scope. on/off and RAPM come from the regular-season pipeline, so
# the overall on/off CSV is attached to reg + all (it's a season metric); the
# postseason scope has no on/off (no postseason RAPM pipeline).
onoff_overall_csv = os.path.join(PROJECT_ROOT, f"mbb_onoff_{SEASON}_v2.csv")
onoff_conf_csv    = os.path.join(PROJECT_ROOT, f"mbb_onoff_{SEASON}_conf_v2.csv")
_exists = lambda path: path if os.path.exists(path) else None

# stint_csv supplies on-court box totals for the stint-based advanced rates; it
# may only be used where its games match the scope (regular & conference), so
# all/post pass None and fall back to the season approximation.
# Points-responsible on-court rate (build_points_resp.py): regular-season table
# for the regular + all-games scopes; a conference-games table for the conf scope.
points_resp_csv      = os.path.join(PROJECT_ROOT, f"points_resp_{SEASON}.csv")
points_resp_conf_csv = os.path.join(PROJECT_ROOT, f"points_resp_conf_{SEASON}.csv")

PLAYER_STATS_SCOPE_SPECS = [
    # (stats_csv, onoff_csv, stint_csv, team_context_csv, assist_share_csv,
    #  min_gp, min_mpg, min_onoff_poss, min_team_games, points_resp_csv, output_json, label)
    # Regular & all-games: no per-player minimum — show every player whose team
    # played >= 20 games (drops non-D-I opponents, keeps deep-bench players).
    # Conf/post keep per-player minimums (their team game counts are too low for
    # a 20-game team gate, so it stays 0 there).
    (f"player_stats_{SEASON}.csv",      onoff_overall_csv, onoff_overall_csv, f"team_context_{SEASON}.csv",      f"assist_share_{SEASON}.csv",
     0, 0, 200, 20, points_resp_csv, "player-stats.json",      "(regular)"),
    (f"player_stats_all_{SEASON}.csv",  onoff_overall_csv, None,              f"team_context_all_{SEASON}.csv",  f"assist_share_{SEASON}_all.csv",
     0, 0, 200, 20, points_resp_csv, "player-stats-all.json",  "(all games)"),
    (f"player_stats_post_{SEASON}.csv", None,              None,              f"team_context_post_{SEASON}.csv", f"assist_share_{SEASON}_post.csv",
     1, 1, 200, 0, None, "player-stats-post.json", "(postseason)"),
    (f"player_stats_conf_{SEASON}.csv", onoff_conf_csv,    onoff_conf_csv,    f"team_context_conf_{SEASON}.csv", f"assist_share_{SEASON}_conf.csv",
     4, 8, 100, 0, points_resp_conf_csv, "player-stats-conf.json", "(conference)"),
]

for stats_csv_name, onoff_csv, stint_csv, team_ctx_name, assist_share_csv, min_gp, min_mpg, min_op, min_tg, points_resp_path, out_json, label in PLAYER_STATS_SCOPE_SPECS:
    records = build_player_stats_json(
        os.path.join(PROJECT_ROOT, stats_csv_name),
        _exists(onoff_csv) if onoff_csv else None,
        bios_dict=all_player_bios, min_games_played=min_gp, min_mpg=min_mpg,
        min_onoff_possessions=min_op, label=label, min_team_games=min_tg,
        team_context_csv_path=os.path.join(PROJECT_ROOT, team_ctx_name),
        stint_csv_path=_exists(stint_csv) if stint_csv else None,
        assist_share_csv_path=_exists(os.path.join(PROJECT_ROOT, assist_share_csv)),
        points_resp_csv_path=points_resp_path,
    )
    if records is not None:
        write_json({"players": records, "meta": build_metadata}, out_json)


# ===========================================================================
# 3c. player-impact.json  +  player-impact-conf.json  (RAPM + on/off)
# ===========================================================================
print("\nBuilding player-impact JSON files…")
from impact_artifact import build_impact_records

overall_impact_records = build_impact_records(
    PROJECT_ROOT, SEASON, conf_map=conference_label_map, conf_variant=False, min_possessions=400
)
if overall_impact_records is not None:
    write_json({"players": overall_impact_records, "meta": build_metadata}, "player-impact.json")
    print(f"  → {len(overall_impact_records)} players (overall)")
else:
    print(f"  [warn] no RAPM file for {SEASON} — player-impact skipped")

conference_impact_records = build_impact_records(
    PROJECT_ROOT, SEASON, conf_map=conference_label_map, conf_variant=True, min_possessions=200
)
if conference_impact_records is not None:
    write_json({"players": conference_impact_records, "meta": build_metadata}, "player-impact-conf.json")
    print(f"  → {len(conference_impact_records)} players (conference)")


# ===========================================================================
# 4. lineup-stats.json  (per-team lineup combo files + index)
# ===========================================================================
print("\nBuilding lineup files…")
LINEUP_COLUMNS_TO_KEEP = [
    "Combo", "Mins", "Avg_Poss", "NetRtg", "ORtg", "DRtg",
    "AST_100", "TOV_100", "REB_100", "STL_100", "BLK_100"
]
# Minimum avg possessions for a combo to appear (to avoid tiny sample noise).
LINEUP_MIN_POSS_OVERALL  = {"1": 100, "2": 75, "3": 75, "5": 25}
LINEUP_MIN_POSS_CONF     = {"1": 50,  "2": 40, "3": 40, "5": 15}
LINEUP_ROUND_COLS = [
    "Mins", "Avg_Poss", "NetRtg", "ORtg", "DRtg",
    "AST_100", "TOV_100", "REB_100", "STL_100", "BLK_100"
]


def load_lineup_csv(combo_size, game_scope, min_avg_possessions):
    """Load one of the 8 lineup CSVs and filter to minimum possessions.

    Returns (dict_by_team, set_of_team_names).
    """
    scope_label  = "overall" if game_scope == "all" else "conference"
    csv_filename = os.path.join(PROJECT_ROOT, f"{combo_size}_man_{scope_label}_stats_{SEASON}.csv")
    if not os.path.exists(csv_filename):
        return {}, set()

    lineup_df = pd.read_csv(csv_filename)
    lineup_df = lineup_df[lineup_df["Avg_Poss"] >= min_avg_possessions].copy()
    columns_present    = [c for c in LINEUP_COLUMNS_TO_KEEP if c in lineup_df.columns]
    round_cols_present = [c for c in LINEUP_ROUND_COLS      if c in lineup_df.columns]
    lineup_df[round_cols_present] = lineup_df[round_cols_present].round(1)

    combos_by_team = {}
    team_names_seen = set()
    for team_name, team_group in lineup_df.groupby("Team"):
        team_names_seen.add(team_name)
        combos_by_team[str(team_name)] = (
            team_group[columns_present].sort_values("NetRtg", ascending=False).to_dict("records")
        )
    return combos_by_team, team_names_seen


all_team_names_across_sizes = set()
overall_combos_by_size      = {}    # {"1": {team: [records]}, "2": ..., ...}
conference_combos_by_size   = {}

for combo_size in ["1", "2", "3", "5"]:
    overall_combos, overall_teams = load_lineup_csv(combo_size, "all",  LINEUP_MIN_POSS_OVERALL[combo_size])
    overall_combos_by_size[combo_size] = overall_combos
    all_team_names_across_sizes |= overall_teams
    print(f"  {combo_size}-man overall:     "
          f"{sum(len(v) for v in overall_combos.values()):5d} combos, {len(overall_combos)} teams")

    conf_combos, _ = load_lineup_csv(combo_size, "conf", LINEUP_MIN_POSS_CONF[combo_size])
    conference_combos_by_size[combo_size] = conf_combos
    print(f"  {combo_size}-man conference:  "
          f"{sum(len(v) for v in conf_combos.values()):5d} combos, {len(conf_combos)} teams")

# Write one JSON file per team so the browser downloads ~20KB instead of ~10MB.
LINEUPS_PER_TEAM_DIR = os.path.join(SEASON_DATA_DIR, "lineups")
os.makedirs(LINEUPS_PER_TEAM_DIR, exist_ok=True)
team_slug_map = {}
for team_name in sorted(all_team_names_across_sizes):
    team_slug = slugify(team_name)
    team_slug_map[team_name] = team_slug
    per_team_data = {
        "team":    team_name,
        "overall": {size: overall_combos_by_size[size].get(team_name, [])     for size in ["1","2","3","5"]},
        "conf":    {size: conference_combos_by_size[size].get(team_name, [])  for size in ["1","2","3","5"]},
    }
    team_lineup_path = os.path.join(LINEUPS_PER_TEAM_DIR, f"{team_slug}.json")
    with open(team_lineup_path, "w") as lineup_file:
        lineup_file.write(json.dumps(_sanitize_for_json(per_team_data), separators=(",", ":")))

print(f"  lineup files:          {len(team_slug_map):5d} teams → lineups/{{slug}}.json")

# Small index file: just the list of teams and their slugs for the dropdown.
write_json(
    {"teams": sorted(all_team_names_across_sizes), "slugs": team_slug_map, "meta": build_metadata},
    "lineup-index.json"
)


# ===========================================================================
# 5. shots-meta.json  +  site/data/shots/{slug}.json
# ===========================================================================
shots_parquet_path    = os.path.join(PROJECT_ROOT, f"shots_{SEASON}.parquet")
box_score_parquet_path = os.path.join(PROJECT_ROOT, f"box_{SEASON}.parquet")

if os.path.exists(shots_parquet_path) and os.path.exists(box_score_parquet_path):
    print("\nBuilding shot data…")
    all_shots_df   = pd.read_parquet(shots_parquet_path)
    box_score_df   = pd.read_parquet(box_score_parquet_path)

    # Build lookup tables from the box score.
    player_identity_df = (
        box_score_df[["athlete_id", "athlete_display_name", "team_id", "team_display_name"]]
        .drop_duplicates("athlete_id").copy()
    )
    team_id_to_display_name = (
        box_score_df[["team_id", "team_display_name"]]
        .drop_duplicates().set_index("team_id")["team_display_name"].to_dict()
    )

    # Attach player names to shot rows.
    all_shots_df = all_shots_df.merge(
        player_identity_df[["athlete_id", "athlete_display_name"]],
        left_on="athlete_id_1", right_on="athlete_id", how="left"
    )

    # Remove free throws and shots with invalid coordinates.
    is_free_throw_shot = (
        all_shots_df["type_text"].str.contains("Free", case=False, na=False)
        | all_shots_df["text"].str.contains("free throw", case=False, na=False)
    )
    field_goal_shots_df = all_shots_df[~is_free_throw_shot].copy()
    field_goal_shots_df = field_goal_shots_df.dropna(subset=["coordinate_x", "coordinate_y"])
    field_goal_shots_df = field_goal_shots_df[
        (field_goal_shots_df["coordinate_x"].abs() <= 50)
        & (field_goal_shots_df["coordinate_y"].abs() <= 30)
    ]

    print("  Classifying zones…")
    field_goal_shots_df["zone"] = classify_shot_zones(field_goal_shots_df)
    field_goal_shots_df = field_goal_shots_df[
        ~field_goal_shots_df["zone"].isin(["Heave", "Unknown"])
    ].copy()

    # --- NCAA per-zone baseline FG% (drives the relative-to-average colours) ---
    zone_baselines = compute_zone_baselines(field_goal_shots_df)
    write_json(
        {"zones": ZONE_NAMES, "baselines": zone_baselines, "meta": build_metadata},
        "zone-baselines.json", output_directory=ROOT_DATA_DIR
    )

    # --- Team zone efficiency ---
    print("  Team zone stats…")
    team_zone_stats = {}
    for team_id, team_shot_group in field_goal_shots_df.groupby("team_id"):
        team_display_name = team_id_to_display_name.get(team_id)
        if not team_display_name:
            continue
        zone_record_list = build_zone_records(team_shot_group)
        if zone_record_list:
            team_zone_stats[team_display_name] = {
                "id":   int(team_id),
                "slug": slugify(team_display_name),
                "z":    zone_record_list
            }

    # --- Player zone efficiency (minimum 30 FGA) ---
    print("  Player zone stats…")
    player_zone_stats = {}
    for (team_id, player_display_name), player_shot_group in (
        field_goal_shots_df.dropna(subset=["athlete_display_name"])
        .groupby(["team_id", "athlete_display_name"])
    ):
        if len(player_shot_group) < 30:
            continue
        team_display_name = team_id_to_display_name.get(team_id, "")
        zone_record_list  = build_zone_records(player_shot_group)
        if zone_record_list:
            player_zone_stats[f"{player_display_name}|{team_display_name}"] = {
                "n": player_display_name,
                "t": team_display_name,
                "z": zone_record_list,
            }

    # --- Territory maps (top scorer per zone per team) ---
    print("  Territory maps…")
    field_goal_shots_df["point_value"] = (
        field_goal_shots_df["scoring_play"].map(lambda made: 1 if made else 0)
        * field_goal_shots_df["zone"].map(lambda z: 3 if z in THREE_POINT_ZONES else 2)
    )
    territory_by_team = {}
    for team_id, team_shot_group in field_goal_shots_df.groupby("team_id"):
        team_display_name = team_id_to_display_name.get(team_id)
        if not team_display_name:
            continue
        top_scorer_per_zone = []
        for zone_name, zone_shot_group in team_shot_group.groupby("zone"):
            if zone_name not in ZONE_NAME_TO_INDEX:
                continue
            named_shots = zone_shot_group.dropna(subset=["athlete_display_name"])
            if named_shots.empty:
                continue
            points_by_player = named_shots.groupby("athlete_display_name")["point_value"].sum()
            top_scorer_name  = points_by_player.idxmax()
            top_scorer_pts   = int(points_by_player.max())
            if top_scorer_pts > 0:
                top_scorer_per_zone.append({
                    "z":   ZONE_NAME_TO_INDEX[zone_name],
                    "n":   top_scorer_name,
                    "pts": top_scorer_pts,
                })
        if top_scorer_per_zone:
            territory_by_team[team_display_name] = top_scorer_per_zone

    # --- Per-team raw shot coordinate files ---
    print("  Raw shot coordinates → shots/{slug}.json…")
    game_schedule_parquet = os.path.join(PROJECT_ROOT, "game_schedule.parquet")
    if os.path.exists(game_schedule_parquet):
        game_schedule_df = pd.read_parquet(game_schedule_parquet)

        # Map team names → IDs so we can join with the schedule.
        display_name_to_team_id = {name: tid for tid, name in team_id_to_display_name.items()}
        schedule_with_team_ids  = game_schedule_df.copy()
        schedule_with_team_ids["team_id"] = schedule_with_team_ids["team"].map(display_name_to_team_id)
        schedule_with_team_ids = schedule_with_team_ids.dropna(subset=["team_id"])
        schedule_with_team_ids["team_id"] = schedule_with_team_ids["team_id"].astype(int)
        schedule_with_team_ids = schedule_with_team_ids.sort_values(["team_id", "date"]).reset_index(drop=True)

        # Assign each team's games a local sequential index (0, 1, 2, …).
        schedule_with_team_ids["local_game_index"] = schedule_with_team_ids.groupby("team_id").cumcount()
        game_index_lookup = schedule_with_team_ids.set_index(["team_id", "game_id"])["local_game_index"]

        # Attach local game index and compressed plot coordinates to every shot.
        fgs_index = pd.MultiIndex.from_arrays([
            field_goal_shots_df["team_id"].astype(int),
            field_goal_shots_df["game_id"].astype(int)
        ])
        field_goal_shots_df = field_goal_shots_df.copy()
        field_goal_shots_df["game_index"]     = game_index_lookup.reindex(fgs_index).fillna(-1).astype(int).values
        # Convert to integer screen coordinates (×10 for sub-foot precision).
        field_goal_shots_df["plot_x_int"] = (-field_goal_shots_df["coordinate_y"] * 10).round().astype(int)
        field_goal_shots_df["plot_y_int"] = ((47 - field_goal_shots_df["coordinate_x"].abs()) * 10).round().astype(int)
        field_goal_shots_df["scored_int"] = field_goal_shots_df["scoring_play"].astype(int)

        # Player index within the team's roster (for compact encoding).
        player_index_parts = []
        for _, team_shots in field_goal_shots_df.groupby("team_id"):
            sorted_player_names = sorted(team_shots["athlete_display_name"].dropna().unique())
            player_name_to_index = pd.Series(range(len(sorted_player_names)), index=sorted_player_names)
            player_index_parts.append(
                team_shots["athlete_display_name"].map(player_name_to_index).fillna(-1).astype(int)
            )
        field_goal_shots_df["player_index"] = (
            pd.concat(player_index_parts).reindex(field_goal_shots_df.index).fillna(-1).astype(int)
        )

        num_shot_files_written = 0
        for team_id_val, team_fgs in field_goal_shots_df.groupby("team_id"):
            team_id_int       = int(team_id_val)
            team_display_name = team_id_to_display_name.get(team_id_int)
            if not team_display_name:
                continue

            team_slug   = slugify(team_display_name)
            team_sched  = schedule_with_team_ids[schedule_with_team_ids["team_id"] == team_id_int].sort_values("date")
            game_records = [
                {"id": int(row.game_id), "opp": row.opponent, "date": str(row.date)[:10], "label": row.label}
                for row in team_sched.itertuples()
            ]

            if team_display_name in team_zone_stats:
                team_zone_stats[team_display_name]["gp"] = int(team_fgs["game_index"].nunique())

            roster_player_names = sorted(team_fgs["athlete_display_name"].dropna().unique().tolist())
            raw_shot_array      = team_fgs[
                ["game_index", "plot_x_int", "plot_y_int", "scored_int", "player_index"]
            ].values

            team_shot_file_path = os.path.join(SHOTS_DATA_DIR, f"{team_slug}.json")
            with open(team_shot_file_path, "w") as shot_file:
                shot_file.write(json.dumps(
                    _sanitize_for_json({
                        "games":   game_records,
                        "players": roster_player_names,
                        "shots":   raw_shot_array.ravel().tolist(),
                    }),
                    separators=(",", ":")
                ))
            num_shot_files_written += 1
        print(f"  → {num_shot_files_written} team shot files")
    else:
        print("  [warn] game_schedule.parquet not found — per-game shots skipped")

    write_json(
        {
            "zones":          ZONE_NAMES,
            "zone_baselines": zone_baselines,
            "team_zones":     team_zone_stats,
            "player_zones":   player_zone_stats,
            "territory":      territory_by_team,
            "meta":           build_metadata,
        },
        "shots-meta.json",
        output_directory=ROOT_DATA_DIR
    )
else:
    print(f"\n  [warn] shots_{SEASON}.parquet not found — shot data skipped")


# ===========================================================================
# seasons.json — list of all built season years for the UI dropdown
# ===========================================================================
built_season_years = sorted(
    [int(dir_name) for dir_name in os.listdir(ROOT_DATA_DIR)
     if dir_name.isdigit() and os.path.isdir(os.path.join(ROOT_DATA_DIR, dir_name))],
    reverse=True
)
write_json(built_season_years, "seasons.json", output_directory=ROOT_DATA_DIR)
print(f"\nAvailable seasons: {built_season_years}")

print("\n  Done. Serve site/ with: python -m http.server 8000 --directory site")


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
#
# SEASON                       int     Calendar year the season ends (e.g. 2026).
# PROJECT_ROOT                 str     Absolute path to the repo root directory.
# ARTIFACTS_DIR                str     Path to artifacts/{SEASON}/ (input files).
# SEASON_DATA_DIR              str     Path to site/data/{SEASON}/ (output JSON).
# SHOTS_DATA_DIR               str     Path to site/data/shots/ (per-team shot JSON).
# ROOT_DATA_DIR                str     Path to site/data/ (parent of per-season dirs).
# ZONE_NAMES                   list    Ordered list of 12 shot-zone names (index = zone ID).
# ZONE_NAME_TO_INDEX           dict    {zone_name: integer index into ZONE_NAMES}.
# THREE_POINT_ZONES            set     Zone names that award 3 points on a make.
#
# build_metadata               dict    Loaded from metadata.json; embedded in every output file.
# data_freshness_date          str     ISO date of the most recent game (used as build stamp).
# model_json_data              dict    Loaded from model.json; None in ratings-only builds.
# teams_ratings_df             DataFrame  Per-team efficiency + pace + home_adv + form.
# net_ratings_raw_df           DataFrame  Full leaderboard from net_ratings.parquet.
# conference_label_map         dict    {team_display_name: conference_abbreviation}.
#
# --- predictor.json section ---
# rank_columns                 list    Rank columns merged from net_ratings_raw_df.
# teams_json_records           list    List of team dicts written into predictor.json.
# wanted_team_columns          list    Columns kept from teams_ratings_df.
#
# --- net-ratings.json section ---
# net_ratings_df               DataFrame  Net ratings with conf column and record string added.
# net_ratings_records          list    to_dict("records") output.
#
# --- Player bios ---
# ESPN_ROSTER_URL_TEMPLATE     str     URL pattern for ESPN's team roster API.
# all_player_bios              dict    {str(athlete_id): {ht, wt, exp, hw, jn}}.
#
# --- build_player_stats_json() ---
# player_stats_df              DataFrame  Loaded from the player-stats CSV.
# field_goals_attempted        Series   FGA column for the current player-stats CSV.
# three_point_field_goals_att  Series   3PA column.
# free_throws_attempted        Series   FTA column.
# turnovers                    Series   TOV column.
# total_points                 Series   PTS column.
# games_played                 Series   GP column.
# total_minutes                Series   Total season minutes.
# team_totals_df               DataFrame  Sum of FGA/FTA/TOV/MIN per team.
# team_possessions             Series   FGA + 0.44*FTA + TOV per team (for usage rate).
# player_possessions           Series   Individual player's possession-ending events.
# hollinger_gamescore          Series   Hollinger's weighted counting-stat composite.
# player_on_court_team_poss    ndarray  Estimated team possessions while player was on court.
# gamescore_per_100_team_poss  ndarray  Game score normalized to per-100-team-possessions.
# league_avg_gamescore_per100  float    Minutes-weighted league mean (used to center BPM).
# on_off_df                    DataFrame  on_off and poss_off_on from the RAPM pipeline.
# columns_to_keep              list    Columns in PLAYER_STATS_COLUMN_RENAME_MAP present in data.
# player_stats_output_df       DataFrame  Renamed subset ready to serialize.
# json_records                 list    to_dict("records") output with None for NaN.
#
# --- Lineup section ---
# LINEUP_COLUMNS_TO_KEEP       list    Stat columns preserved in lineup JSON.
# LINEUP_MIN_POSS_OVERALL/CONF dict    {combo_size: min_avg_poss} thresholds.
# overall_combos_by_size       dict    {"1": {team: [records]}, ...} for all games.
# conference_combos_by_size    dict    Same structure, conference games only.
# all_team_names_across_sizes  set     Union of all team names across combo sizes.
# team_slug_map                dict    {team_name: url_slug}.
# LINEUPS_PER_TEAM_DIR         str     Directory for per-team lineup JSON files.
#
# --- Shot section ---
# all_shots_df                 DataFrame  Raw PBP shot rows from shots_{SEASON}.parquet.
# box_score_df                 DataFrame  Player boxscore from box_{SEASON}.parquet.
# player_identity_df           DataFrame  athlete_id → name/team lookup.
# team_id_to_display_name      dict    {team_id: team_display_name}.
# is_free_throw_shot           Series  Boolean mask for free-throw rows.
# field_goal_shots_df          DataFrame  Non-FT shots with valid coordinates.
# team_zone_stats              dict    {team_name: {id, slug, z: [[zone_idx,m,a],…]}}.
# player_zone_stats            dict    {"{player}|{team}": {n, t, z: …}}.
# territory_by_team            dict    {team_name: [{z, n, pts}, …]} top scorer per zone.
# game_schedule_df             DataFrame  Loaded from game_schedule.parquet.
# schedule_with_team_ids       DataFrame  Schedule with numeric team_id attached.
# game_index_lookup            Series  (team_id, game_id) → local_game_index for fast join.
# raw_shot_array               ndarray  [game_index, plot_x_int, plot_y_int, scored_int, player_index]
#                                       columns packed into a flat list for minimal JSON size.
# built_season_years           list    Sorted (newest-first) list of built season directories.
