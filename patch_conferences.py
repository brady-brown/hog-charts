"""
patch_conferences.py — Backfill conference labels for historical seasons.

WHY THIS FILE EXISTS
--------------------
Historical seasons (2016–2025) were built before the conference-labeling
logic was fully developed, so their player-stats CSV/JSON files have an
empty `conf.` column.  This script patches them all in a single run.

Source of truth for conference membership:  ESPN's season standings API,
which lists every conference and its member teams for that exact season.
This means realignment is handled correctly — Pac-12 members in 2018
appear as Pac-12, not whatever conference they joined in 2024.

Label style alignment:  The 2026 site already shows short human-readable
labels (e.g. "SEC", "Big 12") in the `conf.` column.  To match that style
in historical seasons, we vote on the majority 2026 label for each ESPN
conference ID, then use that label wherever the conference ID appears in
earlier seasons.  For conferences absent in 2026 (e.g. defunct Pac-12) we
fall back to a cleaned version of the ESPN standings name.

Files patched per season:
  - player_stats_{s}.csv
  - player_stats_conf_{s}.csv
  - site/data/{s}/player-stats.json
  - site/data/{s}/player-stats-conf.json
  - site/data/{s}/net-ratings.json

Run:
    python patch_conferences.py
"""
import json
import os
import re
from collections import Counter

import pandas as pd
import requests

REPO_ROOT    = os.path.dirname(os.path.abspath(__file__))
CURRENT_SEASON = 2026                             # season whose labels are authoritative
HISTORICAL_SEASONS = range(2016, CURRENT_SEASON)  # seasons that need patching

# ESPN standings endpoint: returns conference tree with member teams for a season.
ESPN_STANDINGS_URL_TEMPLATE = (
    "https://site.api.espn.com/apis/v2/sports/basketball"
    "/mens-college-basketball/standings?season={season}&level=2"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_conference_name(raw_name):
    """Strip trailing "Conference" / "Conf." from an ESPN standings name.

    "Southeastern Conference" → "Southeastern"
    "Atlantic Coast Conference" → "Atlantic Coast"
    """
    if not raw_name:
        return None
    return re.sub(r"\s+(Conference|Conf\.?)$", "", raw_name).strip()


def fetch_conference_standings(season):
    """Fetch ESPN standings for a season and return team→conf and conf→name maps.

    Returns:
        team_id_to_conference_id   dict  {int(team_id): str(conf_id)}
        conference_id_to_name      dict  {str(conf_id): str(conference_name)}
    """
    raw_json = requests.get(
        ESPN_STANDINGS_URL_TEMPLATE.format(season=season), timeout=30
    ).json()

    team_id_to_conference_id = {}
    conference_id_to_name    = {}

    for conference_node in raw_json.get("children", []):
        conference_id   = str(conference_node["id"])
        conference_name = conference_node.get("name")
        conference_id_to_name[conference_id] = conference_name

        for standings_entry in conference_node.get("standings", {}).get("entries", []):
            try:
                team_id = int(standings_entry["team"]["id"])
                team_id_to_conference_id[team_id] = conference_id
            except (KeyError, TypeError, ValueError):
                pass

    return team_id_to_conference_id, conference_id_to_name


def build_conference_id_to_site_label_map():
    """Build a {conference_id → site-style label} map anchored to the 2026 season.

    Voting procedure:
      1. Load 2026 player_stats to get the site-style conf. label for each team.
      2. Fetch 2026 ESPN standings to map each team to a conference ID.
      3. For each conference ID, the most-common site label among its 2026 members
         wins.  This produces e.g. {"50": "SEC", "2": "Big 12", …}.
    """
    current_season_stats_path = os.path.join(
        REPO_ROOT, f"player_stats_{CURRENT_SEASON}.csv"
    )
    current_season_df = (
        pd.read_csv(current_season_stats_path, usecols=["team_id", "conf."])
        .dropna()
        .drop_duplicates("team_id")
    )
    current_team_id_to_site_label = dict(
        zip(current_season_df["team_id"].astype(int), current_season_df["conf."])
    )

    current_team_id_to_conf_id, _ = fetch_conference_standings(CURRENT_SEASON)

    votes_per_conf_id = {}
    for team_id, conference_id in current_team_id_to_conf_id.items():
        site_label = current_team_id_to_site_label.get(team_id)
        if site_label:
            votes_per_conf_id.setdefault(conference_id, Counter())[site_label] += 1

    return {
        conference_id: vote_counts.most_common(1)[0][0]
        for conference_id, vote_counts in votes_per_conf_id.items()
    }


def build_team_label_map_for_season(season, conference_id_to_site_label):
    """Return {team_id → site-style label} for a historical season.

    Prefers the voted site label; falls back to a cleaned ESPN conference
    name for conferences not present in 2026 (defunct or newly formed).
    """
    team_id_to_conf_id, conf_id_to_name = fetch_conference_standings(season)

    team_id_to_label = {}
    for team_id, conference_id in team_id_to_conf_id.items():
        preferred_label  = conference_id_to_site_label.get(conference_id)
        fallback_label   = clean_conference_name(conf_id_to_name.get(conference_id))
        team_id_to_label[team_id] = preferred_label or fallback_label

    return team_id_to_label


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def patch_csv_file(csv_path, team_id_to_label):
    """Overwrite the conf. column in a player-stats CSV using the label map.

    Returns a short status string like "247/312 labeled" for logging.
    """
    if not os.path.exists(csv_path):
        return f"missing {os.path.basename(csv_path)}"

    player_stats_df = pd.read_csv(csv_path)
    player_stats_df["conf."] = player_stats_df["team_id"].map(team_id_to_label)
    player_stats_df.to_csv(csv_path, index=False)

    labeled_count = int(player_stats_df["conf."].notna().sum())
    return f"{labeled_count}/{len(player_stats_df)} labeled"


def patch_json_file(json_path, team_id_field_name, team_id_to_label):
    """Overwrite the 'conf' field in a player-stats or net-ratings JSON file.

    The JSON files have two structures:
      player JSON:      {"players": [{…, "tid": 123, …}, …]}
      net-ratings JSON: {"net_ratings": [{…, "team_id": 123, …}, …]}

    Returns a short status string for logging.
    """
    if not os.path.exists(json_path):
        return "missing"

    data_object = json.load(open(json_path))
    # Determine which key holds the list of records.
    record_list = data_object.get("players") or data_object.get("net_ratings")
    labeled_count = 0

    for record in record_list:
        raw_team_id = record.get(team_id_field_name)
        if raw_team_id is not None:
            site_label = team_id_to_label.get(int(raw_team_id))
            record["conf"] = site_label
            if site_label:
                labeled_count += 1

    with open(json_path, "w") as output_file:
        output_file.write(json.dumps(data_object, separators=(",", ":")))

    return f"{labeled_count}/{len(record_list)} labeled"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Building conference_id → site-label map from {CURRENT_SEASON} data…")
    conference_id_to_site_label = build_conference_id_to_site_label_map()
    print(f"  {len(conference_id_to_site_label)} conference IDs mapped\n")

    for historical_season in HISTORICAL_SEASONS:
        team_id_to_label = build_team_label_map_for_season(
            historical_season, conference_id_to_site_label
        )
        season_data_dir = os.path.join(REPO_ROOT, "site", "data", str(historical_season))

        patch_results = [
            f"csv={patch_csv_file(os.path.join(REPO_ROOT, f'player_stats_{historical_season}.csv'), team_id_to_label)}",
            f"csv-conf={patch_csv_file(os.path.join(REPO_ROOT, f'player_stats_conf_{historical_season}.csv'), team_id_to_label)}",
            f"players={patch_json_file(os.path.join(season_data_dir, 'player-stats.json'), 'tid', team_id_to_label)}",
            f"players-conf={patch_json_file(os.path.join(season_data_dir, 'player-stats-conf.json'), 'tid', team_id_to_label)}",
            f"net={patch_json_file(os.path.join(season_data_dir, 'net-ratings.json'), 'team_id', team_id_to_label)}",
        ]
        print(f"{historical_season}: " + " | ".join(patch_results))

    print("\nDone.")


if __name__ == "__main__":
    main()


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
#
# REPO_ROOT                       str     Absolute path to the repository root.
# CURRENT_SEASON                  int     2026 — the season whose conf. labels are authoritative.
# HISTORICAL_SEASONS              range   2016–2025 — seasons to be patched.
# ESPN_STANDINGS_URL_TEMPLATE     str     URL pattern for the ESPN standings API.
#
# --- clean_conference_name() ---
# raw_name                        str     ESPN conference name, possibly with " Conference" suffix.
# return value                    str     Cleaned name ("SEC", "Big 12", etc.) or None.
#
# --- fetch_conference_standings() ---
# season                          int     Season to fetch standings for.
# raw_json                        dict    Parsed JSON from the ESPN API.
# conference_node                 dict    One conference subtree in the API response.
# conference_id                   str     ESPN's numeric conference ID as a string.
# conference_name                 str     Full conference name from ESPN.
# standings_entry                 dict    One team entry inside a conference's standings.
# team_id                         int     ESPN team ID for one roster member.
# team_id_to_conference_id        dict    {team_id: conference_id} for every team in the standings.
# conference_id_to_name           dict    {conference_id: conference_name}.
#
# --- build_conference_id_to_site_label_map() ---
# current_season_stats_path       str     Path to player_stats_{CURRENT_SEASON}.csv.
# current_season_df               DataFrame  team_id + conf. for the current season.
# current_team_id_to_site_label   dict    {team_id: site_label} for the current season.
# current_team_id_to_conf_id      dict    {team_id: conference_id} for the current season.
# votes_per_conf_id               dict    {conference_id: Counter({label: vote_count})}.
# return value                    dict    {conference_id: winning_label}.
#
# --- build_team_label_map_for_season() ---
# team_id_to_conf_id              dict    {team_id: conference_id} from ESPN standings.
# conf_id_to_name                 dict    {conference_id: raw_name} from ESPN standings.
# team_id_to_label                dict    {team_id: site_label or cleaned_name}.
# preferred_label                 str     Voted 2026 site label; None if conf not in 2026.
# fallback_label                  str     Cleaned ESPN conference name (for defunct confs).
#
# --- patch_csv_file() ---
# csv_path                        str     Absolute path to the CSV to patch.
# player_stats_df                 DataFrame  Loaded CSV; conf. column is overwritten.
# labeled_count                   int     Number of rows where conf. is not NaN after patch.
#
# --- patch_json_file() ---
# json_path                       str     Absolute path to the JSON file to patch.
# team_id_field_name              str     Key in each record holding the team ID ("tid" or "team_id").
# data_object                     dict    Full parsed JSON ({"players": [...]} or {"net_ratings": [...]}).
# record_list                     list    The list of player or team records inside data_object.
# raw_team_id                     any     The team ID value from the current record.
# site_label                      str     The resolved conference label; None if team not in map.
