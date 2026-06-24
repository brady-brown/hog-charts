"""
conferences.py — Single source of truth for conference labels.

WHY THIS FILE EXISTS
--------------------
Conference labels (e.g. "SEC", "Big 12") used to be read from a hand-maintained
`player_stats.csv` that no longer exists in the repo or in CI.  When that file
went missing, build_player_stats.py silently fell back to an empty map, which is
why the current-season player-stats had blank conferences and the Player Stats
conference filter showed nothing.

This module replaces that fragile dependency with ESPN's season standings API,
which lists every conference and its member teams for an exact season (so
realignment is handled correctly — a team appears in the conference it actually
played in that year).  ESPN returns full names ("Southeastern Conference"); we
map those to the short site-style labels the UI already uses via CONFERENCE_LABELS,
falling back to a cleaned name for anything unmapped.

Used by:
  • build_player_stats.py   — labels every player at build time (incl. CI).
  • impact_artifact.py / patch_conferences.py may also import these helpers.
"""
import re

import requests

# ESPN standings endpoint: conference tree with member teams for a season.
ESPN_STANDINGS_URL_TEMPLATE = (
    "https://site.api.espn.com/apis/v2/sports/basketball"
    "/mens-college-basketball/standings?season={season}&level=2"
)

# ESPN full conference name → short site-style label shown on the site.
# Covers current (2026) conferences plus historical ones (Pac-12, WAC, the
# Colonial→Coastal rename) so older seasons label correctly too.
CONFERENCE_LABELS = {
    "America East Conference":             "America East",
    "American Conference":                 "American",
    "American Athletic Conference":        "American",
    "Atlantic 10 Conference":              "Atlantic 10",
    "Atlantic Coast Conference":           "ACC",
    "Atlantic Sun Conference":             "ASUN",
    "Big 12 Conference":                   "Big 12",
    "Big East Conference":                 "Big East",
    "Big Sky Conference":                  "Big Sky",
    "Big South Conference":                "Big South",
    "Big Ten Conference":                  "Big Ten",
    "Big West Conference":                 "Big West",
    "Coastal Athletic Association":        "Coastal",
    "Colonial Athletic Association":       "Coastal",
    "Conference USA":                      "Conference USA",
    "Horizon League":                      "Horizon",
    "Ivy League":                          "Ivy",
    "Metro Atlantic Athletic Conference":  "MAAC",
    "Mid-American Conference":             "Mid-American",
    "Mid-Eastern Athletic Conference":     "MEAC",
    "Missouri Valley Conference":          "Missouri Valley",
    "Mountain West Conference":            "Mountain West",
    "Northeast Conference":                "Northeast",
    "Ohio Valley Conference":              "Ohio Valley",
    "Pac-12 Conference":                   "Pac-12",
    "Patriot League":                      "Patriot League",
    "Southeastern Conference":             "SEC",
    "Southern Conference":                 "Southern",
    "Southland Conference":                "Southland",
    "Southwestern Athletic Conference":    "SWAC",
    "Summit League":                       "Summit League",
    "The Summit League":                   "Summit League",
    "Sun Belt Conference":                 "Sun Belt",
    "United Athletic Conference":          "United Athletic",
    "West Coast Conference":               "West Coast",
    "Western Athletic Conference":         "WAC",
}


def clean_conference_name(raw_name):
    """Strip a trailing "Conference"/"League" so unmapped names read cleanly.

    "Some New Conference" → "Some New".  Returns None for falsy input.
    """
    if not raw_name:
        return None
    return re.sub(r"\s+(Conference|League|Conf\.?)$", "", raw_name).strip()


def fetch_conference_standings(season):
    """Fetch ESPN standings for a season.

    Returns (team_id_to_conference_id, conference_id_to_name):
        team_id_to_conference_id  {int(team_id): str(conf_id)}
        conference_id_to_name     {str(conf_id): str(full_conference_name)}
    """
    raw_json = requests.get(
        ESPN_STANDINGS_URL_TEMPLATE.format(season=season), timeout=30
    ).json()

    team_id_to_conference_id = {}
    conference_id_to_name = {}
    for conference_node in raw_json.get("children", []):
        conference_id = str(conference_node["id"])
        conference_id_to_name[conference_id] = conference_node.get("name")
        for standings_entry in conference_node.get("standings", {}).get("entries", []):
            try:
                team_id = int(standings_entry["team"]["id"])
                team_id_to_conference_id[team_id] = conference_id
            except (KeyError, TypeError, ValueError):
                pass
    return team_id_to_conference_id, conference_id_to_name


def team_id_to_label(season):
    """Return {team_id: short_site_label} for every team in a season.

    Maps ESPN's full conference name to the short site label via
    CONFERENCE_LABELS, falling back to a cleaned name for anything unmapped.
    Returns {} if the standings request fails (caller keeps prior behavior).
    """
    try:
        team_to_conf_id, conf_id_to_name = fetch_conference_standings(season)
    except Exception:
        return {}

    labels = {}
    for team_id, conf_id in team_to_conf_id.items():
        full_name = conf_id_to_name.get(conf_id)
        labels[team_id] = CONFERENCE_LABELS.get(full_name) or clean_conference_name(full_name)
    return labels


if __name__ == "__main__":
    # Quick self-check: print the resolved labels for a season.
    import os
    season = int(os.environ.get("OVERRIDE_SEASON", "2026"))
    labels = team_id_to_label(season)
    distinct = sorted(set(v for v in labels.values() if v))
    print(f"{season}: {len(labels)} teams, {len(distinct)} conferences")
    print(distinct)
