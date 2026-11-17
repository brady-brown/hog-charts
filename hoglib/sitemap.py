"""site/sitemap.xml — every page a search engine should know about.

The team and player pages render client-side from ?id=, so a crawler only finds
them through links it runs JavaScript to see. Listing them here lets Google
index "{team} stats" / "{player} stats" pages directly.

Reads the site JSON build_site already wrote (no CSVs), so it also runs on its
own:  python -m hoglib.sitemap
URLs carry only ?id= — each page defaults to the live season, so the listed URL
stays the same from one season to the next.
"""
import json
import os
import sys
from datetime import date
from xml.sax.saxutils import escape

SITE_URL = "https://hogcharts.org"

# Tool pages. Not listed: player/team (added per id below), game-chart pages
# (need an id), scout-report (redirects to the team page's Scout tab).
STATIC_PAGES = ["", "prediction.html", "net-ratings.html", "player-stats.html",
                "lineup-stats.html", "lineup-builder.html", "postseason.html",
                "shot-charts.html", "glossary.html"]

RECENT_SEASONS = 2   # players/teams from the live season and the one before it


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_sitemap(site_dir):
    data_dir = os.path.join(site_dir, "data")
    seasons = (_load(os.path.join(data_dir, "seasons.json")) or [])[:RECENT_SEASONS]
    team_ids, player_ids = set(), set()
    for season in seasons:
        teams = _load(os.path.join(data_dir, str(season), "team-stats-all.json")) or {}
        team_ids.update(r["team_id"] for r in teams.get("team_stats", []) if r.get("team_id") is not None)
        # The career file has no games/minutes floor, so every player page that exists is in it.
        players = _load(os.path.join(data_dir, str(season), "player-stats-career.json")) or {}
        player_ids.update(r["id"] for r in players.get("players", []) if r.get("id") is not None)

    today = date.today().isoformat()
    urls = [f"{SITE_URL}/{page}" for page in STATIC_PAGES]
    urls += [f"{SITE_URL}/team.html?id={tid}" for tid in sorted(team_ids)]
    urls += [f"{SITE_URL}/player.html?id={pid}" for pid in sorted(player_ids)]
    body = "\n".join(f"  <url><loc>{escape(u)}</loc><lastmod>{today}</lastmod></url>" for u in urls)
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
           f"{body}\n</urlset>\n")
    with open(os.path.join(site_dir, "sitemap.xml"), "w") as f:
        f.write(xml)
    print(f"  {'sitemap.xml':<35s} ({len(urls)} URLs: {len(team_ids)} teams, {len(player_ids)} players)")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    write_sitemap(sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "site"))
