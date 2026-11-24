"""ESPN's recruiting classes (ESPN 100 and beyond) → recruits.csv at the repo root.

One request per class: core API /recruiting/{class}/athletes?limit=1000 returns
every ranked recruit with grade, national rank, position and the schools on
their list (status Signed / Verbal / Undecided). `alternateId` is the college
athlete id the box scores use, when ESPN has linked it (every past class; only
part of the newest one, so predictor.py also matches by name + signed school).

Class of YYYY = high-school graduates of YYYY = freshmen in season YYYY + 1.

    python -m raw_hogcharts.recruiting                 # classes 2018 … current
    python -m raw_hogcharts.recruiting 2026            # just one class
"""
import csv
import datetime as dt
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "recruits.csv")
BASE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball"
COLUMNS = ["class", "recruit_id", "athlete_id", "name", "pos", "grade", "rank",
           "team_id", "team_status"]
STATUS_ORDER = {"Signed": 0, "Enrolled": 0, "Verbal": 1}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def fetch_class(year):
    items = _get(f"{BASE}/recruiting/{year}/athletes?limit=1000")["items"]
    rows = []
    for item in items:
        a = item["athlete"]
        rank = next((x["value"] for x in item.get("attributes", []) if x["name"] == "rank"), None)
        # the school they signed with (else verbally committed to)
        picks = []
        for s in item.get("schools", []):
            status = (s.get("status") or {}).get("description")
            if status in STATUS_ORDER and "$ref" in (s.get("team") or {}):
                picks.append((STATUS_ORDER[status], s["team"]["$ref"]))
        team_id = status = None
        if picks:
            order, ref = min(picks)
            team_id = int(ref.split("/teams/")[1].split("?")[0])
            status = "signed" if order == 0 else "verbal"
        rows.append({"class": year, "recruit_id": int(a["id"]),
                     "athlete_id": int(a["alternateId"]) if a.get("alternateId") else None,
                     "name": a.get("displayName"), "pos": (a.get("position") or {}).get("abbreviation"),
                     "grade": item.get("grade"), "rank": int(rank) if rank else None,
                     "team_id": team_id, "team_status": status})
    return rows


def main(years):
    existing = []
    if os.path.exists(OUT):
        with open(OUT) as handle:
            existing = [r for r in csv.DictReader(handle) if int(r["class"]) not in years]
    rows = existing
    for year in years:
        got = fetch_class(year)
        print(f"  class {year}: {len(got)} recruits, {sum(r['athlete_id'] is not None for r in got)} linked "
              f"to a college id, {sum(r['team_status'] == 'signed' for r in got)} signed")
        rows += got
    rows.sort(key=lambda r: (int(r["class"]), int(r["rank"]) if r["rank"] not in (None, "") else 10**6))
    with open(OUT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    now = dt.date.today()
    current = now.year if now.month >= 6 else now.year - 1      # newest class already in college
    main([int(y) for y in sys.argv[1:]] or list(range(2018, current + 1)))
