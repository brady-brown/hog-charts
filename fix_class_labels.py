"""
fix_class_labels.py — one-off: give past seasons the class each player had THAT season.

WHY THIS FILE EXISTS
--------------------
build_site.py labels class ("exp") from ESPN's roster endpoint, which only knows a
player's class *today* (asking for an older season still returns today's). So every
season's player-stats JSON carries the classes from whenever it was last rebuilt:
  - 2021-2025 were all rebuilt 2026-06-30 → 2025-26 classes
  - 2026 was last rebuilt 2026-10-21      → 2026-27 classes
  - 2027 (live) is rebuilt nightly       → correct
(Verified: a player's label is identical across 2021-2025, and 2026 matches 2027.)

Rule, per player: take one anchor (class A in season T) from the most recent file that
has one, then for each season s:
  - A below Senior: class(s) = A - (T - s), exact.
  - A Senior: ESPN also calls 5th/6th-year players "Senior", so count from his first
    season F in the data: span = T - F + 1. With span >= 4, class(s) = min(4, s - F + 1);
    otherwise class(s) = 4 - (T - s).
Anything that comes out below Freshman is left blank rather than guessed.

Run once:  python fix_class_labels.py            (rewrites site/data/{2021..2026}/player-stats*.json)
           python fix_class_labels.py --dry-run  (report only)
"""
import glob
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site", "data")
CLASS_YEARS = {"Freshman": 1, "Sophomore": 2, "Junior": 3, "Senior": 4}
YEAR_CLASS = {v: k for k, v in CLASS_YEARS.items()}

# The season whose classes each file actually holds (see module docstring).
LABELS_ARE_FROM = {2021: 2026, 2022: 2026, 2023: 2026, 2024: 2026, 2025: 2026, 2026: 2027, 2027: 2027}
FIX_SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]


def load_players(season):
    with open(os.path.join(DATA_DIR, str(season), "player-stats-all.json")) as f:
        return json.load(f)["players"]


def class_by_season():
    """{athlete_id: {season: class label or None}} for every player in the seasons we fix."""
    seasons_seen = {}          # id -> set of seasons he appears in (career files: no minimums)
    for season in LABELS_ARE_FROM:
        path = os.path.join(DATA_DIR, str(season), "player-stats-career.json")
        with open(path) as f:
            for p in json.load(f)["players"]:
                seasons_seen.setdefault(int(p["id"]), set()).add(season)

    anchor = {}                # id -> (class years A, season T), most recent labelled file wins
    for season in sorted(LABELS_ARE_FROM, reverse=True):
        for p in load_players(season):
            label = p.get("exp")
            if label in CLASS_YEARS and int(p["id"]) not in anchor:
                anchor[int(p["id"])] = (CLASS_YEARS[label], LABELS_ARE_FROM[season])

    result = {}
    for athlete_id, (years, anchor_season) in anchor.items():
        first = min(seasons_seen.get(athlete_id, {anchor_season}))
        span = anchor_season - first + 1
        out = {}
        for season in FIX_SEASONS:
            if years < 4:
                y = years - (anchor_season - season)
            elif span >= 4:
                y = min(4, season - first + 1)
            else:
                y = 4 - (anchor_season - season)
            out[season] = YEAR_CLASS.get(y)        # None below Freshman: unknown, not guessed
        result[athlete_id] = out
    return result


def main():
    dry_run = "--dry-run" in sys.argv
    classes = class_by_season()
    for season in FIX_SEASONS:
        changed = blanked = 0
        for path in sorted(glob.glob(os.path.join(DATA_DIR, str(season), "player-stats*.json"))):
            if path.endswith("player-stats-career.json"):
                continue                            # no class field
            with open(path) as f:
                payload = json.load(f)
            for p in payload["players"]:
                new = classes.get(int(p["id"]), {}).get(season)
                if new != p.get("exp"):
                    changed += 1
                    blanked += new is None
                    p["exp"] = new
            if not dry_run:
                with open(path, "w") as f:
                    f.write(json.dumps(payload, separators=(",", ":"), allow_nan=False))
        print(f"{season}: {changed:,} labels changed ({blanked:,} now blank) across its player-stats files")
    if dry_run:
        print("(dry run — nothing written)")


if __name__ == "__main__":
    main()
