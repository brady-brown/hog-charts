"""
build_history.py — Build site data for historical seasons.

Run once locally to generate data for past seasons. Historical seasons
never change, so this only needs to run for seasons not yet built.

Usage:
    python build_history.py              # builds all missing seasons 2020-2025
    python build_history.py --season 2023  # build one specific season
"""

import argparse
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))

FIRST_SEASON = 2022  # 2022 is the first season with enough prior data to calibrate


def current_season():
    from datetime import date
    today = date.today()
    return today.year + 1 if today.month >= 11 else today.year


def season_built(season):
    """True if this season's site/data dir already has the key files."""
    path = os.path.join(BASE, "site", "data", str(season), "player-stats.json")
    return os.path.exists(path)


def run_script(script, season, env):
    print(f"\n  → python {script}  (SEASON={season})")
    result = subprocess.run(
        [sys.executable, os.path.join(BASE, script)],
        env=env,
    )
    if result.returncode != 0:
        print(f"  [ERROR] {script} failed for season {season}")
        return False
    return True


def build_season(season):
    env = os.environ.copy()
    env["OVERRIDE_SEASON"] = str(season)

    print(f"\n{'='*60}")
    print(f"  Building season {season-1}-{str(season)[2:]}  (SEASON={season})")
    print(f"{'='*60}")

    steps = [
        ("build_artifacts.py",    True),
        ("build_player_stats.py", True),
        ("build_lineups.py",      True),
        ("build_onoff_rapm.py",   False),  # on/off may fail on older data — non-fatal
        ("build_site.py",         True),   # shots skipped automatically (no parquet)
    ]

    for script, required in steps:
        ok = run_script(script, season, env)
        if not ok and required:
            print(f"  Skipping remaining steps for season {season}.")
            return False
        elif not ok:
            print(f"  [WARNING] {script} failed — continuing without on/off data.")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None,
                        help="Build a specific season only")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if already built")
    args = parser.parse_args()

    cur = current_season()

    if args.season:
        seasons = [args.season]
    else:
        seasons = list(range(FIRST_SEASON, cur))  # all historical (not current)

    print(f"Historical build: seasons {seasons}")

    for s in seasons:
        if not args.force and season_built(s):
            print(f"\n  Season {s} already built — skipping (use --force to rebuild)")
            continue
        build_season(s)

    print("\n\nHistorical build complete.")
