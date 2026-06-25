"""
build_history.py — Build site data for historical seasons.

WHY THIS FILE EXISTS
--------------------
The nightly GitHub Actions workflow only rebuilds the current season.  This
script exists to populate the per-season data directories for every past season
so the UI's season dropdown works all the way back to 2016.

Historical seasons never change once they're built, so this script checks
whether a season is already built before doing any work.  Run it manually
once whenever you want to add older seasons to the site.

Usage:
    python build_history.py                 # Builds all missing seasons 2016–(current−1)
    python build_history.py --season 2023   # Build one specific season only
    python build_history.py --force         # Rebuild even if already built
"""

import argparse
import os
import subprocess
import sys

# Absolute path to the repo root (same directory as this script).
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Earliest season with reliable sportsdataverse PBP + box-score data.
EARLIEST_AVAILABLE_SEASON = 2016


def current_season():
    """Return the calendar year the current season ends (November rule)."""
    from datetime import date
    today = date.today()
    return today.year + 1 if today.month >= 11 else today.year


def season_already_built(target_season):
    """Return True if this season's JSON data directory already has player-stats.json.

    player-stats.json is the last file written by build_site.py, so its presence
    means the full pipeline completed successfully for this season.
    """
    player_stats_json_path = os.path.join(
        REPO_ROOT, "site", "data", str(target_season), "player-stats.json"
    )
    return os.path.exists(player_stats_json_path)


def run_pipeline_script(script_filename, target_season, env_vars):
    """Run a single pipeline script with OVERRIDE_SEASON set to target_season.

    Returns True if the script exited with code 0 (success), False otherwise.
    """
    print(f"\n  → python {script_filename}  (SEASON={target_season})")
    completed_process = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, script_filename)],
        env=env_vars,
    )
    if completed_process.returncode != 0:
        print(f"  [ERROR] {script_filename} failed for season {target_season}")
        return False
    return True


def build_single_season(target_season):
    """Run the full pipeline for one historical season.

    The pipeline steps for historical seasons differ from the nightly build:
    - RATINGS_ONLY=1 skips model training (speeds up build_artifacts.py
      significantly; net-ratings are still computed and saved).
    - build_shots_data.py is NOT run because shot parquets are large and
      historical shot data is not committed.  build_site.py handles the
      missing parquet gracefully.
    - build_onoff_rapm.py is non-fatal: some older seasons have PBP data
      quality issues that prevent accurate on/off computation.

    Returns True if all required steps succeeded, False otherwise.
    """
    # Child process environment: override the season and skip model training.
    child_process_env = os.environ.copy()
    child_process_env["OVERRIDE_SEASON"] = str(target_season)
    child_process_env["RATINGS_ONLY"]    = "1"

    print(f"\n{'='*60}")
    print(f"  Building season {target_season-1}–{str(target_season)[2:]}  (SEASON={target_season})")
    print(f"{'='*60}")

    # (script_filename, is_required)
    # Non-required steps log a warning and continue rather than aborting.
    pipeline_steps = [
        ("build_artifacts.py",    True),
        ("build_player_stats.py", True),
        ("build_lineups.py",      True),
        ("build_onoff_rapm.py",   False),   # may fail on older PBP data — non-fatal
        ("build_site.py",         True),    # shot sections skipped automatically if no parquet
    ]

    for script_filename, is_required in pipeline_steps:
        script_succeeded = run_pipeline_script(script_filename, target_season, child_process_env)
        if not script_succeeded and is_required:
            print(f"  Skipping remaining steps for season {target_season}.")
            return False
        elif not script_succeeded:
            print(f"  [WARNING] {script_filename} failed — continuing without on/off data.")

    return True


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="Build historical season data for Hog Charts"
    )
    argument_parser.add_argument(
        "--season", type=int, default=None,
        help="Build a specific season only (e.g. --season 2023)"
    )
    argument_parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if already built"
    )
    parsed_args = argument_parser.parse_args()

    live_current_season = current_season()

    if parsed_args.season:
        seasons_to_build = [parsed_args.season]
    else:
        # All historical seasons (not the current one, which the nightly job handles).
        seasons_to_build = list(range(EARLIEST_AVAILABLE_SEASON, live_current_season))

    print(f"Historical build: seasons {seasons_to_build}")

    for target_season in seasons_to_build:
        if not parsed_args.force and season_already_built(target_season):
            print(f"\n  Season {target_season} already built — skipping (use --force to rebuild)")
            continue
        build_single_season(target_season)

    print("\n\nHistorical build complete.")


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
#
# REPO_ROOT                    str     Absolute path to the repository root.
# EARLIEST_AVAILABLE_SEASON    int     2016 — sportsdataverse PBP is reliable from this year onward.
#
# --- current_season() ---
# today                        date    Today's calendar date (for determining the current season).
# return value                 int     Calendar year the current season ends (November rule).
#
# --- season_already_built() ---
# target_season                int     The season to check.
# player_stats_json_path       str     Path to site/data/{season}/player-stats.json.
# return value                 bool    True if player-stats.json exists (pipeline completed).
#
# --- run_pipeline_script() ---
# script_filename              str     Filename of the script to run (e.g. "build_artifacts.py").
# target_season                int     The season the script should process.
# env_vars                     dict    Environment variables passed to the subprocess.
# completed_process            CompletedProcess  Result of subprocess.run().
# return value                 bool    True if returncode == 0.
#
# --- build_single_season() ---
# target_season                int     The season being built.
# child_process_env            dict    Env copy with OVERRIDE_SEASON + RATINGS_ONLY set.
# pipeline_steps               list    [(script, is_required), …] ordered build steps.
# script_succeeded             bool    Return value from run_pipeline_script().
#
# --- __main__ ---
# argument_parser              ArgumentParser   CLI argument parser.
# parsed_args                  Namespace        Parsed --season and --force flags.
# live_current_season          int              Current season (used as the upper bound).
# seasons_to_build             list             All target_season values to process.
