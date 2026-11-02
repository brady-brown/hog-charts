"""Run the build pipeline inside the mirror workspace, on ESPN-sourced feeds.

Same scripts as the root pipeline, same order, same dependency chain — the only
difference is where they read from and write to. build_ingest.py (step 0 of the
root pipeline) is replaced by raw_hogcharts/ingest.py, which pulls from ESPN.

    python -m raw_hogcharts.run_pipeline --season 2026
    python -m raw_hogcharts.run_pipeline --season 2026 --from build_site.py
    python -m raw_hogcharts.run_pipeline --season 2026 --only build_player_stats.py
    python -m raw_hogcharts.run_pipeline --season 2026 --ingest      # pull first

Output lands in raw_hogcharts/build/site/ — the duplicate website. Serve it with:

    python -m http.server -d raw_hogcharts/build/site 8001
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "raw_hogcharts"

from . import ingest, workspace

# The nightly order from CLAUDE.md, minus build_ingest.py.
PIPELINE = (
    "build_artifacts.py",
    "build_shots_data.py",
    "build_player_stats.py",
    "build_lineups.py",
    "build_onoff_rapm.py",
    "build_points_resp.py",
    "build_shot_diet.py",
    "build_site.py",
    "build_scout.py",
    "build_postseason.py",
)


def check_feeds(season):
    """Refuse to start if the season's ESPN feeds aren't ingested yet."""
    missing = [feed for feed in ingest.FEEDS if not os.path.exists(ingest.feed_path(season, feed))]
    if missing:
        raise SystemExit(
            f"season {season} is missing ESPN feeds: {', '.join(missing)}\n"
            f"run: python -m raw_hogcharts.ingest --season {season} --full"
        )


def run_script(script, mirror, season, extra_env=None):
    environment = dict(os.environ)
    environment["OVERRIDE_SEASON"] = str(season)
    # The mirror leads PYTHONPATH so `import sportsdataverse` hits the shim.
    environment["PYTHONPATH"] = os.pathsep.join(
        [mirror] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else []))
    if extra_env:
        environment.update(extra_env)

    started = time.time()
    print(f"\n{'=' * 70}\n▶ {script}  (season {season})\n{'=' * 70}", flush=True)
    result = subprocess.run([sys.executable, script], cwd=mirror, env=environment, check=False)
    elapsed = time.time() - started
    status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"◀ {script}: {status} in {elapsed:.1f}s", flush=True)
    return result.returncode, elapsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--ingest", action="store_true",
                        help="pull fresh data from ESPN before building")
    parser.add_argument("--full-ingest", action="store_true",
                        help="with --ingest, rescan the whole season window")
    parser.add_argument("--only", action="append", metavar="SCRIPT",
                        help="run just these scripts (repeatable)")
    parser.add_argument("--from", dest="start_at", metavar="SCRIPT",
                        help="start at this script and run the rest")
    parser.add_argument("--skip", action="append", default=[], metavar="SCRIPT",
                        help="skip these scripts (repeatable)")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue after a failing step instead of stopping")
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args(argv)

    season = arguments.season or ingest.current_season()

    if arguments.ingest:
        ingest.ingest_season(season, full=arguments.full_ingest, workers=arguments.workers)

    check_feeds(season)
    mirror = workspace.ensure()

    scripts = list(arguments.only) if arguments.only else list(PIPELINE)
    if arguments.start_at:
        if arguments.start_at not in scripts:
            raise SystemExit(f"--from {arguments.start_at} is not in the pipeline")
        scripts = scripts[scripts.index(arguments.start_at):]
    scripts = [script for script in scripts if script not in arguments.skip]

    print(f"\nbuilding season {season} from ESPN data in {mirror}")
    print(f"steps: {' → '.join(scripts)}")

    results, failures = [], 0
    for script in scripts:
        code, elapsed = run_script(script, mirror, season)
        results.append((script, code, elapsed))
        if code != 0:
            failures += 1
            if not arguments.keep_going:
                break

    print(f"\n{'=' * 70}\nsummary — season {season}, {dt.datetime.now():%Y-%m-%d %H:%M}")
    for script, code, elapsed in results:
        print(f"  {'ok  ' if code == 0 else 'FAIL'}  {script:<26} {elapsed:>7.1f}s")
    remaining = len(scripts) - len(results)
    if remaining:
        print(f"  ....  {remaining} step(s) not reached")
    print(f"\nduplicate site: {os.path.join(mirror, 'site')}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
