"""Build the four raw feed tables for a season, straight from ESPN.

Writes raw_hogcharts/data/raw_{season}/{feed}.parquet — the same filenames and
column shapes the pipeline already reads from the sportsdataverse cache, so the
builders can run against it unchanged.

The cost model this is built around: past seasons are immutable, so they get
pulled once and then never touched again; only the current season is refreshed,
and only for games that are new or that changed since the last run.

    # one-time archive pull (slow, run locally, then leave alone forever)
    python -m raw_hogcharts.ingest --season 2025 --full
    python -m raw_hogcharts.ingest --backfill 2021 2025

    # nightly: rescan the last few days of the current season and merge
    python -m raw_hogcharts.ingest --season 2026

    # everything again, ignoring the archive lock
    python -m raw_hogcharts.ingest --season 2025 --full --force

Cost per season: ~6,300 games × one summary call (~400 KB, ~0.3 s). At the
default 8 workers a full season lands in roughly 5-10 minutes; a nightly
incremental run is a handful of scoreboard calls plus the day's ~100 games.
"""
import argparse
import datetime as dt
import json
import os
import sys

import pandas as pd

if __package__ in (None, ""):        # allow `python raw_hogcharts/ingest.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "raw_hogcharts"

from .espn import client, normalize

FEEDS = ("pbp", "player_box", "team_box", "schedule")

# Generous bounds on a season: first exhibitions in late October through the
# title game in early April. Scanning empty days is one cheap call each.
SEASON_START = (10, 15)     # (month, day) in season_year - 1
SEASON_END = (4, 20)        # (month, day) in season_year

# How far back an incremental run re-scans. Covers late box-score corrections
# and games that were still in progress when the previous run happened.
INCREMENTAL_LOOKBACK_DAYS = 4


def data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def season_dir(season):
    return os.path.join(data_dir(), f"raw_{season}")


def feed_path(season, feed):
    return os.path.join(season_dir(season), f"{feed}.parquet")


def manifest_path(season):
    return os.path.join(season_dir(season), "manifest.json")


def read_manifest(season):
    try:
        with open(manifest_path(season), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def write_manifest(season, **fields):
    manifest = read_manifest(season)
    manifest.update(fields)
    manifest["season"] = season
    manifest["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    os.makedirs(season_dir(season), exist_ok=True)
    with open(manifest_path(season), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def current_season(today=None):
    """Same rule as every build script: Nov or later belongs to the next season."""
    today = today or dt.date.today()
    override = os.environ.get("OVERRIDE_SEASON")
    if override:
        return int(override)
    return today.year + 1 if today.month >= 11 else today.year


def season_window(season):
    start = dt.date(season - 1, *SEASON_START)
    end = dt.date(season, *SEASON_END)
    return start, end


def date_strings(start, end):
    """Every ESPN date key (YYYYMMDD) from start to end inclusive."""
    days = (end - start).days
    return [(start + dt.timedelta(days=offset)).strftime("%Y%m%d") for offset in range(days + 1)]


def _progress(total, label):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=label, unit="req")
    except ImportError:
        return None


# ── step 1: the schedule, from the scoreboard ────────────────────────────────
def scan_schedule(season, dates, workers=8):
    """One scoreboard call per date → the season's schedule rows."""
    bar = _progress(len(dates), f"scoreboard {season}")
    payloads = client.map_concurrent(
        client.fetch_scoreboard, dates, workers=workers, on_error="collect", progress=bar)
    if bar is not None:
        bar.close()

    rows, seen = [], set()
    for events in payloads:
        for event in events or []:
            row = normalize.normalize_event(event)
            game_id = row.get("game_id")
            if game_id is None or game_id in seen:
                continue        # a game can appear on two dates around midnight
            seen.add(game_id)
            rows.append(row)

    frame = normalize.to_frame(rows, "schedule")
    if len(frame):
        frame = frame[frame["season"] == season].reset_index(drop=True)
    return frame


# ── step 2: the game feeds, from the summary endpoint ────────────────────────
def _fetch_round(game_ids, workers, label):
    """One pass of summary fetches. Returns (rows by feed, ids that came back empty)."""
    bar = _progress(len(game_ids), label)

    def one(game_id):
        payload = client.fetch_summary(game_id)
        if not payload:
            return None
        return normalize.normalize_summary(payload)

    payloads = client.map_concurrent(
        one, game_ids, workers=workers, on_error="collect", progress=bar)
    if bar is not None:
        bar.close()

    collected = {feed: [] for feed in ("pbp", "player_box", "team_box")}
    missing = []
    for game_id, normalized in zip(game_ids, payloads):
        if not normalized:
            missing.append(game_id)
            continue
        for feed, rows in normalized.items():
            collected[feed].extend(rows)
    return collected, missing


def fetch_games(game_ids, workers=8, label="summaries", sweeps=2):
    """Fetch and normalize summaries. Returns {feed: DataFrame} plus the misses.

    ESPN throttles hard on long runs, and a game whose request was throttled to
    death is indistinguishable at this level from one that genuinely has no
    summary. So anything that comes back empty gets another pass at lower
    concurrency before it's called missing — during the 2021-2025 backfill that
    was the difference between 32 dropped games and none.
    """
    collected = {feed: [] for feed in ("pbp", "player_box", "team_box")}
    remaining = list(game_ids)

    for sweep in range(sweeps + 1):
        if not remaining:
            break
        sweep_workers = workers if sweep == 0 else max(2, workers // (2 * sweep))
        sweep_label = label if sweep == 0 else f"{label} retry {sweep}"
        rows, remaining = _fetch_round(remaining, sweep_workers, sweep_label)
        for feed, feed_rows in rows.items():
            collected[feed].extend(feed_rows)
        if remaining and sweep < sweeps:
            print(f"  {len(remaining)} game(s) came back empty — retrying at "
                  f"{max(2, workers // (2 * (sweep + 1)))} workers")

    frames = {feed: normalize.to_frame(rows, feed) for feed, rows in collected.items()}
    return frames, remaining


# ── step 3: merge into the season cache ──────────────────────────────────────
def load_cached(season, feed):
    path = feed_path(season, feed)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def merge_and_write(season, feed, fresh):
    """Replace every refreshed game's rows, keep the rest, write the parquet."""
    cached = load_cached(season, feed)
    if cached is not None and len(cached):
        if fresh is not None and len(fresh):
            refreshed = set(fresh["game_id"].dropna().astype("int64"))
            kept = cached[~cached["game_id"].astype("int64").isin(refreshed)]
            merged = pd.concat([kept, fresh], ignore_index=True)
        else:
            merged = cached
    else:
        merged = fresh if fresh is not None else pd.DataFrame()

    if not len(merged):
        return 0

    sort_columns = [column for column in ("game_id", "sequence_number") if column in merged.columns]
    merged = merged.sort_values(sort_columns).reset_index(drop=True)

    os.makedirs(season_dir(season), exist_ok=True)
    merged.to_parquet(feed_path(season, feed), index=False)
    return len(merged)


def games_needing_fetch(schedule, season, full, lookback_days=INCREMENTAL_LOOKBACK_DAYS):
    """Which game ids to pull summaries for.

    A full run takes every completed game. An incremental run takes completed
    games that either aren't in the cache yet or finished inside the lookback
    window (box scores get corrected for a day or two after the final buzzer).
    """
    if not len(schedule):
        return []
    completed = schedule[schedule["status_type_completed"].fillna(False).astype(bool)]
    candidates = set(completed["game_id"].dropna().astype("int64"))

    if full:
        return sorted(candidates)

    cached = load_cached(season, "player_box")
    already = set()
    if cached is not None and len(cached):
        already = set(cached["game_id"].dropna().astype("int64"))

    cutoff = dt.date.today() - dt.timedelta(days=lookback_days)
    recent_frame = completed[pd.to_datetime(completed["game_date"].astype("string")).dt.date >= cutoff]
    recent = set(recent_frame["game_id"].dropna().astype("int64"))

    return sorted((candidates - already) | (candidates & recent))


def repair_season(season, workers=4):
    """Re-fetch just the games a previous run failed to get, and merge them.

    Cheaper and safer than re-pulling the season: it touches only the gaps, so
    the rest of an archive stays exactly as it was first captured.
    """
    manifest = read_manifest(season)
    missing = [int(game_id) for game_id in manifest.get("missing_summaries", [])]

    schedule = load_cached(season, "schedule")
    cached_box = load_cached(season, "player_box")
    if schedule is not None and cached_box is not None:
        completed = schedule[schedule["status_type_completed"].fillna(False).astype(bool)]
        expected = set(completed["game_id"].dropna().astype("int64"))
        have = set(cached_box["game_id"].dropna().astype("int64"))
        missing = sorted(set(missing) | (expected - have))

    if not missing:
        print(f"[{season}] nothing to repair")
        return []

    print(f"[{season}] repairing {len(missing)} game(s) at {workers} workers")
    frames, still_missing = fetch_games(missing, workers=workers,
                                        label=f"repair {season}", sweeps=3)
    for feed in ("pbp", "player_box", "team_box"):
        rows = merge_and_write(season, feed, frames.get(feed))
        print(f"[{season}] {feed:<11} {rows:>9,} rows")

    recovered = len(missing) - len(still_missing)
    print(f"[{season}] recovered {recovered} of {len(missing)}")
    write_manifest(
        season,
        missing_summaries=[int(game_id) for game_id in still_missing],
        archived=bool(season < current_season() and not still_missing),
    )
    return still_missing


def ingest_season(season, full=False, workers=8, force=False, dates=None):
    manifest = read_manifest(season)
    if manifest.get("archived") and not force:
        print(f"season {season} is archived ({manifest.get('games', '?')} games, "
              f"updated {manifest.get('updated_at', '?')}) — skipping. Use --force to rebuild.")
        return

    if dates:
        scan_dates = dates
        full = True
    elif full:
        scan_dates = date_strings(*season_window(season))
    else:
        start, end = season_window(season)
        today = dt.date.today()
        window_start = max(start, min(today, end) - dt.timedelta(days=INCREMENTAL_LOOKBACK_DAYS))
        window_end = min(end, today)
        if window_end < window_start:
            print(f"season {season} is outside its window; nothing to scan.")
            return
        scan_dates = date_strings(window_start, window_end)

    print(f"[{season}] scanning {len(scan_dates)} date(s): {scan_dates[0]} → {scan_dates[-1]}")
    schedule = scan_schedule(season, scan_dates, workers=workers)
    print(f"[{season}] scoreboard returned {len(schedule)} games")

    game_ids = games_needing_fetch(schedule, season, full=full)
    print(f"[{season}] fetching summaries for {len(game_ids)} game(s)")

    frames, missing = ({}, [])
    if game_ids:
        frames, missing = fetch_games(game_ids, workers=workers, label=f"summaries {season}")
        if missing:
            print(f"[{season}] {len(missing)} game(s) had no summary payload")

    counts = {}
    for feed in ("pbp", "player_box", "team_box"):
        counts[feed] = merge_and_write(season, feed, frames.get(feed))
    counts["schedule"] = merge_and_write(season, "schedule", schedule)

    for feed, rows in counts.items():
        print(f"[{season}] {feed:<11} {rows:>9,} rows → {feed_path(season, feed)}")

    is_past_season = season < current_season()
    write_manifest(
        season,
        games=int(counts["schedule"]),
        rows={feed: int(count) for feed, count in counts.items()},
        last_mode="full" if full else "incremental",
        missing_summaries=[int(game_id) for game_id in missing],
        archived=bool(is_past_season and full and not missing),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, default=None,
                        help="season to ingest (defaults to the current season)")
    parser.add_argument("--backfill", nargs=2, type=int, metavar=("FIRST", "LAST"),
                        help="ingest a range of past seasons in full")
    parser.add_argument("--full", action="store_true",
                        help="rescan the whole season window instead of the recent days")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the season is marked archived")
    parser.add_argument("--dates", nargs="+", metavar="YYYYMMDD",
                        help="ingest specific dates only")
    parser.add_argument("--repair", action="store_true",
                        help="re-fetch only the games a previous run failed to get")
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent requests (default 8; be polite)")
    arguments = parser.parse_args(argv)

    if arguments.backfill:
        first, last = arguments.backfill
        for season in range(first, last + 1):
            if arguments.repair:
                repair_season(season, workers=max(2, arguments.workers // 2))
            else:
                ingest_season(season, full=True, workers=arguments.workers,
                              force=arguments.force)
        return

    season = arguments.season or current_season()
    if arguments.repair:
        repair_season(season, workers=max(2, arguments.workers // 2))
        return
    ingest_season(season, full=arguments.full, workers=arguments.workers,
                  force=arguments.force, dates=arguments.dates)


if __name__ == "__main__":
    main()
