"""
build_ingest.py — nightly STEP 0: download the four ESPN feeds once.

WHY THIS FILE EXISTS
--------------------
The pipeline needs the same four raw feeds (play-by-play, player box, team box,
schedule) in several builders. Before this step each builder downloaded them
independently, so the nightly pulled the ~1GB PBP four separate times — the load
that nearly OOMs the runner. This script downloads each feed ONCE and caches it
as raw_{SEASON}/{feed}.parquet; every downstream builder reads the cache via
hoglib.feeds (falling back to a live download only for ad-hoc single-script
runs). This is the one place that talks to ESPN for these feeds.

Outputs (into raw_{SEASON}/):
    pbp.parquet          full play-by-play
    player_box.parquet   per-player box scores
    team_box.parquet     per-team box scores
    schedule.parquet     season schedule

Run locally:
    python build_ingest.py
    OVERRIDE_SEASON=2025 python build_ingest.py
"""
import os

from hoglib.season import detect_season
from hoglib import feeds

SEASON = detect_season()

print(f"Ingesting {SEASON} ESPN feeds → {os.path.relpath(feeds.raw_dir(SEASON))}/ …")
counts = feeds.write_cache(SEASON)
print("\nDone.")
for feed in feeds.FEEDS:
    print(f"  {feed + '.parquet':<22s} — {counts[feed]:,} rows")
