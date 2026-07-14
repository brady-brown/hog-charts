"""Shared ESPN feed access for the build pipeline.

Four raw feeds come from ESPN via sportsdataverse: pbp (play-by-play),
player_box, team_box, schedule. Before this, several builders each downloaded
the same feeds independently — the nightly pulled the ~1GB PBP four separate
times, which is what nearly OOMs the runner.

build_ingest.py (nightly step 0) downloads each feed ONCE and caches it as
raw_{season}/{feed}.parquet. Every builder then reads the cache via the loaders
here. When the cache is absent (an ad-hoc local run of a single builder) the
loader falls back to a live download, so scripts still work standalone.

Loaders return a pandas DataFrame for a single season — the same shape the
builders previously got from mbb.load_mbb_*(seasons=[season], return_as_pandas=True).
"""
import os

FEEDS = ("pbp", "player_box", "team_box", "schedule")


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def raw_dir(season):
    return os.path.join(_repo_root(), f"raw_{season}")


def cache_path(season, feed):
    return os.path.join(raw_dir(season), f"{feed}.parquet")


# ── live downloaders (the fallback + what build_ingest writes) ───────────────
def _download(feed, season):
    import sportsdataverse.mbb as mbb
    loaders = {
        "pbp":        mbb.load_mbb_pbp,
        "player_box": mbb.load_mbb_player_boxscore,
        "team_box":   mbb.load_mbb_team_boxscore,
        "schedule":   mbb.load_mbb_schedule,
    }
    return loaders[feed](seasons=[season], return_as_pandas=True)


def _load(feed, season):
    path = cache_path(season, feed)
    if os.path.exists(path):
        import pandas as pd
        return pd.read_parquet(path)
    return _download(feed, season)


def load_pbp(season):        return _load("pbp", season)
def load_player_box(season): return _load("player_box", season)
def load_team_box(season):   return _load("team_box", season)
def load_schedule(season):   return _load("schedule", season)


def write_cache(season, feeds=FEEDS):
    """Download each feed once and write raw_{season}/{feed}.parquet. Returns
    {feed: row_count}. Used by build_ingest.py."""
    os.makedirs(raw_dir(season), exist_ok=True)
    counts = {}
    for feed in feeds:
        df = _download(feed, season)
        df.to_parquet(cache_path(season, feed), index=False)
        counts[feed] = len(df)
    return counts
