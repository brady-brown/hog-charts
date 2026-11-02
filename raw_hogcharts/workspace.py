"""Assemble a mirror workspace where the existing builders run on ESPN data.

Nothing in the root pipeline is edited or copied. Instead raw_hogcharts/build/
is populated with symlinks back to the real scripts, and the builders — which
all root their paths at `os.path.dirname(os.path.abspath(__file__))` — resolve
that root to the mirror instead of the repo. os.path.abspath does not follow
symlinks, which is the whole trick:

    raw_hogcharts/build/build_site.py   → ../../build_site.py   (symlink)
    PROJECT_ROOT inside it              = raw_hogcharts/build
    its outputs                         → raw_hogcharts/build/site/data/...

Three pieces make the mirror self-contained:

  raw_{season}/     symlinked to raw_hogcharts/data/raw_{season}, so
                    hoglib.feeds finds ESPN-sourced feeds where it would
                    normally find the sportsdataverse cache.

  sportsdataverse/  a local shim package. The mirror directory leads sys.path,
                    so `import sportsdataverse.mbb` resolves here rather than to
                    the installed library, and the few builders that still call
                    load_mbb_* directly are served from the ESPN parquet. Any
                    call the shim can't serve raises instead of hitting the
                    network — the point is that no code path can silently fall
                    back to sportsdataverse.

  site/             the duplicate website. Markup, css, js and images are
                    symlinked to the live site so the two never drift; data/ is
                    a real directory the mirror pipeline fills in itself.
"""
import os
import shutil

MIRROR_NAME = "build"

# Scripts and packages the builders import or exec. Symlinked, never copied.
LINKED_SCRIPTS = (
    "build_artifacts.py", "build_shots_data.py", "build_player_stats.py",
    "build_lineups.py", "build_onoff_rapm.py", "build_points_resp.py",
    "build_shot_diet.py", "build_site.py", "build_scout.py", "build_postseason.py",
    "build_ingest.py", "build_conf_stats.py", "build_history.py",
    "conferences.py", "patch_conferences.py", "predictor.py", "impact_artifact.py",
)
LINKED_PACKAGES = ("hoglib", "coaches_suite")

# Static site assets that should always match the live site.
SITE_LINKED_ENTRIES = ("css", "js", "img", "_headers", "index.html", "glossary.html",
                       "lineup-stats.html", "net-ratings.html", "player-stats.html",
                       "postseason.html", "prediction.html", "scout-report.html",
                       "shot-charts.html")
# Small hand-maintained inputs that live under site/data and are not rebuilt.
SITE_DATA_SEEDS = ("zone-baselines.json",)


def package_dir():
    return os.path.dirname(os.path.abspath(__file__))


def repo_root():
    return os.path.dirname(package_dir())


def mirror_dir():
    return os.path.join(package_dir(), MIRROR_NAME)


def _link(source, destination):
    """Point destination at source, replacing whatever is there."""
    if os.path.islink(destination):
        if os.path.realpath(destination) == os.path.realpath(source):
            return
        os.unlink(destination)
    elif os.path.isdir(destination):
        raise RuntimeError(f"{destination} is a real directory; refusing to replace it")
    elif os.path.exists(destination):
        os.remove(destination)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    os.symlink(os.path.relpath(source, os.path.dirname(destination)), destination)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


# ── the sportsdataverse shim ─────────────────────────────────────────────────
_SHIM_LOADERS = '''"""Local stand-in for sportsdataverse.mbb, serving ESPN-sourced parquet.

The mirror workspace leads sys.path, so builders that still call
`mbb_loaders.load_mbb_*` get these functions instead of the real library. They
read raw_{season}/{feed}.parquet — the tables raw_hogcharts/ingest.py builds
from ESPN — and return polars frames, matching what the library returns.

There is deliberately no network path here. If a season has not been ingested,
this raises rather than reaching for sportsdataverse's parquet releases.
"""
import os

import pandas as pd
import polars as pl

MIRROR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEED_FILES = {
    "pbp": "pbp.parquet",
    "player_box": "player_box.parquet",
    "team_box": "team_box.parquet",
    "schedule": "schedule.parquet",
}


class MissingSeasonError(RuntimeError):
    pass


def _read(feed, seasons):
    if isinstance(seasons, int):
        seasons = [seasons]
    frames = []
    for season in seasons:
        path = os.path.join(MIRROR_ROOT, f"raw_{int(season)}", FEED_FILES[feed])
        if not os.path.exists(path):
            raise MissingSeasonError(
                f"no ESPN {feed} for season {season} ({path}). "
                f"Run: python -m raw_hogcharts.ingest --season {season} --full"
            )
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _out(frame, return_as_pandas):
    return frame if return_as_pandas else pl.from_pandas(frame)


def load_mbb_pbp(seasons, return_as_pandas=False):
    return _out(_read("pbp", seasons), return_as_pandas)


def load_mbb_player_boxscore(seasons, return_as_pandas=False):
    return _out(_read("player_box", seasons), return_as_pandas)


def load_mbb_team_boxscore(seasons, return_as_pandas=False):
    return _out(_read("team_box", seasons), return_as_pandas)


def load_mbb_schedule(seasons, return_as_pandas=False):
    return _out(_read("schedule", seasons), return_as_pandas)
'''

_SHIM_MBB_INIT = '''"""Shim namespace mirroring sportsdataverse.mbb's loader surface."""
from . import mbb_loaders
from .mbb_loaders import (
    load_mbb_pbp,
    load_mbb_player_boxscore,
    load_mbb_schedule,
    load_mbb_team_boxscore,
)

__all__ = [
    "mbb_loaders",
    "load_mbb_pbp",
    "load_mbb_player_boxscore",
    "load_mbb_schedule",
    "load_mbb_team_boxscore",
]


def __getattr__(name):
    raise AttributeError(
        f"sportsdataverse.mbb.{name} is not available in raw_hogcharts. "
        "This workspace reads ESPN endpoints only; add what you need to "
        "raw_hogcharts/espn/ rather than reaching for sportsdataverse."
    )
'''

_SHIM_ROOT_INIT = '''"""Shadow package: raw_hogcharts serves these tables from ESPN, not from
sportsdataverse's parquet releases. Only the mbb loader surface exists here.
"""
from . import mbb

__all__ = ["mbb"]
'''


def write_shim(mirror):
    shim = os.path.join(mirror, "sportsdataverse")
    _write(os.path.join(shim, "__init__.py"), _SHIM_ROOT_INIT)
    _write(os.path.join(shim, "mbb", "__init__.py"), _SHIM_MBB_INIT)
    _write(os.path.join(shim, "mbb", "mbb_loaders.py"), _SHIM_LOADERS)


# ── the duplicate site tree ──────────────────────────────────────────────────
def build_site_tree(mirror):
    live_site = os.path.join(repo_root(), "site")
    mirror_site = os.path.join(mirror, "site")
    os.makedirs(mirror_site, exist_ok=True)

    for entry in SITE_LINKED_ENTRIES:
        source = os.path.join(live_site, entry)
        if os.path.exists(source):
            _link(source, os.path.join(mirror_site, entry))

    mirror_data = os.path.join(mirror_site, "data")
    os.makedirs(mirror_data, exist_ok=True)
    for seed in SITE_DATA_SEEDS:
        source = os.path.join(live_site, "data", seed)
        destination = os.path.join(mirror_data, seed)
        if os.path.exists(source) and not os.path.exists(destination):
            shutil.copy2(source, destination)


def link_seasons(mirror):
    """Expose every ingested season as raw_{season}/ inside the mirror."""
    data_root = os.path.join(package_dir(), "data")
    linked = []
    if not os.path.isdir(data_root):
        return linked
    for entry in sorted(os.listdir(data_root)):
        if entry.startswith("raw_") and os.path.isdir(os.path.join(data_root, entry)):
            _link(os.path.join(data_root, entry), os.path.join(mirror, entry))
            linked.append(entry)
    return linked


def ensure(verbose=True):
    """Create or refresh the mirror. Safe to run repeatedly."""
    mirror = mirror_dir()
    os.makedirs(mirror, exist_ok=True)

    for script in LINKED_SCRIPTS:
        source = os.path.join(repo_root(), script)
        if os.path.exists(source):
            _link(source, os.path.join(mirror, script))
    for package in LINKED_PACKAGES:
        source = os.path.join(repo_root(), package)
        if os.path.exists(source):
            _link(source, os.path.join(mirror, package))

    write_shim(mirror)
    build_site_tree(mirror)
    seasons = link_seasons(mirror)

    _write(os.path.join(mirror, ".gitignore"), "*\n")

    if verbose:
        print(f"mirror workspace ready: {mirror}")
        print(f"  linked scripts   {len(LINKED_SCRIPTS)}")
        print(f"  linked seasons   {', '.join(seasons) if seasons else 'none ingested yet'}")
        print(f"  duplicate site   {os.path.join(mirror, 'site')}")
    return mirror


if __name__ == "__main__":
    ensure()
