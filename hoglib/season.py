"""Season detection — one copy for the whole pipeline.

SEASON is the calendar year a season ENDS. November/December games belong to the
next year's season (e.g. Nov 2025 → season 2026). The OVERRIDE_SEASON env var
forces a specific season (build_history.py uses it to regenerate past seasons).

This is lifted verbatim from the block every build_*.py used to re-declare:

    _season_override = os.environ.get("OVERRIDE_SEASON")
    if _season_override:
        SEASON = int(_season_override)
    else:
        _today = date.today()
        SEASON = _today.year + 1 if _today.month >= 11 else _today.year
"""
import math
import os
from datetime import date


def detect_season(override_env="OVERRIDE_SEASON"):
    """Return the current season year, honouring the OVERRIDE_SEASON env var."""
    override = os.environ.get(override_env)
    if override:
        return int(override)
    today = date.today()
    return today.year + 1 if today.month >= 11 else today.year


# Below this many games the scaled gates are too low to exclude non-D-I opponents
# (they play 4 or fewer), so builders drop teams with no conference instead.
NON_D1_GATE_GAMES = 8

# Possession floors were set for a full schedule, so they scale up to this many games.
FULL_SEASON_GAMES = 30

# Below this many field goal attempts league-wide, zone rates are too noisy to use
# as baselines (roughly the first month), so last season's stand in.
MIN_BASELINE_SHOTS = 100_000


def leader_games(games, rank=5):
    """Games played by the season's leaders, for scaling the gates below.

    Uses the 5th-highest count rather than the max: a non-D-I school stacking
    guarantee games can out-play every D-I team early (Bethesda had 6 in the
    first week of 2027 when no D-I team had more than 4), and taking its count
    as the leader pushes every scaled gate up and drops real teams.
    """
    games = [g for g in games if g == g]     # drop NaN
    if not games:
        return float("nan")
    return sorted(games, reverse=True)[min(rank, len(games)) - 1]


# The live season's games gates stop rising here: three games behind the leader
# climbs past what slower-scheduled teams have played and hides them.
LIVE_MIN_GAMES = 4


def is_live_season(override_env="OVERRIDE_SEASON"):
    """True for the nightly's current season; False for a past-season regen."""
    return not os.environ.get(override_env)


def scaled_min_games(full_min, most_played, live=None):
    """Scale a games-played gate to how far the season has got.

    The full gates (20 team games, 5 for net ratings, ...) would blank the live
    season's tables for weeks after opening night. Early on the gate is three games
    behind whoever has played the most. In the live season it then holds at
    LIVE_MIN_GAMES; a finished season (past-season regen) climbs to full_min.
    live defaults to is_live_season(). site/js/common.js scaledMinGames applies the
    same rule to the pages' default filters.
    """
    if not most_played > 0:      # also False for NaN (no games at all)
        return full_min
    gate = max(1, int(most_played) - 3)
    if is_live_season() if live is None else live:
        gate = min(gate, LIVE_MIN_GAMES)
    return min(full_min, gate)


def needs_conference_gate(most_played):
    """Whether builders should drop teams with no conference (non-D-I opponents).

    A games gate only screens them out once it's above the ~4 games they play:
    early in any season, and all of the live season (its gate holds at 4).
    """
    return is_live_season() or not most_played >= NON_D1_GATE_GAMES


def scaled_min_possessions(full_min, most_played):
    """Scale a possessions floor to how many games the leader has played."""
    if not most_played > 0:      # also False for NaN (no games at all)
        return full_min
    return min(full_min, math.ceil(full_min * most_played / FULL_SEASON_GAMES))
