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
import os
from datetime import date


def detect_season(override_env="OVERRIDE_SEASON"):
    """Return the current season year, honouring the OVERRIDE_SEASON env var."""
    override = os.environ.get(override_env)
    if override:
        return int(override)
    today = date.today()
    return today.year + 1 if today.month >= 11 else today.year
