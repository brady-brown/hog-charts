"""JSON sanitizing + slug helpers for the build pipeline.

sanitize_for_json is the NaN-cleaning rule that MUST stay identical across every
builder: the browser's JSON.parse rejects bare NaN/Infinity and the page hangs
on 'Loading…'. It maps non-finite floats and pandas/numpy nulls to None and
downcasts numpy scalars to plain Python types.

This is a superset of the three former per-script copies (build_site,
build_scout, build_postseason) and produces identical output for each: their
data never carried the scalar types a given copy happened to omit (json.dump
would have raised otherwise), so the extra branches are no-ops there.

write_json is intentionally NOT centralized here — each script's wrapper differs
in its directory default, logging, and allow_nan, and those are kept per-script.
They all route through sanitize_for_json, which is the part that must not drift.
"""
import re

import numpy as np
import pandas as pd


def slugify(name):
    """Team/player name → URL-safe slug. MUST match site/js/common.js slugify()."""
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def _is_nonfinite(x):
    return x != x or x in (float("inf"), float("-inf"))


def sanitize_for_json(obj):
    """Recursively replace NaN/Inf and pandas/numpy nulls with None (browser-safe)."""
    if isinstance(obj, float):
        # np.float64 subclasses float, so this branch also catches it.
        return None if _is_nonfinite(obj) else obj
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if _is_nonfinite(v) else v
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    return obj
