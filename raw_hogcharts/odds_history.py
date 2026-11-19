"""Historical betting lines from ESPN's core odds API, for model studies.

    python -m raw_hogcharts.odds_history 2021 2022 ... [--workers 8] [--force]

For every completed game in raw_{season}/schedule.parquet, one request to

    sports.core.api.espn.com/v2/.../events/{id}/competitions/{id}/odds

which lists every book ESPN tracked for that game. One row per (game, book) is
written to raw_hogcharts/data/raw_{season}/odds_history.parquet:

    game_id, provider, open_spread, close_spread, open_total, close_total

Spreads are home-relative (negative = home favored), as everywhere else.
Live-odds books are skipped. What ESPN kept varies by season: 2021-22 carry
only the final pre-game number ("current", stored as close); open + close from
2024 on. DraftKings is listed 2021-24 and 2026; 2025 is ESPN BET only.

The live season's lines come from the nightly ingest instead (lines.parquet).
Resumable: a season already written is skipped unless --force.
"""
import argparse
import os

import pandas as pd

from raw_hogcharts import ingest
from raw_hogcharts.espn import client

CORE = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/"
        "mens-college-basketball/events/{g}/competitions/{g}/odds")


def _num(text):
    if text is None:
        return None
    text = str(text).strip().lower().lstrip("ou")
    if text in ("pk", "pick", "even", "ev"):
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def _point(side_odds, when):
    return _num((((side_odds or {}).get(when) or {}).get("pointSpread") or {}).get("american"))


def _total(item, when):
    return _num((((item.get(when) or {}).get("total")) or {}).get("american"))


def _plausible_spread(x):
    return x if x is not None and abs(x) <= 60 else None


def _plausible_total(x):
    return x if x is not None and 90 <= x <= 260 else None


def parse_item(game_id, item):
    """The close comes from the item's top-level spread/overUnder (the final
    pre-game number, home-relative; matches the "details" text). The nested
    open/close blocks sometimes hold prices where lines should be (2023 ESPN BET:
    "-120"), so they're used only for the open, and only when plausible."""
    name = (item.get("provider") or {}).get("name") or ""
    if "live" in name.lower():
        return None
    home = item.get("homeTeamOdds") or {}
    close_spread = _plausible_spread(_num(item.get("spread")))
    if close_spread is None:
        close_spread = _plausible_spread(_point(home, "close")) or _plausible_spread(_point(home, "current"))
    close_total = _plausible_total(_num(item.get("overUnder"))) or _plausible_total(_total(item, "close"))
    return {
        "game_id": int(game_id), "provider": name,
        "open_spread": _plausible_spread(_point(home, "open")), "close_spread": close_spread,
        "open_total": _plausible_total(_total(item, "open")), "close_total": close_total,
    }


def fetch_game(game_id):
    payload = client.get_json(CORE.format(g=int(game_id))) or {}
    rows = [parse_item(game_id, item) for item in payload.get("items", [])]
    return [row for row in rows if row and row["close_spread"] is not None]


def output_path(season):
    return os.path.join(ingest.season_dir(season), "odds_history.parquet")


def fetch_season(season, workers=8, force=False):
    path = output_path(season)
    if os.path.exists(path) and not force:
        print(f"[{season}] already have {os.path.relpath(path)} — skipping (--force to redo)")
        return
    schedule = pd.read_parquet(ingest.feed_path(season, "schedule"),
                               columns=["game_id", "status_type_completed"])
    ids = schedule[schedule["status_type_completed"].fillna(False).astype(bool)]["game_id"] \
        .dropna().astype("int64").drop_duplicates().tolist()
    print(f"[{season}] fetching odds for {len(ids):,} games")
    bar = ingest._progress(len(ids), f"odds {season}")
    results = client.map_concurrent(fetch_game, ids, workers=workers, on_error="collect", progress=bar)
    if bar is not None:
        bar.close()
    rows = [row for game_rows in results if game_rows for row in game_rows]
    frame = pd.DataFrame(rows, columns=["game_id", "provider", "open_spread", "close_spread",
                                        "open_total", "close_total"])
    frame.to_parquet(path, index=False)
    with_line = frame["game_id"].nunique()
    print(f"[{season}] {len(frame):,} rows, {with_line:,} of {len(ids):,} games have a line → {os.path.relpath(path)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seasons", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for season in args.seasons:
        fetch_season(season, workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
