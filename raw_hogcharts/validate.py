"""Diff the ESPN-sourced feeds against the sportsdataverse cache.

This is what makes the swap safe: before any builder runs on ESPN-sourced data,
prove the tables are the same tables. It compares row-for-row on a natural key
and reports per-column mismatch counts, restricted to the games present in both
caches so a partial ingest still validates cleanly.

    python -m raw_hogcharts.validate --season 2026
    python -m raw_hogcharts.validate --season 2026 --all-columns
    python -m raw_hogcharts.validate --season 2026 --feed pbp --samples 10

The reference cache is raw_{season}/ at the repo root (written by build_ingest.py
from the sportsdataverse parquet releases). If it isn't there, run build_ingest
first or point --reference at another directory.
"""
import argparse
import os
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "raw_hogcharts"

from . import ingest

# The columns the build pipeline actually reads. A mismatch in any of these is a
# real defect; a mismatch outside them is informational (--all-columns shows it).
PIPELINE_COLUMNS = {
    "pbp": [
        "athlete_id_1", "athlete_id_2", "away_score", "away_team_id", "clock_minutes",
        "clock_seconds", "coordinate_x", "coordinate_y", "game_date", "game_id",
        "home_score", "home_team_id", "period_number", "points_attempted",
        "score_value", "scoring_play", "season", "season_type", "sequence_number",
        "shooting_play", "team_id", "text", "type_text",
    ],
    "player_box": [
        "active", "assists", "athlete_display_name", "athlete_id", "athlete_jersey",
        "athlete_position_name", "blocks", "defensive_rebounds", "did_not_play",
        "field_goals_attempted", "field_goals_made", "fouls", "free_throws_attempted",
        "free_throws_made", "game_date", "game_id", "minutes", "offensive_rebounds",
        "opponent_team_id", "opponent_team_score", "points", "rebounds", "season",
        "season_type", "starter", "steals", "team_display_name", "team_id",
        "team_location", "team_name", "team_score", "team_winner",
        "three_point_field_goals_attempted", "three_point_field_goals_made", "turnovers",
    ],
    "team_box": [
        "assists", "blocks", "defensive_rebounds", "field_goals_attempted",
        "field_goals_made", "fouls", "free_throws_attempted", "free_throws_made",
        "game_date", "game_id", "offensive_rebounds", "opponent_team_id",
        "opponent_team_score", "season", "season_type", "steals", "team_display_name",
        "team_home_away", "team_id", "team_location", "team_name", "team_score",
        "team_winner", "three_point_field_goals_attempted",
        "three_point_field_goals_made", "total_rebounds", "total_turnovers", "turnovers",
    ],
    "schedule": [
        "away_conference_id", "away_display_name", "away_id", "away_location",
        "away_name", "away_score", "conference_competition", "date", "game_date",
        "game_id", "home_conference_id", "home_display_name", "home_id",
        "home_location", "home_name", "home_score", "home_winner", "id",
        "neutral_site", "notes_headline", "season", "season_type",
        "status_type_completed", "tournament_id",
    ],
}

KEYS = {
    "pbp": ["game_id", "sequence_number"],
    "player_box": ["game_id", "team_id", "athlete_id"],
    "team_box": ["game_id", "team_id"],
    "schedule": ["game_id"],
}


# sportsdataverse publishes its parquet as plain GitHub release assets, so a
# past season can be pulled for comparison without importing the library. This
# is a test fixture only — nothing in the pipeline reads these.
SDV_RELEASE_URLS = {
    "pbp": ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
            "espn_mens_college_basketball_pbp/play_by_play_{season}.parquet"),
    "player_box": ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
                   "espn_mens_college_basketball_player_boxscores/player_box_{season}.parquet"),
    "team_box": ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
                 "espn_mens_college_basketball_team_boxscores/team_box_{season}.parquet"),
    "schedule": ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
                 "espn_mens_college_basketball_schedules/mbb_schedule_{season}.parquet"),
}


def reference_dir(season, override=None):
    if override:
        return override
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, f"raw_{season}")


def download_reference(season, feeds):
    """Fetch the sportsdataverse parquet for a season into a scratch cache.

    Lets a freshly backfilled archive be checked against the old source once,
    before it's frozen. Downloaded files live under data/.reference/ and are
    never read by the pipeline.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    destination = os.path.join(package_dir, "data", ".reference", str(season))
    os.makedirs(destination, exist_ok=True)
    for feed in feeds:
        path = os.path.join(destination, f"{feed}.parquet")
        if os.path.exists(path):
            continue
        url = SDV_RELEASE_URLS[feed].format(season=season)
        print(f"  downloading reference {feed} for {season} …")
        pd.read_parquet(url).to_parquet(path, index=False)
    return destination


def _plain(frame, columns=None):
    """Drop pyarrow backing so comparisons don't trip over dtype identity."""
    out = pd.DataFrame(index=frame.index)
    for column in (columns if columns is not None else frame.columns):
        if column not in frame.columns:
            continue
        series = frame[column]
        if str(series.dtype).startswith(("date32", "timestamp")):
            out[column] = pd.to_datetime(
                series.astype("string"), utc=True, errors="coerce").dt.date.astype("string")
        elif str(series.dtype).startswith(("large_string", "string")):
            out[column] = series.astype("string")
        elif str(series.dtype).startswith("bool"):
            out[column] = series.astype("boolean")
        else:
            out[column] = pd.to_numeric(series, errors="coerce").astype("Float64")
    return out


def compare_feed(feed, mine, theirs, columns, samples=3, tolerance=1e-6):
    key = KEYS[feed]
    shared_games = set(mine["game_id"].dropna().astype("int64")) & set(
        theirs["game_id"].dropna().astype("int64"))
    mine = mine[mine["game_id"].astype("int64").isin(shared_games)]
    theirs = theirs[theirs["game_id"].astype("int64").isin(shared_games)]

    report = {
        "feed": feed,
        "shared_games": len(shared_games),
        "rows_mine": len(mine),
        "rows_theirs": len(theirs),
        "missing_keys": 0,
        "extra_keys": 0,
        "mismatches": {},
        "samples": {},
    }
    if not shared_games:
        return report

    wanted = sorted(set(columns) | set(key))
    mine_keyed = _plain(mine, wanted).set_index(
        [mine[column].astype("Int64") for column in key])
    theirs_keyed = _plain(theirs, wanted).set_index(
        [theirs[column].astype("Int64") for column in key])
    mine_keyed = mine_keyed[~mine_keyed.index.duplicated()]
    theirs_keyed = theirs_keyed[~theirs_keyed.index.duplicated()]

    report["missing_keys"] = len(theirs_keyed.index.difference(mine_keyed.index))
    report["extra_keys"] = len(mine_keyed.index.difference(theirs_keyed.index))

    common_index = mine_keyed.index.intersection(theirs_keyed.index)
    report["compared_rows"] = len(common_index)
    left = mine_keyed.loc[common_index]
    right = theirs_keyed.loc[common_index]

    for column in columns:
        if column not in left.columns or column not in right.columns:
            report["mismatches"][column] = "absent"
            continue
        left_values, right_values = left[column], right[column]
        both_null = left_values.isna() & right_values.isna()
        if str(left_values.dtype) == "Float64" and str(right_values.dtype) == "Float64":
            differs = ((left_values - right_values).abs() > tolerance).fillna(True)
        else:
            differs = (left_values != right_values).fillna(True)
        differs = differs & ~both_null
        count = int(differs.sum())
        if count:
            report["mismatches"][column] = count
            sample = pd.DataFrame({"espn": left_values[differs], "sdv": right_values[differs]})
            report["samples"][column] = sample.head(samples)
    return report


def print_report(report):
    feed = report["feed"]
    print(f"\n── {feed} " + "─" * (60 - len(feed)))
    print(f"   shared games   {report['shared_games']:,}")
    print(f"   rows           espn {report['rows_mine']:,}  |  sdv {report['rows_theirs']:,}"
          f"  |  compared {report.get('compared_rows', 0):,}")
    if report["missing_keys"] or report["extra_keys"]:
        print(f"   key gaps       missing from espn {report['missing_keys']:,}"
              f"  |  extra in espn {report['extra_keys']:,}")
    if not report["mismatches"]:
        print("   columns        all match")
        return True
    print("   columns        MISMATCH")
    compared = max(report.get("compared_rows", 0), 1)
    for column, count in sorted(report["mismatches"].items(),
                                key=lambda item: -item[1] if isinstance(item[1], int) else 0):
        if isinstance(count, int):
            print(f"     {column:<38} {count:>8,}  ({100 * count / compared:.3f}%)")
        else:
            print(f"     {column:<38} {count}")
        if column in report["samples"]:
            for line in report["samples"][column].to_string().splitlines():
                print(f"        {line}")
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--feed", choices=ingest.FEEDS, action="append")
    parser.add_argument("--reference", default=None,
                        help="directory holding the sportsdataverse parquet (default raw_{season}/)")
    parser.add_argument("--all-columns", action="store_true",
                        help="compare every shared column, not just the ones the pipeline reads")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--download-reference", action="store_true",
                        help="pull the sportsdataverse parquet for this season to compare against "
                             "(one-off check for a backfilled archive)")
    arguments = parser.parse_args(argv)

    season = arguments.season or ingest.current_season()
    feeds = arguments.feed or list(ingest.FEEDS)
    reference = (download_reference(season, feeds) if arguments.download_reference
                 else reference_dir(season, arguments.reference))

    print(f"comparing raw_hogcharts/data/raw_{season} against {reference}")
    all_clean = True
    for feed in feeds:
        mine_path = ingest.feed_path(season, feed)
        theirs_path = os.path.join(reference, f"{feed}.parquet")
        if not os.path.exists(mine_path):
            print(f"\n── {feed}: no ESPN parquet yet ({mine_path})")
            all_clean = False
            continue
        if not os.path.exists(theirs_path):
            print(f"\n── {feed}: no reference parquet ({theirs_path})")
            continue
        mine = pd.read_parquet(mine_path)
        theirs = pd.read_parquet(theirs_path)
        columns = (sorted(set(mine.columns) & set(theirs.columns))
                   if arguments.all_columns else PIPELINE_COLUMNS[feed])
        all_clean &= print_report(compare_feed(feed, mine, theirs, columns, arguments.samples))

    print("\n" + ("all feeds match the reference" if all_clean else "differences found — see above"))
    return 0 if all_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
