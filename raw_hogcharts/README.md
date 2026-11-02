# raw_hogcharts

A second copy of the Hog Charts pipeline that gets its data straight from ESPN's
public endpoints instead of from sportsdataverse's parquet releases. Everything
lives in this folder; nothing in the root pipeline is edited.

The reason it exists: `sportsdataverse` is a middleman. Its `load_mbb_*`
functions download pre-built parquet from a GitHub release, and that parquet is
itself a normalization of the same two ESPN endpoints this package calls. If
those releases stop being published, or the library goes away, the nightly build
dies with it. ESPN is upstream of both.

## What talks to ESPN

Two endpoints supply the entire pipeline.

| endpoint | gives | calls |
|---|---|---|
| `scoreboard?dates=YYYYMMDD&groups=50&limit=500` | every D-I game on one Eastern date: teams, conference ids, scores, status, `tournamentId`, notes headline | one per date (~185/season) |
| `summary?event={game_id}` | one game: plays (with substitutions and shot coordinates), player box, team box, header | one per game (~6,300/season) |

## Layout

```
raw_hogcharts/
  espn/client.py       HTTP: retry/backoff on 429+5xx, bounded concurrency
  espn/normalize.py    ESPN JSON → the four feed tables, sdv-shaped
  ingest.py            build data/raw_{season}/{feed}.parquet
  validate.py          diff those tables against the sportsdataverse cache
  workspace.py         assemble build/ — the mirror the builders run in
  run_pipeline.py      run the builders in nightly order inside the mirror
  data/                ingested feeds (gitignored)
  build/               mirror workspace + the duplicate site (gitignored)
```

## Running it

```bash
# one-time: pull a past season, then never touch it again
python -m raw_hogcharts.ingest --season 2025 --full
python -m raw_hogcharts.ingest --backfill 2021 2025

# prove the tables match sportsdataverse before trusting them
python -m raw_hogcharts.validate --season 2026

# nightly: refresh the current season, then build
python -m raw_hogcharts.ingest --season 2026
python -m raw_hogcharts.run_pipeline --season 2026

# or in one step
python -m raw_hogcharts.run_pipeline --season 2026 --ingest

# look at the result
python -m http.server -d raw_hogcharts/build/site 8001
```

A full season takes ~3 minutes at 8 workers (~40 requests/second sustained). A
nightly incremental run is a handful of scoreboard calls plus that day's games.

## The mirror workspace

The builders all root their paths at `os.path.dirname(os.path.abspath(__file__))`.
`os.path.abspath` doesn't follow symlinks, so a symlink to `build_site.py` inside
`build/` makes that script treat `build/` as the project root — reading
`build/raw_2026/`, writing `build/site/data/`. The scripts run verbatim.

Three pieces complete the isolation:

- **`build/raw_{season}/`** → symlink to `data/raw_{season}/`, so `hoglib.feeds`
  finds ESPN-sourced feeds where it would normally find the sportsdataverse cache.
- **`build/sportsdataverse/`** → a shim package. The mirror leads `sys.path`, so
  `import sportsdataverse` resolves here, not to the installed library. The few
  builders that still call `load_mbb_*` directly (`build_artifacts.py`,
  `predictor.py`) are served from the ESPN parquet, and anything the shim can't
  serve raises. No code path can silently fall back to sportsdataverse.
- **`build/site/`** → markup, css, js and images symlinked to the live site so
  they never drift; `data/` is a real directory this pipeline fills itself.

## Ingest model

Past seasons are immutable, so they're pulled once and frozen. A completed full
ingest of a past season writes `"archived": true` into that season's
`manifest.json`, and later runs refuse to touch it without `--force`.

The current season runs incrementally: rescan the last few days of scoreboard,
fetch summaries for games that are either new or finished inside a 4-day
lookback (box scores get corrected for a day or two), then merge — every
refreshed game's rows are replaced, the rest are kept.

This matters beyond speed. ESPN occasionally remaps athlete ids; re-pulling a
finished season can shift them under stats that were computed from the old ones.
Freezing archives avoids that entirely.

## Validation

`validate.py` diffs the ESPN tables against `raw_{season}/` at the repo root
(sportsdataverse's cache) on the columns the pipeline actually reads, keyed on
`(game_id, sequence_number)` for plays and `(game_id, team_id, athlete_id)` for
boxes.

For a season the root cache doesn't hold, `--download-reference` pulls
sportsdataverse's parquet straight from its GitHub release URL — no library
import — into `data/.reference/`. Use it to check a freshly backfilled archive
once, before freezing it:

```bash
python -m raw_hogcharts.validate --season 2025 --download-reference
```

Across all of 2026: **2,915,731 play rows on both sides**, every column
identical — coordinates, clocks, scores, play types, team and athlete
attribution. The differences that remain are all ESPN being *more current* than
the snapshot:

- Saint Francis is "Red Wolves" live and "Red Flash" in the cached parquet.
- Two players gained a "Jr." suffix.
- One athlete id was remapped (Terry Anderson: 5241590 → 5108084, same team,
  same 26 minutes and 17 points).

Nothing numeric differs.

The stronger check is the derived output — the numbers the site actually shows.
Running the builders on each source and diffing their CSVs, for 2026:

| output | result |
|---|---|
| `player_stats_all_2026.csv` | 9,986 players, all 35 numeric columns identical |
| `shots_2026.parquet` | 991,836 rows, identical |
| `mbb_onoff_2026_v2.csv` | 9,922 rows, all 49 numeric columns identical |
| `mbb_rapm_202526.csv` | 3,116 rows, identical |
| `points_resp_2026.csv` | 9,650 rows, identical |
| `shot_diet_2026_all.csv` | 9,393 rows, identical |
| `team_stats_all_2026.csv` | 727 teams, all 31 numeric columns identical |
| `net_ratings_all_2026.csv` | 727 teams, identical |

The on/off and RAPM match matters most: those run through the stint engine,
which reconstructs lineups from substitution events and the game clock. Getting
them bit-identical means the play stream is right down to sub timing.

## Conventions worth knowing

- **`game_date` is the US/Eastern date**, not UTC. 3,406 of 6,318 games in 2026
  fall on a different UTC date, and ESPN's `dates=` parameter is Eastern too.
- **Coordinates.** ESPN gives half-court integers; the feeds carry a full-court
  frame: `x = s·(y_raw − 41.75)`, `y = s·(x_raw − 25)`, where `s` is −1 for the
  home team and +1 otherwise, with no flip at halftime. Raw values are kept
  alongside as `coordinate_x_raw` / `coordinate_y_raw`.
- **`notes_headline` is `""` when absent**, not null — `build_postseason.py`
  keys off it.
- **DNP players carry null stats**, not zeros, matching the cache.
- `sdv`'s play `id` is a float64 and loses precision past 2^53; it's reproduced
  as-is for compatibility, with the exact value alongside as `play_id` (string).

## Not done yet

- Past-season backfill hasn't been run — only 2026 is ingested.
- The duplicate site isn't wired to a deploy target; it's local-only.
- No nightly automation yet. The root nightly action is untouched.
