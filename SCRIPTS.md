# Hog Charts — Python Script Reference

Every Python file in the project, grouped by role.

---

## 1. Data Pipeline — Nightly Build (GitHub Actions)

These scripts run automatically each night via `.github/workflows/nightly-build.yml`.
They fetch the latest ESPN data and regenerate all site artifacts.

| Script | Purpose |
|--------|---------|
| `build_onoff_rapm.py` | Core analytics engine. Reconstructs lineups from play-by-play, computes on/off net-rating splits, and runs Ridge regression (RAPM) to isolate each player's individual contribution to net efficiency. Outputs `mbb_onoff_{SEASON}_v2.csv` and `mbb_rapm_{SEASON}.csv`. |
| `build_player_stats.py` | Aggregates per-game player box scores to season totals + per-game averages. Outputs `player_stats_{SEASON}.csv` and `player_stats_conf_{SEASON}.csv`. |
| `build_lineups.py` | Generates 1/2/3/5-man lineup combo stats for every team (overall + conference). Outputs 8 CSVs named `{n}_man_{scope}_stats_{SEASON}.csv`. |
| `build_shots_data.py` | Filters the full play-by-play to shooting plays only and saves `shots_{SEASON}.parquet` + `box_{SEASON}.parquet`. Keeps the cloud app free of large data downloads. |
| `build_artifacts.py` | Trains the `TempoPredictor` spread model on past seasons, runs the iterative adjusted-efficiency algorithm, and saves `model.json`, `teams.parquet`, `net_ratings.parquet`, and `metadata.json` to `artifacts/{SEASON}/`. |
| `build_site.py` | Master export step. Reads all parquet/CSV artifacts and writes compact JSON files into `site/data/{SEASON}/` for the static HTML site and the Streamlit app to consume. |

---

## 2. Prediction Model

| Script | Purpose |
|--------|---------|
| `predictor.py` | Defines the three predictor classes. `BasePredictor` loads data and computes iterative opponent-adjusted efficiency ratings. `IsotonicPredictor` adds isotonic calibration (maps raw score spread → win probability). `TempoPredictor` adds the pace-adjusted linear spread model and walk-forward backtesting. |
| `model_runtime.py` | sklearn-free runtime predictor for the deployed cloud app. Loads `model.json` + `teams.parquet` and reproduces `predict_game()` using only numpy — no pickle, no version skew. |

---

## 3. Visualization

| Script | Purpose |
|--------|---------|
| `mbb_viz.py` | Matplotlib zone-efficiency shot charts. `MBBZoneEfficiencyVisualizer` classifies shots into 12 named zones (At Rim, Paint, Mid-Range, 3PT corners/wings/center) and renders zone FG% labels on an NCAA half-court. Also provides `get_team_zone_leaders()` used by the Plotly layer. |
| `shot_charts_plotly.py` | Interactive Plotly shot charts for the Streamlit app. Thin wrapper around `mbb_viz.py`'s data methods; replaces matplotlib rendering with `go.Figure` (scatter, density, zone, territory). |

---

## 4. Streamlit App

| Script | Purpose |
|--------|---------|
| `app.py` | Streamlit entry point. Wires together all pages: Game Predictor, Net Ratings, Player Stats, Shot Charts, Lineup Stats, and Player Impact. Loads runtime artifacts via `model_runtime.py` and visualization via `shot_charts_plotly.py`. |
| `ui.py` | Shared CSS/HTML components: page header, color theme variables, card layouts. Imported by `app.py` to keep styling consistent across all pages. |

---

## 5. Supplemental / One-Time Scripts

These run locally (not in CI) to build data that doesn't need nightly updates.

| Script | Purpose |
|--------|---------|
| `build_history.py` | Orchestrates historical season builds (2016–present). Loops over past seasons, calling `build_artifacts.py → build_player_stats.py → build_lineups.py → build_onoff_rapm.py → build_site.py` with `OVERRIDE_SEASON` and `RATINGS_ONLY=1` set. |
| `build_synergy.py` | Computes two-man WOWY ("With Or Without You") synergy: net rating when a duo is together vs. apart. Uses the WOWY identity `apart(A,B) = on(A) + on(B) − 2·both(A,B)`. Outputs `site/data/{SEASON}/synergy/{slug}.json`. |
| `build_conf_stats.py` | Standalone script to regenerate conference-only player stats for a single season. Functionally equivalent to the conference branch inside `build_player_stats.py`. |
| `impact_artifact.py` | Builds `player-impact.json` by joining RAPM ratings with on/off splits into one per-player impact record. Also imported by `build_site.py` so it runs in the nightly build. |
| `patch_conferences.py` | Backfills conference labels in historical season CSVs and JSON files. Sources labels from ESPN's standings API; votes 2026 site-style labels (e.g. "SEC") onto each ESPN conference ID so historical data matches current style. |

---

## Key Data Flow

```
sportsdataverse (ESPN API)
        │
        ▼
build_onoff_rapm.py ──► mbb_onoff_{SEASON}_v2.csv
build_player_stats.py ──► player_stats_{SEASON}.csv
build_lineups.py ──────► {n}_man_{scope}_stats_{SEASON}.csv
build_shots_data.py ───► shots_{SEASON}.parquet
        │
        ▼
build_artifacts.py ────► artifacts/{SEASON}/{model.json, teams.parquet, net_ratings.parquet}
        │
        ▼
build_site.py ─────────► site/data/{SEASON}/{predictor.json, player-stats.json, ...}
        │
        ▼
site/index.html (static) ◄── JavaScript fetch()
app.py (Streamlit) ────── model_runtime.py + shot_charts_plotly.py
```

---

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `OVERRIDE_SEASON` | auto (Nov rule) | Force a specific season year (e.g. `2025`). |
| `RATINGS_ONLY` | `0` | Set to `1` in `build_artifacts.py` to skip model training (faster historical builds). |
