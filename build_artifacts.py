"""
build_artifacts.py — Train the predictor once and dump everything the website
needs to load instantly.

WHY THIS FILE EXISTS
--------------------
The Streamlit app and the static site both need efficiency ratings, a trained
spread model, and a win-probability calibrator.  Training on a live call at
request time would take 30+ seconds and require network access to ESPN.
Instead, this script runs nightly in GitHub Actions, saves the results as
small data-only files, and the web app reads those files instantly.

Outputs (written into artifacts/{SEASON}/):
    model.json          Model coefficients + isotonic calibration thresholds
                        (everything needed to reproduce predict_game without sklearn).
    teams.parquet       Per-team efficiency, pace, home-court, and form ratings.
    net_ratings.parquet Public leaderboard: one row per D-I team with rank, record, SOS.
    metadata.json       Build info, backtest metrics, season + calibration years.

Run:
    python build_artifacts.py
    RATINGS_ONLY=1 python build_artifacts.py     # skip model training (for historical builds)
    OVERRIDE_SEASON=2025 python build_artifacts.py
"""

import json
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

import sportsdataverse.mbb.mbb_loaders as mbb_loaders
from predictor import TempoPredictor

import os as _os

# ---------------------------------------------------------------------------
# Season detection
# ---------------------------------------------------------------------------
_season_override = _os.environ.get("OVERRIDE_SEASON")
if _season_override:
    _season = int(_season_override)
else:
    _today = _date.today()
    _season = _today.year + 1 if _today.month >= 11 else _today.year

# Set RATINGS_ONLY=1 to skip model training (faster for historical seasons
# where we don't need a fresh predictor, just the ratings table).
RATINGS_ONLY = bool(_os.environ.get("RATINGS_ONLY"))

ARTIFACTS_DIR        = Path(__file__).parent / "artifacts" / str(_season)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# All fully completed seasons before the current one — used to train the model.
CALIBRATION_YEARS    = list(range(2021, _season))
# Most recently completed season — used as the held-out backtest.
BACKTEST_YEAR        = _season - 1
# Teams with fewer than this many games are excluded from the ratings table.
MIN_GAMES_TO_QUALIFY = 5


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_net_ratings_table(trained_predictor):
    """Assemble the public net-ratings leaderboard from a loaded predictor.

    Columns in the output:
        rank, team, wins, losses, net_eff, off_eff, def_eff,
        off_rank, def_rank, sos, sos_rank, pace, home_court, form, games, team_id

    Strength of schedule (SOS) = mean adjusted net efficiency of opponents faced.
    home_court is scaled by the spread model's home-court coefficient so the
    displayed value represents the actual point contribution in a prediction.
    """
    current_season = trained_predictor.current_season
    season_team_game_df = mbb_loaders.load_mbb_team_boxscore(seasons=[current_season]).to_pandas()

    # Win/loss record per team.
    team_record_df = (
        season_team_game_df.groupby('team_id')
        .agg(wins=('team_winner', 'sum'), games=('team_winner', 'count'))
        .reset_index()
    )
    team_record_df['losses'] = team_record_df['games'] - team_record_df['wins']

    # Strength of schedule: average net efficiency of opponents.
    net_eff_by_team_id = trained_predictor.current_efficiency['net_eff']
    games_vs_rated_opponents = season_team_game_df[
        season_team_game_df['opponent_team_id'].isin(net_eff_by_team_id.index)
    ].copy()
    games_vs_rated_opponents['opponent_net_eff'] = games_vs_rated_opponents['opponent_team_id'].map(net_eff_by_team_id)
    sos_per_team = (
        games_vs_rated_opponents.groupby('team_id')['opponent_net_eff']
        .mean().rename('sos').reset_index()
    )

    # Start from the efficiency ratings DataFrame, add supplemental columns.
    efficiency_df = trained_predictor.current_efficiency.reset_index()
    efficiency_df['team']       = efficiency_df['team_id'].map(trained_predictor.team_id_to_name)
    efficiency_df['pace']       = efficiency_df['team_id'].map(
        lambda tid: trained_predictor.current_pace.get(tid, np.nan)
    )

    # Scale raw home-court advantage by the spread model coefficient so the
    # displayed value = points contributed to a prediction.
    # In ratings-only mode (no trained model) fall back to a neutral scale of 1.0.
    home_court_spread_coef = (
        float(trained_predictor.spread_regression_model.coef_[1])
        if trained_predictor.spread_regression_model is not None
        else 1.0
    )
    efficiency_df['home_court'] = efficiency_df['team_id'].map(
        lambda tid: trained_predictor.home_adv.get(tid, np.nan)
    ) * home_court_spread_coef
    efficiency_df['form'] = efficiency_df['team_id'].map(
        lambda tid: trained_predictor.current_form.get(tid, 0.0)
    )

    net_ratings_df = (
        efficiency_df
        .merge(team_record_df[['team_id', 'wins', 'losses', 'games']], on='team_id', how='left')
        .merge(sos_per_team, on='team_id', how='left')
    )
    net_ratings_df = net_ratings_df[net_ratings_df['games'].fillna(0) >= MIN_GAMES_TO_QUALIFY].copy()
    net_ratings_df = net_ratings_df.dropna(subset=['team'])

    net_ratings_df = net_ratings_df.sort_values('net_eff', ascending=False).reset_index(drop=True)
    net_ratings_df['rank']     = np.arange(1, len(net_ratings_df) + 1)
    net_ratings_df['off_rank'] = net_ratings_df['off_eff'].rank(ascending=False, method='min').astype(int)
    net_ratings_df['def_rank'] = net_ratings_df['def_eff'].rank(ascending=True,  method='min').astype(int)
    net_ratings_df['sos_rank'] = net_ratings_df['sos'].rank(ascending=False, method='min').astype(int)

    final_columns = [
        'rank', 'team', 'wins', 'losses', 'net_eff', 'off_eff', 'def_eff',
        'off_rank', 'def_rank', 'sos', 'sos_rank', 'pace', 'home_court', 'form',
        'games', 'team_id'
    ]
    return net_ratings_df[final_columns].round({
        'net_eff': 1, 'off_eff': 1, 'def_eff': 1,
        'sos': 1, 'pace': 1, 'home_court': 1, 'form': 1
    })


def build_teams_table(trained_predictor):
    """Per-team raw inputs the runtime predictor needs (raw, unscaled).

    The web app reads this parquet and uses the raw home_adv (not scaled by
    the spread coefficient) so that model_runtime.py can apply the coefficient
    itself at prediction time.
    """
    efficiency_df = trained_predictor.current_efficiency.reset_index()
    efficiency_df['team']     = efficiency_df['team_id'].map(trained_predictor.team_id_to_name)
    efficiency_df['pace']     = efficiency_df['team_id'].map(
        lambda tid: trained_predictor.current_pace.get(tid, trained_predictor._league_avg_pace)
    )
    efficiency_df['home_adv'] = efficiency_df['team_id'].map(
        lambda tid: trained_predictor.home_adv.get(tid, trained_predictor._league_home_adv)
    )
    efficiency_df['form'] = efficiency_df['team_id'].map(
        lambda tid: trained_predictor.current_form.get(tid, 0.0)
    )
    efficiency_df = efficiency_df.dropna(subset=['team'])
    return efficiency_df[['team_id', 'team', 'off_eff', 'def_eff', 'net_eff',
                           'pace', 'home_adv', 'form']]


def build_model_json(trained_predictor, backtest_metrics):
    """Serialize everything model_runtime.py needs to reproduce predict_game.

    Stores the regression coefficients and isotonic regression lookup tables
    as plain lists so the web app never needs to import sklearn or pickle.
    """
    return {
        "built_at":          pd.Timestamp.now().isoformat(),
        "season":            int(trained_predictor.current_season),
        "calibration_years": CALIBRATION_YEARS,
        "form_mode":         trained_predictor.FORM_MODE,
        "form_games":        trained_predictor.FORM_GAMES,
        "league_avg_pace":   float(trained_predictor._league_avg_pace),
        "league_home_adv":   float(trained_predictor._league_home_adv),
        # Linear regression coefficients: [tempo_adj, home_court, form_diff]
        "coef":              [float(c) for c in trained_predictor.spread_regression_model.coef_],
        "intercept":         float(trained_predictor.spread_regression_model.intercept_),
        # Isotonic regression lookup: np.interp(spread, iso_x, iso_y) → P(win)
        "iso_x":             [float(x) for x in trained_predictor.calibrator.X_thresholds_],
        "iso_y":             [float(y) for y in trained_predictor.calibrator.y_thresholds_],
        "backtest_year":     BACKTEST_YEAR,
        "backtest":          backtest_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    build_mode_label = "ratings-only" if RATINGS_ONLY else "full"
    print("=" * 60)
    print(f"Building predictor artifacts  [{build_mode_label}]")
    print("=" * 60)

    trained_predictor = TempoPredictor()

    if not RATINGS_ONLY:
        trained_predictor.train(calibration_year=CALIBRATION_YEARS)

    trained_predictor._build_ratings(_season)

    backtest_metrics = {}
    if not RATINGS_ONLY and BACKTEST_YEAR is not None:
        print(f"\nBacktesting on {BACKTEST_YEAR} (held out)...")
        validation_results, _, _ = trained_predictor.validate_walk_forward(BACKTEST_YEAR)
        backtest_metrics = {metric_name: float(metric_val)
                            for metric_name, metric_val in validation_results.items()}
        print(
            f"   accuracy={validation_results['accuracy']:.4f}  "
            f"logloss={validation_results['logloss']:.4f}  "
            f"auc={validation_results['auc']:.4f}  "
            f"spread_mae={validation_results['spread_mae']:.2f}"
        )

    # --- Write teams.parquet ---
    teams_ratings_df = build_teams_table(trained_predictor)
    teams_ratings_df.to_parquet(ARTIFACTS_DIR / "teams.parquet", index=False)
    print(f"\nWrote {ARTIFACTS_DIR / 'teams.parquet'}  ({len(teams_ratings_df)} teams)")

    # --- Write model.json ---
    if not RATINGS_ONLY:
        model_json_dict = build_model_json(trained_predictor, backtest_metrics)
        with open(ARTIFACTS_DIR / "model.json", "w") as model_file:
            json.dump(model_json_dict, model_file, indent=2)
        print(f"Wrote {ARTIFACTS_DIR / 'model.json'}")

    # --- Write net_ratings.parquet ---
    net_ratings_df = build_net_ratings_table(trained_predictor)
    net_ratings_df.to_parquet(ARTIFACTS_DIR / "net_ratings.parquet", index=False)
    print(f"Wrote {ARTIFACTS_DIR / 'net_ratings.parquet'}  ({len(net_ratings_df)} teams)")

    # --- Write metadata.json ---
    build_metadata = {
        "built_at":          pd.Timestamp.now().isoformat(),
        "season":            int(trained_predictor.current_season),
        "calibration_years": CALIBRATION_YEARS,
        "n_teams":           int(len(net_ratings_df)),
        "form_mode":         trained_predictor.FORM_MODE,
        "form_games":        trained_predictor.FORM_GAMES,
        "league_avg_pace":   round(trained_predictor._league_avg_pace, 2),
        "league_home_adv":   round(trained_predictor._league_home_adv, 2),
        "backtest_year":     BACKTEST_YEAR,
        "backtest":          backtest_metrics,
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w") as metadata_file:
        json.dump(build_metadata, metadata_file, indent=2)
    print(f"Wrote {ARTIFACTS_DIR / 'metadata.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()


# ===========================================================================
# VARIABLE GLOSSARY
# ===========================================================================
#
# _season                      int     Calendar year the season ends (e.g. 2026).
# _season_override             str     Value of OVERRIDE_SEASON env var; None if not set.
# RATINGS_ONLY                 bool    If True, skip model training (faster for historical builds).
# ARTIFACTS_DIR                Path    Directory where output files are written: artifacts/{season}/.
# CALIBRATION_YEARS            list    All complete seasons before the current one; training data.
# BACKTEST_YEAR                int     The most recently completed season, held out to measure accuracy.
# MIN_GAMES_TO_QUALIFY         int     Teams below this game count are excluded from net_ratings.
#
# --- build_net_ratings_table() ---
# current_season               int     Season number from the loaded predictor.
# season_team_game_df          DataFrame  Every team-game box score for the current season.
# team_record_df               DataFrame  Wins, losses, games played per team.
# net_eff_by_team_id           Series   Indexed by team_id; net efficiency for SOS computation.
# games_vs_rated_opponents     DataFrame  Rows where the opponent has a net-eff rating.
# sos_per_team                 DataFrame  Mean opponent net-eff per team (strength of schedule).
# efficiency_df                DataFrame  Base efficiency ratings from the predictor.
# home_court_spread_coef       float    Regression coefficient for home_court; scales raw adv → pts.
# net_ratings_df               DataFrame  Full merged table before filtering and ranking.
# final_columns                list     Column order for the output parquet.
#
# --- build_teams_table() ---
# efficiency_df                DataFrame  Efficiency ratings with pace/home_adv/form columns added.
#   home_adv                   float    RAW home-court advantage (not scaled by coef).
#
# --- build_model_json() ---
# model_json_dict              dict     JSON-serializable representation of the trained model.
#   coef                       list[float]  [tempo_adj_coef, home_court_coef, form_diff_coef].
#   intercept                  float    Regression intercept (typically near 0).
#   iso_x                      list[float]  Predicted margin thresholds for the isotonic curve.
#   iso_y                      list[float]  Win-probability values at each threshold.
#   backtest_year              int      Year used to compute the held-out accuracy metrics.
#   backtest                   dict     {accuracy, logloss, auc, spread_mae, spread_rmse}.
#
# --- main() ---
# trained_predictor            TempoPredictor  Fully loaded predictor with ratings + model.
# backtest_metrics             dict     Validation metrics, empty dict in RATINGS_ONLY mode.
# validation_results           dict     Output of validate_walk_forward().
# teams_ratings_df             DataFrame  Output of build_teams_table(); written to parquet.
# net_ratings_df               DataFrame  Output of build_net_ratings_table(); written to parquet.
# model_json_dict              dict     Output of build_model_json(); written to model.json.
# build_metadata               dict     Build info written to metadata.json.
